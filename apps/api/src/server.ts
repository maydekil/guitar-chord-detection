import { mkdir } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

import type { SaveSongAnalysisRequest } from "@gcd/shared/library";
import { cleanupApiAssets, saveUploadedAudioAsset } from "./audioAssets.js";
import { streamAudioFile, streamSongAudio } from "./audioStreaming.js";
import { createEngineCommands } from "./engineCommands.js";
import type { EngineJobRequest, EngineJobStore } from "./engineJobs.js";
import { createEngineJobStore, initializeEngineJobTables, markInterruptedJobs } from "./engineJobs.js";
import { buildChordSheetExport, buildLrcExport } from "./exportFormats.js";
import { readJsonBody, safeFileName, sendJson } from "./httpUtils.js";
import { normalizeAssetCleanupMaxAgeMs, normalizeTransposeSemitones } from "./queryParams.js";
import { deleteSong, getSong, initializeSongTables, listSongs, saveSong } from "./songStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const port = Number.parseInt(process.env.PORT ?? "8787", 10);
const dataDir = process.env.GCD_API_DATA_DIR ?? path.resolve(__dirname, "..", ".data");
const audioDir = path.join(dataDir, "audio");
const engineOutputDir = path.join(dataDir, "engine-output");
const assetCleanupMaxAgeMs = normalizeAssetCleanupMaxAgeMs(process.env.GCD_ASSET_CLEANUP_MAX_AGE_HOURS);
const repositoryRoot = process.env.GCD_REPOSITORY_ROOT ?? path.resolve(__dirname, "../../..");
const engineCwd = path.join(repositoryRoot, "engine");
const pythonExecutable = process.env.GCD_PYTHON ?? path.join(repositoryRoot, ".venv", "bin", "python");
const publicBaseUrl = (process.env.GCD_API_PUBLIC_URL ?? `http://localhost:${port}`).replace(/\/+$/, "");

await mkdir(dataDir, { recursive: true });
await mkdir(audioDir, { recursive: true });
await mkdir(engineOutputDir, { recursive: true });

const database = new DatabaseSync(path.join(dataDir, "guitar-chord-detection.sqlite"));
initializeDatabase(database);
markInterruptedJobs(database);

let engineJobStore: EngineJobStore;
const engineCommands = createEngineCommands({
    pythonExecutable,
    engineCwd,
    engineOutputDir,
    audioDir,
    publicBaseUrl,
    processRegistry: {
        register(jobId, child) {
            engineJobStore.registerProcess(jobId, child);
        },
        unregister(jobId) {
            engineJobStore.unregisterProcess(jobId);
        },
        isCancelled(jobId) {
            return engineJobStore.isCancelled(jobId);
        }
    }
});
engineJobStore = createEngineJobStore(database, engineCommands);

const server = http.createServer((request, response) => {
    void handleRequest(request, response).catch((error) => {
        console.error("[api] request failed", error);
        sendJson(response, 500, { message: "Internal server error" });
    });
});

server.listen(port, () => {
    console.log(`[api] listening on http://localhost:${port}`);
});
void runAssetCleanup().catch((error: unknown) => console.error("[api] asset cleanup failed", error));
setInterval(() => {
    void runAssetCleanup().catch((error: unknown) => console.error("[api] asset cleanup failed", error));
}, 60 * 60_000).unref();

async function handleRequest(request: http.IncomingMessage, response: http.ServerResponse): Promise<void> {
    const url = new URL(request.url ?? "/", publicBaseUrl);
    const method = request.method ?? "GET";

    if (method === "GET" && url.pathname === "/health") {
        sendJson(response, 200, { ok: true });
        return;
    }

    if (method === "GET" && url.pathname === "/songs") {
        sendJson(response, 200, listSongs(database, url));
        return;
    }

    if (method === "POST" && url.pathname === "/audio/upload") {
        const upload = await saveUploadedAudioAsset(request, url, audioDir, publicBaseUrl);
        sendJson(response, 201, upload);
        return;
    }

    if (method === "POST" && url.pathname === "/assets/cleanup") {
        sendJson(response, 200, await runAssetCleanup());
        return;
    }

    if (method === "POST" && url.pathname === "/jobs") {
        const body = await readJsonBody<EngineJobRequest>(request);
        sendJson(response, 202, engineJobStore.createEngineJob(body));
        return;
    }

    const jobMatch = /^\/jobs\/([^/]+)$/.exec(url.pathname);
    if (jobMatch && method === "GET") {
        const job = engineJobStore.getEngineJob(decodeURIComponent(jobMatch[1]));
        if (!job) {
            sendJson(response, 404, { message: "Job not found" });
            return;
        }
        sendJson(response, 200, engineJobStore.toEngineJobSnapshot(job));
        return;
    }

    const cancelJobMatch = /^\/jobs\/([^/]+)\/cancel$/.exec(url.pathname);
    if (cancelJobMatch && method === "POST") {
        const job = engineJobStore.cancelEngineJob(decodeURIComponent(cancelJobMatch[1]));
        if (!job) {
            sendJson(response, 404, { message: "Job not found" });
            return;
        }
        sendJson(response, 200, engineJobStore.toEngineJobSnapshot(job));
        return;
    }

    if (method === "POST" && url.pathname === "/songs") {
        const body = await readJsonBody<SaveSongAnalysisRequest>(request);
        sendJson(response, 201, saveSong(database, body, publicBaseUrl));
        return;
    }

    const songMatch = /^\/songs\/([^/]+)$/.exec(url.pathname);
    if (songMatch && method === "GET") {
        const song = getSong(database, decodeURIComponent(songMatch[1]));
        if (!song) {
            sendJson(response, 404, { message: "Song not found" });
            return;
        }
        sendJson(response, 200, song);
        return;
    }

    if (songMatch && method === "DELETE") {
        const deleted = deleteSong(database, decodeURIComponent(songMatch[1]), audioDir);
        sendJson(response, 200, { deleted });
        return;
    }

    const exportMatch = /^\/songs\/([^/]+)\/export$/.exec(url.pathname);
    if (exportMatch && method === "GET") {
        exportSong(
            response,
            decodeURIComponent(exportMatch[1]),
            url.searchParams.get("format") ?? "txt",
            normalizeTransposeSemitones(Number.parseInt(url.searchParams.get("transpose") ?? "0", 10))
        );
        return;
    }

    const streamMatch = /^\/songs\/([^/]+)\/audio\/(original|instrumental)\/stream$/.exec(url.pathname);
    if (streamMatch && method === "GET") {
        await streamSongAudio(
            request,
            response,
            decodeURIComponent(streamMatch[1]),
            streamMatch[2] as "original" | "instrumental",
            (id) => getSong(database, id)
        );
        return;
    }

    const uploadedAudioStreamMatch = /^\/audio\/([a-f0-9]{64})(\.[a-z0-9]+)\/stream$/.exec(url.pathname);
    if (uploadedAudioStreamMatch && method === "GET") {
        await streamAudioFile(request, response, path.join(audioDir, `${uploadedAudioStreamMatch[1]}${uploadedAudioStreamMatch[2]}`));
        return;
    }

    if (method === "POST" && url.pathname === "/analysis") {
        sendJson(response, 200, await engineCommands.analyzeAudio(await readJsonBody(request)));
        return;
    }

    if (method === "POST" && url.pathname === "/lyrics/transcribe") {
        sendJson(response, 200, await engineCommands.transcribeLyrics(await readJsonBody(request)));
        return;
    }

    if (method === "POST" && url.pathname === "/vocals/remove") {
        sendJson(response, 200, await engineCommands.removeVocals(await readJsonBody(request)));
        return;
    }

    if (method === "POST" && url.pathname === "/audio/pitch-shift") {
        sendJson(response, 200, await engineCommands.pitchShiftAudio(await readJsonBody(request)));
        return;
    }

    sendJson(response, 404, { message: "Not found" });
}

function exportSong(response: http.ServerResponse, id: string, format: string, transposeSemitones: number): void {
    const song = getSong(database, id);
    if (!song) {
        sendJson(response, 404, { message: "Song not found" });
        return;
    }

    const safeFormat = format === "lrc" ? "lrc" : "txt";
    const content = safeFormat === "lrc" ? buildLrcExport(song, transposeSemitones) : buildChordSheetExport(song, transposeSemitones);
    const fileName = `${safeFileName(song.artist)}-${safeFileName(song.title)}.${safeFormat}`;

    response.writeHead(200, {
        "content-disposition": `attachment; filename="${fileName}"`,
        "content-type": "text/plain; charset=utf-8"
    });
    response.end(content);
}

async function runAssetCleanup() {
    return await cleanupApiAssets(database, audioDir, engineOutputDir, assetCleanupMaxAgeMs);
}

function initializeDatabase(db: DatabaseSync): void {
    initializeSongTables(db);
    initializeEngineJobTables(db);
}
