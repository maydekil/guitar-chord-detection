import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdir, readdir, readFile, rm, stat, unlink, writeFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { SaveSongAnalysisRequest } from "@gcd/shared/library";
import { buildChordSheetExport, buildLrcExport } from "./exportFormats.js";
import { isAlreadyExistsError, parseRangeHeader, readJsonBody, readRawBody, resolveAudioContentType, resolveSafeAudioExtension, safeFileName, sendJson } from "./httpUtils.js";
import { collectReferencedAudioPaths, deleteSong, getSong, initializeSongTables, listSongs, saveSong } from "./songStore.js";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

interface AnalyzeAudioRequest {
    audioPath: string;
    forceRefresh?: boolean;
}

interface GenerateLyricsRequest {
    audioPath: string;
    model?: string;
}

interface RemoveVocalsRequest {
    audioPath: string;
}

interface PitchShiftAudioRequest {
    audioPath: string;
    semitones: number;
}

interface EngineCommandOptions {
    timeoutMs: number;
    env?: NodeJS.ProcessEnv;
    jobId?: string;
}

type EngineJobKind = "analysis" | "lyrics" | "vocals" | "pitch-shift";
type EngineJobStatus = "queued" | "running" | "succeeded" | "failed";

interface EngineJobRequest {
    kind: EngineJobKind;
    payload: unknown;
}

interface EngineJob {
    id: string;
    kind: EngineJobKind;
    status: EngineJobStatus;
    progress: number;
    result: unknown;
    error: { code: string; message: string } | null;
    createdAt: string;
    updatedAt: string;
}

interface EngineJobRow {
    id: string;
    kind: string;
    status: string;
    progress: number;
    result_json: string | null;
    error_json: string | null;
    created_at: string;
    updated_at: string;
}

interface AssetCleanupResult {
    deletedAudioFiles: number;
    deletedEngineOutputFiles: number;
    skippedReferencedFiles: number;
    reclaimedBytes: number;
}

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
const engineJobs = new Map<string, EngineJob>();
const activeEngineProcesses = new Map<string, ChildProcessWithoutNullStreams>();
const cancelledEngineJobs = new Set<string>();

const server = http.createServer((request, response) => {
    void handleRequest(request, response).catch((error) => {
        console.error("[api] request failed", error);
        sendJson(response, 500, { message: "Internal server error" });
    });
});

server.listen(port, () => {
    console.log(`[api] listening on http://localhost:${port}`);
});
void cleanupApiAssets().catch((error: unknown) => console.error("[api] asset cleanup failed", error));
setInterval(() => {
    void cleanupApiAssets().catch((error: unknown) => console.error("[api] asset cleanup failed", error));
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
        const upload = await saveUploadedAudio(request, url);
        sendJson(response, 201, upload);
        return;
    }

    if (method === "POST" && url.pathname === "/assets/cleanup") {
        sendJson(response, 200, await cleanupApiAssets());
        return;
    }

    if (method === "POST" && url.pathname === "/jobs") {
        const body = await readJsonBody<EngineJobRequest>(request);
        sendJson(response, 202, createEngineJob(body));
        return;
    }

    const jobMatch = /^\/jobs\/([^/]+)$/.exec(url.pathname);
    if (jobMatch && method === "GET") {
        const job = getEngineJob(decodeURIComponent(jobMatch[1]));
        if (!job) {
            sendJson(response, 404, { message: "Job not found" });
            return;
        }
        sendJson(response, 200, toEngineJobSnapshot(job));
        return;
    }

    const cancelJobMatch = /^\/jobs\/([^/]+)\/cancel$/.exec(url.pathname);
    if (cancelJobMatch && method === "POST") {
        const job = cancelEngineJob(decodeURIComponent(cancelJobMatch[1]));
        if (!job) {
            sendJson(response, 404, { message: "Job not found" });
            return;
        }
        sendJson(response, 200, toEngineJobSnapshot(job));
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
        await streamSongAudio(request, response, decodeURIComponent(streamMatch[1]), streamMatch[2] as "original" | "instrumental");
        return;
    }

    const uploadedAudioStreamMatch = /^\/audio\/([a-f0-9]{64})(\.[a-z0-9]+)\/stream$/.exec(url.pathname);
    if (uploadedAudioStreamMatch && method === "GET") {
        await streamAudioFile(request, response, path.join(audioDir, `${uploadedAudioStreamMatch[1]}${uploadedAudioStreamMatch[2]}`));
        return;
    }

    if (method === "POST" && url.pathname === "/analysis") {
        const body = await readJsonBody<AnalyzeAudioRequest>(request);
        sendJson(response, 200, await analyzeAudio(body));
        return;
    }

    if (method === "POST" && url.pathname === "/lyrics/transcribe") {
        const body = await readJsonBody<GenerateLyricsRequest>(request);
        sendJson(response, 200, await transcribeLyrics(body));
        return;
    }

    if (method === "POST" && url.pathname === "/vocals/remove") {
        const body = await readJsonBody<RemoveVocalsRequest>(request);
        sendJson(response, 200, await removeVocals(body));
        return;
    }

    if (method === "POST" && url.pathname === "/audio/pitch-shift") {
        const body = await readJsonBody<PitchShiftAudioRequest>(request);
        sendJson(response, 200, await pitchShiftAudio(body));
        return;
    }

    sendJson(response, 404, { message: "Not found" });
}

async function streamSongAudio(request: http.IncomingMessage, response: http.ServerResponse, id: string, kind: "original" | "instrumental"): Promise<void> {
    const song = getSong(database, id);
    if (!song) {
        sendJson(response, 404, { message: "Song not found" });
        return;
    }

    const audioPath = kind === "original" ? song.audioPath : song.instrumentalAudioPath;
    if (!audioPath) {
        sendJson(response, 404, { message: "Audio stream not found" });
        return;
    }

    const fileStat = await stat(audioPath).catch(() => null);
    if (!fileStat) {
        sendJson(response, 404, { message: "Audio file not found" });
        return;
    }

    await streamAudioFile(request, response, audioPath);
}

async function streamAudioFile(request: http.IncomingMessage, response: http.ServerResponse, audioPath: string): Promise<void> {
    const fileStat = await stat(audioPath).catch(() => null);
    if (!fileStat) {
        sendJson(response, 404, { message: "Audio file not found" });
        return;
    }

    const range = parseRangeHeader(request.headers.range, fileStat.size);
    if (range) {
        response.writeHead(206, {
            "accept-ranges": "bytes",
            "content-length": range.end - range.start + 1,
            "content-range": `bytes ${range.start}-${range.end}/${fileStat.size}`,
            "content-type": resolveAudioContentType(audioPath)
        });
        createReadStream(audioPath, { start: range.start, end: range.end }).pipe(response);
        return;
    }

    response.writeHead(200, {
        "accept-ranges": "bytes",
        "content-length": fileStat.size,
        "content-type": resolveAudioContentType(audioPath)
    });
    createReadStream(audioPath).pipe(response);
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

async function saveUploadedAudio(request: http.IncomingMessage, url: URL): Promise<{ audioPath: string; audioStreamUrl: string; fileHash: string }> {
    const body = await readRawBody(request);
    if (body.length === 0) {
        throw new Error("Uploaded audio is empty");
    }

    const fileHash = createHash("sha256").update(body).digest("hex");
    const originalFileName = url.searchParams.get("filename") ?? "audio";
    const extension = resolveSafeAudioExtension(originalFileName);
    const audioPath = path.join(audioDir, `${fileHash}${extension}`);

    await writeFile(audioPath, body, { flag: "wx" }).catch(async (error: unknown) => {
        if (!isAlreadyExistsError(error)) {
            throw error;
        }
    });

    return {
        audioPath,
        audioStreamUrl: `${publicBaseUrl}/audio/${encodeURIComponent(fileHash)}${extension}/stream`,
        fileHash
    };
}

function createEngineJob(request: EngineJobRequest): EngineJob {
    const now = new Date().toISOString();
    const job: EngineJob = {
        id: randomUUID(),
        kind: request.kind,
        status: "queued",
        progress: 0,
        result: null,
        error: null,
        createdAt: now,
        updatedAt: now
    };

    engineJobs.set(job.id, job);
    persistEngineJob(job);
    void runEngineJob(job, request.payload);
    return toEngineJobSnapshot(job);
}

async function runEngineJob(job: EngineJob, payload: unknown): Promise<void> {
    updateEngineJob(job, { status: "running", progress: 5 });
    const progressTimer = startEstimatedProgress(job);

    try {
        const result = await executeEngineJob(job, payload);
        if (result && typeof result === "object" && "error" in result) {
            const error = (result as { error: { code?: unknown; message?: unknown } }).error;
            updateEngineJob(job, {
                status: "failed",
                progress: 100,
                result,
                error: {
                    code: String(error.code ?? "ENGINE_JOB_FAILED"),
                    message: String(error.message ?? "Engine job failed")
                }
            });
            return;
        }

        updateEngineJob(job, {
            status: "succeeded",
            progress: 100,
            result,
            error: null
        });
    } catch (error) {
        updateEngineJob(job, {
            status: "failed",
            progress: 100,
            result: null,
            error: {
                code: "ENGINE_JOB_EXCEPTION",
                message: error instanceof Error ? error.message : "Engine job failed"
            }
        });
    } finally {
        clearInterval(progressTimer);
        activeEngineProcesses.delete(job.id);
        cancelledEngineJobs.delete(job.id);
    }
}

async function executeEngineJob(job: EngineJob, payload: unknown): Promise<unknown> {
    if (job.kind === "analysis") {
        return await analyzeAudio(payload as AnalyzeAudioRequest, job.id);
    }
    if (job.kind === "lyrics") {
        return await transcribeLyrics(payload as GenerateLyricsRequest, job.id);
    }
    if (job.kind === "vocals") {
        return await removeVocals(payload as RemoveVocalsRequest, job.id);
    }
    if (job.kind === "pitch-shift") {
        return await pitchShiftAudio(payload as PitchShiftAudioRequest, job.id);
    }
    return enginePlaceholder("ENGINE_JOB_UNKNOWN_KIND", "Unknown engine job kind");
}

function cancelEngineJob(id: string): EngineJob | null {
    const job = getEngineJob(id);
    if (!job) {
        return null;
    }
    if (job.status !== "queued" && job.status !== "running") {
        return job;
    }

    cancelledEngineJobs.add(id);
    const child = activeEngineProcesses.get(id);
    if (child) {
        child.kill("SIGTERM");
        setTimeout(() => {
            if (child.exitCode === null && child.signalCode === null) {
                child.kill("SIGKILL");
            }
        }, 1200);
    }
    updateEngineJob(job, {
        status: "failed",
        progress: 100,
        error: {
            code: "ENGINE_JOB_CANCELLED",
            message: "Engine job cancelled"
        }
    });
    engineJobs.set(job.id, job);
    return job;
}

function startEstimatedProgress(job: EngineJob): NodeJS.Timeout {
    const startedAt = Date.now();
    const expectedMs = getExpectedJobDurationMs(job.kind);
    return setInterval(() => {
        if (job.status !== "running") {
            return;
        }
        const elapsed = Date.now() - startedAt;
        const estimated = 5 + Math.min(90, Math.round((elapsed / expectedMs) * 90));
        if (estimated > job.progress && estimated < 96) {
            updateEngineJob(job, { progress: estimated });
        }
    }, 1000);
}

function getExpectedJobDurationMs(kind: EngineJobKind): number {
    if (kind === "lyrics") {
        return 120_000;
    }
    if (kind === "vocals") {
        return 240_000;
    }
    if (kind === "pitch-shift") {
        return 45_000;
    }
    return 90_000;
}

function updateEngineJob(job: EngineJob, patch: Partial<EngineJob>): void {
    Object.assign(job, patch, { updatedAt: new Date().toISOString() });
    persistEngineJob(job);
}

function toEngineJobSnapshot(job: EngineJob): EngineJob {
    return { ...job };
}

function getEngineJob(id: string): EngineJob | null {
    const memoryJob = engineJobs.get(id);
    if (memoryJob) {
        return memoryJob;
    }

    const row = database.prepare("SELECT * FROM engine_jobs WHERE id = ?").get(id) as EngineJobRow | undefined;
    return row ? rowToEngineJob(row) : null;
}

function persistEngineJob(job: EngineJob): void {
    database.prepare(`
        INSERT INTO engine_jobs (
            id, kind, status, progress, result_json, error_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            kind = excluded.kind,
            status = excluded.status,
            progress = excluded.progress,
            result_json = excluded.result_json,
            error_json = excluded.error_json,
            updated_at = excluded.updated_at
    `).run(
        job.id,
        job.kind,
        job.status,
        job.progress,
        JSON.stringify(job.result),
        JSON.stringify(job.error),
        job.createdAt,
        job.updatedAt
    );
}

function rowToEngineJob(row: EngineJobRow): EngineJob {
    return {
        id: row.id,
        kind: row.kind as EngineJobKind,
        status: row.status as EngineJobStatus,
        progress: row.progress,
        result: row.result_json ? JSON.parse(row.result_json) as unknown : null,
        error: row.error_json ? JSON.parse(row.error_json) as EngineJob["error"] : null,
        createdAt: row.created_at,
        updatedAt: row.updated_at
    };
}

async function analyzeAudio(request: AnalyzeAudioRequest, jobId?: string): Promise<ChordAnalysisResult> {
    if (!request.audioPath) {
        return enginePlaceholder("ENGINE_INVALID_INPUT", "Audio path is required") satisfies ChordAnalysisResult;
    }

    return await runEngineJson<ChordAnalysisResult>(
        ["-m", "chord_engine.cli", "analyze", request.audioPath],
        "ENGINE",
        { timeoutMs: 5 * 60_000, jobId }
    );
}

async function transcribeLyrics(request: GenerateLyricsRequest, jobId?: string): Promise<LyricsTranscriptionResult> {
    if (!request.audioPath) {
        return enginePlaceholder("LYRICS_INVALID_INPUT", "Audio path is required") satisfies LyricsTranscriptionResult;
    }

    return await runEngineJson<LyricsTranscriptionResult>(
        ["-m", "chord_engine.cli", "transcribe-lyrics", request.audioPath, "--model", normalizeLyricsModel(request.model)],
        "LYRICS",
        { timeoutMs: 10 * 60_000, jobId }
    );
}

async function removeVocals(request: RemoveVocalsRequest, jobId?: string): Promise<VocalRemovalResult> {
    if (!request.audioPath) {
        return enginePlaceholder("VOCAL_REMOVAL_INVALID_INPUT", "Audio path is required") satisfies VocalRemovalResult;
    }

    const result = await runEngineJson<VocalRemovalResult>(
        ["-m", "chord_engine.cli", "remove-vocals", request.audioPath, "--output-root", path.join(engineOutputDir, "vocal-removal"), "--model", "htdemucs"],
        "VOCAL_REMOVAL",
        {
            timeoutMs: 20 * 60_000,
            jobId,
            env: {
                ...process.env,
                DEMUCS_CACHE: path.join(engineOutputDir, "vocal-removal", "model-cache", "demucs"),
                HF_HOME: path.join(engineOutputDir, "vocal-removal", "model-cache", "huggingface"),
                TORCH_HOME: path.join(engineOutputDir, "vocal-removal", "model-cache", "torch")
            }
        }
    );
    return await withStreamableAudioResult(result);
}

async function pitchShiftAudio(request: PitchShiftAudioRequest, jobId?: string): Promise<PitchShiftResult> {
    if (!request.audioPath) {
        return enginePlaceholder("PITCH_SHIFT_INVALID_INPUT", "Audio path is required") satisfies PitchShiftResult;
    }

    const result = await runEngineJson<PitchShiftResult>(
        [
            "-m",
            "chord_engine.cli",
            "pitch-shift-audio",
            request.audioPath,
            "--output-root",
            path.join(engineOutputDir, "pitch-shift"),
            "--semitones",
            String(normalizeTransposeSemitones(request.semitones))
        ],
        "PITCH_SHIFT",
        { timeoutMs: 5 * 60_000, jobId }
    );
    return await withStreamableAudioResult(result);
}

async function withStreamableAudioResult<T extends { audio?: { path: string; streamUrl?: string }; error?: unknown }>(result: T): Promise<T> {
    if (result.error || !result.audio?.path) {
        return result;
    }

    const uploaded = await materializeAudioAsset(result.audio.path);
    return {
        ...result,
        audio: {
            ...result.audio,
            path: uploaded.audioPath,
            streamUrl: uploaded.audioStreamUrl
        }
    };
}

async function materializeAudioAsset(sourcePath: string): Promise<{ audioPath: string; audioStreamUrl: string; fileHash: string }> {
    const bytes = await readFile(sourcePath);
    const fileHash = createHash("sha256").update(bytes).digest("hex");
    const extension = resolveSafeAudioExtension(sourcePath);
    const audioPath = path.join(audioDir, `${fileHash}${extension}`);

    await writeFile(audioPath, bytes, { flag: "wx" }).catch(async (error: unknown) => {
        if (!isAlreadyExistsError(error)) {
            throw error;
        }
    });

    return {
        audioPath,
        audioStreamUrl: `${publicBaseUrl}/audio/${encodeURIComponent(fileHash)}${extension}/stream`,
        fileHash
    };
}

async function cleanupApiAssets(): Promise<AssetCleanupResult> {
    const referencedAudioPaths = collectReferencedAudioPaths(database);
    const result: AssetCleanupResult = {
        deletedAudioFiles: 0,
        deletedEngineOutputFiles: 0,
        skippedReferencedFiles: 0,
        reclaimedBytes: 0
    };

    for (const filePath of await listFiles(audioDir)) {
        if (referencedAudioPaths.has(path.resolve(filePath))) {
            result.skippedReferencedFiles += 1;
            continue;
        }
        const deletedBytes = await removeFileIfOlderThan(filePath, assetCleanupMaxAgeMs);
        if (deletedBytes > 0) {
            result.deletedAudioFiles += 1;
            result.reclaimedBytes += deletedBytes;
        }
    }

    for (const filePath of await listFiles(engineOutputDir)) {
        if (isModelCachePath(filePath)) {
            continue;
        }
        const deletedBytes = await removeFileIfOlderThan(filePath, assetCleanupMaxAgeMs);
        if (deletedBytes > 0) {
            result.deletedEngineOutputFiles += 1;
            result.reclaimedBytes += deletedBytes;
        }
    }

    return result;
}

async function listFiles(root: string): Promise<string[]> {
    try {
        const entries = await readdir(root, { withFileTypes: true });
        const files: string[] = [];
        for (const entry of entries) {
            const entryPath = path.join(root, entry.name);
            if (entry.isDirectory()) {
                files.push(...await listFiles(entryPath));
            } else if (entry.isFile()) {
                files.push(entryPath);
            }
        }
        return files;
    } catch (error) {
        if (error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "ENOENT") {
            return [];
        }
        throw error;
    }
}

async function removeFileIfOlderThan(filePath: string, maxAgeMs: number): Promise<number> {
    const fileStat = await stat(filePath).catch(() => null);
    if (!fileStat?.isFile()) {
        return 0;
    }
    if (Date.now() - fileStat.mtimeMs < maxAgeMs) {
        return 0;
    }
    await unlink(filePath);
    await removeEmptyParents(path.dirname(filePath), filePath.startsWith(engineOutputDir) ? engineOutputDir : audioDir);
    return fileStat.size;
}

async function removeEmptyParents(directoryPath: string, stopAt: string): Promise<void> {
    const resolvedStop = path.resolve(stopAt);
    let current = path.resolve(directoryPath);
    while (current.startsWith(`${resolvedStop}${path.sep}`)) {
        await rm(current, { recursive: false }).catch(() => undefined);
        current = path.dirname(current);
    }
}

function isModelCachePath(filePath: string): boolean {
    return path.resolve(filePath).split(path.sep).includes("model-cache");
}

async function runEngineJson<T extends { version: string; error?: { code: string; message: string } }>(
    args: string[],
    errorPrefix: string,
    options: EngineCommandOptions
): Promise<T> {
    return await new Promise<T>((resolve) => {
        let settled = false;
        let stdout = "";
        let stderr = "";
        const child = spawn(pythonExecutable, args, {
            cwd: engineCwd,
            env: options.env ?? process.env
        });
        if (options.jobId) {
            activeEngineProcesses.set(options.jobId, child);
        }

        const finish = (result: T): void => {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timeoutId);
            if (options.jobId) {
                activeEngineProcesses.delete(options.jobId);
            }
            if (stderr.trim()) {
                console.error(`[api engine stderr] ${stderr.trim()}`);
            }
            resolve(result);
        };

        const timeoutId = setTimeout(() => {
            child.kill("SIGKILL");
            finish(enginePlaceholder(`${errorPrefix}_TIMEOUT`, "Engine request timed out") as T);
        }, options.timeoutMs);

        child.stdout.on("data", (chunk: Buffer | string) => {
            stdout += String(chunk);
        });

        child.stderr.on("data", (chunk: Buffer | string) => {
            stderr += String(chunk);
        });

        child.on("error", () => {
            finish(enginePlaceholder(`${errorPrefix}_SPAWN_FAILED`, "Could not start engine process") as T);
        });

        child.on("close", (code) => {
            if (options.jobId && cancelledEngineJobs.has(options.jobId)) {
                finish(enginePlaceholder(`${errorPrefix}_CANCELLED`, "Engine job cancelled") as T);
                return;
            }
            const parsed = parseEngineJson<T>(stdout);
            if (!parsed) {
                finish(enginePlaceholder(`${errorPrefix}_INVALID_JSON`, "Engine returned invalid JSON output") as T);
                return;
            }
            if (code !== 0 && !parsed.error) {
                finish(enginePlaceholder(`${errorPrefix}_EXIT_NONZERO`, "Engine failed with non-zero exit code") as T);
                return;
            }
            finish(parsed);
        });
    });
}

function initializeDatabase(db: DatabaseSync): void {
    initializeSongTables(db);
    db.exec(`
        CREATE TABLE IF NOT EXISTS engine_jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL,
            result_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_engine_jobs_updated_at ON engine_jobs(updated_at);
    `);
}

function markInterruptedJobs(db: DatabaseSync): void {
    const now = new Date().toISOString();
    db.prepare(`
        UPDATE engine_jobs
        SET status = 'failed',
            progress = 100,
            error_json = ?,
            updated_at = ?
        WHERE status IN ('queued', 'running')
    `).run(
        JSON.stringify({
            code: "ENGINE_JOB_INTERRUPTED",
            message: "API server restarted before this job finished"
        }),
        now
    );
}

function enginePlaceholder(code: string, message: string) {
    return {
        version: "1",
        error: {
            code,
            message
        }
    };
}

function parseEngineJson<T>(stdout: string): T | null {
    const trimmed = stdout.trim();
    if (!trimmed) {
        return null;
    }

    try {
        return JSON.parse(trimmed) as T;
    } catch {
        return null;
    }
}

function normalizeLyricsModel(model: string | undefined): string {
    const allowedModels = new Set(["tiny", "base", "small", "medium", "large"]);
    return model && allowedModels.has(model) ? model : "small";
}

function normalizeTransposeSemitones(semitones: number): number {
    if (!Number.isFinite(semitones)) {
        return 0;
    }
    return Math.max(-11, Math.min(11, Math.trunc(semitones)));
}

function normalizeAssetCleanupMaxAgeMs(rawHours: string | undefined): number {
    const hours = rawHours ? Number.parseFloat(rawHours) : 24;
    if (!Number.isFinite(hours) || hours <= 0) {
        return 24 * 60 * 60_000;
    }
    return Math.max(1, hours) * 60 * 60_000;
}

function normalizeSearch(value: string): string {
    return value.trim().toLocaleLowerCase();
}

function normalizePage(value: number): number {
    return Number.isFinite(value) ? Math.max(1, Math.trunc(value)) : 1;
}

function normalizePageSize(value: number): number {
    return Number.isFinite(value) ? Math.max(5, Math.min(100, Math.trunc(value))) : 10;
}
