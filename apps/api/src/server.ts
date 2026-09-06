import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdir, stat, unlink, writeFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { SaveSongAnalysisRequest, SongLibraryListResult, SongLibraryRecord } from "@gcd/shared/library";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

interface SongRow {
    id: string;
    title: string;
    artist: string;
    audio_path: string;
    audio_stream_url: string | null;
    lyrics: string;
    instrumental_audio_path: string | null;
    instrumental_audio_stream_url: string | null;
    file_hash: string;
    algorithm: string;
    contract_version: string;
    duration: number;
    analysis_json: string;
    created_at: string;
    updated_at: string;
}

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
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const port = Number.parseInt(process.env.PORT ?? "8787", 10);
const dataDir = process.env.GCD_API_DATA_DIR ?? path.resolve(__dirname, "..", ".data");
const audioDir = path.join(dataDir, "audio");
const engineOutputDir = path.join(dataDir, "engine-output");
const repositoryRoot = process.env.GCD_REPOSITORY_ROOT ?? path.resolve(__dirname, "../../..");
const engineCwd = path.join(repositoryRoot, "engine");
const pythonExecutable = process.env.GCD_PYTHON ?? path.join(repositoryRoot, ".venv", "bin", "python");
const publicBaseUrl = (process.env.GCD_API_PUBLIC_URL ?? `http://localhost:${port}`).replace(/\/+$/, "");

await mkdir(dataDir, { recursive: true });
await mkdir(audioDir, { recursive: true });
await mkdir(engineOutputDir, { recursive: true });
const database = new DatabaseSync(path.join(dataDir, "guitar-chord-detection.sqlite"));
initializeDatabase(database);

const server = http.createServer((request, response) => {
    void handleRequest(request, response).catch((error) => {
        console.error("[api] request failed", error);
        sendJson(response, 500, { message: "Internal server error" });
    });
});

server.listen(port, () => {
    console.log(`[api] listening on http://localhost:${port}`);
});

async function handleRequest(request: http.IncomingMessage, response: http.ServerResponse): Promise<void> {
    const url = new URL(request.url ?? "/", publicBaseUrl);
    const method = request.method ?? "GET";

    if (method === "GET" && url.pathname === "/health") {
        sendJson(response, 200, { ok: true });
        return;
    }

    if (method === "GET" && url.pathname === "/songs") {
        sendJson(response, 200, listSongs(url));
        return;
    }

    if (method === "POST" && url.pathname === "/audio/upload") {
        const upload = await saveUploadedAudio(request, url);
        sendJson(response, 201, upload);
        return;
    }

    if (method === "POST" && url.pathname === "/songs") {
        const body = await readJsonBody<SaveSongAnalysisRequest>(request);
        sendJson(response, 201, saveSong(body));
        return;
    }

    const songMatch = /^\/songs\/([^/]+)$/.exec(url.pathname);
    if (songMatch && method === "GET") {
        const song = getSong(decodeURIComponent(songMatch[1]));
        if (!song) {
            sendJson(response, 404, { message: "Song not found" });
            return;
        }
        sendJson(response, 200, song);
        return;
    }

    if (songMatch && method === "DELETE") {
        const deleted = deleteSong(decodeURIComponent(songMatch[1]));
        sendJson(response, 200, { deleted });
        return;
    }

    const streamMatch = /^\/songs\/([^/]+)\/audio\/(original|instrumental)\/stream$/.exec(url.pathname);
    if (streamMatch && method === "GET") {
        await streamSongAudio(response, decodeURIComponent(streamMatch[1]), streamMatch[2] as "original" | "instrumental");
        return;
    }

    const uploadedAudioStreamMatch = /^\/audio\/([a-f0-9]{64})(\.[a-z0-9]+)\/stream$/.exec(url.pathname);
    if (uploadedAudioStreamMatch && method === "GET") {
        await streamAudioFile(response, path.join(audioDir, `${uploadedAudioStreamMatch[1]}${uploadedAudioStreamMatch[2]}`));
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

function listSongs(url: URL): SongLibraryListResult {
    const query = normalizeSearch(url.searchParams.get("query") ?? "");
    const page = normalizePage(Number.parseInt(url.searchParams.get("page") ?? "1", 10));
    const pageSize = normalizePageSize(Number.parseInt(url.searchParams.get("pageSize") ?? "10", 10));
    const offset = (page - 1) * pageSize;
    const countStatement = query
        ? database.prepare("SELECT COUNT(*) AS total FROM songs WHERE lower(title || ' ' || artist || ' ' || audio_path) LIKE ?")
        : database.prepare("SELECT COUNT(*) AS total FROM songs");
    const total = query
        ? (countStatement.get(`%${query}%`) as { total: number }).total
        : (countStatement.get() as { total: number }).total;
    const rows = query
        ? database.prepare("SELECT * FROM songs WHERE lower(title || ' ' || artist || ' ' || audio_path) LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?").all(`%${query}%`, pageSize, offset) as unknown as SongRow[]
        : database.prepare("SELECT * FROM songs ORDER BY updated_at DESC LIMIT ? OFFSET ?").all(pageSize, offset) as unknown as SongRow[];

    return {
        records: rows.map(rowToRecord),
        total,
        page,
        pageSize,
        totalPages: Math.max(1, Math.ceil(total / pageSize))
    };
}

function getSong(id: string): SongLibraryRecord | null {
    const row = database.prepare("SELECT * FROM songs WHERE id = ?").get(id) as SongRow | undefined;
    return row ? rowToRecord(row) : null;
}

function saveSong(request: SaveSongAnalysisRequest): SongLibraryRecord {
    const now = new Date().toISOString();
    const fileHash = request.analysis.source.path || request.audioPath;
    const id = `${fileHash}:${request.analysis.analysis.algorithm}:${request.analysis.version}`;
    const existing = getSong(id);
    const record: SongLibraryRecord = {
        id,
        title: request.metadata.title.trim(),
        artist: request.metadata.artist.trim(),
        audioPath: request.audioPath,
        audioStreamUrl: `${publicBaseUrl}/songs/${encodeURIComponent(id)}/audio/original/stream`,
        lyrics: request.lyrics ?? existing?.lyrics ?? "",
        instrumentalAudioPath: request.instrumentalAudioPath ?? existing?.instrumentalAudioPath,
        instrumentalAudioStreamUrl: request.instrumentalAudioPath
            ? `${publicBaseUrl}/songs/${encodeURIComponent(id)}/audio/instrumental/stream`
            : existing?.instrumentalAudioStreamUrl,
        fileHash,
        algorithm: request.analysis.analysis.algorithm,
        contractVersion: request.analysis.version,
        duration: request.analysis.source.duration,
        analysis: request.analysis,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now
    };

    database.prepare(`
        INSERT INTO songs (
            id, title, artist, audio_path, audio_stream_url, lyrics,
            instrumental_audio_path, instrumental_audio_stream_url, file_hash,
            algorithm, contract_version, duration, analysis_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            artist = excluded.artist,
            audio_path = excluded.audio_path,
            audio_stream_url = excluded.audio_stream_url,
            lyrics = excluded.lyrics,
            instrumental_audio_path = excluded.instrumental_audio_path,
            instrumental_audio_stream_url = excluded.instrumental_audio_stream_url,
            file_hash = excluded.file_hash,
            algorithm = excluded.algorithm,
            contract_version = excluded.contract_version,
            duration = excluded.duration,
            analysis_json = excluded.analysis_json,
            updated_at = excluded.updated_at
    `).run(
        record.id,
        record.title,
        record.artist,
        record.audioPath,
        record.audioStreamUrl ?? null,
        record.lyrics ?? "",
        record.instrumentalAudioPath ?? null,
        record.instrumentalAudioStreamUrl ?? null,
        record.fileHash,
        record.algorithm,
        record.contractVersion,
        record.duration,
        JSON.stringify(record.analysis),
        record.createdAt,
        record.updatedAt
    );

    return record;
}

function deleteSong(id: string): boolean {
    const existing = getSong(id);
    const result = database.prepare("DELETE FROM songs WHERE id = ?").run(id);
    const deleted = Number(result.changes) > 0;
    if (deleted && existing) {
        void unlinkManagedAudio(existing.audioPath);
        if (existing.instrumentalAudioPath) {
            void unlinkManagedAudio(existing.instrumentalAudioPath);
        }
    }
    return deleted;
}

async function streamSongAudio(response: http.ServerResponse, id: string, kind: "original" | "instrumental"): Promise<void> {
    const song = getSong(id);
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

    await streamAudioFile(response, audioPath);
}

async function streamAudioFile(response: http.ServerResponse, audioPath: string): Promise<void> {
    const fileStat = await stat(audioPath).catch(() => null);
    if (!fileStat) {
        sendJson(response, 404, { message: "Audio file not found" });
        return;
    }

    response.writeHead(200, {
        "content-length": fileStat.size,
        "content-type": resolveAudioContentType(audioPath)
    });
    createReadStream(audioPath).pipe(response);
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

async function analyzeAudio(request: AnalyzeAudioRequest): Promise<ChordAnalysisResult> {
    if (!request.audioPath) {
        return enginePlaceholder("ENGINE_INVALID_INPUT", "Audio path is required") satisfies ChordAnalysisResult;
    }

    return await runEngineJson<ChordAnalysisResult>(
        ["-m", "chord_engine.cli", "analyze", request.audioPath],
        "ENGINE",
        { timeoutMs: 5 * 60_000 }
    );
}

async function transcribeLyrics(request: GenerateLyricsRequest): Promise<LyricsTranscriptionResult> {
    if (!request.audioPath) {
        return enginePlaceholder("LYRICS_INVALID_INPUT", "Audio path is required") satisfies LyricsTranscriptionResult;
    }

    return await runEngineJson<LyricsTranscriptionResult>(
        ["-m", "chord_engine.cli", "transcribe-lyrics", request.audioPath, "--model", normalizeLyricsModel(request.model)],
        "LYRICS",
        { timeoutMs: 10 * 60_000 }
    );
}

async function removeVocals(request: RemoveVocalsRequest): Promise<VocalRemovalResult> {
    if (!request.audioPath) {
        return enginePlaceholder("VOCAL_REMOVAL_INVALID_INPUT", "Audio path is required") satisfies VocalRemovalResult;
    }

    return await runEngineJson<VocalRemovalResult>(
        ["-m", "chord_engine.cli", "remove-vocals", request.audioPath, "--output-root", path.join(engineOutputDir, "vocal-removal"), "--model", "htdemucs"],
        "VOCAL_REMOVAL",
        {
            timeoutMs: 20 * 60_000,
            env: {
                ...process.env,
                DEMUCS_CACHE: path.join(engineOutputDir, "vocal-removal", "model-cache", "demucs"),
                HF_HOME: path.join(engineOutputDir, "vocal-removal", "model-cache", "huggingface"),
                TORCH_HOME: path.join(engineOutputDir, "vocal-removal", "model-cache", "torch")
            }
        }
    );
}

async function pitchShiftAudio(request: PitchShiftAudioRequest): Promise<PitchShiftResult> {
    if (!request.audioPath) {
        return enginePlaceholder("PITCH_SHIFT_INVALID_INPUT", "Audio path is required") satisfies PitchShiftResult;
    }

    return await runEngineJson<PitchShiftResult>(
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
        { timeoutMs: 5 * 60_000 }
    );
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

        const finish = (result: T): void => {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timeoutId);
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
    db.exec(`
        CREATE TABLE IF NOT EXISTS songs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            audio_stream_url TEXT,
            lyrics TEXT NOT NULL DEFAULT '',
            instrumental_audio_path TEXT,
            instrumental_audio_stream_url TEXT,
            file_hash TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            duration REAL NOT NULL,
            analysis_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_songs_updated_at ON songs(updated_at);
        CREATE INDEX IF NOT EXISTS idx_songs_title_artist ON songs(title, artist);
    `);
}

function rowToRecord(row: SongRow): SongLibraryRecord {
    return {
        id: row.id,
        title: row.title,
        artist: row.artist,
        audioPath: row.audio_path,
        audioStreamUrl: row.audio_stream_url ?? undefined,
        lyrics: row.lyrics,
        instrumentalAudioPath: row.instrumental_audio_path ?? undefined,
        instrumentalAudioStreamUrl: row.instrumental_audio_stream_url ?? undefined,
        fileHash: row.file_hash,
        algorithm: row.algorithm,
        contractVersion: row.contract_version,
        duration: row.duration,
        analysis: JSON.parse(row.analysis_json) as SongLibraryRecord["analysis"],
        createdAt: row.created_at,
        updatedAt: row.updated_at
    };
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

async function readJsonBody<T>(request: http.IncomingMessage): Promise<T> {
    return JSON.parse((await readRawBody(request)).toString("utf-8")) as T;
}

async function readRawBody(request: http.IncomingMessage): Promise<Buffer> {
    const chunks: Buffer[] = [];
    for await (const chunk of request) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
}

function sendJson(response: http.ServerResponse, statusCode: number, payload: unknown): void {
    response.writeHead(statusCode, { "content-type": "application/json" });
    response.end(JSON.stringify(payload));
}

function resolveAudioContentType(audioPath: string): string {
    const extension = path.extname(audioPath).toLowerCase();
    if (extension === ".mp3") {
        return "audio/mpeg";
    }
    if (extension === ".wav") {
        return "audio/wav";
    }
    return "application/octet-stream";
}

function resolveSafeAudioExtension(fileName: string): string {
    const extension = path.extname(fileName).toLowerCase();
    if (extension === ".mp3" || extension === ".wav") {
        return extension;
    }
    return ".audio";
}

function isAlreadyExistsError(error: unknown): boolean {
    return error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "EEXIST";
}

async function unlinkManagedAudio(audioPath: string): Promise<void> {
    if (!path.resolve(audioPath).startsWith(`${path.resolve(audioDir)}${path.sep}`)) {
        return;
    }

    await unlink(audioPath).catch((error: unknown) => {
        if (!(error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "ENOENT")) {
            console.error("[api] failed to delete managed audio", error);
        }
    });
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
