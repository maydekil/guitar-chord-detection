import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { createReadStream, readFileSync } from "node:fs";
import { mkdir, readFile, stat, unlink, writeFile } from "node:fs/promises";
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

interface LyricLine {
    time: number | null;
    text: string;
}

interface LyricChordMarker {
    label: string;
    left: number;
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
        ? database.prepare(`
            SELECT *
            FROM songs
            WHERE lower(title || ' ' || artist || ' ' || audio_path) LIKE ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        `).all(`%${query}%`, pageSize, offset) as unknown as SongRow[]
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
    const fileHash = buildStableFileHash(request.audioPath, request.analysis.source.path);
    const id = buildSongId(fileHash);
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

async function streamSongAudio(request: http.IncomingMessage, response: http.ServerResponse, id: string, kind: "original" | "instrumental"): Promise<void> {
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
    const song = getSong(id);
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

function buildChordSheetExport(song: SongLibraryRecord, transposeSemitones: number): string {
    const lines = [
        `${song.artist} - ${song.title}`,
        `Duration: ${formatExportTime(song.duration)}`,
        `Transpose: ${formatTranspose(transposeSemitones)}`,
        ""
    ];

    if (song.lyrics?.trim()) {
        lines.push("Chord Sheet:");
        lines.push(...buildChordOverLyricLines(parseLyrics(song.lyrics), song.analysis.analysis.chords, transposeSemitones));
        lines.push("");
        lines.push("Timeline:");
    } else {
        lines.push("Timeline:");
    }

    lines.push(...song.analysis.analysis.chords.map((segment) => (
        `${formatExportTime(segment.start)} - ${formatExportTime(segment.end)}  ${transposeChordLabel(segment.chord, transposeSemitones)}`
    )));

    return `${lines.join("\n")}\n`;
}

function buildLrcExport(song: SongLibraryRecord, transposeSemitones: number): string {
    if (song.lyrics?.trim()) {
        return `${buildChordTaggedLrcLines(parseLyrics(song.lyrics), song.analysis.analysis.chords, transposeSemitones).join("\n")}\n`;
    }

    return `${song.analysis.analysis.chords.map((segment) => (
        `[${formatExportTime(segment.start)}]${transposeChordLabel(segment.chord, transposeSemitones)}`
    )).join("\n")}\n`;
}

function buildChordOverLyricLines(
    lines: LyricLine[],
    segments: SongLibraryRecord["analysis"]["analysis"]["chords"],
    transposeSemitones: number
): string[] {
    const output: string[] = [];
    for (const [index, line] of lines.entries()) {
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        if (line.time === null) {
            output.push("");
            output.push(line.text);
            continue;
        }
        const markers = buildLyricChordMarkers(segments, line.time, nextTimedLine?.time ?? line.time + 5, transposeSemitones);
        output.push(renderChordLineAboveLyric(line.text, markers));
        output.push(line.text);
    }
    return output;
}

function buildChordTaggedLrcLines(
    lines: LyricLine[],
    segments: SongLibraryRecord["analysis"]["analysis"]["chords"],
    transposeSemitones: number
): string[] {
    return lines.map((line, index) => {
        if (line.time === null) {
            return line.text;
        }
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        const markers = buildLyricChordMarkers(segments, line.time, nextTimedLine?.time ?? line.time + 5, transposeSemitones);
        const chordTags = markers.length > 0 ? `${markers.map((marker) => `[${marker.label}]`).join("")} ` : "";
        return `[${formatExportTime(line.time)}]${chordTags}${line.text}`;
    });
}

function buildLyricChordMarkers(
    segments: SongLibraryRecord["analysis"]["analysis"]["chords"],
    startTime: number,
    endTime: number,
    transposeSemitones: number
): LyricChordMarker[] {
    const safeEndTime = Math.max(startTime + 0.25, endTime);
    const windowDuration = safeEndTime - startTime;
    const markers: LyricChordMarker[] = [];
    const openingChord = findActiveChord(segments, startTime);

    if (openingChord) {
        markers.push({
            label: transposeChordLabel(openingChord, transposeSemitones),
            left: 0
        });
    }

    for (const segment of segments) {
        if (segment.start <= startTime || segment.start >= safeEndTime) {
            continue;
        }
        const label = transposeChordLabel(segment.chord, transposeSemitones);
        const previous = markers[markers.length - 1];
        if (previous?.label === label) {
            continue;
        }
        markers.push({
            label,
            left: Math.min(92, Math.max(0, ((segment.start - startTime) / windowDuration) * 100))
        });
    }

    return markers;
}

function renderChordLineAboveLyric(text: string, markers: LyricChordMarker[]): string {
    if (markers.length === 0) {
        return "";
    }
    const width = Math.max(24, text.length);
    const chars = Array.from({ length: width }, () => " ");
    for (const marker of markers) {
        const position = Math.min(width - 1, Math.max(0, Math.round((marker.left / 100) * Math.max(1, width - 1))));
        for (let index = 0; index < marker.label.length && position + index < chars.length; index += 1) {
            chars[position + index] = marker.label[index] ?? " ";
        }
    }
    return chars.join("").trimEnd();
}

function parseLyrics(lyrics: string): LyricLine[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .filter((line) => line.length > 0)
        .map((line) => {
            const match = /^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$/.exec(line);
            if (!match) {
                return { time: null, text: line };
            }

            const minutes = Number.parseInt(match[1], 10);
            const seconds = Number.parseInt(match[2], 10);
            const fraction = match[3] ? Number.parseFloat(`0.${match[3]}`) : 0;
            const time = minutes * 60 + seconds + fraction;
            if (!Number.isFinite(time) || seconds >= 60) {
                return { time: null, text: line };
            }

            return {
                time,
                text: match[4].trim()
            };
        });
}

function findActiveChord(segments: SongLibraryRecord["analysis"]["analysis"]["chords"], currentTimeSeconds: number): string | null {
    if (!Number.isFinite(currentTimeSeconds)) {
        return null;
    }
    return segments.find((segment) => segment.start <= currentTimeSeconds && currentTimeSeconds < segment.end)?.chord ?? null;
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
    cleanupDuplicateSongRows(db);
    db.exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_file_hash_unique ON songs(file_hash);");
}

function cleanupDuplicateSongRows(db: DatabaseSync): void {
    db.exec(`
        DELETE FROM songs
        WHERE EXISTS (
            SELECT 1
            FROM songs newer
            WHERE newer.file_hash = songs.file_hash
                AND (
                    newer.updated_at > songs.updated_at
                    OR (newer.updated_at = songs.updated_at AND newer.id > songs.id)
                )
        );

        UPDATE songs
        SET id = 'song:' || file_hash
        WHERE id <> 'song:' || file_hash
            AND NOT EXISTS (
                SELECT 1
                FROM songs existing
                WHERE existing.id = 'song:' || songs.file_hash
            );
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

function buildStableFileHash(audioPath: string, fallback: string): string {
    try {
        return createHash("sha256").update(readFileSync(audioPath)).digest("hex");
    } catch {
        return fallback || audioPath;
    }
}

function buildSongId(fileHash: string): string {
    return `song:${fileHash}`;
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

const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

function formatTranspose(semitones: number): string {
    if (semitones === 0) {
        return "0";
    }
    return semitones > 0 ? `+${semitones}` : `${semitones}`;
}

function transposeChordLabel(chord: string, semitones: number): string {
    if (chord === "N" || semitones === 0) {
        return chord;
    }

    const match = /^(C#|D#|F#|G#|A#|C|D|E|F|G|A|B)(.*)$/.exec(chord);
    if (!match) {
        return chord;
    }

    const rootIndex = SHARP_ROOTS.indexOf(match[1] as (typeof SHARP_ROOTS)[number]);
    if (rootIndex < 0) {
        return chord;
    }

    const nextIndex = modulo(rootIndex + semitones, SHARP_ROOTS.length);
    return `${SHARP_ROOTS[nextIndex]}${match[2]}`;
}

function modulo(value: number, divisor: number): number {
    return ((value % divisor) + divisor) % divisor;
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

function safeFileName(value: string): string {
    const safe = value.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    return safe || "song";
}

function formatExportTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}

function parseRangeHeader(rangeHeader: string | undefined, fileSize: number): { start: number; end: number } | null {
    if (!rangeHeader) {
        return null;
    }

    const match = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader.trim());
    if (!match) {
        return null;
    }

    const rawStart = match[1];
    const rawEnd = match[2];
    if (!rawStart && !rawEnd) {
        return null;
    }

    if (!rawStart) {
        const suffixLength = Number.parseInt(rawEnd, 10);
        if (!Number.isFinite(suffixLength) || suffixLength <= 0) {
            return null;
        }
        const start = Math.max(0, fileSize - suffixLength);
        return { start, end: fileSize - 1 };
    }

    const start = Number.parseInt(rawStart, 10);
    const end = rawEnd ? Number.parseInt(rawEnd, 10) : fileSize - 1;
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || start >= fileSize) {
        return null;
    }

    return {
        start,
        end: Math.min(end, fileSize - 1)
    };
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
