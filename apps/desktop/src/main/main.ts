import path from "node:path";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import electronMain from "electron/main";
import type { BrowserWindow as ElectronBrowserWindow } from "electron/main";
import type { IpcMainInvokeEvent } from "electron";
import type { ChordAnalysisResult, ChordAnalysisSuccess } from "@gcd/shared/analysis";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { SaveSongAnalysisRequest, SongLibrarySearchOptions, SongMetadataInput } from "@gcd/shared/library";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

import { AnalysisCache, resolveAnalysisCachePath } from "./analysisCache.js";
import { analyzeAudioInEngine, pitchShiftAudioInEngine, removeVocalsInEngine, transcribeLyricsInEngine } from "./engineProcess.js";
import { buildAudioFileDialogOptions, toFileSelectionResult } from "./fileDialog.js";
import { SongLibraryStore, resolveSongLibraryPath } from "./songLibrary.js";
import { buildMainWindowOptions } from "./window.js";

const { app, BrowserWindow, dialog, ipcMain } = electronMain;

type AudioPlaybackSource = AudioPlaybackBytesSource | AudioPlaybackUrlSource;

interface AudioPlaybackBytesSource {
    kind: "bytes";
    mimeType: "audio/mpeg" | "audio/wav";
    bytes: Uint8Array;
}

interface AudioPlaybackUrlSource {
    kind: "url";
    url: string;
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

interface AudioUploadResult {
    audioPath: string;
    audioStreamUrl: string;
    fileHash: string;
}

interface ApiStatus {
    mode: "api" | "local";
    baseUrl: string | null;
    healthy: boolean;
}

interface ApiConfig {
    baseUrl: string | null;
}

type ApiJobKind = "analysis" | "lyrics" | "vocals" | "pitch-shift";
type ApiJobStatus = "queued" | "running" | "succeeded" | "failed";

interface ApiJob<T> {
    id: string;
    kind: ApiJobKind;
    status: ApiJobStatus;
    progress: number;
    result: T | null;
    error: { code: string; message: string } | null;
}

interface ApiJobProgressEvent {
    id: string;
    kind: ApiJobKind;
    status: ApiJobStatus;
    progress: number;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ANALYSIS_CONTRACT_VERSION = "1";
const ANALYSIS_ALGORITHM = "chroma-template-v12";
const WINDOW_ICON_PATH = path.resolve(app.getAppPath(), "assets", "otehdekil.ico");
const DOCK_ICON_PATH = path.resolve(app.getAppPath(), "assets", "otehdekil.png");

let analysisCache: AnalysisCache | null = null;
let songLibrary: SongLibraryStore | null = null;
let apiBaseUrl: string | null = normalizeApiBaseUrl(process.env.GCD_API_BASE_URL);
const apiUploadCache = new Map<string, AudioUploadResult>();

function resolveAudioMimeType(audioPath: string): AudioPlaybackBytesSource["mimeType"] | null {
    const extension = path.extname(audioPath).toLowerCase();
    if (extension === ".mp3") {
        return "audio/mpeg";
    }
    if (extension === ".wav") {
        return "audio/wav";
    }
    return null;
}

function createMainWindow(): ElectronBrowserWindow {
    const preloadPath = path.resolve(__dirname, "../preload/index.js");
    const window = new BrowserWindow(buildMainWindowOptions(preloadPath, WINDOW_ICON_PATH));

    const devServerUrl = process.env.VITE_DEV_SERVER_URL;
    if (devServerUrl) {
        void window.loadURL(devServerUrl);
    } else {
        void window.loadFile(path.resolve(__dirname, "../index.html"));
    }

    return window;
}

function registerIpcHandlers(): void {
    ipcMain.handle("app:getVersion", () => app.getVersion());
    ipcMain.handle("app:isApiMode", () => Boolean(apiBaseUrl));
    ipcMain.handle("app:getApiStatus", async (): Promise<ApiStatus> => {
        if (!apiBaseUrl) {
            return { mode: "local", baseUrl: null, healthy: false };
        }

        try {
            await getApiJson("health");
            return { mode: "api", baseUrl: apiBaseUrl, healthy: true };
        } catch {
            return { mode: "api", baseUrl: apiBaseUrl, healthy: false };
        }
    });
    ipcMain.handle("app:testApiConfig", async (_event, request: ApiConfig): Promise<ApiStatus> => {
        const nextBaseUrl = normalizeApiBaseUrl(request.baseUrl ?? undefined);
        if (!nextBaseUrl) {
            return { mode: "local", baseUrl: null, healthy: false };
        }
        try {
            await fetch(`${nextBaseUrl}/health`);
            return { mode: "api", baseUrl: nextBaseUrl, healthy: true };
        } catch {
            return { mode: "api", baseUrl: nextBaseUrl, healthy: false };
        }
    });
    ipcMain.handle("app:saveApiConfig", async (_event, request: ApiConfig): Promise<ApiStatus> => {
        apiBaseUrl = normalizeApiBaseUrl(request.baseUrl ?? undefined);
        apiUploadCache.clear();
        await saveApiConfig({ baseUrl: apiBaseUrl });
        if (!apiBaseUrl) {
            return { mode: "local", baseUrl: null, healthy: false };
        }
        try {
            await getApiJson("health");
            return { mode: "api", baseUrl: apiBaseUrl, healthy: true };
        } catch {
            return { mode: "api", baseUrl: apiBaseUrl, healthy: false };
        }
    });
    ipcMain.handle("engine:analyzeAudio", async (event, request: string | AnalyzeAudioRequest) => {
        const normalized = typeof request === "string" ? { audioPath: request, forceRefresh: false } : request;
        if (apiBaseUrl) {
            const uploadedAudio = await uploadAudioForApi(normalized.audioPath);
            return await runApiEngineJob<ChordAnalysisResult>(event, "analysis", {
                ...normalized,
                audioPath: uploadedAudio?.audioPath ?? normalized.audioPath
            });
        }

        const cache = getAnalysisCache();
        const result = await cache.analyzeWithCache({
            audioPath: normalized.audioPath,
            algorithm: ANALYSIS_ALGORITHM,
            contractVersion: ANALYSIS_CONTRACT_VERSION,
            forceRefresh: normalized.forceRefresh === true,
            analyze: (pathToAnalyze) => analyzeAudioInEngine(pathToAnalyze, { appPath: app.getAppPath() })
        });
        return result;
    });
    ipcMain.handle("engine:generateLyrics", async (event, request: GenerateLyricsRequest): Promise<LyricsTranscriptionResult> => {
        if (apiBaseUrl) {
            const uploadedAudio = await uploadAudioForApi(request.audioPath);
            return await runApiEngineJob<LyricsTranscriptionResult>(event, "lyrics", {
                audioPath: uploadedAudio?.audioPath ?? request.audioPath,
                model: normalizeLyricsModel(request.model)
            });
        }

        return await transcribeLyricsInEngine(request.audioPath, { appPath: app.getAppPath() }, normalizeLyricsModel(request.model));
    });
    ipcMain.handle("engine:removeVocals", async (event, request: RemoveVocalsRequest): Promise<VocalRemovalResult> => {
        if (apiBaseUrl) {
            const uploadedAudio = await uploadAudioForApi(request.audioPath);
            return await runApiEngineJob<VocalRemovalResult>(event, "vocals", {
                audioPath: uploadedAudio?.audioPath ?? request.audioPath
            });
        }

        return await removeVocalsInEngine(
            request.audioPath,
            path.resolve(app.getPath("userData"), "vocal-removal"),
            { appPath: app.getAppPath() }
        );
    });
    ipcMain.handle("engine:pitchShiftAudio", async (event, request: PitchShiftAudioRequest): Promise<PitchShiftResult> => {
        if (apiBaseUrl) {
            const uploadedAudio = await uploadAudioForApi(request.audioPath);
            return await runApiEngineJob<PitchShiftResult>(event, "pitch-shift", {
                audioPath: uploadedAudio?.audioPath ?? request.audioPath,
                semitones: normalizeTransposeSemitones(request.semitones)
            });
        }

        return await pitchShiftAudioInEngine(
            request.audioPath,
            path.resolve(app.getPath("userData"), "pitch-shift"),
            normalizeTransposeSemitones(request.semitones),
            { appPath: app.getAppPath() }
        );
    });
    ipcMain.handle("library:saveAnalysis", async (_event, request: SaveSongAnalysisRequest) => {
        const metadata = normalizeSongMetadata(request.metadata);
        if (!metadata) {
            throw new Error("Song title and artist are required");
        }
        if (apiBaseUrl) {
            const uploadedAudio = await uploadAudioForApi(request.audioPath);
            const uploadedInstrumentalAudio = request.instrumentalAudioPath
                ? await uploadAudioForApi(request.instrumentalAudioPath)
                : null;
            const analysis = uploadedAudio
                ? rewriteAnalysisSourcePath(request.analysis, uploadedAudio.audioPath)
                : request.analysis;
            return await postApiJson("songs", {
                ...request,
                audioPath: uploadedAudio?.audioPath ?? request.audioPath,
                analysis,
                metadata,
                instrumentalAudioPath: uploadedInstrumentalAudio?.audioPath ?? request.instrumentalAudioPath
            });
        }

        return await getSongLibrary().upsertAnalysis({
            audioPath: request.audioPath,
            algorithm: ANALYSIS_ALGORITHM,
            contractVersion: ANALYSIS_CONTRACT_VERSION,
            analysis: request.analysis,
            metadata,
            lyrics: request.lyrics,
            instrumentalAudioPath: request.instrumentalAudioPath
        });
    });
    ipcMain.handle("library:listSongs", async (_event, options?: SongLibrarySearchOptions) => {
        if (apiBaseUrl) {
            const params = new URLSearchParams();
            if (options?.query) {
                params.set("query", options.query);
            }
            if (options?.page) {
                params.set("page", String(options.page));
            }
            if (options?.pageSize) {
                params.set("pageSize", String(options.pageSize));
            }
            const query = params.toString();
            return await getApiJson(`songs${query ? `?${query}` : ""}`);
        }

        return await getSongLibrary().listSongs(options);
    });
    ipcMain.handle("library:getSong", async (_event, id: string) => {
        if (apiBaseUrl) {
            return await getApiJson(`songs/${encodeURIComponent(id)}`);
        }

        return await getSongLibrary().getSong(id);
    });
    ipcMain.handle("library:deleteSong", async (_event, id: string) => {
        if (apiBaseUrl) {
            return await deleteApiJson(`songs/${encodeURIComponent(id)}`);
        }

        return await getSongLibrary().deleteSong(id);
    });
    ipcMain.handle("library:getExportUrl", (_event, request: { id: string; format: "txt" | "lrc"; transpose?: number }) => {
        if (!apiBaseUrl) {
            return { url: null };
        }
        const params = new URLSearchParams({
            format: request.format,
            transpose: String(normalizeTransposeSemitones(request.transpose ?? 0))
        });
        return {
            url: buildApiUrl(`songs/${encodeURIComponent(request.id)}/export?${params.toString()}`)
        };
    });
    ipcMain.handle("engine:cancelApiJob", async (_event, id: string) => {
        if (!apiBaseUrl) {
            return null;
        }
        return await postApiJson(`jobs/${encodeURIComponent(id)}/cancel`, {});
    });
    ipcMain.handle("file:selectAudio", async () => {
        const response = await dialog.showOpenDialog(buildAudioFileDialogOptions());
        if (response.canceled) {
            return toFileSelectionResult([]);
        }
        return toFileSelectionResult(response.filePaths);
    });
    ipcMain.handle("file:getAudioPlaybackSource", async (_event, audioPath: string): Promise<AudioPlaybackSource> => {
        if (isHttpUrl(audioPath)) {
            return {
                kind: "url",
                url: audioPath
            };
        }

        const mimeType = resolveAudioMimeType(audioPath);
        if (!mimeType) {
            throw new Error("Unsupported audio file type");
        }

        const bytes = await readFile(audioPath);
        return {
            kind: "bytes",
            mimeType,
            bytes
        };
    });
}

function getAnalysisCache(): AnalysisCache {
    if (!analysisCache) {
        const cacheFilePath = resolveAnalysisCachePath(app.getPath("userData"));
        analysisCache = new AnalysisCache(cacheFilePath);
    }
    return analysisCache;
}

function getSongLibrary(): SongLibraryStore {
    if (!songLibrary) {
        songLibrary = new SongLibraryStore(resolveSongLibraryPath(app.getPath("userData")));
    }
    return songLibrary;
}

function normalizeSongMetadata(metadata: SongMetadataInput): SongMetadataInput | null {
    const title = metadata.title.trim();
    const artist = metadata.artist.trim();
    if (!title || !artist) {
        return null;
    }
    return { title, artist };
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

function isHttpUrl(value: string): boolean {
    return /^https?:\/\//i.test(value);
}

function normalizeApiBaseUrl(value: string | undefined): string | null {
    if (!value || value.trim().length === 0) {
        return null;
    }
    return value.replace(/\/+$/, "");
}

function buildApiUrl(pathname: string): string {
    if (!apiBaseUrl) {
        throw new Error("API base URL is not configured");
    }
    return `${apiBaseUrl}/${pathname.replace(/^\/+/, "")}`;
}

async function getApiJson(pathname: string): Promise<unknown> {
    const response = await fetch(buildApiUrl(pathname));
    return await readApiJsonResponse(response);
}

async function postApiJson(pathname: string, body: unknown): Promise<unknown> {
    const response = await fetch(buildApiUrl(pathname), {
        method: "POST",
        headers: {
            "content-type": "application/json"
        },
        body: JSON.stringify(body)
    });
    return await readApiJsonResponse(response);
}

async function runApiEngineJob<T>(event: IpcMainInvokeEvent, kind: ApiJobKind, payload: unknown): Promise<T> {
    const created = await postApiJson("jobs", { kind, payload }) as ApiJob<T>;
    let latest = created;
    sendApiJobProgress(event, latest);

    while (latest.status === "queued" || latest.status === "running") {
        await delay(900);
        latest = await getApiJson(`jobs/${encodeURIComponent(latest.id)}`) as ApiJob<T>;
        sendApiJobProgress(event, latest);
    }

    if (latest.status === "succeeded" && latest.result) {
        return latest.result;
    }

    if (latest.result) {
        return latest.result;
    }

    throw new Error(latest.error?.message ?? "API engine job failed");
}

async function loadApiConfig(): Promise<void> {
    if (process.env.GCD_API_BASE_URL) {
        return;
    }
    const config = await readFile(resolveApiConfigPath(), "utf-8")
        .then((content) => JSON.parse(content) as ApiConfig)
        .catch(() => null);
    apiBaseUrl = normalizeApiBaseUrl(config?.baseUrl ?? undefined);
}

async function saveApiConfig(config: ApiConfig): Promise<void> {
    const configPath = resolveApiConfigPath();
    await mkdir(path.dirname(configPath), { recursive: true });
    await writeFile(configPath, JSON.stringify(config, null, 2), "utf-8");
}

function resolveApiConfigPath(): string {
    return path.join(app.getPath("userData"), "api-config.json");
}

async function postApiBytes(pathname: string, bytes: Buffer, contentType: string): Promise<unknown> {
    const response = await fetch(buildApiUrl(pathname), {
        method: "POST",
        headers: {
            "content-type": contentType,
            "content-length": String(bytes.byteLength)
        },
        body: toArrayBuffer(bytes)
    });
    return await readApiJsonResponse(response);
}

async function deleteApiJson(pathname: string): Promise<unknown> {
    const response = await fetch(buildApiUrl(pathname), {
        method: "DELETE"
    });
    return await readApiJsonResponse(response);
}

async function readApiJsonResponse(response: Response): Promise<unknown> {
    const payload = await response.json() as unknown;
    if (!response.ok) {
        const message = payload && typeof payload === "object" && "message" in payload
            ? String((payload as { message: unknown }).message)
            : `API request failed with ${response.status}`;
        throw new Error(message);
    }
    return payload;
}

async function uploadAudioForApi(audioPath: string): Promise<AudioUploadResult | null> {
    if (isHttpUrl(audioPath)) {
        return null;
    }

    const mimeType = resolveAudioMimeType(audioPath);
    if (!mimeType) {
        throw new Error("Unsupported audio file type");
    }

    const fileStat = await stat(audioPath).catch(() => null);
    if (!fileStat) {
        return null;
    }
    const cacheKey = `${audioPath}:${fileStat.size}:${fileStat.mtimeMs}`;
    const cached = apiUploadCache.get(cacheKey);
    if (cached) {
        return cached;
    }

    const bytes = await readFile(audioPath);
    const uploaded = await postApiBytes(
        `audio/upload?filename=${encodeURIComponent(path.basename(audioPath))}`,
        bytes,
        mimeType
    ) as AudioUploadResult;
    apiUploadCache.set(cacheKey, uploaded);
    return uploaded;
}

function rewriteAnalysisSourcePath(analysis: ChordAnalysisSuccess, audioPath: string): ChordAnalysisSuccess {
    return {
        ...analysis,
        source: {
            ...analysis.source,
            path: audioPath
        }
    };
}

function toArrayBuffer(bytes: Buffer): ArrayBuffer {
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

function sendApiJobProgress(event: IpcMainInvokeEvent, job: ApiJob<unknown>): void {
    const progress: ApiJobProgressEvent = {
        id: job.id,
        kind: job.kind,
        status: job.status,
        progress: job.progress
    };
    event.sender.send("api:jobProgress", progress);
}

async function bootstrap(): Promise<void> {
    if (process.platform === "darwin" && app.dock) {
        app.dock.setIcon(DOCK_ICON_PATH);
    }

    await loadApiConfig();
    registerIpcHandlers();
    createMainWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });
}

void app.whenReady().then(bootstrap);

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
        app.quit();
    }
});
