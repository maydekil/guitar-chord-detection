import path from "node:path";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, dialog, ipcMain } from "electron";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { SaveSongAnalysisRequest, SongLibrarySearchOptions, SongMetadataInput } from "@gcd/shared/library";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

import { AnalysisCache, resolveAnalysisCachePath } from "./analysisCache.js";
import { analyzeAudioInEngine, pitchShiftAudioInEngine, removeVocalsInEngine, transcribeLyricsInEngine } from "./engineProcess.js";
import { buildAudioFileDialogOptions, toFileSelectionResult } from "./fileDialog.js";
import { SongLibraryStore, resolveSongLibraryPath } from "./songLibrary.js";
import { buildMainWindowOptions } from "./window.js";

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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ANALYSIS_CONTRACT_VERSION = "1";
const ANALYSIS_ALGORITHM = "chroma-template-v3";
const WINDOW_ICON_PATH = path.resolve(app.getAppPath(), "assets", "otehdekil.ico");
const DOCK_ICON_PATH = path.resolve(app.getAppPath(), "assets", "otehdekil.png");
const API_BASE_URL = normalizeApiBaseUrl(process.env.GCD_API_BASE_URL);

let analysisCache: AnalysisCache | null = null;
let songLibrary: SongLibraryStore | null = null;

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

function createMainWindow(): BrowserWindow {
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
    ipcMain.handle("engine:analyzeAudio", async (_event, request: string | AnalyzeAudioRequest) => {
        const normalized = typeof request === "string" ? { audioPath: request, forceRefresh: false } : request;
        if (API_BASE_URL) {
            return await postApiJson("analysis", normalized);
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
    ipcMain.handle("engine:generateLyrics", async (_event, request: GenerateLyricsRequest): Promise<LyricsTranscriptionResult> => {
        if (API_BASE_URL) {
            return await postApiJson("lyrics/transcribe", {
                audioPath: request.audioPath,
                model: normalizeLyricsModel(request.model)
            }) as LyricsTranscriptionResult;
        }

        return await transcribeLyricsInEngine(request.audioPath, { appPath: app.getAppPath() }, normalizeLyricsModel(request.model));
    });
    ipcMain.handle("engine:removeVocals", async (_event, request: RemoveVocalsRequest): Promise<VocalRemovalResult> => {
        if (API_BASE_URL) {
            return await postApiJson("vocals/remove", request) as VocalRemovalResult;
        }

        return await removeVocalsInEngine(
            request.audioPath,
            path.resolve(app.getPath("userData"), "vocal-removal"),
            { appPath: app.getAppPath() }
        );
    });
    ipcMain.handle("engine:pitchShiftAudio", async (_event, request: PitchShiftAudioRequest): Promise<PitchShiftResult> => {
        if (API_BASE_URL) {
            return await postApiJson("audio/pitch-shift", {
                audioPath: request.audioPath,
                semitones: normalizeTransposeSemitones(request.semitones)
            }) as PitchShiftResult;
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
        if (API_BASE_URL) {
            return await postApiJson("songs", {
                ...request,
                metadata
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
        if (API_BASE_URL) {
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
        if (API_BASE_URL) {
            return await getApiJson(`songs/${encodeURIComponent(id)}`);
        }

        return await getSongLibrary().getSong(id);
    });
    ipcMain.handle("library:deleteSong", async (_event, id: string) => {
        if (API_BASE_URL) {
            return await deleteApiJson(`songs/${encodeURIComponent(id)}`);
        }

        return await getSongLibrary().deleteSong(id);
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
    if (!API_BASE_URL) {
        throw new Error("API base URL is not configured");
    }
    return `${API_BASE_URL}/${pathname.replace(/^\/+/, "")}`;
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

async function bootstrap(): Promise<void> {
    if (process.platform === "darwin" && app.dock) {
        app.dock.setIcon(DOCK_ICON_PATH);
    }

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
