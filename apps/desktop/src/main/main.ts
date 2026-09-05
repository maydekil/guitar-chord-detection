import path from "node:path";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, dialog, ipcMain } from "electron";

import { AnalysisCache, resolveAnalysisCachePath } from "./analysisCache.js";
import { analyzeAudioInEngine } from "./engineProcess.js";
import { buildAudioFileDialogOptions, toFileSelectionResult } from "./fileDialog.js";
import { buildMainWindowOptions } from "./window.js";

interface AudioPlaybackSource {
    mimeType: "audio/mpeg" | "audio/wav";
    bytes: Uint8Array;
}

interface AnalyzeAudioRequest {
    audioPath: string;
    forceRefresh?: boolean;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ANALYSIS_CONTRACT_VERSION = "1";
const ANALYSIS_ALGORITHM = "chroma-template-v3";

let analysisCache: AnalysisCache | null = null;

function resolveAudioMimeType(audioPath: string): AudioPlaybackSource["mimeType"] | null {
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
    const window = new BrowserWindow(buildMainWindowOptions(preloadPath));

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
        const cache = getAnalysisCache();
        return await cache.analyzeWithCache({
            audioPath: normalized.audioPath,
            algorithm: ANALYSIS_ALGORITHM,
            contractVersion: ANALYSIS_CONTRACT_VERSION,
            forceRefresh: normalized.forceRefresh === true,
            analyze: (pathToAnalyze) => analyzeAudioInEngine(pathToAnalyze, { appPath: app.getAppPath() })
        });
    });
    ipcMain.handle("file:selectAudio", async () => {
        const response = await dialog.showOpenDialog(buildAudioFileDialogOptions());
        if (response.canceled) {
            return toFileSelectionResult([]);
        }
        return toFileSelectionResult(response.filePaths);
    });
    ipcMain.handle("file:getAudioPlaybackSource", async (_event, audioPath: string): Promise<AudioPlaybackSource> => {
        const mimeType = resolveAudioMimeType(audioPath);
        if (!mimeType) {
            throw new Error("Unsupported audio file type");
        }

        const bytes = await readFile(audioPath);
        return {
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

async function bootstrap(): Promise<void> {
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
