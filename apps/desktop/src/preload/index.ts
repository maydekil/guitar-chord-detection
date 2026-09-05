import { contextBridge, ipcRenderer } from "electron";
import type { ChordAnalysisResult } from "@gcd/shared/analysis";

export interface FileSelectionResult {
    canceled: boolean;
    path: string | null;
    fileName: string | null;
}

export interface AudioPlaybackSource {
    mimeType: "audio/mpeg" | "audio/wav";
    bytes: Uint8Array;
}

export interface AnalyzeAudioOptions {
    forceRefresh?: boolean;
}

export interface DesktopApi {
    getAppVersion(): Promise<string>;
    selectAudioFile(): Promise<FileSelectionResult>;
    analyzeAudio?(audioPath: string, options?: AnalyzeAudioOptions): Promise<ChordAnalysisResult>;
    getAudioPlaybackSource?(audioPath: string): Promise<AudioPlaybackSource>;
}

const desktopApi: DesktopApi = {
    getAppVersion: () => ipcRenderer.invoke("app:getVersion") as Promise<string>,
    selectAudioFile: () => ipcRenderer.invoke("file:selectAudio") as Promise<FileSelectionResult>,
    analyzeAudio: (audioPath: string, options?: AnalyzeAudioOptions) =>
        ipcRenderer.invoke("engine:analyzeAudio", {
            audioPath,
            forceRefresh: options?.forceRefresh === true
        }) as Promise<ChordAnalysisResult>,
    getAudioPlaybackSource: (audioPath: string) =>
        ipcRenderer.invoke("file:getAudioPlaybackSource", audioPath) as Promise<AudioPlaybackSource>
};

contextBridge.exposeInMainWorld("gcd", desktopApi);
