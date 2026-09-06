import { contextBridge, ipcRenderer } from "electron";
import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { DeleteSongResult, SaveSongAnalysisRequest, SongLibraryRecord, SongLibrarySearchOptions } from "@gcd/shared/library";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";

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

export interface GenerateLyricsOptions {
    model?: string;
}

export interface DesktopApi {
    getAppVersion(): Promise<string>;
    selectAudioFile(): Promise<FileSelectionResult>;
    analyzeAudio?(audioPath: string, options?: AnalyzeAudioOptions): Promise<ChordAnalysisResult>;
    generateLyricsFromAudio?(audioPath: string, options?: GenerateLyricsOptions): Promise<LyricsTranscriptionResult>;
    getAudioPlaybackSource?(audioPath: string): Promise<AudioPlaybackSource>;
    listSongs?(options?: SongLibrarySearchOptions): Promise<SongLibraryRecord[]>;
    getSong?(id: string): Promise<SongLibraryRecord | null>;
    saveSongAnalysis?(request: SaveSongAnalysisRequest): Promise<SongLibraryRecord>;
    deleteSong?(id: string): Promise<DeleteSongResult>;
}

const desktopApi: DesktopApi = {
    getAppVersion: () => ipcRenderer.invoke("app:getVersion") as Promise<string>,
    selectAudioFile: () => ipcRenderer.invoke("file:selectAudio") as Promise<FileSelectionResult>,
    analyzeAudio: (audioPath: string, options?: AnalyzeAudioOptions) =>
        ipcRenderer.invoke("engine:analyzeAudio", {
            audioPath,
            forceRefresh: options?.forceRefresh === true
        }) as Promise<ChordAnalysisResult>,
    generateLyricsFromAudio: (audioPath: string, options?: GenerateLyricsOptions) =>
        ipcRenderer.invoke("engine:generateLyrics", { audioPath, model: options?.model }) as Promise<LyricsTranscriptionResult>,
    getAudioPlaybackSource: (audioPath: string) =>
        ipcRenderer.invoke("file:getAudioPlaybackSource", audioPath) as Promise<AudioPlaybackSource>,
    listSongs: (options?: SongLibrarySearchOptions) =>
        ipcRenderer.invoke("library:listSongs", options) as Promise<SongLibraryRecord[]>,
    getSong: (id: string) => ipcRenderer.invoke("library:getSong", id) as Promise<SongLibraryRecord | null>,
    saveSongAnalysis: (request: SaveSongAnalysisRequest) =>
        ipcRenderer.invoke("library:saveAnalysis", request) as Promise<SongLibraryRecord>,
    deleteSong: (id: string) => ipcRenderer.invoke("library:deleteSong", id) as Promise<DeleteSongResult>
};

contextBridge.exposeInMainWorld("gcd", desktopApi);
