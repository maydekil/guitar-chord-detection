import { contextBridge, ipcRenderer } from "electron";
import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { DeleteSongResult, SaveSongAnalysisRequest, SongLibraryListResult, SongLibraryRecord, SongLibrarySearchOptions } from "@gcd/shared/library";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

export interface FileSelectionResult {
    canceled: boolean;
    path: string | null;
    fileName: string | null;
}

export interface AudioPlaybackBytesSource {
    kind: "bytes";
    mimeType: "audio/mpeg" | "audio/wav";
    bytes: Uint8Array;
}

export interface AudioPlaybackUrlSource {
    kind: "url";
    url: string;
}

export type AudioPlaybackSource = AudioPlaybackBytesSource | AudioPlaybackUrlSource;

export interface AnalyzeAudioOptions {
    forceRefresh?: boolean;
}

export interface GenerateLyricsOptions {
    model?: string;
}

export interface PitchShiftOptions {
    semitones: number;
}

export interface DesktopApi {
    getAppVersion(): Promise<string>;
    isApiMode?(): Promise<boolean>;
    selectAudioFile(): Promise<FileSelectionResult>;
    analyzeAudio?(audioPath: string, options?: AnalyzeAudioOptions): Promise<ChordAnalysisResult>;
    generateLyricsFromAudio?(audioPath: string, options?: GenerateLyricsOptions): Promise<LyricsTranscriptionResult>;
    pitchShiftAudio?(audioPath: string, options: PitchShiftOptions): Promise<PitchShiftResult>;
    removeVocals?(audioPath: string): Promise<VocalRemovalResult>;
    getAudioPlaybackSource?(audioPath: string): Promise<AudioPlaybackSource>;
    listSongs?(options?: SongLibrarySearchOptions): Promise<SongLibraryListResult | SongLibraryRecord[]>;
    getSong?(id: string): Promise<SongLibraryRecord | null>;
    saveSongAnalysis?(request: SaveSongAnalysisRequest): Promise<SongLibraryRecord>;
    deleteSong?(id: string): Promise<DeleteSongResult>;
}

const desktopApi: DesktopApi = {
    getAppVersion: () => ipcRenderer.invoke("app:getVersion") as Promise<string>,
    isApiMode: () => ipcRenderer.invoke("app:isApiMode") as Promise<boolean>,
    selectAudioFile: () => ipcRenderer.invoke("file:selectAudio") as Promise<FileSelectionResult>,
    analyzeAudio: (audioPath: string, options?: AnalyzeAudioOptions) =>
        ipcRenderer.invoke("engine:analyzeAudio", {
            audioPath,
            forceRefresh: options?.forceRefresh === true
        }) as Promise<ChordAnalysisResult>,
    generateLyricsFromAudio: (audioPath: string, options?: GenerateLyricsOptions) =>
        ipcRenderer.invoke("engine:generateLyrics", { audioPath, model: options?.model }) as Promise<LyricsTranscriptionResult>,
    pitchShiftAudio: (audioPath: string, options: PitchShiftOptions) =>
        ipcRenderer.invoke("engine:pitchShiftAudio", { audioPath, semitones: options.semitones }) as Promise<PitchShiftResult>,
    removeVocals: (audioPath: string) =>
        ipcRenderer.invoke("engine:removeVocals", { audioPath }) as Promise<VocalRemovalResult>,
    getAudioPlaybackSource: (audioPath: string) =>
        ipcRenderer.invoke("file:getAudioPlaybackSource", audioPath) as Promise<AudioPlaybackSource>,
    listSongs: (options?: SongLibrarySearchOptions) =>
        ipcRenderer.invoke("library:listSongs", options) as Promise<SongLibraryListResult | SongLibraryRecord[]>,
    getSong: (id: string) => ipcRenderer.invoke("library:getSong", id) as Promise<SongLibraryRecord | null>,
    saveSongAnalysis: (request: SaveSongAnalysisRequest) =>
        ipcRenderer.invoke("library:saveAnalysis", request) as Promise<SongLibraryRecord>,
    deleteSong: (id: string) => ipcRenderer.invoke("library:deleteSong", id) as Promise<DeleteSongResult>
};

contextBridge.exposeInMainWorld("gcd", desktopApi);
