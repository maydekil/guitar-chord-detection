import type { ChordAnalysisSuccess } from "./analysis.js";

export interface SongLibraryRecord {
    id: string;
    title: string;
    artist: string;
    audioPath: string;
    lyrics?: string;
    fileHash: string;
    algorithm: string;
    contractVersion: string;
    duration: number;
    analysis: ChordAnalysisSuccess;
    createdAt: string;
    updatedAt: string;
}

export interface SongLibrarySearchOptions {
    query?: string;
}

export interface SongMetadataInput {
    title: string;
    artist: string;
}

export interface SaveSongAnalysisRequest {
    audioPath: string;
    metadata: SongMetadataInput;
    analysis: ChordAnalysisSuccess;
    lyrics?: string;
}

export interface DeleteSongResult {
    deleted: boolean;
}
