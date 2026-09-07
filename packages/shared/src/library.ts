import type { ChordAnalysisSuccess } from "./analysis.js";

export interface SongLibrarySummary {
    id: string;
    title: string;
    artist: string;
    audioPath: string;
    audioStreamUrl?: string;
    lyrics?: string;
    instrumentalAudioPath?: string;
    instrumentalAudioStreamUrl?: string;
    fileHash: string;
    algorithm: string;
    contractVersion: string;
    duration: number;
    chordCount: number;
    keyEstimate?: string;
    averageConfidence?: number;
    createdAt: string;
    updatedAt: string;
}

export interface SongLibraryRecord extends SongLibrarySummary {
    analysis: ChordAnalysisSuccess;
}

export interface SongLibrarySearchOptions {
    query?: string;
    page?: number;
    pageSize?: number;
}

export interface SongLibraryListResult {
    records: SongLibrarySummary[];
    total: number;
    page: number;
    pageSize: number;
    totalPages: number;
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
    instrumentalAudioPath?: string;
}

export interface DeleteSongResult {
    deleted: boolean;
}
