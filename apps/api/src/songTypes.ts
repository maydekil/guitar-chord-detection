import type { ChordAnalysisSuccess } from "@gcd/shared/analysis";

export interface SongRow {
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
    chord_count: number | null;
    key_estimate: string | null;
    average_confidence: number | null;
    search_index: string | null;
    analysis_json: string;
    created_at: string;
    updated_at: string;
}

export interface AnalysisSummary {
    chordCount: number;
    keyEstimate: string | null;
    averageConfidence: number | null;
}

export type SongAnalysis = ChordAnalysisSuccess;
