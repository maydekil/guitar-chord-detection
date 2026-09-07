import type { SongLibraryRecord, SongLibrarySummary } from "@gcd/shared/library";

import type { SongRow } from "./songTypes.js";

export function rowToRecord(row: SongRow): SongLibraryRecord {
    return {
        ...rowToSummary(row),
        analysis: JSON.parse(row.analysis_json) as SongLibraryRecord["analysis"],
    };
}

export function rowToSummary(row: SongRow): SongLibrarySummary {
    return {
        id: row.id,
        title: row.title,
        artist: row.artist,
        audioPath: row.audio_path,
        audioStreamUrl: row.audio_stream_url ?? undefined,
        lyrics: row.lyrics || undefined,
        instrumentalAudioPath: row.instrumental_audio_path ?? undefined,
        instrumentalAudioStreamUrl: row.instrumental_audio_stream_url ?? undefined,
        fileHash: row.file_hash,
        algorithm: row.algorithm,
        contractVersion: row.contract_version,
        duration: row.duration,
        chordCount: row.chord_count ?? 0,
        keyEstimate: row.key_estimate ?? undefined,
        averageConfidence: row.average_confidence ?? undefined,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
    };
}
