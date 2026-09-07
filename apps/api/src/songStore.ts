import { unlink } from "node:fs/promises";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import type { SaveSongAnalysisRequest, SongLibraryListResult, SongLibraryRecord } from "@gcd/shared/library";

import { normalizePage, normalizePageSize, normalizeSearch } from "./queryParams.js";
import { buildSearchIndex, buildSongId, buildStableFileHash, summarizeAnalysis } from "./songMetadata.js";
import { rowToRecord, rowToSummary } from "./songMappers.js";
import { initializeSongTables } from "./songSchema.js";
import type { SongRow } from "./songTypes.js";

export { initializeSongTables };

export function listSongs(db: DatabaseSync, url: URL): SongLibraryListResult {
    const query = normalizeSearch(url.searchParams.get("query") ?? "");
    const page = normalizePage(Number.parseInt(url.searchParams.get("page") ?? "1", 10));
    const pageSize = normalizePageSize(Number.parseInt(url.searchParams.get("pageSize") ?? "10", 10));
    const offset = (page - 1) * pageSize;
    const countStatement = query
        ? db.prepare("SELECT COUNT(*) AS total FROM songs WHERE search_index LIKE ?")
        : db.prepare("SELECT COUNT(*) AS total FROM songs");
    const total = query
        ? (countStatement.get(`%${query}%`) as { total: number }).total
        : (countStatement.get() as { total: number }).total;
    const rows = query
        ? db.prepare(`
            SELECT * FROM songs
            WHERE search_index LIKE ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        `).all(`%${query}%`, pageSize, offset) as unknown as SongRow[]
        : db.prepare(`
            SELECT * FROM songs
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        `).all(pageSize, offset) as unknown as SongRow[];

    return {
        records: rows.map(rowToSummary),
        total,
        page,
        pageSize,
        totalPages: Math.max(1, Math.ceil(total / pageSize)),
    };
}

export function getSong(db: DatabaseSync, id: string): SongLibraryRecord | null {
    const row = db.prepare("SELECT * FROM songs WHERE id = ?").get(id) as SongRow | undefined;
    return row ? rowToRecord(row) : null;
}

export function saveSong(db: DatabaseSync, request: SaveSongAnalysisRequest, publicBaseUrl: string): SongLibraryRecord {
    const now = new Date().toISOString();
    const fileHash = buildStableFileHash(request.audioPath, request.analysis.source.path);
    const id = buildSongId(fileHash);
    const existing = getSong(db, id);
    const analysisSummary = summarizeAnalysis(request.analysis);
    const record: SongLibraryRecord = {
        id,
        title: request.metadata.title.trim(),
        artist: request.metadata.artist.trim(),
        audioPath: request.audioPath,
        audioStreamUrl: `${publicBaseUrl}/songs/${encodeURIComponent(id)}/audio/original/stream`,
        lyrics: request.lyrics ?? existing?.lyrics,
        instrumentalAudioPath: request.instrumentalAudioPath ?? existing?.instrumentalAudioPath,
        instrumentalAudioStreamUrl: request.instrumentalAudioPath || existing?.instrumentalAudioPath
            ? `${publicBaseUrl}/songs/${encodeURIComponent(id)}/audio/instrumental/stream`
            : undefined,
        fileHash,
        algorithm: request.analysis.analysis.algorithm,
        contractVersion: request.analysis.version,
        duration: request.analysis.source.duration,
        chordCount: analysisSummary.chordCount,
        keyEstimate: analysisSummary.keyEstimate ?? undefined,
        averageConfidence: analysisSummary.averageConfidence ?? undefined,
        analysis: request.analysis,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
    };

    db.prepare(`
        INSERT INTO songs (
            id, title, artist, audio_path, audio_stream_url, lyrics,
            instrumental_audio_path, instrumental_audio_stream_url, file_hash,
            algorithm, contract_version, duration, chord_count, key_estimate,
            average_confidence, search_index, analysis_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            artist = excluded.artist,
            audio_path = excluded.audio_path,
            audio_stream_url = excluded.audio_stream_url,
            lyrics = excluded.lyrics,
            instrumental_audio_path = excluded.instrumental_audio_path,
            instrumental_audio_stream_url = excluded.instrumental_audio_stream_url,
            algorithm = excluded.algorithm,
            contract_version = excluded.contract_version,
            duration = excluded.duration,
            chord_count = excluded.chord_count,
            key_estimate = excluded.key_estimate,
            average_confidence = excluded.average_confidence,
            search_index = excluded.search_index,
            analysis_json = excluded.analysis_json,
            updated_at = excluded.updated_at
    `).run(
        record.id,
        record.title,
        record.artist,
        record.audioPath,
        record.audioStreamUrl ?? null,
        record.lyrics ?? "",
        record.instrumentalAudioPath ?? null,
        record.instrumentalAudioStreamUrl ?? null,
        record.fileHash,
        record.algorithm,
        record.contractVersion,
        record.duration,
        record.chordCount,
        record.keyEstimate ?? null,
        record.averageConfidence ?? null,
        buildSearchIndex(record.title, record.artist, record.audioPath),
        JSON.stringify(record.analysis),
        record.createdAt,
        record.updatedAt,
    );

    return record;
}

export async function deleteSong(db: DatabaseSync, id: string, audioDir: string): Promise<boolean> {
    const existing = getSong(db, id);
    if (!existing) {
        return false;
    }
    const result = db.prepare("DELETE FROM songs WHERE id = ?").run(id);
    if (result.changes > 0) {
        await unlinkManagedAudio(existing.audioPath, audioDir);
        await unlinkManagedAudio(existing.instrumentalAudioPath, audioDir);
    }
    return result.changes > 0;
}

export function collectReferencedAudioPaths(db: DatabaseSync): Set<string> {
    const rows = db.prepare("SELECT audio_path, instrumental_audio_path FROM songs").all() as Array<{
        audio_path: string | null;
        instrumental_audio_path: string | null;
    }>;
    const paths = new Set<string>();
    for (const row of rows) {
        addManagedAudioReference(paths, row.audio_path);
        addManagedAudioReference(paths, row.instrumental_audio_path);
    }
    return paths;
}

function addManagedAudioReference(paths: Set<string>, audioPath: string | null | undefined): void {
    if (audioPath) {
        paths.add(path.resolve(audioPath));
    }
}

async function unlinkManagedAudio(audioPath: string | null | undefined, audioDir: string): Promise<void> {
    if (!audioPath) {
        return;
    }
    const resolvedPath = path.resolve(audioPath);
    const resolvedAudioDir = path.resolve(audioDir);
    if (!resolvedPath.startsWith(`${resolvedAudioDir}${path.sep}`)) {
        return;
    }
    try {
        await unlink(resolvedPath);
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
            console.error("Failed to remove managed audio asset", error);
        }
    }
}
