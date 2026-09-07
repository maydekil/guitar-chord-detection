import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";

import { buildSearchIndex, buildSongId, summarizeAnalysis } from "./songMetadata.js";
import type { SongRow } from "./songTypes.js";

export function initializeSongTables(db: DatabaseSync): void {
    db.exec(`
        CREATE TABLE IF NOT EXISTS songs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            audio_stream_url TEXT,
            lyrics TEXT NOT NULL DEFAULT '',
            instrumental_audio_path TEXT,
            instrumental_audio_stream_url TEXT,
            file_hash TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            duration REAL NOT NULL,
            chord_count INTEGER NOT NULL,
            key_estimate TEXT,
            average_confidence REAL,
            search_index TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    `);
    ensureSongColumn(db, "audio_stream_url", "TEXT");
    ensureSongColumn(db, "lyrics", "TEXT NOT NULL DEFAULT ''");
    ensureSongColumn(db, "instrumental_audio_path", "TEXT");
    ensureSongColumn(db, "instrumental_audio_stream_url", "TEXT");
    ensureSongColumn(db, "search_index", "TEXT NOT NULL DEFAULT ''");
    cleanupDuplicateSongRows(db);
    db.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_file_hash_unique ON songs(file_hash);`);
    db.exec(`CREATE INDEX IF NOT EXISTS idx_songs_search ON songs(search_index);`);
    backfillSongSummaries(db);
}

function ensureSongColumn(db: DatabaseSync, columnName: string, definition: string): void {
    const columns = db.prepare("PRAGMA table_info(songs)").all() as Array<{ name: string }>;
    if (!columns.some((column) => column.name === columnName)) {
        db.exec(`ALTER TABLE songs ADD COLUMN ${columnName} ${definition}`);
    }
}

function backfillSongSummaries(db: DatabaseSync): void {
    const rows = db.prepare(`
        SELECT * FROM songs
        WHERE search_index = '' OR chord_count IS NULL OR chord_count = 0 OR average_confidence IS NULL
    `).all() as unknown as SongRow[];

    const statement = db.prepare(`
        UPDATE songs
        SET chord_count = ?, key_estimate = ?, average_confidence = ?, search_index = ?
        WHERE id = ?
    `);

    for (const row of rows) {
        try {
            const analysis = JSON.parse(row.analysis_json) as SongRowAnalysis;
            const summary = summarizeAnalysis(analysis);
            statement.run(
                summary.chordCount,
                summary.keyEstimate,
                summary.averageConfidence,
                buildSearchIndex(row.title, row.artist, row.audio_path),
                row.id,
            );
        } catch {
            statement.run(
                row.chord_count ?? 0,
                row.key_estimate,
                row.average_confidence,
                buildSearchIndex(row.title, row.artist, row.audio_path),
                row.id,
            );
        }
    }
}

function cleanupDuplicateSongRows(db: DatabaseSync): void {
    migrateInvalidSongFileHashes(db);
    db.exec(`
        DELETE FROM songs
        WHERE rowid NOT IN (
            SELECT rowid
            FROM (
                SELECT rowid, ROW_NUMBER() OVER (
                    PARTITION BY file_hash
                    ORDER BY updated_at DESC, created_at DESC, rowid DESC
                ) AS duplicate_rank
                FROM songs
            ) ranked
            WHERE duplicate_rank = 1
        );
    `);
    db.exec(`
        UPDATE songs
        SET id = 'song:' || file_hash
        WHERE id <> 'song:' || file_hash
          AND NOT EXISTS (
              SELECT 1 FROM songs other_song WHERE other_song.id = 'song:' || songs.file_hash
          );
    `);
}

function migrateInvalidSongFileHashes(db: DatabaseSync): void {
    const rows = db.prepare("SELECT id, audio_path, file_hash, updated_at, created_at FROM songs").all() as Array<{
        id: string;
        audio_path: string;
        file_hash: string;
        updated_at: string;
        created_at: string;
    }>;

    const update = db.prepare("UPDATE songs SET id = ?, file_hash = ? WHERE id = ?");
    const remove = db.prepare("DELETE FROM songs WHERE id = ?");
    const existingHashes = new Map(rows.map((row) => [row.file_hash, row.id]));

    for (const row of rows) {
        if (/^[a-f0-9]{64}$/i.test(row.file_hash)) {
            continue;
        }
        try {
            const fileHash = createHash("sha256").update(readFileSync(row.audio_path)).digest("hex");
            const canonicalId = buildSongId(fileHash);
            const duplicateId = existingHashes.get(fileHash);
            if (duplicateId && duplicateId !== row.id) {
                remove.run(row.id);
                continue;
            }
            update.run(canonicalId, fileHash, row.id);
            existingHashes.set(fileHash, canonicalId);
        } catch {
            remove.run(row.id);
        }
    }
}

type SongRowAnalysis = Parameters<typeof summarizeAnalysis>[0];
