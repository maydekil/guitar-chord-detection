import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { unlink } from "node:fs/promises";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import type { SaveSongAnalysisRequest, SongLibraryListResult, SongLibraryRecord, SongLibrarySummary } from "@gcd/shared/library";

interface SongRow {
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
            chord_count INTEGER NOT NULL DEFAULT 0,
            key_estimate TEXT,
            average_confidence REAL,
            search_index TEXT NOT NULL DEFAULT '',
            analysis_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_songs_updated_at ON songs(updated_at);
        CREATE INDEX IF NOT EXISTS idx_songs_title_artist ON songs(title, artist);
    `);
    ensureSongColumn(db, "chord_count", "INTEGER NOT NULL DEFAULT 0");
    ensureSongColumn(db, "key_estimate", "TEXT");
    ensureSongColumn(db, "average_confidence", "REAL");
    ensureSongColumn(db, "search_index", "TEXT NOT NULL DEFAULT ''");
    db.exec("CREATE INDEX IF NOT EXISTS idx_songs_search_index ON songs(search_index);");
    backfillSongSummaries(db);
    cleanupDuplicateSongRows(db);
    db.exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_file_hash_unique ON songs(file_hash);");
}

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
            SELECT id, title, artist, audio_path, audio_stream_url, lyrics,
                instrumental_audio_path, instrumental_audio_stream_url, file_hash,
                algorithm, contract_version, duration, chord_count, key_estimate,
                average_confidence, created_at, updated_at
            FROM songs
            WHERE search_index LIKE ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        `).all(`%${query}%`, pageSize, offset) as unknown as SongRow[]
        : db.prepare(`
            SELECT id, title, artist, audio_path, audio_stream_url, lyrics,
                instrumental_audio_path, instrumental_audio_stream_url, file_hash,
                algorithm, contract_version, duration, chord_count, key_estimate,
                average_confidence, created_at, updated_at
            FROM songs
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        `).all(pageSize, offset) as unknown as SongRow[];

    return {
        records: rows.map(rowToSummary),
        total,
        page,
        pageSize,
        totalPages: Math.max(1, Math.ceil(total / pageSize))
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
        lyrics: request.lyrics ?? existing?.lyrics ?? "",
        instrumentalAudioPath: request.instrumentalAudioPath ?? existing?.instrumentalAudioPath,
        instrumentalAudioStreamUrl: request.instrumentalAudioPath
            ? `${publicBaseUrl}/songs/${encodeURIComponent(id)}/audio/instrumental/stream`
            : existing?.instrumentalAudioStreamUrl,
        fileHash,
        algorithm: request.analysis.analysis.algorithm,
        contractVersion: request.analysis.version,
        duration: request.analysis.source.duration,
        chordCount: analysisSummary.chordCount,
        keyEstimate: analysisSummary.keyEstimate,
        averageConfidence: analysisSummary.averageConfidence,
        analysis: request.analysis,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now
    };

    const statement = db.prepare(`
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
            file_hash = excluded.file_hash,
            algorithm = excluded.algorithm,
            contract_version = excluded.contract_version,
            duration = excluded.duration,
            chord_count = excluded.chord_count,
            key_estimate = excluded.key_estimate,
            average_confidence = excluded.average_confidence,
            search_index = excluded.search_index,
            analysis_json = excluded.analysis_json,
            updated_at = excluded.updated_at
    `);
    statement.run(
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
        analysisSummary.chordCount,
        analysisSummary.keyEstimate ?? null,
        analysisSummary.averageConfidence ?? null,
        buildSearchIndex(record.title, record.artist, record.audioPath),
        JSON.stringify(record.analysis),
        record.createdAt,
        record.updatedAt
    );

    return record;
}

export function deleteSong(db: DatabaseSync, id: string, audioDir: string): boolean {
    const existing = getSong(db, id);
    const result = db.prepare("DELETE FROM songs WHERE id = ?").run(id);
    const deleted = Number(result.changes) > 0;
    if (deleted && existing) {
        void unlinkManagedAudio(existing.audioPath, audioDir);
        if (existing.instrumentalAudioPath) {
            void unlinkManagedAudio(existing.instrumentalAudioPath, audioDir);
        }
    }
    return deleted;
}

export function collectReferencedAudioPaths(db: DatabaseSync): Set<string> {
    const rows = db.prepare("SELECT audio_path, instrumental_audio_path FROM songs").all() as unknown as Array<{
        audio_path: string | null;
        instrumental_audio_path: string | null;
    }>;
    const referenced = new Set<string>();
    for (const row of rows) {
        addManagedAudioReference(referenced, row.audio_path);
        addManagedAudioReference(referenced, row.instrumental_audio_path);
    }
    return referenced;
}

function cleanupDuplicateSongRows(db: DatabaseSync): void {
    migrateInvalidSongFileHashes(db);
    db.exec(`
        DELETE FROM songs
        WHERE EXISTS (
            SELECT 1
            FROM songs newer
            WHERE newer.file_hash = songs.file_hash
                AND (
                    newer.updated_at > songs.updated_at
                    OR (newer.updated_at = songs.updated_at AND newer.id > songs.id)
                )
        );

        UPDATE songs
        SET id = 'song:' || file_hash
        WHERE id <> 'song:' || file_hash
            AND NOT EXISTS (
                SELECT 1
                FROM songs existing
                WHERE existing.id = 'song:' || songs.file_hash
            );
    `);
}

function ensureSongColumn(db: DatabaseSync, columnName: string, definition: string): void {
    const columns = db.prepare("PRAGMA table_info(songs)").all() as unknown as Array<{ name: string }>;
    if (columns.some((column) => column.name === columnName)) {
        return;
    }
    db.exec(`ALTER TABLE songs ADD COLUMN ${columnName} ${definition};`);
}

function backfillSongSummaries(db: DatabaseSync): void {
    const rows = db.prepare(`
        SELECT id, title, artist, audio_path, analysis_json
        FROM songs
        WHERE search_index = ''
            OR chord_count = 0
            OR average_confidence IS NULL
    `).all() as unknown as Array<{ id: string; title: string; artist: string; audio_path: string; analysis_json: string }>;
    const statement = db.prepare(`
        UPDATE songs
        SET chord_count = ?, key_estimate = ?, average_confidence = ?, search_index = ?
        WHERE id = ?
    `);

    for (const row of rows) {
        try {
            const analysis = JSON.parse(row.analysis_json) as SongLibraryRecord["analysis"];
            const summary = summarizeAnalysis(analysis);
            statement.run(
                summary.chordCount,
                summary.keyEstimate ?? null,
                summary.averageConfidence ?? null,
                buildSearchIndex(row.title, row.artist, row.audio_path),
                row.id
            );
        } catch {
            statement.run(0, null, null, buildSearchIndex(row.title, row.artist, row.audio_path), row.id);
        }
    }
}

function migrateInvalidSongFileHashes(db: DatabaseSync): void {
    const rows = db.prepare(`
        SELECT id, audio_path, file_hash
        FROM songs
        WHERE file_hash LIKE '/%'
            OR file_hash LIKE '%.mp3'
            OR file_hash LIKE '%.wav'
            OR id LIKE 'song:/%'
    `).all() as unknown as Array<{ id: string; audio_path: string; file_hash: string }>;

    for (const row of rows) {
        let fileHash: string;
        try {
            fileHash = createHash("sha256").update(readFileSync(row.audio_path)).digest("hex");
        } catch {
            db.prepare("DELETE FROM songs WHERE id = ?").run(row.id);
            continue;
        }

        const nextId = buildSongId(fileHash);
        const existing = db.prepare("SELECT id FROM songs WHERE (id = ? OR file_hash = ?) AND id <> ?").get(nextId, fileHash, row.id);
        if (existing) {
            db.prepare("DELETE FROM songs WHERE id = ?").run(row.id);
            continue;
        }

        db.prepare("UPDATE songs SET id = ?, file_hash = ? WHERE id = ?").run(nextId, fileHash, row.id);
    }
}

function rowToRecord(row: SongRow): SongLibraryRecord {
    return {
        ...rowToSummary(row),
        analysis: JSON.parse(row.analysis_json) as SongLibraryRecord["analysis"]
    };
}

function rowToSummary(row: SongRow): SongLibrarySummary {
    return {
        id: row.id,
        title: row.title,
        artist: row.artist,
        audioPath: row.audio_path,
        audioStreamUrl: row.audio_stream_url ?? undefined,
        lyrics: row.lyrics,
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
        updatedAt: row.updated_at
    };
}

function buildStableFileHash(audioPath: string, fallback: string): string {
    try {
        return createHash("sha256").update(readFileSync(audioPath)).digest("hex");
    } catch {
        return fallback || audioPath;
    }
}

function buildSongId(fileHash: string): string {
    return `song:${fileHash}`;
}

function buildSearchIndex(title: string, artist: string, audioPath: string): string {
    return normalizeSearch(`${title} ${artist} ${audioPath}`);
}

function summarizeAnalysis(analysis: SongLibraryRecord["analysis"]): Pick<SongLibrarySummary, "averageConfidence" | "chordCount" | "keyEstimate"> {
    const segments = analysis.analysis.chords;
    const chordCount = segments.length;
    const averageConfidence = chordCount > 0
        ? segments.reduce((total, segment) => total + segment.confidence, 0) / chordCount
        : undefined;
    const durations = new Map<string, number>();
    for (const segment of segments) {
        if (segment.chord === "N") {
            continue;
        }
        const root = segment.chord.endsWith("m") ? segment.chord.slice(0, -1) : segment.chord;
        durations.set(root, (durations.get(root) ?? 0) + Math.max(0, segment.end - segment.start));
    }
    const keyEstimate = [...durations.entries()].sort((left, right) => right[1] - left[1])[0]?.[0];
    return { averageConfidence, chordCount, keyEstimate };
}

function addManagedAudioReference(referenced: Set<string>, audioPath: string | null | undefined): void {
    if (!audioPath) {
        return;
    }
    referenced.add(path.resolve(audioPath));
}

async function unlinkManagedAudio(audioPath: string, audioDir: string): Promise<void> {
    if (!path.resolve(audioPath).startsWith(`${path.resolve(audioDir)}${path.sep}`)) {
        return;
    }

    await unlink(audioPath).catch((error: unknown) => {
        if (!(error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "ENOENT")) {
            console.error("[api] failed to delete managed audio", error);
        }
    });
}

function normalizeSearch(value: string): string {
    return value.trim().toLocaleLowerCase();
}

function normalizePage(value: number): number {
    return Number.isFinite(value) ? Math.max(1, Math.trunc(value)) : 1;
}

function normalizePageSize(value: number): number {
    return Number.isFinite(value) ? Math.max(5, Math.min(100, Math.trunc(value))) : 10;
}
