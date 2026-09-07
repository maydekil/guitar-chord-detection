import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { copyFile, mkdir, readFile, unlink } from "node:fs/promises";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import type { ChordAnalysisSuccess } from "@gcd/shared/analysis";
import type { DeleteSongResult, SongLibraryListResult, SongLibraryRecord, SongLibrarySearchOptions, SongLibrarySummary, SongMetadataInput } from "@gcd/shared/library";

interface SongLibraryStoreFile {
    records: Record<string, SongLibraryRecord>;
}

interface SongRow {
    id: string;
    title: string;
    artist: string;
    audio_path: string;
    lyrics: string | null;
    instrumental_audio_path: string | null;
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

interface UpsertSongAnalysisOptions {
    audioPath: string;
    algorithm: string;
    contractVersion: string;
    analysis: ChordAnalysisSuccess;
    metadata: SongMetadataInput;
    lyrics?: string;
    instrumentalAudioPath?: string;
}

interface SongLibraryDependencies {
    readFile: typeof readFile;
    mkdir: typeof mkdir;
    copyFile: typeof copyFile;
    unlink: typeof unlink;
    now: () => Date;
}

export class SongLibraryStore {
    private readonly storeFilePath: string;
    private readonly deps: SongLibraryDependencies;
    private database: DatabaseSync | null = null;
    private initialized = false;

    constructor(storeFilePath: string, deps?: Partial<SongLibraryDependencies>) {
        this.storeFilePath = storeFilePath;
        this.deps = {
            readFile,
            mkdir,
            copyFile,
            unlink,
            now: () => new Date(),
            ...deps
        };
    }

    async listSongs(options?: SongLibrarySearchOptions): Promise<SongLibraryListResult> {
        const database = await this.getDatabase();
        const query = normalizeSearch(options?.query ?? "");
        const pageSize = normalizePageSize(options?.pageSize);
        const page = normalizePage(options?.page);
        const offset = (page - 1) * pageSize;
        const countStatement = query
            ? database.prepare("SELECT COUNT(*) AS total FROM songs WHERE search_index LIKE ?")
            : database.prepare("SELECT COUNT(*) AS total FROM songs");
        const total = query
            ? (countStatement.get(`%${query}%`) as { total: number }).total
            : (countStatement.get() as { total: number }).total;
        const statement = query
            ? database.prepare(`
                SELECT id, title, artist, audio_path, lyrics, instrumental_audio_path, file_hash,
                    algorithm, contract_version, duration, chord_count, key_estimate,
                    average_confidence, created_at, updated_at
                FROM songs
                WHERE search_index LIKE ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            `)
            : database.prepare(`
                SELECT id, title, artist, audio_path, lyrics, instrumental_audio_path, file_hash,
                    algorithm, contract_version, duration, chord_count, key_estimate,
                    average_confidence, created_at, updated_at
                FROM songs
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            `);
        const rows = query
            ? statement.all(`%${query}%`, pageSize, offset) as unknown as SongRow[]
            : statement.all(pageSize, offset) as unknown as SongRow[];
        return {
            records: rows.map(rowToSummary),
            total,
            page,
            pageSize,
            totalPages: Math.max(1, Math.ceil(total / pageSize))
        };
    }

    async getSong(id: string): Promise<SongLibraryRecord | null> {
        const database = await this.getDatabase();
        const row = database.prepare("SELECT * FROM songs WHERE id = ?").get(id) as SongRow | undefined;
        return row ? rowToRecord(row) : null;
    }

    async upsertAnalysis(options: UpsertSongAnalysisOptions): Promise<SongLibraryRecord> {
        const database = await this.getDatabase();
        const fileHash = buildFileHash(await this.deps.readFile(options.audioPath));
        const id = buildSongId(fileHash);
        const existing = await this.getSong(id);
        const timestamp = this.deps.now().toISOString();
        const title = options.metadata.title.trim();
        const artist = options.metadata.artist.trim();
        const managedAudioPath = this.resolveManagedAudioPath(fileHash, options.algorithm, options.contractVersion, options.audioPath);

        await this.deps.mkdir(path.dirname(managedAudioPath), { recursive: true });
        if (path.resolve(options.audioPath) !== path.resolve(managedAudioPath)) {
            await this.deps.copyFile(options.audioPath, managedAudioPath);
        }

        const summary = summarizeAnalysis(options.analysis);
        const record: SongLibraryRecord = {
            id,
            title,
            artist,
            audioPath: managedAudioPath,
            lyrics: options.lyrics ?? existing?.lyrics ?? "",
            instrumentalAudioPath: options.instrumentalAudioPath ?? existing?.instrumentalAudioPath,
            fileHash,
            algorithm: options.algorithm,
            contractVersion: options.contractVersion,
            duration: options.analysis.source.duration,
            chordCount: summary.chordCount,
            keyEstimate: summary.keyEstimate,
            averageConfidence: summary.averageConfidence,
            analysis: options.analysis,
            createdAt: existing?.createdAt ?? timestamp,
            updatedAt: timestamp
        };

        this.upsertRecord(database, record);
        return record;
    }

    async deleteSong(id: string): Promise<DeleteSongResult> {
        const database = await this.getDatabase();
        const existing = await this.getSong(id);
        if (!existing) {
            return { deleted: false };
        }

        database.prepare("DELETE FROM songs WHERE id = ?").run(id);

        try {
            await this.deps.unlink(existing.audioPath);
        } catch (error) {
            if (!isMissingFileError(error)) {
                console.error("[library] failed to delete managed audio file", error);
            }
        }

        if (existing.instrumentalAudioPath) {
            try {
                await this.deps.unlink(existing.instrumentalAudioPath);
            } catch (error) {
                if (!isMissingFileError(error)) {
                    console.error("[library] failed to delete instrumental audio file", error);
                }
            }
        }

        return { deleted: true };
    }

    private async getDatabase(): Promise<DatabaseSync> {
        if (!this.database) {
            await this.deps.mkdir(path.dirname(this.storeFilePath), { recursive: true });
            this.database = new DatabaseSync(this.storeFilePath);
        }
        if (!this.initialized) {
            this.initializeDatabase(this.database);
            await this.migrateLegacyJsonIfNeeded(this.database);
            this.initialized = true;
        }
        return this.database;
    }

    private initializeDatabase(database: DatabaseSync): void {
        database.exec(`
            CREATE TABLE IF NOT EXISTS songs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                lyrics TEXT NOT NULL DEFAULT '',
                instrumental_audio_path TEXT,
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
        this.ensureColumn(database, "chord_count", "INTEGER NOT NULL DEFAULT 0");
        this.ensureColumn(database, "key_estimate", "TEXT");
        this.ensureColumn(database, "average_confidence", "REAL");
        this.ensureColumn(database, "search_index", "TEXT NOT NULL DEFAULT ''");
        database.exec("CREATE INDEX IF NOT EXISTS idx_songs_search_index ON songs(search_index);");
        this.backfillSongSummaries(database);
        this.cleanupDuplicateRows(database);
        database.exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_file_hash_unique ON songs(file_hash);");
    }

    private ensureColumn(database: DatabaseSync, columnName: string, definition: string): void {
        const columns = database.prepare("PRAGMA table_info(songs)").all() as unknown as Array<{ name: string }>;
        if (columns.some((column) => column.name === columnName)) {
            return;
        }
        database.exec(`ALTER TABLE songs ADD COLUMN ${columnName} ${definition};`);
    }

    private backfillSongSummaries(database: DatabaseSync): void {
        const rows = database.prepare(`
            SELECT id, title, artist, audio_path, analysis_json
            FROM songs
            WHERE search_index = ''
                OR chord_count = 0
                OR average_confidence IS NULL
        `).all() as unknown as Array<{ id: string; title: string; artist: string; audio_path: string; analysis_json: string }>;

        const statement = database.prepare(`
            UPDATE songs
            SET chord_count = ?, key_estimate = ?, average_confidence = ?, search_index = ?
            WHERE id = ?
        `);
        for (const row of rows) {
            try {
                const analysis = JSON.parse(row.analysis_json) as ChordAnalysisSuccess;
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

    private cleanupDuplicateRows(database: DatabaseSync): void {
        this.migrateInvalidFileHashRows(database);
        database.exec(`
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

    private migrateInvalidFileHashRows(database: DatabaseSync): void {
        const rows = database.prepare(`
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
                fileHash = buildFileHash(readFileSync(row.audio_path));
            } catch {
                database.prepare("DELETE FROM songs WHERE id = ?").run(row.id);
                continue;
            }

            const nextId = buildSongId(fileHash);
            const existing = database.prepare("SELECT id FROM songs WHERE (id = ? OR file_hash = ?) AND id <> ?").get(nextId, fileHash, row.id);
            if (existing) {
                database.prepare("DELETE FROM songs WHERE id = ?").run(row.id);
                continue;
            }

            database.prepare("UPDATE songs SET id = ?, file_hash = ? WHERE id = ?").run(nextId, fileHash, row.id);
        }
    }

    private async migrateLegacyJsonIfNeeded(database: DatabaseSync): Promise<void> {
        const count = database.prepare("SELECT COUNT(*) AS count FROM songs").get() as { count: number };
        if (count.count > 0) {
            return;
        }

        try {
            const content = await this.deps.readFile(resolveLegacySongLibraryPath(path.dirname(this.storeFilePath)), "utf-8");
            const parsed = JSON.parse(content) as SongLibraryStoreFile;
            if (!parsed || typeof parsed !== "object" || !parsed.records || typeof parsed.records !== "object") {
                return;
            }

            for (const record of Object.values(parsed.records)) {
                this.upsertRecord(database, record);
            }
        } catch (error) {
            if (!isMissingFileError(error)) {
                console.error("[library] failed to migrate legacy JSON song library", error);
            }
        }
    }

    private upsertRecord(database: DatabaseSync, record: SongLibraryRecord): void {
        const statement = database.prepare(`
            INSERT INTO songs (
                id,
                title,
                artist,
                audio_path,
                lyrics,
                instrumental_audio_path,
                file_hash,
                algorithm,
                contract_version,
                duration,
                chord_count,
                key_estimate,
                average_confidence,
                search_index,
                analysis_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                artist = excluded.artist,
                audio_path = excluded.audio_path,
                lyrics = excluded.lyrics,
                instrumental_audio_path = excluded.instrumental_audio_path,
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
        const summary = summarizeAnalysis(record.analysis);
        statement.run(
            record.id,
            record.title,
            record.artist,
            record.audioPath,
            record.lyrics ?? "",
            record.instrumentalAudioPath ?? null,
            record.fileHash,
            record.algorithm,
            record.contractVersion,
            record.duration,
            summary.chordCount,
            summary.keyEstimate ?? null,
            summary.averageConfidence ?? null,
            buildSearchIndex(record.title, record.artist, record.audioPath),
            JSON.stringify(record.analysis),
            record.createdAt,
            record.updatedAt
        );
    }

    private resolveManagedAudioPath(fileHash: string, algorithm: string, contractVersion: string, sourceAudioPath: string): string {
        const extension = path.extname(sourceAudioPath).toLowerCase() || ".audio";
        const safeAlgorithm = algorithm.replace(/[^a-zA-Z0-9.-]/g, "-");
        const safeContractVersion = contractVersion.replace(/[^a-zA-Z0-9.-]/g, "-");
        return path.join(path.dirname(this.storeFilePath), "audio", `${fileHash}-${safeAlgorithm}-${safeContractVersion}${extension}`);
    }
}

export function resolveSongLibraryPath(userDataPath: string): string {
    return path.join(userDataPath, "song-library-v1.sqlite");
}

function resolveLegacySongLibraryPath(userDataPath: string): string {
    return path.join(userDataPath, "song-library-v1.json");
}

function rowToRecord(row: SongRow): SongLibraryRecord {
    return {
        ...rowToSummary(row),
        analysis: JSON.parse(row.analysis_json) as ChordAnalysisSuccess
    };
}

function rowToSummary(row: SongRow): SongLibrarySummary {
    return {
        id: row.id,
        title: row.title,
        artist: row.artist,
        audioPath: row.audio_path,
        lyrics: row.lyrics ?? "",
        instrumentalAudioPath: row.instrumental_audio_path ?? undefined,
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

function buildFileHash(content: Buffer): string {
    return createHash("sha256").update(content).digest("hex");
}

function buildSongId(fileHash: string): string {
    return `song:${fileHash}`;
}

function normalizeSearch(value: string): string {
    return value.trim().toLocaleLowerCase();
}

function buildSearchIndex(title: string, artist: string, audioPath: string): string {
    return normalizeSearch(`${title} ${artist} ${audioPath}`);
}

function summarizeAnalysis(analysis: ChordAnalysisSuccess): Pick<SongLibrarySummary, "averageConfidence" | "chordCount" | "keyEstimate"> {
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

function normalizePage(value: number | undefined): number {
    if (!value || !Number.isFinite(value)) {
        return 1;
    }
    return Math.max(1, Math.trunc(value));
}

function normalizePageSize(value: number | undefined): number {
    if (!value || !Number.isFinite(value)) {
        return 10;
    }
    return Math.max(5, Math.min(100, Math.trunc(value)));
}

function isMissingFileError(error: unknown): boolean {
    if (!error || typeof error !== "object") {
        return false;
    }
    return (error as { code?: unknown }).code === "ENOENT";
}
