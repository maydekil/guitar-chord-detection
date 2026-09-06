import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ChordAnalysisSuccess } from "@gcd/shared/analysis";
import type { DeleteSongResult, SongLibraryRecord, SongLibrarySearchOptions, SongMetadataInput } from "@gcd/shared/library";

interface SongLibraryStoreFile {
    records: Record<string, SongLibraryRecord>;
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
    writeFile: typeof writeFile;
    mkdir: typeof mkdir;
    copyFile: typeof copyFile;
    unlink: typeof unlink;
    now: () => Date;
}

export class SongLibraryStore {
    private readonly storeFilePath: string;
    private readonly deps: SongLibraryDependencies;

    constructor(storeFilePath: string, deps?: Partial<SongLibraryDependencies>) {
        this.storeFilePath = storeFilePath;
        this.deps = {
            readFile,
            writeFile,
            mkdir,
            copyFile,
            unlink,
            now: () => new Date(),
            ...deps
        };
    }

    async listSongs(options?: SongLibrarySearchOptions): Promise<SongLibraryRecord[]> {
        const store = await this.loadStore();
        const query = normalizeSearch(options?.query ?? "");
        const records = Object.values(store.records).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

        if (!query) {
            return records;
        }

        return records.filter((record) => {
            const haystack = normalizeSearch(`${record.title} ${record.artist} ${record.audioPath}`);
            return haystack.includes(query);
        });
    }

    async getSong(id: string): Promise<SongLibraryRecord | null> {
        const store = await this.loadStore();
        return store.records[id] ?? null;
    }

    async upsertAnalysis(options: UpsertSongAnalysisOptions): Promise<SongLibraryRecord> {
        const store = await this.loadStore();
        const fileHash = buildFileHash(await this.deps.readFile(options.audioPath));
        const id = `${fileHash}:${options.algorithm}:${options.contractVersion}`;
        const existing = store.records[id];
        const timestamp = this.deps.now().toISOString();
        const title = options.metadata.title.trim();
        const artist = options.metadata.artist.trim();
        const managedAudioPath = this.resolveManagedAudioPath(fileHash, options.algorithm, options.contractVersion, options.audioPath);

        await this.deps.mkdir(path.dirname(managedAudioPath), { recursive: true });
        if (path.resolve(options.audioPath) !== path.resolve(managedAudioPath)) {
            await this.deps.copyFile(options.audioPath, managedAudioPath);
        }

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
            analysis: options.analysis,
            createdAt: existing?.createdAt ?? timestamp,
            updatedAt: timestamp
        };

        store.records[id] = record;
        await this.saveStore(store);
        return record;
    }

    async deleteSong(id: string): Promise<DeleteSongResult> {
        const store = await this.loadStore();
        const existing = store.records[id];
        if (!existing) {
            return { deleted: false };
        }

        delete store.records[id];
        await this.saveStore(store);

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

    private async loadStore(): Promise<SongLibraryStoreFile> {
        try {
            const content = await this.deps.readFile(this.storeFilePath, "utf-8");
            const parsed = JSON.parse(content) as SongLibraryStoreFile;
            if (!parsed || typeof parsed !== "object" || !parsed.records || typeof parsed.records !== "object") {
                return { records: {} };
            }
            return parsed;
        } catch (error) {
            if (isMissingFileError(error)) {
                return { records: {} };
            }
            console.error("[library] failed to load song library, treating as empty", error);
            return { records: {} };
        }
    }

    private async saveStore(store: SongLibraryStoreFile): Promise<void> {
        try {
            await this.deps.mkdir(path.dirname(this.storeFilePath), { recursive: true });
            await this.deps.writeFile(this.storeFilePath, JSON.stringify(store), "utf-8");
        } catch (error) {
            console.error("[library] failed to persist song library", error);
        }
    }

    private resolveManagedAudioPath(fileHash: string, algorithm: string, contractVersion: string, sourceAudioPath: string): string {
        const extension = path.extname(sourceAudioPath).toLowerCase() || ".audio";
        const safeAlgorithm = algorithm.replace(/[^a-zA-Z0-9.-]/g, "-");
        const safeContractVersion = contractVersion.replace(/[^a-zA-Z0-9.-]/g, "-");
        return path.join(path.dirname(this.storeFilePath), "audio", `${fileHash}-${safeAlgorithm}-${safeContractVersion}${extension}`);
    }
}

export function resolveSongLibraryPath(userDataPath: string): string {
    return path.join(userDataPath, "song-library-v1.json");
}

function buildFileHash(content: Buffer): string {
    return createHash("sha256").update(content).digest("hex");
}

function normalizeSearch(value: string): string {
    return value.trim().toLocaleLowerCase();
}

function isMissingFileError(error: unknown): boolean {
    if (!error || typeof error !== "object") {
        return false;
    }
    return (error as { code?: unknown }).code === "ENOENT";
}
