import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ChordAnalysisResult } from "@gcd/shared/analysis";

interface CacheRecord {
    result: ChordAnalysisResult;
    cachedAt: string;
}

interface CacheStore {
    entries: Record<string, CacheRecord>;
}

interface AnalyzeWithCacheOptions {
    audioPath: string;
    algorithm: string;
    contractVersion: string;
    forceRefresh?: boolean;
    analyze: (audioPath: string) => Promise<ChordAnalysisResult>;
}

interface AnalysisCacheDependencies {
    readFile: typeof readFile;
    writeFile: typeof writeFile;
    mkdir: typeof mkdir;
    now: () => Date;
}

export class AnalysisCache {
    private readonly cacheFilePath: string;
    private readonly deps: AnalysisCacheDependencies;

    constructor(cacheFilePath: string, deps?: Partial<AnalysisCacheDependencies>) {
        this.cacheFilePath = cacheFilePath;
        this.deps = {
            readFile,
            writeFile,
            mkdir,
            now: () => new Date(),
            ...deps
        };
    }

    async analyzeWithCache(options: AnalyzeWithCacheOptions): Promise<ChordAnalysisResult> {
        const key = await buildAnalysisCacheKey(options.audioPath, options.algorithm, options.contractVersion, this.deps.readFile);
        const store = await this.loadStore();

        if (!options.forceRefresh) {
            const cached = store.entries[key];
            if (cached) {
                return cached.result;
            }
        }

        const result = await options.analyze(options.audioPath);
        if ("error" in result) {
            return result;
        }

        store.entries[key] = {
            result,
            cachedAt: this.deps.now().toISOString()
        };
        await this.saveStore(store);

        return result;
    }

    private async loadStore(): Promise<CacheStore> {
        try {
            const content = await this.deps.readFile(this.cacheFilePath, "utf-8");
            const parsed = JSON.parse(content) as CacheStore;
            if (!parsed || typeof parsed !== "object" || !parsed.entries || typeof parsed.entries !== "object") {
                return { entries: {} };
            }
            return parsed;
        } catch (error) {
            if (isMissingFileError(error)) {
                return { entries: {} };
            }
            console.error("[cache] failed to load cache store, treating as empty", error);
            return { entries: {} };
        }
    }

    private async saveStore(store: CacheStore): Promise<void> {
        try {
            const dir = path.dirname(this.cacheFilePath);
            await this.deps.mkdir(dir, { recursive: true });
            await this.deps.writeFile(this.cacheFilePath, JSON.stringify(store), "utf-8");
        } catch (error) {
            console.error("[cache] failed to persist cache store", error);
        }
    }
}

export async function buildAnalysisCacheKey(
    audioPath: string,
    algorithm: string,
    contractVersion: string,
    readFileFn: typeof readFile = readFile
): Promise<string> {
    const content = await readFileFn(audioPath);
    const contentHash = createHash("sha256").update(content).digest("hex");
    return `${contentHash}:${algorithm}:${contractVersion}`;
}

export function resolveAnalysisCachePath(userDataPath: string): string {
    return path.join(userDataPath, "analysis-cache-v1.json");
}

function isMissingFileError(error: unknown): boolean {
    if (!error || typeof error !== "object") {
        return false;
    }
    const maybeCode = (error as { code?: unknown }).code;
    return maybeCode === "ENOENT";
}