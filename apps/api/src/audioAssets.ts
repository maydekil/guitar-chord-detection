import { createHash } from "node:crypto";
import { readdir, readFile, rm, stat, unlink, writeFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";

import { collectReferencedAudioPaths } from "./songStore.js";
import { isAlreadyExistsError, readRawBody, resolveSafeAudioExtension } from "./httpUtils.js";
import type { DatabaseSync } from "node:sqlite";

export interface ManagedAudioAsset {
    audioPath: string;
    audioStreamUrl: string;
    fileHash: string;
}

export interface AssetCleanupResult {
    deletedAudioFiles: number;
    deletedEngineOutputFiles: number;
    skippedReferencedFiles: number;
    reclaimedBytes: number;
}

export async function saveUploadedAudioAsset(
    request: http.IncomingMessage,
    url: URL,
    audioDir: string,
    publicBaseUrl: string
): Promise<ManagedAudioAsset> {
    const body = await readRawBody(request);
    if (body.length === 0) {
        throw new Error("Uploaded audio is empty");
    }

    const originalFileName = url.searchParams.get("filename") ?? "audio";
    return await persistManagedAudioBytes(body, originalFileName, { audioDir, publicBaseUrl });
}

export async function materializeAudioAsset(
    sourcePath: string,
    audioDir: string,
    publicBaseUrl: string
): Promise<ManagedAudioAsset> {
    const bytes = await readFile(sourcePath);
    return await persistManagedAudioBytes(bytes, sourcePath, { audioDir, publicBaseUrl });
}

export async function cleanupApiAssets(
    db: DatabaseSync,
    audioDir: string,
    engineOutputDir: string,
    maxAgeMs: number
): Promise<AssetCleanupResult> {
    const referencedAudioPaths = collectReferencedAudioPaths(db);
    const result: AssetCleanupResult = {
        deletedAudioFiles: 0,
        deletedEngineOutputFiles: 0,
        skippedReferencedFiles: 0,
        reclaimedBytes: 0
    };

    for (const filePath of await listFiles(audioDir)) {
        if (referencedAudioPaths.has(path.resolve(filePath))) {
            result.skippedReferencedFiles += 1;
            continue;
        }
        const deletedBytes = await removeFileIfOlderThan(filePath, maxAgeMs, { audioDir, engineOutputDir });
        if (deletedBytes > 0) {
            result.deletedAudioFiles += 1;
            result.reclaimedBytes += deletedBytes;
        }
    }

    for (const filePath of await listFiles(engineOutputDir)) {
        if (isModelCachePath(filePath)) {
            continue;
        }
        const deletedBytes = await removeFileIfOlderThan(filePath, maxAgeMs, { audioDir, engineOutputDir });
        if (deletedBytes > 0) {
            result.deletedEngineOutputFiles += 1;
            result.reclaimedBytes += deletedBytes;
        }
    }

    return result;
}

async function persistManagedAudioBytes(
    bytes: Buffer,
    fileName: string,
    options: { audioDir: string; publicBaseUrl: string }
): Promise<ManagedAudioAsset> {
    const fileHash = createHash("sha256").update(bytes).digest("hex");
    const extension = resolveSafeAudioExtension(fileName);
    const audioPath = path.join(options.audioDir, `${fileHash}${extension}`);

    await writeFile(audioPath, bytes, { flag: "wx" }).catch(async (error: unknown) => {
        if (!isAlreadyExistsError(error)) {
            throw error;
        }
    });

    return {
        audioPath,
        audioStreamUrl: `${options.publicBaseUrl}/audio/${encodeURIComponent(fileHash)}${extension}/stream`,
        fileHash
    };
}

async function listFiles(root: string): Promise<string[]> {
    try {
        const entries = await readdir(root, { withFileTypes: true });
        const files: string[] = [];
        for (const entry of entries) {
            const entryPath = path.join(root, entry.name);
            if (entry.isDirectory()) {
                files.push(...await listFiles(entryPath));
            } else if (entry.isFile()) {
                files.push(entryPath);
            }
        }
        return files;
    } catch (error) {
        if (error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "ENOENT") {
            return [];
        }
        throw error;
    }
}

async function removeFileIfOlderThan(
    filePath: string,
    maxAgeMs: number,
    options: { audioDir: string; engineOutputDir: string }
): Promise<number> {
    const fileStat = await stat(filePath).catch(() => null);
    if (!fileStat?.isFile()) {
        return 0;
    }
    if (Date.now() - fileStat.mtimeMs < maxAgeMs) {
        return 0;
    }
    await unlink(filePath);
    await removeEmptyParents(path.dirname(filePath), filePath.startsWith(options.engineOutputDir) ? options.engineOutputDir : options.audioDir);
    return fileStat.size;
}

async function removeEmptyParents(directoryPath: string, stopAt: string): Promise<void> {
    const resolvedStop = path.resolve(stopAt);
    let current = path.resolve(directoryPath);
    while (current.startsWith(`${resolvedStop}${path.sep}`)) {
        await rm(current, { recursive: false }).catch(() => undefined);
        current = path.dirname(current);
    }
}

function isModelCachePath(filePath: string): boolean {
    return path.resolve(filePath).split(path.sep).includes("model-cache");
}
