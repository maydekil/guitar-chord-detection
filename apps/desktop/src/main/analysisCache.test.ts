import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, vi } from "vitest";

import { AnalysisCache, buildAnalysisCacheKey, resolveAnalysisCachePath } from "./analysisCache";

async function createTempAudio(content: string): Promise<string> {
    const root = await mkdtemp(path.join(os.tmpdir(), "gcd-cache-test-"));
    const filePath = path.join(root, "audio.wav");
    await writeFile(filePath, content, "utf-8");
    return filePath;
}

function successResult(filePath: string) {
    return {
        version: "1",
        source: {
            path: filePath,
            duration: 1,
            sampleRate: 22050
        },
        analysis: {
            algorithm: "chroma-template-v1",
            chords: []
        }
    };
}

describe("buildAnalysisCacheKey", () => {
    it("includes SHA-256 file content and contract scope", async () => {
        const audioPath = await createTempAudio("abc");

        const key = await buildAnalysisCacheKey(audioPath, "chroma-template-v1", "1");

        expect(key).toMatch(/^[0-9a-f]{64}:chroma-template-v1:1$/);
    });
});

describe("AnalysisCache", () => {
    it("hits cache for same content and same version scope", async () => {
        const audioPath = await createTempAudio("same-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));
        const cache = new AnalysisCache(cachePath);

        const analyze = vi.fn(async () => successResult(audioPath));

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });
        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });

        expect(analyze).toHaveBeenCalledTimes(1);
    });

    it("misses cache when file content changes", async () => {
        const audioPath = await createTempAudio("v1-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));
        const cache = new AnalysisCache(cachePath);

        const analyze = vi.fn(async () => successResult(audioPath));

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });

        await writeFile(audioPath, "v2-content", "utf-8");

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });

        expect(analyze).toHaveBeenCalledTimes(2);
    });

    it("misses cache when contract version changes", async () => {
        const audioPath = await createTempAudio("same-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));
        const cache = new AnalysisCache(cachePath);

        const analyze = vi.fn(async () => successResult(audioPath));

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });
        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "2",
            analyze
        });

        expect(analyze).toHaveBeenCalledTimes(2);
    });

    it("bypasses cache on force refresh and reruns analysis", async () => {
        const audioPath = await createTempAudio("same-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));
        const cache = new AnalysisCache(cachePath);

        const analyze = vi.fn(async () => successResult(audioPath));

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v2",
            contractVersion: "1",
            analyze
        });

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v2",
            contractVersion: "1",
            forceRefresh: true,
            analyze
        });

        expect(analyze).toHaveBeenCalledTimes(2);
    });

    it("overwrites cached result with force refresh success", async () => {
        const audioPath = await createTempAudio("same-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));
        const cache = new AnalysisCache(cachePath);

        const initial = {
            ...successResult(audioPath),
            analysis: {
                algorithm: "chroma-template-v2",
                chords: [{ start: 0, end: 1, chord: "C", confidence: 0.8 }]
            }
        };
        const refreshed = {
            ...successResult(audioPath),
            analysis: {
                algorithm: "chroma-template-v2",
                chords: [{ start: 0, end: 1, chord: "Am", confidence: 0.9 }]
            }
        };

        const analyze = vi
            .fn()
            .mockResolvedValueOnce(initial)
            .mockResolvedValueOnce(refreshed)
            .mockResolvedValueOnce(successResult(audioPath));

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v2",
            contractVersion: "1",
            analyze
        });
        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v2",
            contractVersion: "1",
            forceRefresh: true,
            analyze
        });

        const afterRefresh = await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v2",
            contractVersion: "1",
            analyze
        });

        expect(analyze).toHaveBeenCalledTimes(2);
        expect("analysis" in afterRefresh).toBe(true);
        if ("analysis" in afterRefresh) {
            expect(afterRefresh.analysis.chords[0]?.chord).toBe("Am");
        }
    });

    it("treats corrupted cache file as miss and heals on write", async () => {
        const audioPath = await createTempAudio("same-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));
        await writeFile(cachePath, "not-json", "utf-8");

        const cache = new AnalysisCache(cachePath);
        const analyze = vi.fn(async () => successResult(audioPath));

        await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });

        const healed = await readFile(cachePath, "utf-8");
        const parsed = JSON.parse(healed) as { entries: Record<string, unknown> };
        expect(parsed.entries).toBeDefined();
        expect(Object.keys(parsed.entries).length).toBe(1);
    });

    it("returns analysis result even when cache persistence fails", async () => {
        const audioPath = await createTempAudio("same-content");
        const cachePath = resolveAnalysisCachePath(path.dirname(audioPath));

        const cache = new AnalysisCache(cachePath, {
            writeFile: async () => {
                throw new Error("disk full");
            }
        });
        const analyze = vi.fn(async () => successResult(audioPath));
        const errorSpy = vi.spyOn(console, "error").mockImplementation(() => { });

        const result = await cache.analyzeWithCache({
            audioPath,
            algorithm: "chroma-template-v1",
            contractVersion: "1",
            analyze
        });

        expect("analysis" in result).toBe(true);
        expect(errorSpy).toHaveBeenCalled();
        errorSpy.mockRestore();
    });
});
