import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { normalizeSearch } from "./queryParams.js";
import type { AnalysisSummary, SongAnalysis } from "./songTypes.js";

export function buildStableFileHash(audioPath: string, fallbackSourcePath?: string): string {
    try {
        return createHash("sha256").update(readFileSync(audioPath)).digest("hex");
    } catch {
        return fallbackSourcePath || audioPath;
    }
}

export function buildSongId(fileHash: string): string {
    return `song:${fileHash}`;
}

export function buildSearchIndex(title: string, artist: string, audioPath: string): string {
    return normalizeSearch(`${title} ${artist} ${audioPath}`);
}

export function summarizeAnalysis(analysis: SongAnalysis): AnalysisSummary {
    const chordCount = analysis.analysis.chords.length;
    const confidenceValues = analysis.analysis.chords
        .map((segment) => segment.confidence)
        .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const averageConfidence = confidenceValues.length
        ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
        : null;

    const rootDurations = new Map<string, number>();
    for (const segment of analysis.analysis.chords) {
        if (segment.chord === "N") {
            continue;
        }
        const root = segment.chord.replace(/m$/, "");
        rootDurations.set(root, (rootDurations.get(root) ?? 0) + Math.max(0, segment.end - segment.start));
    }

    const keyEstimate = [...rootDurations.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
    return { chordCount, keyEstimate, averageConfidence };
}
