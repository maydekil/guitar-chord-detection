import type { ChordSegment } from "@gcd/shared/analysis";

export const SPLIT_SNAP_GRID_SECONDS = 0.5;

export function normalizeTimelineSegments(segments: ChordSegment[], durationSeconds: number): ChordSegment[] {
    const safeDuration = Number.isFinite(durationSeconds) ? Math.max(0, durationSeconds) : 0;
    return segments
        .map((segment) => ({
            ...segment,
            start: Math.max(0, Math.min(segment.start, safeDuration)),
            end: Math.max(0, Math.min(segment.end, safeDuration)),
        }))
        .filter((segment) => segment.end > segment.start)
        .sort((left, right) => left.start - right.start || left.end - right.end);
}

export function findActiveChord(segments: ChordSegment[], currentTimeSeconds: number): string | null {
    if (!Number.isFinite(currentTimeSeconds)) {
        return null;
    }
    return segments.find((segment) => currentTimeSeconds >= segment.start && currentTimeSeconds < segment.end)?.chord ?? null;
}

export function cloneTimelineSegments(segments: ChordSegment[]): ChordSegment[] {
    return segments.map((segment) => ({ ...segment }));
}

export function snapSplitTime(rawSplitTime: number, segment: ChordSegment): number {
    const snapped = Math.round(rawSplitTime / SPLIT_SNAP_GRID_SECONDS) * SPLIT_SNAP_GRID_SECONDS;
    return Math.min(segment.end - 0.1, Math.max(segment.start + 0.1, snapped));
}
