import type { ChordAnalysisSuccess, ChordSegment } from "./analysis.js";

export interface LyricChordLine {
    time: number | null;
    text: string;
}

export interface LyricChordMarker {
    label: string;
    left: number;
}

/** Positions are percentages of the lyric's time window, not word alignment. */
export function buildPhraseAwareLyricChordMarkers(
    segments: ChordSegment[],
    startTime: number,
    endTime: number,
): LyricChordMarker[] {
    if (!Number.isFinite(startTime) || !Number.isFinite(endTime) || endTime <= startTime) {
        return [];
    }
    return segments
        .filter((segment) => segment.end > startTime && segment.start < endTime)
        .map((segment) => ({
            label: segment.chord,
            left: ((Math.max(startTime, segment.start) - startTime) / (endTime - startTime)) * 100,
        }));
}

export function buildLeadSheetLyricChordMarkers(
    lines: LyricChordLine[],
    lineIndex: number,
    segments: ChordSegment[],
    endTime: number,
): LyricChordMarker[] {
    const line = lines[lineIndex];
    if (!line || line.time === null) {
        return [];
    }
    return buildPhraseAwareLyricChordMarkers(segments, line.time, endTime);
}

/** Include the whole last line instead of truncating its chords after five seconds. */
export function lyricChordWindowEnd(
    lines: LyricChordLine[],
    lineIndex: number,
    segments: ChordSegment[],
): number {
    const start = lines[lineIndex]?.time ?? 0;
    const next = lines.slice(lineIndex + 1).find((line) => line.time !== null);
    return next?.time ?? segments.reduce((end, segment) => Math.max(end, segment.end), start);
}

export function buildLeadSheetTimelineSegments(
    _lines: LyricChordLine[],
    sourceSegments: ChordSegment[],
    durationSeconds: number,
): ChordSegment[] {
    return buildLeadSheetTimelineFromSegments(sourceSegments, durationSeconds);
}

/** Compatibility adapter: layout must not arrange or relabel an engine timeline. */
export function buildLeadSheetTimelineFromSegments(
    sourceSegments: ChordSegment[],
    _durationSeconds: number,
): ChordSegment[] {
    return sourceSegments.map((segment) => ({ ...segment }));
}

export function selectPlayableChordSegments(analysis: ChordAnalysisSuccess): ChordSegment[] {
    return analysis.analysis.chords;
}

export function buildLeadSheetAnalysis(
    analysis: ChordAnalysisSuccess,
    _lyrics: string,
): ChordAnalysisSuccess {
    const chords = buildLeadSheetTimelineFromSegments(selectPlayableChordSegments(analysis), analysis.source.duration);
    return {
        ...analysis,
        analysis: {
            ...analysis.analysis,
            chords,
            detectedChords: analysis.analysis.detectedChords ?? analysis.analysis.chords,
            leadSheetChords: chords,
            leadSheetSource: "audio",
        },
    };
}

export function renderChordLineAboveLyric(text: string, markers: LyricChordMarker[]): string {
    if (markers.length === 0) {
        return "";
    }
    const width = Math.max(24, text.length);
    const chars = Array.from({ length: width }, () => " ");
    for (const marker of markers) {
        const position = Math.min(width - 1, Math.max(0, Math.round((marker.left / 100) * Math.max(1, width - 1))));
        for (let index = 0; index < marker.label.length && position + index < chars.length; index += 1) {
            chars[position + index] = marker.label[index] ?? " ";
        }
    }
    return chars.join("").trimEnd();
}
