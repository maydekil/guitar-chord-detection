import type { ChordSegment } from "@gcd/shared/analysis";

import { formatTranspose, transposeChordLabel } from "./chords.js";
import { buildChordOverLyricLines, buildChordTaggedLrcLines, parseLyrics } from "./lyrics.js";
import { formatPreciseTime } from "./time.js";

export function buildLocalChordSheetExport(
    artist: string,
    title: string,
    durationSeconds: number,
    segments: ChordSegment[],
    lyrics: string,
    transposeSemitones: number,
): string {
    const lines = [
        `${artist || "Unknown Artist"} - ${title || "Untitled"}`,
        `Duration: ${formatPreciseTime(durationSeconds)}`,
        `Transpose: ${formatTranspose(transposeSemitones)}`,
        "",
    ];

    if (lyrics.trim()) {
        lines.push("Chord Sheet:", ...buildChordOverLyricLines(parseLyrics(lyrics), segments, transposeSemitones), "", "Timeline:");
    } else {
        lines.push("Timeline:");
    }

    lines.push(...segments.map((segment) => (
        `${formatPreciseTime(segment.start)} - ${formatPreciseTime(segment.end)}  ${transposeChordLabel(segment.chord, transposeSemitones)}`
    )));

    return `${lines.join("\n")}\n`;
}

export function buildLocalLrcExport(lyrics: string, segments: ChordSegment[], transposeSemitones: number): string {
    if (lyrics.trim()) {
        return `${buildChordTaggedLrcLines(parseLyrics(lyrics), segments, transposeSemitones).join("\n")}\n`;
    }
    return `${segments.map((segment) => `[${formatPreciseTime(segment.start)}]${transposeChordLabel(segment.chord, transposeSemitones)}`).join("\n")}\n`;
}
