import type { SongLibraryRecord } from "@gcd/shared/library";

import { formatTranspose, transposeChordLabel } from "./chordTranspose.js";
import { formatExportTime } from "./exportTime.js";
import { buildChordOverLyricLines, buildChordTaggedLrcLines } from "./lyricChordLayout.js";
import { parseLyrics } from "./lyricsParser.js";

export function buildChordSheetExport(song: SongLibraryRecord, transposeSemitones: number): string {
    const lines: string[] = [
        `${song.artist} - ${song.title}`,
        `Duration: ${formatExportTime(song.duration)}`,
        `Transpose: ${formatTranspose(transposeSemitones)}`,
        "",
    ];

    if (song.lyrics?.trim()) {
        lines.push("Chord Sheet:");
        lines.push(...buildChordOverLyricLines(parseLyrics(song.lyrics), song.analysis.analysis.chords, transposeSemitones));
        lines.push("", "Timeline:");
    } else {
        lines.push("Timeline:");
    }

    for (const segment of song.analysis.analysis.chords) {
        lines.push(
            `${formatExportTime(segment.start)} - ${formatExportTime(segment.end)}  ${transposeChordLabel(
                segment.chord,
                transposeSemitones,
            )}`,
        );
    }
    return `${lines.join("\n")}\n`;
}

export function buildLrcExport(song: SongLibraryRecord, transposeSemitones: number): string {
    if (song.lyrics?.trim()) {
        return `${buildChordTaggedLrcLines(parseLyrics(song.lyrics), song.analysis.analysis.chords, transposeSemitones).join("\n")}\n`;
    }
    return `${song.analysis.analysis.chords
        .map((segment) => `[${formatExportTime(segment.start)}]${transposeChordLabel(segment.chord, transposeSemitones)}`)
        .join("\n")}\n`;
}
