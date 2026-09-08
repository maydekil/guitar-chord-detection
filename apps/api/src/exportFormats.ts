import type { SongLibraryRecord } from "@gcd/shared/library";
import { buildLeadSheetAnalysis, buildLeadSheetTimelineFromSegments, selectPlayableChordSegments } from "@gcd/shared/lyricChordLayout";

import { formatTranspose, transposeChordLabel } from "./chordTranspose.js";
import { formatExportTime } from "./exportTime.js";
import { buildChordOverLyricLines, buildChordTaggedLrcLines } from "./lyricChordLayout.js";
import { parseLyrics } from "./lyricsParser.js";

export function buildChordSheetExport(song: SongLibraryRecord, transposeSemitones: number): string {
    const playableAnalysis = buildLeadSheetAnalysis(song.analysis, song.lyrics ?? "");
    const exportSegments = song.lyrics?.trim()
        ? selectPlayableChordSegments(playableAnalysis)
        : buildLeadSheetTimelineFromSegments(song.analysis.analysis.chords, song.duration);
    const lines: string[] = [
        `${song.artist} - ${song.title}`,
        `Duration: ${formatExportTime(song.duration)}`,
        `Transpose: ${formatTranspose(transposeSemitones)}`,
        "",
    ];

    if (song.lyrics?.trim()) {
        lines.push("Chord Sheet:");
        lines.push(...buildChordOverLyricLines(parseLyrics(song.lyrics), exportSegments, transposeSemitones));
        lines.push("", "Timeline:");
    } else {
        lines.push("Timeline:");
    }

    for (const segment of exportSegments) {
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
    const playableAnalysis = buildLeadSheetAnalysis(song.analysis, song.lyrics ?? "");
    if (song.lyrics?.trim()) {
        return `${buildChordTaggedLrcLines(parseLyrics(song.lyrics), selectPlayableChordSegments(playableAnalysis), transposeSemitones).join("\n")}\n`;
    }
    return `${buildLeadSheetTimelineFromSegments(song.analysis.analysis.chords, song.duration)
        .map((segment) => `[${formatExportTime(segment.start)}]${transposeChordLabel(segment.chord, transposeSemitones)}`)
        .join("\n")}\n`;
}
