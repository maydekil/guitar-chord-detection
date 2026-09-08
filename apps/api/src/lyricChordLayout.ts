import type { SongAnalysis } from "./songTypes.js";
import { buildLeadSheetLyricChordMarkers, lyricChordWindowEnd, renderChordLineAboveLyric } from "@gcd/shared/lyricChordLayout";
import { transposeChordLabel } from "./chordTranspose.js";
import { formatExportTime } from "./exportTime.js";
import type { LyricLine } from "./lyricsParser.js";

interface LyricChordMarker {
    label: string;
    left: number;
}

export function buildChordOverLyricLines(
    lines: LyricLine[],
    segments: SongAnalysis["analysis"]["chords"],
    transposeSemitones: number,
): string[] {
    const output: string[] = [];
    for (const [index, line] of lines.entries()) {
        if (line.time === null) {
            output.push("", line.text);
            continue;
        }
        const markers = buildLyricChordMarkers(
            lines,
            index,
            segments,
            lyricChordWindowEnd(lines, index, segments),
            transposeSemitones,
        );
        output.push(renderChordLineAboveLyric(line.text, markers), line.text);
    }
    return output;
}

export function buildChordTaggedLrcLines(
    lines: LyricLine[],
    segments: SongAnalysis["analysis"]["chords"],
    transposeSemitones: number,
): string[] {
    return lines.map((line, index) => {
        if (line.time === null) {
            return line.text;
        }
        const markers = buildLyricChordMarkers(
            lines,
            index,
            segments,
            lyricChordWindowEnd(lines, index, segments),
            transposeSemitones,
        );
        const chordTags = markers.map((marker) => `[${marker.label}]`).join("");
        return `[${formatExportTime(line.time)}]${chordTags}${line.text}`;
    });
}

function buildLyricChordMarkers(
    lines: LyricLine[],
    lineIndex: number,
    segments: SongAnalysis["analysis"]["chords"],
    endTime: number,
    transposeSemitones: number,
): LyricChordMarker[] {
    return buildLeadSheetLyricChordMarkers(lines, lineIndex, segments, endTime)
        .map((marker) => ({
            ...marker,
            label: transposeChordLabel(marker.label, transposeSemitones),
        }));
}
