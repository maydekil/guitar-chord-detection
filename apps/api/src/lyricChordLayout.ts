import type { SongAnalysis } from "./songTypes.js";
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
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        if (line.time === null) {
            output.push("", line.text);
            continue;
        }
        const markers = buildLyricChordMarkers(
            segments,
            line.time,
            nextTimedLine?.time ?? line.time + 5,
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
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        const markers = buildLyricChordMarkers(
            segments,
            line.time,
            nextTimedLine?.time ?? line.time + 5,
            transposeSemitones,
        );
        const chordTags = markers.map((marker) => `[${marker.label}]`).join("");
        return `[${formatExportTime(line.time)}]${chordTags}${line.text}`;
    });
}

function buildLyricChordMarkers(
    segments: SongAnalysis["analysis"]["chords"],
    startTime: number,
    endTime: number,
    transposeSemitones: number,
): LyricChordMarker[] {
    const safeEndTime = Math.max(startTime + 0.25, endTime);
    const windowDuration = safeEndTime - startTime;
    const markers: LyricChordMarker[] = [];
    const openingChord = findActiveChord(segments, startTime);
    if (openingChord) {
        markers.push({ label: transposeChordLabel(openingChord, transposeSemitones), left: 0 });
    }
    for (const segment of segments) {
        if (segment.start <= startTime || segment.start >= safeEndTime) {
            continue;
        }
        const label = transposeChordLabel(segment.chord, transposeSemitones);
        if (markers.at(-1)?.label === label) {
            continue;
        }
        markers.push({
            label,
            left: Math.min(92, Math.max(0, ((segment.start - startTime) / windowDuration) * 100)),
        });
    }
    return markers;
}

function renderChordLineAboveLyric(text: string, markers: LyricChordMarker[]): string {
    const width = Math.max(24, text.length);
    const chars = Array.from({ length: width }, () => " ");
    for (const marker of markers) {
        const position = Math.min(width - 1, Math.max(0, Math.round((marker.left / 100) * width)));
        marker.label.split("").forEach((char, offset) => {
            if (position + offset < chars.length) {
                chars[position + offset] = char;
            }
        });
    }
    return chars.join("").trimEnd();
}

function findActiveChord(segments: SongAnalysis["analysis"]["chords"], currentTimeSeconds: number): string | null {
    if (!Number.isFinite(currentTimeSeconds)) {
        return null;
    }
    return segments.find((segment) => currentTimeSeconds >= segment.start && currentTimeSeconds < segment.end)?.chord ?? null;
}
