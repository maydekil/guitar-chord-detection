import type { ChordSegment } from "@gcd/shared/analysis";

import { transposeChordLabel } from "./chords.js";
import { formatLrcTime, formatTime } from "./time.js";

export interface LyricLine {
    time: number | null;
    text: string;
}

interface LyricChordMarker {
    label: string;
    left: number;
}

export function renderLyricsPreview(
    lyrics: string,
    segments: ChordSegment[],
    durationSeconds: number,
    currentTimeSeconds: number,
    transposeSemitones: number,
) {
    const lines = parseLyrics(lyrics);
    if (lines.length === 0) {
        return <p className="lyrics-empty">Belum ada lyric. Paste text biasa atau format LRC untuk sinkron timestamp.</p>;
    }

    return (
        <div className="lyrics-preview-lines">
            {lines.map((line, index) => {
                const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
                const lineStart = line.time ?? 0;
                const lineEnd = nextTimedLine?.time ?? durationSeconds;
                const isActive = line.time !== null && currentTimeSeconds >= lineStart && currentTimeSeconds < lineEnd;
                const markers = line.time === null
                    ? []
                    : buildLyricChordMarkers(segments, lineStart, lineEnd, transposeSemitones);
                return (
                    <div
                        key={`${line.time ?? "plain"}-${line.text}-${index}`}
                        className="lyrics-preview-line"
                        data-active={isActive ? "true" : "false"}
                        data-active-lyric={isActive ? "true" : "false"}
                    >
                        <span className="lyrics-preview-time">{line.time === null ? "" : formatTime(line.time)}</span>
                        <span className="lyrics-preview-content">
                            <span className="lyrics-preview-chords" aria-hidden="true">
                                {markers.map((marker, markerIndex) => (
                                    <span
                                        key={`${marker.label}-${marker.left}-${markerIndex}`}
                                        className="lyrics-preview-chord"
                                        style={{ left: `${marker.left}%` }}
                                    >
                                        {marker.label}
                                    </span>
                                ))}
                            </span>
                            <span className="lyrics-preview-text">{line.text}</span>
                        </span>
                    </div>
                );
            })}
        </div>
    );
}

export function buildChordOverLyricLines(lines: LyricLine[], segments: ChordSegment[], transposeSemitones: number): string[] {
    const output: string[] = [];
    for (const [index, line] of lines.entries()) {
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        if (line.time === null) {
            output.push("", line.text);
            continue;
        }
        const markers = buildLyricChordMarkers(segments, line.time, nextTimedLine?.time ?? line.time + 5, transposeSemitones);
        output.push(renderChordLineAboveLyric(line.text, markers), line.text);
    }
    return output;
}

export function buildChordTaggedLrcLines(lines: LyricLine[], segments: ChordSegment[], transposeSemitones: number): string[] {
    return lines.map((line, index) => {
        if (line.time === null) {
            return line.text;
        }
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        const markers = buildLyricChordMarkers(segments, line.time, nextTimedLine?.time ?? line.time + 5, transposeSemitones);
        const chordTags = markers.length > 0 ? `${markers.map((marker) => `[${marker.label}]`).join("")} ` : "";
        return `[${formatLrcTime(line.time)}]${chordTags}${line.text}`;
    });
}

export function parseLyrics(lyrics: string): LyricLine[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .filter(Boolean)
        .map((line) => {
            const match = line.match(/^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$/);
            if (!match) {
                return { time: null, text: line };
            }
            const minutes = Number.parseInt(match[1], 10);
            const seconds = Number.parseInt(match[2], 10);
            const fraction = match[3] ? Number.parseFloat(`0.${match[3].padEnd(3, "0")}`) : 0;
            const time = minutes * 60 + seconds + fraction;
            return { time: Number.isFinite(time) ? time : null, text: match[4].trim() };
        });
}

export function stripLyricsTiming(lyrics: string): string {
    return parseLyrics(lyrics).map((line) => line.text).join("\n");
}

export function autoSyncLyrics(lyrics: string, durationSeconds: number, segments: ChordSegment[]): string {
    const plainLines = extractPlainLyricLines(lyrics);
    if (plainLines.length === 0) {
        return "";
    }
    const times = buildLyricSyncTimes(plainLines.length, durationSeconds, segments);
    return plainLines.map((line, index) => `[${formatLrcTime(times[index] ?? 0)}]${line}`).join("\n");
}

function buildLyricChordMarkers(
    segments: ChordSegment[],
    startTime: number,
    endTime: number,
    transposeSemitones: number,
): LyricChordMarker[] {
    const safeEndTime = Math.max(startTime + 0.25, endTime);
    const windowDuration = safeEndTime - startTime;
    const markers: LyricChordMarker[] = [];
    const openingChord = segments.find((segment) => startTime >= segment.start && startTime < segment.end)?.chord;
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

function extractPlainLyricLines(lyrics: string): string[] {
    return parseLyrics(lyrics)
        .map((line) => line.text.trim())
        .filter(Boolean);
}

function buildLyricSyncTimes(lineCount: number, durationSeconds: number, segments: ChordSegment[]): number[] {
    if (lineCount <= 0) {
        return [];
    }
    const safeDuration = Number.isFinite(durationSeconds) ? Math.max(0, durationSeconds) : 0;
    if (lineCount === 1) {
        return [0];
    }
    const anchors = selectMusicalLyricAnchors(lineCount, safeDuration, segments);
    if (anchors.length >= lineCount) {
        return anchors.slice(0, lineCount);
    }
    return spreadAnchors(anchors, lineCount);
}

function selectMusicalLyricAnchors(lineCount: number, durationSeconds: number, segments: ChordSegment[]): number[] {
    const minGap = Math.max(2.5, durationSeconds / Math.max(24, lineCount * 1.8));
    const anchors: number[] = [0];
    for (const segment of segments) {
        const start = Math.max(0, segment.start);
        const previous = anchors.at(-1) ?? 0;
        if (start <= 0.25 || start - previous < minGap) {
            continue;
        }
        anchors.push(start);
        if (anchors.length >= lineCount) {
            break;
        }
    }
    if (anchors.at(-1) !== durationSeconds && durationSeconds > 0) {
        anchors.push(durationSeconds);
    }
    return anchors;
}

function spreadAnchors(anchors: number[], lineCount: number): number[] {
    const start = anchors[0] ?? 0;
    const end = anchors.at(-1) ?? start;
    if (lineCount <= 1) {
        return [start];
    }
    return Array.from({ length: lineCount }, (_, index) => start + ((end - start) * index) / (lineCount - 1));
}
