import type { SongLibraryRecord } from "@gcd/shared/library";

interface LyricLine {
    time: number | null;
    text: string;
}

interface LyricChordMarker {
    label: string;
    left: number;
}

const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

export function buildChordSheetExport(song: SongLibraryRecord, transposeSemitones: number): string {
    const lines = [
        `${song.artist} - ${song.title}`,
        `Duration: ${formatExportTime(song.duration)}`,
        `Transpose: ${formatTranspose(transposeSemitones)}`,
        ""
    ];

    if (song.lyrics?.trim()) {
        lines.push("Chord Sheet:");
        lines.push(...buildChordOverLyricLines(parseLyrics(song.lyrics), song.analysis.analysis.chords, transposeSemitones));
        lines.push("");
        lines.push("Timeline:");
    } else {
        lines.push("Timeline:");
    }

    lines.push(...song.analysis.analysis.chords.map((segment) => (
        `${formatExportTime(segment.start)} - ${formatExportTime(segment.end)}  ${transposeChordLabel(segment.chord, transposeSemitones)}`
    )));

    return `${lines.join("\n")}\n`;
}

export function buildLrcExport(song: SongLibraryRecord, transposeSemitones: number): string {
    if (song.lyrics?.trim()) {
        return `${buildChordTaggedLrcLines(parseLyrics(song.lyrics), song.analysis.analysis.chords, transposeSemitones).join("\n")}\n`;
    }

    return `${song.analysis.analysis.chords.map((segment) => (
        `[${formatExportTime(segment.start)}]${transposeChordLabel(segment.chord, transposeSemitones)}`
    )).join("\n")}\n`;
}

function buildChordOverLyricLines(
    lines: LyricLine[],
    segments: SongLibraryRecord["analysis"]["analysis"]["chords"],
    transposeSemitones: number
): string[] {
    const output: string[] = [];
    for (const [index, line] of lines.entries()) {
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        if (line.time === null) {
            output.push("");
            output.push(line.text);
            continue;
        }
        const markers = buildLyricChordMarkers(segments, line.time, nextTimedLine?.time ?? line.time + 5, transposeSemitones);
        output.push(renderChordLineAboveLyric(line.text, markers));
        output.push(line.text);
    }
    return output;
}

function buildChordTaggedLrcLines(
    lines: LyricLine[],
    segments: SongLibraryRecord["analysis"]["analysis"]["chords"],
    transposeSemitones: number
): string[] {
    return lines.map((line, index) => {
        if (line.time === null) {
            return line.text;
        }
        const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
        const markers = buildLyricChordMarkers(segments, line.time, nextTimedLine?.time ?? line.time + 5, transposeSemitones);
        const chordTags = markers.length > 0 ? `${markers.map((marker) => `[${marker.label}]`).join("")} ` : "";
        return `[${formatExportTime(line.time)}]${chordTags}${line.text}`;
    });
}

function buildLyricChordMarkers(
    segments: SongLibraryRecord["analysis"]["analysis"]["chords"],
    startTime: number,
    endTime: number,
    transposeSemitones: number
): LyricChordMarker[] {
    const safeEndTime = Math.max(startTime + 0.25, endTime);
    const windowDuration = safeEndTime - startTime;
    const markers: LyricChordMarker[] = [];
    const openingChord = findActiveChord(segments, startTime);

    if (openingChord) {
        markers.push({
            label: transposeChordLabel(openingChord, transposeSemitones),
            left: 0
        });
    }

    for (const segment of segments) {
        if (segment.start <= startTime || segment.start >= safeEndTime) {
            continue;
        }
        const label = transposeChordLabel(segment.chord, transposeSemitones);
        const previous = markers[markers.length - 1];
        if (previous?.label === label) {
            continue;
        }
        markers.push({
            label,
            left: Math.min(92, Math.max(0, ((segment.start - startTime) / windowDuration) * 100))
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

function parseLyrics(lyrics: string): LyricLine[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .filter((line) => line.length > 0)
        .map((line) => {
            const match = /^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$/.exec(line);
            if (!match) {
                return { time: null, text: line };
            }

            const minutes = Number.parseInt(match[1], 10);
            const seconds = Number.parseInt(match[2], 10);
            const fraction = match[3] ? Number.parseFloat(`0.${match[3]}`) : 0;
            const time = minutes * 60 + seconds + fraction;
            if (!Number.isFinite(time) || seconds >= 60) {
                return { time: null, text: line };
            }

            return {
                time,
                text: match[4].trim()
            };
        });
}

function findActiveChord(segments: SongLibraryRecord["analysis"]["analysis"]["chords"], currentTimeSeconds: number): string | null {
    if (!Number.isFinite(currentTimeSeconds)) {
        return null;
    }
    return segments.find((segment) => segment.start <= currentTimeSeconds && currentTimeSeconds < segment.end)?.chord ?? null;
}

function formatTranspose(semitones: number): string {
    if (semitones === 0) {
        return "0";
    }
    return semitones > 0 ? `+${semitones}` : String(semitones);
}

function transposeChordLabel(chord: string, semitones: number): string {
    if (chord === "N" || semitones === 0) {
        return chord;
    }
    const match = /^([A-G]#?)(m?)$/.exec(chord);
    if (!match) {
        return chord;
    }
    const rootIndex = SHARP_ROOTS.indexOf(match[1] as (typeof SHARP_ROOTS)[number]);
    if (rootIndex < 0) {
        return chord;
    }
    const nextRoot = SHARP_ROOTS[modulo(rootIndex + semitones, SHARP_ROOTS.length)];
    return `${nextRoot}${match[2]}`;
}

function modulo(value: number, divisor: number): number {
    return ((value % divisor) + divisor) % divisor;
}

function formatExportTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}
