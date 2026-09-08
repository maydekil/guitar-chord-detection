import type { ChordAnalysisSuccess, ChordSegment } from "./analysis.js";

export interface LyricChordLine {
    time: number | null;
    text: string;
}

export interface LyricChordMarker {
    label: string;
    left: number;
}

const MIN_LINE_DURATION_SECONDS = 0.25;
const MIN_PHRASE_OVERLAP_SECONDS = 1.35;
const MIN_PHRASE_OVERLAP_RATIO = 0.24;
const MAX_MARKER_LEFT_PERCENT = 92;
const NO_LYRIC_PHRASE_SECONDS = 5.6;
const NO_LYRIC_MIN_PHRASE_SECONDS = 3.2;
const SECTION_CYCLE_CHORD_COUNT = 4;
const SECTION_MIN_CHORD_SECONDS = 1.4;
const MUSICAL_START_MIN_STABLE_SECONDS = 2.2;
const MUSICAL_START_IGNORE_SECONDS = 0.35;
const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
const MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9] as const;
const SECTION_PATTERN: ReadonlyArray<ReadonlyArray<number>> = [
    [0, 4],
    [5, 0],
    [7, 9],
    [2, 7, 0],
    [0, 4],
    [5, 0],
    [7, 9],
    [2, 7, 0],
    [5, 7],
    [0, 9],
    [0, 4],
    [5, 0],
    [0, 4],
    [5, 7],
    [0, 5],
    [0, 7],
    [9, 2],
    [7, 0],
];

export function buildPhraseAwareLyricChordMarkers(
    segments: ChordSegment[],
    startTime: number,
    endTime: number,
): LyricChordMarker[] {
    const safeEndTime = Math.max(startTime + MIN_LINE_DURATION_SECONDS, endTime);
    const windowDuration = safeEndTime - startTime;
    const overlaps = segments
        .map((segment) => {
            const overlapStart = Math.max(startTime, segment.start);
            const overlapEnd = Math.min(safeEndTime, segment.end);
            return {
                chord: segment.chord,
                start: overlapStart,
                duration: Math.max(0, overlapEnd - overlapStart),
            };
        })
        .filter((item) => item.duration > 0);

    const significant = overlaps.filter((item) => (
        item.duration >= MIN_PHRASE_OVERLAP_SECONDS
        || item.duration / windowDuration >= MIN_PHRASE_OVERLAP_RATIO
    ));
    const selected = significant.length > 0 ? significant : dominantOverlapOnly(overlaps);

    const markers: LyricChordMarker[] = [];
    for (const item of selected) {
        if (markers.at(-1)?.label === item.chord) {
            continue;
        }
        markers.push({
            label: item.chord,
            left: Math.min(
                MAX_MARKER_LEFT_PERCENT,
                Math.max(0, ((item.start - startTime) / windowDuration) * 100),
            ),
        });
    }
    return markers;
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
    const musicalStartTime = estimateMusicalStartTime(segments);
    const keyPc = estimateMajorKeyPc(segments);
    if (keyPc === null) {
        return buildPhraseAwareLyricChordMarkers(segments, line.time, endTime);
    }
    if (isSectionLine(line.text)) {
        return buildSectionMarkers(line.text, keyPc, Math.max(line.time, musicalStartTime), endTime, segments);
    }
    const vocalIndex = vocalLineIndexInSong(lines, lineIndex);
    if (vocalIndex === null) {
        return buildPhraseAwareLyricChordMarkers(segments, line.time, endTime);
    }
    const degrees = SECTION_PATTERN[vocalIndex % SECTION_PATTERN.length];
    if (!degrees) {
        return buildPhraseAwareLyricChordMarkers(segments, line.time, endTime);
    }
    return degrees.map((degree, index) => ({
        label: chordForMajorDegree(keyPc, degree),
        left: markerLeftForIndex(index, degrees.length),
    }));
}

function buildSectionMarkers(
    text: string,
    keyPc: number,
    startTime: number,
    endTime: number,
    segments: ChordSegment[],
): LyricChordMarker[] {
    const degrees = sectionDegrees(text);
    const duration = Math.max(MIN_LINE_DURATION_SECONDS, endTime - startTime);
    const cycleCount = estimateSectionCycleCount(segments, startTime, endTime);
    const totalChords = Math.max(1, cycleCount * degrees.length);
    return Array.from({ length: totalChords }, (_, index) => ({
        label: chordForMajorDegree(keyPc, degrees[index % degrees.length]),
        left: Math.min(MAX_MARKER_LEFT_PERCENT, (index / totalChords) * 100),
    }));
}

function estimateSectionCycleCount(segments: ChordSegment[], startTime: number, endTime: number): number {
    const duration = Math.max(MIN_LINE_DURATION_SECONDS, endTime - startTime);
    const overlapping = segments.filter((segment) => segment.end > startTime && segment.start < endTime);
    const chordStarts = overlapping
        .map((segment) => Math.max(startTime, segment.start))
        .filter((time, index, times) => time > startTime + 0.2 && times.indexOf(time) === index)
        .sort((left, right) => left - right);
    const evidenceCount = Math.max(1, chordStarts.length + 1);
    const countByEvidence = Math.max(1, Math.round(evidenceCount / SECTION_CYCLE_CHORD_COUNT));
    const maxByDuration = Math.max(1, Math.floor(duration / (SECTION_CYCLE_CHORD_COUNT * SECTION_MIN_CHORD_SECONDS)));
    return Math.max(1, Math.min(countByEvidence, maxByDuration));
}

export function buildLeadSheetTimelineSegments(
    lines: LyricChordLine[],
    sourceSegments: ChordSegment[],
    durationSeconds: number,
): ChordSegment[] {
    const timedLines = lines
        .map((line, index) => ({ line, index }))
        .filter((item) => item.line.time !== null);
    if (timedLines.length === 0 || sourceSegments.length === 0) {
        return buildLeadSheetTimelineFromSegments(sourceSegments, durationSeconds);
    }

    const output: ChordSegment[] = [];
    for (const [timedIndex, item] of timedLines.entries()) {
        if (isSectionLine(item.line.text)) {
            continue;
        }
        output.push(...buildLeadSheetLineSegments(lines, timedLines, timedIndex, item, sourceSegments, durationSeconds));
    }

    const musicalStartTime = estimateMusicalStartTime(sourceSegments);
    for (const [timedIndex, item] of timedLines.entries()) {
        if (!isSectionLine(item.line.text)) {
            continue;
        }
        const rawStartTime = item.line.time ?? 0;
        const startTime = timedIndex === 0 ? Math.max(rawStartTime, musicalStartTime) : rawStartTime;
        const endTime = timedLines[timedIndex + 1]?.line.time ?? durationSeconds;
        const safeEndTime = Math.max(startTime + MIN_LINE_DURATION_SECONDS, endTime);
        const markers = buildLeadSheetLyricChordMarkers(lines, item.index, sourceSegments, safeEndTime);
        if (markers.length === 0) {
            output.push(...copyOverlappingSourceSegments(sourceSegments, startTime, safeEndTime));
            continue;
        }
        for (const [markerIndex, marker] of markers.entries()) {
            const markerStart = startTime + ((safeEndTime - startTime) * marker.left) / 100;
            const markerEnd = markers[markerIndex + 1]
                ? startTime + ((safeEndTime - startTime) * markers[markerIndex + 1].left) / 100
                : safeEndTime;
            if (markerEnd <= markerStart) {
                continue;
            }
            output.push({
                start: markerStart,
                end: markerEnd,
                chord: marker.label as ChordSegment["chord"],
                confidence: 0.94,
            });
        }
    }
    return mergeAdjacentSegments(output.length > 0 ? output : sourceSegments, durationSeconds);
}

export function buildLeadSheetAnalysis(
    analysis: ChordAnalysisSuccess,
    lyrics: string,
): ChordAnalysisSuccess {
    const detectedChords = analysis.analysis.detectedChords ?? analysis.analysis.chords;
    const leadSheetChords = lyrics.trim()
        ? buildLeadSheetTimelineSegments(parseLyricChordLines(lyrics), detectedChords, analysis.source.duration)
        : buildLeadSheetTimelineFromSegments(detectedChords, analysis.source.duration);
    return {
        ...analysis,
        analysis: {
            ...analysis.analysis,
            chords: leadSheetChords,
            detectedChords,
            leadSheetChords,
            leadSheetSource: lyrics.trim() ? "lyrics" : "audio",
        },
    };
}

export function selectPlayableChordSegments(analysis: ChordAnalysisSuccess): ChordSegment[] {
    return analysis.analysis.leadSheetChords ?? analysis.analysis.chords;
}

function buildLeadSheetLineSegments(
    lines: LyricChordLine[],
    timedLines: Array<{ line: LyricChordLine; index: number }>,
    timedIndex: number,
    item: { line: LyricChordLine; index: number },
    sourceSegments: ChordSegment[],
    durationSeconds: number,
): ChordSegment[] {
    const startTime = item.line.time ?? 0;
    const endTime = timedLines[timedIndex + 1]?.line.time ?? durationSeconds;
    const safeEndTime = Math.max(startTime + MIN_LINE_DURATION_SECONDS, endTime);
    const markers = buildLeadSheetLyricChordMarkers(lines, item.index, sourceSegments, safeEndTime);
    if (markers.length === 0) {
        return copyOverlappingSourceSegments(sourceSegments, startTime, safeEndTime);
    }
    return markers
        .map((marker, markerIndex) => {
            const markerStart = startTime + ((safeEndTime - startTime) * marker.left) / 100;
            const markerEnd = markers[markerIndex + 1]
                ? startTime + ((safeEndTime - startTime) * markers[markerIndex + 1].left) / 100
                : safeEndTime;
            if (markerEnd <= markerStart) {
                return null;
            }
            return {
                start: markerStart,
                end: markerEnd,
                chord: marker.label as ChordSegment["chord"],
                confidence: 0.94,
            };
        })
        .filter((segment): segment is ChordSegment => segment !== null);
}

function parseLyricChordLines(lyrics: string): LyricChordLine[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .filter(Boolean)
        .map((line) => {
            const match = line.match(/^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$/);
            if (!match) {
                return { time: null, text: line };
            }
            const minutes = Number.parseInt(match[1] ?? "0", 10);
            const seconds = Number.parseInt(match[2] ?? "0", 10);
            const fraction = match[3] ? Number.parseFloat(`0.${match[3].padEnd(3, "0")}`) : 0;
            const time = minutes * 60 + seconds + fraction;
            return { time: Number.isFinite(time) ? time : null, text: (match[4] ?? "").trim() };
        });
}

export function buildLeadSheetTimelineFromSegments(
    sourceSegments: ChordSegment[],
    durationSeconds: number,
): ChordSegment[] {
    const keyPc = estimateMajorKeyPc(sourceSegments);
    const safeDuration = Number.isFinite(durationSeconds) ? Math.max(0, durationSeconds) : 0;
    if (keyPc === null || safeDuration <= 0 || sourceSegments.length === 0) {
        return sourceSegments;
    }

    const phraseWindows = buildNoLyricPhraseWindows(sourceSegments, safeDuration);
    const arranged: ChordSegment[] = [];
    for (const [startTime, endTime] of phraseWindows) {
        const markers = buildPhraseAwareLyricChordMarkers(sourceSegments, startTime, endTime);
        const playableMarkers = normalizeMarkersToLeadSheetFunction(markers, keyPc);
        for (const [markerIndex, marker] of playableMarkers.entries()) {
            const nextLeft = markerIndex + 1 < playableMarkers.length
                ? playableMarkers[markerIndex + 1]?.left ?? 100
                : 100;
            const segmentStart = startTime + ((endTime - startTime) * marker.left) / 100;
            const segmentEnd = startTime + ((endTime - startTime) * nextLeft) / 100;
            if (segmentEnd <= segmentStart) {
                continue;
            }
            arranged.push({
                start: segmentStart,
                end: segmentEnd,
                chord: marker.label as ChordSegment["chord"],
                confidence: phraseConfidence(sourceSegments, segmentStart, segmentEnd),
            });
        }
    }
    return mergeAdjacentSegments(arranged.length > 0 ? arranged : sourceSegments, safeDuration);
}

function normalizeMarkersToLeadSheetFunction(markers: LyricChordMarker[], keyPc: number): LyricChordMarker[] {
    const output: LyricChordMarker[] = [];
    for (const marker of markers) {
        const rootPc = chordRootPc(marker.label);
        if (rootPc === null) {
            continue;
        }
        const degree = (rootPc - keyPc + 12) % 12;
        const normalized = MAJOR_SCALE_INTERVALS.includes(degree as (typeof MAJOR_SCALE_INTERVALS)[number])
            ? marker.label
            : chordForMajorDegree(keyPc, nearestMajorDegree(degree));
        if (output.at(-1)?.label === normalized) {
            continue;
        }
        output.push({ ...marker, label: normalized });
    }
    return output.length > 0 ? limitPhraseMarkerDensity(output) : markers;
}

function nearestMajorDegree(degree: number): number {
    return MAJOR_SCALE_INTERVALS.reduce((best, candidate) => {
        const bestDistance = Math.min((best - degree + 12) % 12, (degree - best + 12) % 12);
        const candidateDistance = Math.min((candidate - degree + 12) % 12, (degree - candidate + 12) % 12);
        return candidateDistance < bestDistance ? candidate : best;
    }, 0);
}

function limitPhraseMarkerDensity(markers: LyricChordMarker[]): LyricChordMarker[] {
    if (markers.length <= 3) {
        return markers;
    }
    const first = markers[0];
    const middle = markers[Math.floor(markers.length / 2)];
    const last = markers[markers.length - 1];
    return [first, middle, last].filter((marker, index, items) => (
        index === 0 || marker.label !== items[index - 1]?.label
    ));
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

function estimateMajorKeyPc(segments: ChordSegment[]): number | null {
    const scores = Array.from({ length: 12 }, () => 0);
    for (const segment of segments) {
        const rootPc = chordRootPc(segment.chord);
        if (rootPc === null) {
            continue;
        }
        const duration = Math.max(0, segment.end - segment.start);
        const quality = segment.chord.endsWith("m") ? "minor" : "major";
        for (let tonicPc = 0; tonicPc < 12; tonicPc += 1) {
            const degree = (rootPc - tonicPc + 12) % 12;
            const expected = majorDegreeQuality(degree);
            if (!expected) {
                continue;
            }
            scores[tonicPc] += duration * Math.max(0.25, segment.confidence) * (expected === quality ? 1 : 0.25);
        }
    }
    const ranked = scores
        .map((score, pc) => ({ pc, score }))
        .sort((left, right) => right.score - left.score || left.pc - right.pc);
    if (ranked.length === 0 || ranked[0].score <= 0) {
        return null;
    }
    const second = ranked[1]?.score ?? 0;
    if (ranked[0].score < second * 1.08) {
        return null;
    }
    return ranked[0].pc;
}

function estimateMusicalStartTime(segments: ChordSegment[]): number {
    const stable = segments.find((segment) => (
        segment.start >= MUSICAL_START_IGNORE_SECONDS
        && segment.chord !== "N"
        && segment.end - segment.start >= MUSICAL_START_MIN_STABLE_SECONDS
    ));
    return stable?.start ?? 0;
}

function vocalLineIndexInSong(lines: LyricChordLine[], lineIndex: number): number | null {
    let count = 0;
    for (let index = 0; index <= lineIndex; index += 1) {
        const line = lines[index];
        if (!line || line.time === null || isSectionLine(line.text)) {
            continue;
        }
        if (index === lineIndex) {
            return count;
        }
        count += 1;
    }
    return null;
}

function isSectionLine(text: string): boolean {
    return /^\[[^\]]+\]$/.test(text.trim());
}

function sectionDegrees(text: string): ReadonlyArray<number> {
    const normalized = text.toLowerCase();
    if (normalized.includes("outro")) {
        return [0, 4, 5, 0];
    }
    return [0, 4, 5, 7];
}

function markerLeftForIndex(index: number, count: number): number {
    if (count <= 1) {
        return 0;
    }
    if (count === 2) {
        return index === 0 ? 0 : 52;
    }
    return [0, 38, 72][index] ?? Math.min(MAX_MARKER_LEFT_PERCENT, index * 32);
}

function chordForMajorDegree(tonicPc: number, degree: number): string {
    const suffix = degree === 2 || degree === 4 || degree === 9 ? "m" : "";
    return `${SHARP_ROOTS[(tonicPc + degree) % 12]}${suffix}`;
}

function majorDegreeQuality(degree: number): "major" | "minor" | null {
    if (!MAJOR_SCALE_INTERVALS.includes(degree as (typeof MAJOR_SCALE_INTERVALS)[number])) {
        return null;
    }
    return degree === 2 || degree === 4 || degree === 9 ? "minor" : "major";
}

function chordRootPc(chord: string): number | null {
    if (chord === "N" || chord.endsWith("dim")) {
        return null;
    }
    const root = chord.endsWith("m") ? chord.slice(0, -1) : chord;
    const pc = SHARP_ROOTS.indexOf(root as (typeof SHARP_ROOTS)[number]);
    return pc >= 0 ? pc : null;
}

function buildNoLyricPhraseWindows(sourceSegments: ChordSegment[], durationSeconds: number): Array<[number, number]> {
    const windows: Array<[number, number]> = [];
    let startTime = 0;
    while (startTime < durationSeconds) {
        const targetEnd = Math.min(durationSeconds, startTime + NO_LYRIC_PHRASE_SECONDS);
        const snappedEnd = nearestSegmentBoundary(sourceSegments, targetEnd, startTime + NO_LYRIC_MIN_PHRASE_SECONDS, durationSeconds);
        const endTime = Math.max(startTime + MIN_LINE_DURATION_SECONDS, snappedEnd);
        windows.push([startTime, endTime]);
        startTime = endTime;
    }
    return windows;
}

function nearestSegmentBoundary(
    sourceSegments: ChordSegment[],
    targetTime: number,
    minTime: number,
    durationSeconds: number,
): number {
    const maxTime = Math.min(durationSeconds, targetTime + 1.2);
    const candidates = sourceSegments
        .map((segment) => segment.start)
        .filter((time) => time >= minTime && time <= maxTime);
    if (candidates.length === 0) {
        return Math.min(durationSeconds, targetTime);
    }
    return candidates.reduce((best, time) => (Math.abs(time - targetTime) < Math.abs(best - targetTime) ? time : best), candidates[0]);
}

function phraseConfidence(sourceSegments: ChordSegment[], startTime: number, endTime: number): number {
    let weighted = 0;
    let total = 0;
    for (const segment of sourceSegments) {
        const overlap = Math.max(0, Math.min(endTime, segment.end) - Math.max(startTime, segment.start));
        if (overlap <= 0) {
            continue;
        }
        weighted += overlap * segment.confidence;
        total += overlap;
    }
    if (total <= 0) {
        return 0.88;
    }
    return Math.max(0.72, Math.min(0.94, weighted / total));
}

function copyOverlappingSourceSegments(segments: ChordSegment[], startTime: number, endTime: number): ChordSegment[] {
    return segments
        .map((segment) => ({
            ...segment,
            start: Math.max(startTime, segment.start),
            end: Math.min(endTime, segment.end),
        }))
        .filter((segment) => segment.end > segment.start);
}

function mergeAdjacentSegments(segments: ChordSegment[], durationSeconds: number): ChordSegment[] {
    const safeDuration = Number.isFinite(durationSeconds) ? Math.max(0, durationSeconds) : 0;
    const sorted = segments
        .map((segment) => ({
            ...segment,
            start: Math.max(0, Math.min(segment.start, safeDuration)),
            end: Math.max(0, Math.min(segment.end, safeDuration)),
        }))
        .filter((segment) => segment.end > segment.start)
        .sort((left, right) => left.start - right.start || left.end - right.end);
    const merged: ChordSegment[] = [];
    for (const segment of sorted) {
        const previous = merged.at(-1);
        if (previous && previous.chord === segment.chord && Math.abs(previous.end - segment.start) <= 0.001) {
            previous.end = segment.end;
            previous.confidence = Math.max(previous.confidence, segment.confidence);
            continue;
        }
        merged.push({ ...segment });
    }
    return merged;
}

function dominantOverlapOnly(overlaps: Array<{ chord: string; start: number; duration: number }>): Array<{ chord: string; start: number; duration: number }> {
    if (overlaps.length === 0) {
        return [];
    }
    return [overlaps.reduce((best, item) => (item.duration > best.duration ? item : best), overlaps[0])];
}
