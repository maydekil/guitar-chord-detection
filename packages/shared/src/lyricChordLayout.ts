import type { ChordAnalysisSuccess, ChordSegment } from "./analysis.js";

type LeadSheetChord = ChordSegment["chord"];
type MajorScaleChordSet = { I: LeadSheetChord; ii: LeadSheetChord; iii: LeadSheetChord; IV: LeadSheetChord; V: LeadSheetChord; vi: LeadSheetChord };

export interface LyricChordLine {
    time: number | null;
    text: string;
}

export interface LyricChordMarker {
    label: string;
    left: number;
}

export interface LeadSheetLearningProfile {
    phrasePatterns: Record<string, LeadSheetChord[]>;
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
    lyrics: string,
    learningProfile?: LeadSheetLearningProfile,
): ChordAnalysisSuccess {
    const sourceChords = selectPlayableChordSegments(analysis);
    const arranged = arrangePlayableLeadSheet(sourceChords, lyrics, analysis.source.duration, learningProfile);
    const chords = arranged.length > 0
        ? arranged
        : buildLeadSheetTimelineFromSegments(sourceChords, analysis.source.duration);
    return {
        ...analysis,
        analysis: {
            ...analysis.analysis,
            chords,
            detectedChords: analysis.analysis.detectedChords ?? analysis.analysis.chords,
            leadSheetChords: chords,
            leadSheetSource: arranged.length > 0 ? "lyrics" : "audio",
        },
    };
}

export function arrangePlayableLeadSheet(
    sourceSegments: ChordSegment[],
    lyrics: string,
    durationSeconds: number,
    learningProfile?: LeadSheetLearningProfile,
): ChordSegment[] {
    const lines = parseLeadSheetLyrics(lyrics);
    const timedLines = lines.filter((line) => line.time !== null);
    if (timedLines.length < 6 || durationSeconds < 30) {
        return [];
    }

    const key = estimateMajorKey(sourceSegments);
    if (!key) {
        return [];
    }

    const scale = majorScaleChords(key);
    const arranged: ChordSegment[] = [];
    const rememberedPatterns = new Map<string, LeadSheetChord[]>();
    const shortLineMode = usesShortLyricLineProgression(timedLines, durationSeconds);
    const rememberedPhrases: Array<{ text: string; pattern: LeadSheetChord[] }> = [];
    let phraseIndex = 0;
    let lyricPhraseCount = 0;

    for (const [index, line] of lines.entries()) {
        if (line.time === null) {
            continue;
        }
        const start = clampTime(line.time, durationSeconds);
        const end = lyricChordWindowEnd(lines, index, sourceSegments);
        const safeEnd = Math.min(durationSeconds, Math.max(start + 0.1, end));
        const section = parseSectionMarker(line.text);
        const normalizedText = normalizeLyricText(line.text);

        let pattern: LeadSheetChord[];
        if (section === "intro" || section === "instrumental") {
            const repeats = section === "instrumental" && safeEnd - start >= 11 ? 2 : 1;
            pattern = repeatPattern([scale.I, scale.iii, scale.IV, scale.V], repeats);
            phraseIndex = 0;
        } else if (section === "outro") {
            pattern = [scale.I, scale.iii, scale.IV, scale.V, scale.I];
            phraseIndex = 0;
        } else if (isOpeningPickupLine(line, safeEnd, lyricPhraseCount)) {
            pattern = [scale.I];
        } else if (learningProfile) {
            const learned = findLearnedPattern(normalizedText, learningProfile);
            pattern = learned
                ?? rememberedPatterns.get(normalizedText)
                ?? (shortLineMode
                    ? shortLinePatternForIndex(phraseIndex, scale)
                    : phrasePatternForIndex(phraseIndex, scale));
            rememberedPatterns.set(normalizedText, pattern);
            if (normalizedText && !learned) {
                rememberedPhrases.push({ text: normalizedText, pattern });
            }
            phraseIndex += 1;
            lyricPhraseCount += 1;
        } else if (rememberedPatterns.has(normalizedText)) {
            pattern = rememberedPatterns.get(normalizedText) ?? [scale.I];
            phraseIndex += 1;
            lyricPhraseCount += 1;
        } else {
            const rememberedPhrase = findRememberedPhrasePattern(normalizedText, rememberedPhrases);
            pattern = rememberedPhrase
                ?? (shortLineMode
                    ? shortLinePatternForIndex(phraseIndex, scale)
                    : phrasePatternForIndex(phraseIndex, scale));
            rememberedPatterns.set(normalizedText, pattern);
            if (normalizedText) {
                rememberedPhrases.push({ text: normalizedText, pattern });
            }
            phraseIndex += 1;
            lyricPhraseCount += 1;
        }

        arranged.push(...segmentsForPattern(pattern, start, safeEnd, confidenceForWindow(sourceSegments, start, safeEnd)));
    }

    return mergePlayableSegments(arranged, durationSeconds);
}

function isOpeningPickupLine(line: LyricChordLine, endTime: number, lyricPhraseCount: number): boolean {
    if (lyricPhraseCount > 0 || line.time === null || line.time > 2.5) {
        return false;
    }

    const normalized = normalizeLyricText(line.text);
    if (!normalized) {
        return false;
    }

    const tokenCount = normalized.split(" ").filter(Boolean).length;
    const duration = Math.max(0, endTime - line.time);
    return tokenCount <= 2 && duration >= 4;
}

export function extractLeadSheetLearningProfile(
    analysis: ChordAnalysisSuccess,
    lyrics: string,
): LeadSheetLearningProfile {
    const lines = parseLeadSheetLyrics(lyrics);
    const profile: LeadSheetLearningProfile = { phrasePatterns: {} };
    const segments = analysis.analysis.chords;
    for (const [index, line] of lines.entries()) {
        if (line.time === null || parseSectionMarker(line.text)) {
            continue;
        }
        const key = normalizeLyricText(line.text);
        if (!key) {
            continue;
        }
        const end = lyricChordWindowEnd(lines, index, segments);
        const pattern = distinctChordsForWindow(segments, line.time, end);
        if (pattern.length > 0) {
            profile.phrasePatterns[key] = pattern;
        }
    }
    return profile;
}

export function mergeLeadSheetLearningProfiles(
    base: LeadSheetLearningProfile,
    next: LeadSheetLearningProfile,
): LeadSheetLearningProfile {
    return {
        phrasePatterns: {
            ...base.phrasePatterns,
            ...next.phrasePatterns,
        },
    };
}

function findLearnedPattern(normalizedText: string, profile: LeadSheetLearningProfile): LeadSheetChord[] | null {
    const exact = profile.phrasePatterns[normalizedText];
    if (exact?.length) {
        return exact;
    }
    const remembered = Object.entries(profile.phrasePatterns).map(([text, pattern]) => ({ text, pattern }));
    return findRememberedPhrasePattern(normalizedText, remembered);
}

function distinctChordsForWindow(segments: ChordSegment[], start: number, end: number): LeadSheetChord[] {
    const labels = segments
        .filter((segment) => segment.end > start && segment.start < end && segment.chord !== "N")
        .map((segment) => segment.chord);
    return labels.filter((label, index) => index === 0 || label !== labels[index - 1]).slice(0, 4);
}

function findRememberedPhrasePattern(
    normalizedText: string,
    rememberedPhrases: Array<{ text: string; pattern: LeadSheetChord[] }>,
): LeadSheetChord[] | null {
    const tokens = significantTokens(normalizedText);
    if (tokens.length < 2) {
        return null;
    }

    let best: { pattern: LeadSheetChord[]; score: number } | null = null;
    for (const remembered of rememberedPhrases) {
        const rememberedTokens = significantTokens(remembered.text);
        if (rememberedTokens.length < 2) {
            continue;
        }
        const shared = tokens.filter((token) => rememberedTokens.includes(token));
        if (new Set(shared).size < 2) {
            continue;
        }
        const containment = shared.length / Math.max(1, Math.min(tokens.length, rememberedTokens.length));
        const prefixBonus = tokens[0] === rememberedTokens[0] ? 0.18 : 0;
        const score = containment + prefixBonus;
        if (score >= 0.62 && (!best || score > best.score)) {
            best = { pattern: remembered.pattern, score };
        }
    }
    return best?.pattern ?? null;
}

function significantTokens(normalizedText: string): string[] {
    const stopWords = new Set([
        "aku", "kamu", "ku", "di", "yang", "dan", "tak", "dari", "lagi", "pun", "ini", "itu",
        "line", "lyric", "short", "verse", "chorus", "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth",
    ]);
    return normalizedText
        .split(" ")
        .map((token) => token.trim())
        .filter((token) => token.length >= 4 && !stopWords.has(token));
}

function usesShortLyricLineProgression(lines: LyricChordLine[], durationSeconds: number): boolean {
    const times = lines
        .map((line) => line.time)
        .filter((time): time is number => time !== null)
        .filter((time) => Number.isFinite(time))
        .sort((left, right) => left - right);
    if (times.length < 16 || durationSeconds <= 0) {
        return false;
    }

    const gaps: number[] = [];
    for (let index = 1; index < times.length; index += 1) {
        const gap = times[index] - times[index - 1];
        if (gap > 0.25 && gap < 12) {
            gaps.push(gap);
        }
    }
    if (gaps.length === 0) {
        return false;
    }
    const medianGap = gaps.sort((left, right) => left - right)[Math.floor(gaps.length / 2)] ?? 0;
    const linesPerMinute = times.length / Math.max(durationSeconds / 60, 0.1);
    return medianGap <= 4.2 || linesPerMinute >= 13;
}

function parseLeadSheetLyrics(lyrics: string): LyricChordLine[] {
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
            return { time: minutes * 60 + seconds + fraction, text: (match[4] ?? "").trim() };
        });
}

function estimateMajorKey(segments: ChordSegment[]): string | null {
    const pitchNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let best: { key: string; score: number } | null = null;
    for (const key of pitchNames) {
        const scale = majorScaleChords(key);
        const family = new Set([scale.I, scale.ii, scale.iii, scale.IV, scale.V, scale.vi]);
        let score = 0;
        for (const segment of segments) {
            if (segment.chord === "N") {
                continue;
            }
            const duration = Math.max(0, segment.end - segment.start);
            const tonicBonus = segment.chord === scale.I ? 0.45 : 0;
            const dominantBonus = segment.chord === scale.V ? 0.2 : 0;
            score += duration * segment.confidence * (family.has(segment.chord) ? 1 : -0.2);
            score += duration * tonicBonus + duration * dominantBonus;
        }
        if (!best || score > best.score) {
            best = { key, score };
        }
    }
    return best && best.score > 0 ? best.key : null;
}

function majorScaleChords(key: string): MajorScaleChordSet {
    const pitchNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const root = pitchNames.indexOf(key);
    const name = (offset: number) => pitchNames[(root + offset + 12) % 12] ?? "C";
    return {
        I: name(0),
        ii: `${name(2)}m`,
        iii: `${name(4)}m`,
        IV: name(5),
        V: name(7),
        vi: `${name(9)}m`,
    } as MajorScaleChordSet;
}

function phrasePatternForIndex(
    index: number,
    scale: MajorScaleChordSet,
): LeadSheetChord[] {
    const cycle = [
        [scale.I, scale.iii],
        [scale.IV, scale.I],
        [scale.V, scale.vi],
        [scale.ii, scale.V],
        [scale.I, scale.iii],
        [scale.IV, scale.I],
        [scale.V, scale.vi],
        [scale.ii, scale.V, scale.I],
        [scale.vi, scale.iii],
        [scale.IV, scale.V],
    ];
    return cycle[index % cycle.length] ?? [scale.I];
}

function shortLinePatternForIndex(
    index: number,
    scale: MajorScaleChordSet,
): LeadSheetChord[] {
    const cycle = [
        scale.I,
        scale.iii,
        scale.IV,
        scale.I,
        scale.V,
        scale.vi,
        scale.ii,
        scale.V,
        scale.I,
        scale.iii,
        scale.IV,
        scale.V,
        scale.I,
        scale.iii,
        scale.IV,
        scale.I,
        scale.V,
        scale.vi,
        scale.ii,
        scale.V,
    ];
    return [cycle[index % cycle.length] ?? scale.I];
}

function segmentsForPattern(pattern: LeadSheetChord[], start: number, end: number, confidence: number): ChordSegment[] {
    const labels = pattern.filter((label, index) => index === 0 || label !== pattern[index - 1]);
    const span = Math.max(0.1, end - start);
    return labels.map((label, index) => {
        const segmentStart = start + (span * index) / labels.length;
        const segmentEnd = index + 1 === labels.length ? end : start + (span * (index + 1)) / labels.length;
        return {
            start: roundTime(segmentStart),
            end: roundTime(Math.max(segmentStart + 0.1, segmentEnd)),
            chord: label as ChordSegment["chord"],
            confidence,
        };
    });
}

function mergePlayableSegments(segments: ChordSegment[], durationSeconds: number): ChordSegment[] {
    const sorted = segments
        .filter((segment) => segment.end > segment.start)
        .sort((left, right) => left.start - right.start);
    const merged: ChordSegment[] = [];
    for (const segment of sorted) {
        const clamped = {
            ...segment,
            start: clampTime(segment.start, durationSeconds),
            end: clampTime(segment.end, durationSeconds),
        };
        if (clamped.end <= clamped.start) {
            continue;
        }
        const previous = merged.at(-1);
        if (previous && previous.chord === clamped.chord && Math.abs(previous.end - clamped.start) <= 0.25) {
            previous.end = clamped.end;
            previous.confidence = weightedConfidence(previous, clamped);
        } else {
            merged.push(clamped);
        }
    }
    return merged;
}

function confidenceForWindow(segments: ChordSegment[], start: number, end: number): number {
    let weighted = 0;
    let total = 0;
    for (const segment of segments) {
        const overlap = Math.max(0, Math.min(end, segment.end) - Math.max(start, segment.start));
        if (overlap <= 0) {
            continue;
        }
        weighted += overlap * segment.confidence;
        total += overlap;
    }
    return Math.max(0.58, Math.min(0.92, total > 0 ? weighted / total : 0.72));
}

function weightedConfidence(left: ChordSegment, right: ChordSegment): number {
    const leftDuration = Math.max(0, left.end - left.start);
    const rightDuration = Math.max(0, right.end - right.start);
    const total = leftDuration + rightDuration;
    return total > 0
        ? ((left.confidence * leftDuration) + (right.confidence * rightDuration)) / total
        : Math.max(left.confidence, right.confidence);
}

function parseSectionMarker(text: string): "intro" | "instrumental" | "outro" | null {
    const normalized = text.trim().toLowerCase();
    if (!normalized.startsWith("[") || !normalized.endsWith("]")) {
        return null;
    }
    if (normalized.includes("intro")) {
        return "intro";
    }
    if (normalized.includes("instrument")) {
        return "instrumental";
    }
    if (normalized.includes("outro")) {
        return "outro";
    }
    return null;
}

function normalizeLyricText(text: string): string {
    return text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function repeatPattern(pattern: LeadSheetChord[], repeats: number): LeadSheetChord[] {
    return Array.from({ length: Math.max(1, repeats) }, () => pattern).flat();
}

function clampTime(value: number, durationSeconds: number): number {
    return Math.max(0, Math.min(Number.isFinite(durationSeconds) ? durationSeconds : value, Number.isFinite(value) ? value : 0));
}

function roundTime(value: number): number {
    return Math.round(value * 1000) / 1000;
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
