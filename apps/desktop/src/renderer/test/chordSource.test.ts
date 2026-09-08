import { describe, expect, it } from "vitest";
import type { ChordAnalysisSuccess, ChordSegment } from "@gcd/shared/analysis";
import {
    buildLeadSheetAnalysis, buildLeadSheetLyricChordMarkers, buildLeadSheetTimelineSegments,
    buildLeadSheetTimelineFromSegments, lyricChordWindowEnd, selectPlayableChordSegments,
    extractLeadSheetLearningProfile,
} from "@gcd/shared/lyricChordLayout";
import { buildChordTaggedLrcLines, parseLyrics } from "../lib/lyrics";

const chords: ChordSegment[] = [
    { start: 0, end: 0.6, chord: "N", confidence: 0.87 },
    { start: 0.6, end: 4, chord: "Am", confidence: 0.82 },
    { start: 4, end: 4.2, chord: "D#", confidence: 0.76 },
    { start: 4.2, end: 12, chord: "E", confidence: 0.8 },
];

function analysis(): ChordAnalysisSuccess {
    return {
        version: "1", source: { path: "song.wav", duration: 12, sampleRate: 22050 },
        analysis: { algorithm: "test", chords },
    };
}

describe("one playable chord source", () => {
    it("preserves timing, confidence, fast changes, and non-diatonic labels", () => {
        const lines = parseLyrics("[00:00]first line\n[00:05]second line");
        expect(buildLeadSheetTimelineSegments(lines, chords, 12)).toEqual(chords);
        expect(buildLeadSheetTimelineFromSegments(chords, 12)).toEqual(chords);
        const markers = buildLeadSheetLyricChordMarkers(lines, 0, chords, 5);
        expect(markers.map((m) => m.label)).toEqual(chords.map((s) => s.chord));
        markers.forEach((m, index) => expect(m.left * 5 / 100).toBeCloseTo(chords[index].start));
    });

    it("lyric text, line count, and timing cannot rewrite chords", () => {
        const first = buildLeadSheetAnalysis(analysis(), "[00:00]one line");
        const second = buildLeadSheetAnalysis(first, "[00:01]different\n[00:03]words\n[00:08]here");
        expect(second.analysis.chords).toEqual(chords);
        expect(second.analysis.leadSheetChords).toEqual(chords);
    });

    it("keeps the carry-over chord and excludes a change exactly at line end", () => {
        expect(buildLeadSheetLyricChordMarkers([{ time: 2, text: "words" }], 0, chords, 4))
            .toEqual([{ label: "Am", left: 0 }]);
    });

    it("preserves saved corrections over original detector and stale lead-sheet copies", () => {
        const edited = chords.map((s) => ({ ...s, chord: "Fm" as const }));
        const current = analysis();
        current.analysis = { ...current.analysis, chords: edited, detectedChords: chords, leadSheetChords: chords };
        const saved = buildLeadSheetAnalysis(current, "[00:00]lyrics");
        const loaded: ChordAnalysisSuccess = JSON.parse(JSON.stringify(saved));
        expect(selectPlayableChordSegments(loaded)).toEqual(edited);
        expect(loaded.analysis.detectedChords).toEqual(chords);
        expect(buildLeadSheetAnalysis(loaded, "").analysis.chords).toEqual(edited);
    });

    it("preserves explicit splits and does not mutate source objects", () => {
        const split: ChordSegment[] = [
            { start: 0, end: 2, chord: "Am", confidence: 0.6 },
            { start: 2, end: 4, chord: "Am", confidence: 0.9 },
        ];
        const result = buildLeadSheetTimelineFromSegments(split, 4);
        expect(result).toEqual(split);
        result[0].end = 1;
        expect(split[0].end).toBe(2);
    });

    it("includes late final-line changes and applies transpose once", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 10, chord: "C", confidence: 0.8 },
            { start: 10, end: 12, chord: "G", confidence: 0.8 },
        ];
        const lines = [{ time: 0, text: "last lyric" }];
        expect(lyricChordWindowEnd(lines, 0, source)).toBe(12);
        expect(buildChordTaggedLrcLines(lines, source, 2)[0]).toContain("[D][A]");
        expect(source[0].chord).toBe("C");
    });

    it("does not fabricate chords for untimed lyrics or empty audio", () => {
        expect(buildLeadSheetLyricChordMarkers([{ time: null, text: "untimed" }], 0, chords, 12)).toEqual([]);
        expect(buildLeadSheetTimelineSegments([{ time: 0, text: "lyrics" }], [], 12)).toEqual([]);
    });

    it("arranges timed lyrics into a repeated playable major-key lead sheet", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 32, chord: "A", confidence: 0.86 },
            { start: 32, end: 64, chord: "C#m", confidence: 0.82 },
            { start: 64, end: 96, chord: "D", confidence: 0.80 },
            { start: 96, end: 128, chord: "E", confidence: 0.84 },
            { start: 128, end: 160, chord: "F#m", confidence: 0.78 },
            { start: 160, end: 192, chord: "Bm", confidence: 0.77 },
        ];
        const timedLyrics = [
            "[00:00][Intro]",
            "[00:10][Intro]",
            "[00:25]first verse line",
            "[00:30]second verse line",
            "[00:36]third verse line",
            "[00:42]fourth verse line",
            "[00:48]fifth verse line",
            "[00:53]sixth verse line",
            "[00:59]seventh verse line",
            "[01:04]eighth verse line",
            "[01:09]pre line one",
            "[01:15]pre line two",
            "[01:21]chorus line one",
            "[01:26]chorus line two",
            "[01:32]chorus line three",
            "[01:37]chorus line four",
            "[01:43]chorus line one",
            "[01:49]chorus line two",
            "[01:54]chorus line three",
            "[01:59]chorus line four",
            "[02:05][Outro]",
        ].join("\n");
        const arranged = buildLeadSheetAnalysis({
            version: "1",
            source: { path: "song.wav", duration: 132, sampleRate: 22050 },
            analysis: { algorithm: "test", chords: source },
        }, timedLyrics);

        expect(arranged.analysis.leadSheetSource).toBe("lyrics");
        expect(arranged.analysis.chords.slice(0, 8).map((segment) => segment.chord)).toEqual([
            "A", "C#m", "D", "E", "A", "C#m", "D", "E",
        ]);
        expect(arranged.analysis.chords.some((segment) => segment.chord === "F#m")).toBe(true);
        expect(arranged.analysis.chords.some((segment) => segment.chord === "Bm")).toBe(true);
        const repeatedChorusChords = arranged.analysis.chords
            .filter((segment) => segment.start >= 81 && segment.start < 103)
            .map((segment) => segment.chord);
        expect(repeatedChorusChords).toEqual(["A", "C#m", "D", "A", "E", "F#m", "Bm", "E"]);
    });

    it("uses one chord per short lyric line for dense verse timing", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 50, chord: "A", confidence: 0.86 },
            { start: 50, end: 90, chord: "C#m", confidence: 0.82 },
            { start: 90, end: 130, chord: "D", confidence: 0.80 },
            { start: 130, end: 180, chord: "E", confidence: 0.84 },
            { start: 180, end: 220, chord: "F#m", confidence: 0.78 },
            { start: 220, end: 262, chord: "Bm", confidence: 0.77 },
        ];
        const timedLyrics = Array.from({ length: 40 }, (_, index) => {
            const seconds = 15 + (index * 3);
            const minute = Math.floor(seconds / 60);
            const second = String(seconds % 60).padStart(2, "0");
            return `[0${minute}:${second}]short lyric ${index + 1}`;
        }).join("\n");

        const arranged = buildLeadSheetAnalysis({
            version: "1",
            source: { path: "song.wav", duration: 262, sampleRate: 22050 },
            analysis: { algorithm: "test", chords: source },
        }, `[00:00][Intro]\n${timedLyrics}`);

        const postIntro = arranged.analysis.chords.filter((segment) => segment.start >= 15);
        expect(postIntro.slice(0, 8).map((segment) => segment.chord)).toEqual([
            "A", "C#m", "D", "A", "E", "F#m", "Bm", "E",
        ]);
        expect(postIntro.length).toBeLessThanOrEqual(41);
    });

    it("keeps an opening pickup ad-lib from shifting the first verse progression", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 80, chord: "C", confidence: 0.86 },
            { start: 80, end: 120, chord: "Em", confidence: 0.82 },
            { start: 120, end: 160, chord: "F", confidence: 0.80 },
            { start: 160, end: 213, chord: "G", confidence: 0.84 },
        ];
        const timedLyrics = [
            "[00:00]Woyah!",
            "[00:11]Baru kemarin kita bertemu",
            "[00:17]Hari ini kepikiran kamu",
            "[00:23]Kubuka ponsel berkali-kali",
            "[00:29]Siapa tau kamu menghubungi",
            "[00:34]Ketik pesan lalu ku hapus",
            "[00:40]Ketik lagi kok jadi serius",
            "[00:46]Akhirnya cuman ku tulis hai",
            "[00:52]5 menit, belumku kirim juga",
        ].join("\n");

        const arranged = buildLeadSheetAnalysis({
            version: "1",
            source: { path: "song.wav", duration: 90, sampleRate: 22050 },
            analysis: { algorithm: "test", chords: source },
        }, timedLyrics);
        const chordAt = (time: number) => arranged.analysis.chords.find((segment) => segment.start <= time && segment.end > time)?.chord;

        expect(chordAt(11)).toBe("C");
        expect(chordAt(14.1)).toBe("Em");
        expect(chordAt(17)).toBe("F");
    });

    it("reuses fuzzy phrase patterns for similar repeated lyric ideas", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 80, chord: "A", confidence: 0.86 },
            { start: 80, end: 150, chord: "C#m", confidence: 0.82 },
            { start: 150, end: 220, chord: "D", confidence: 0.80 },
            { start: 220, end: 280, chord: "E", confidence: 0.84 },
        ];
        const timedLyrics = [
            "[00:00][Intro]",
            "[00:15]Malu Malu Mau Kenalan",
            "[00:18]Dekat Dekat Malah Degdegan",
            "[00:21]Begitu Melulu Sudah Dekat",
            "[00:24]Senyum Senyum",
            "[00:27]Malu Malu Mau Bertanya",
            "[00:30]Dari Tadi Begitu Melulu",
            "[00:33]Senyum Baru",
            "[00:36]Malu Malu Mau Kenalan",
            "[00:39]Dari Tadi Begitu Melulu Lagi",
        ].join("\n");

        const arranged = buildLeadSheetAnalysis({
            version: "1",
            source: { path: "song.wav", duration: 60, sampleRate: 22050 },
            analysis: { algorithm: "test", chords: source },
        }, timedLyrics);
        const chordAt = (time: number) => arranged.analysis.chords.find((segment) => segment.start <= time && segment.end > time)?.chord;

        expect(chordAt(15)).toBe(chordAt(36));
        expect(chordAt(21)).toBe(chordAt(30));
        expect(chordAt(21)).toBe(chordAt(39));
        expect(chordAt(24)).not.toBe(chordAt(33));
    });

    it("learns edited phrase chords and applies them to later similar lyrics", () => {
        const edited: ChordSegment[] = [
            { start: 0, end: 4, chord: "A", confidence: 0.9 },
            { start: 4, end: 8, chord: "E", confidence: 0.9 },
            { start: 8, end: 12, chord: "F#m", confidence: 0.9 },
        ];
        const learned = extractLeadSheetLearningProfile({
            version: "1",
            source: { path: "saved.wav", duration: 12, sampleRate: 22050 },
            analysis: { algorithm: "manual-edit", chords: edited },
        }, "[00:00]Malu Malu Mau Kenalan\n[00:04]Dari Tadi Begitu Melulu\n[00:08]Akhirnya Kita Tau Nama");
        const freshSource: ChordSegment[] = [
            { start: 0, end: 30, chord: "A", confidence: 0.8 },
            { start: 30, end: 60, chord: "C#m", confidence: 0.8 },
            { start: 60, end: 90, chord: "D", confidence: 0.8 },
            { start: 90, end: 120, chord: "E", confidence: 0.8 },
        ];
        const arranged = buildLeadSheetAnalysis({
            version: "1",
            source: { path: "new.wav", duration: 80, sampleRate: 22050 },
            analysis: { algorithm: "test", chords: freshSource },
        }, [
            "[00:00][Intro]",
            "[00:15]Malu Malu Mau Bertanya",
            "[00:18]Dari Tadi Begitu Melulu Lagi",
            "[00:21]Akhirnya Kita Tau Nama",
            "[00:24]another line",
            "[00:27]another line two",
            "[00:30]another line three",
        ].join("\n"), learned);
        const chordAt = (time: number) => arranged.analysis.chords.find((segment) => segment.start <= time && segment.end > time)?.chord;

        expect(chordAt(15)).toBe("A");
        expect(chordAt(18)).toBe("E");
        expect(chordAt(21)).toBe("F#m");
    });
});
