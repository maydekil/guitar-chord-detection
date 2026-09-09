import { describe, expect, it } from "vitest";
import type { ChordAnalysisSuccess, ChordSegment } from "@gcd/shared/analysis";
import {
    buildLeadSheetAnalysis, buildLeadSheetLyricChordMarkers, buildLeadSheetTimelineSegments,
    buildLeadSheetTimelineFromSegments, lyricChordWindowEnd, selectPlayableChordSegments,
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

    it("restores detector chords from legacy lyric-arranged analyses", () => {
        const arranged = chords.map((s) => ({ ...s, chord: "C" as const }));
        const legacy = analysis();
        legacy.analysis = {
            ...legacy.analysis,
            chords: arranged,
            detectedChords: chords,
            leadSheetChords: arranged,
            leadSheetSource: "lyrics",
        };

        const restored = buildLeadSheetAnalysis(legacy, "[00:00]lyrics");
        expect(selectPlayableChordSegments(legacy)).toEqual(chords);
        expect(restored.analysis.chords).toEqual(chords);
        expect(restored.analysis.leadSheetSource).toBe("audio");
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

    it("normalizes lyric capitalization while preserving section markers", () => {
        const lines = parseLyrics([
            "[00:00][Intro]",
            "[00:11]KAMU BILANG MAU KEMANAH?",
            "[00:23]KU Jawa Belum Ada Rencana",
            "BUKAN KENCAN",
        ].join("\n"));

        expect(lines.map((line) => line.text)).toEqual([
            "[Intro]",
            "Kamu bilang mau kemanah?",
            "Ku jawa belum ada rencana",
            "Bukan kencan",
        ]);
    });

    it("keeps detected chord timeline unchanged when timed lyrics are present", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 18, chord: "D", confidence: 0.86 },
            { start: 18, end: 23, chord: "F#m", confidence: 0.82 },
            { start: 23, end: 28, chord: "G", confidence: 0.80 },
            { start: 28, end: 32, chord: "D", confidence: 0.84 },
            { start: 32, end: 38, chord: "A", confidence: 0.78 },
        ];
        const timedLyrics = [
            "[00:00][Intro]",
            "[00:18]Kamu bilang mau kemana",
            "[00:23]Ku jawab belum ada rencana",
            "[00:28]Kamu ajak jalan berdua",
            "[00:32]Aku bilang boleh juga",
        ].join("\n");
        const result = buildLeadSheetAnalysis({
            version: "1",
            source: { path: "song.wav", duration: 38, sampleRate: 22050 },
            analysis: { algorithm: "test", chords: source },
        }, timedLyrics);

        expect(result.analysis.leadSheetSource).toBe("audio");
        expect(result.analysis.chords).toEqual(source);
        expect(result.analysis.leadSheetChords).toEqual(source);
        expect(result.analysis.detectedChords).toEqual(source);
    });
});
