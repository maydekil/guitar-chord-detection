import { describe, expect, it } from "vitest";
import type { ChordSegment } from "@gcd/shared/analysis";
import { buildLeadSheetLyricChordMarkers, buildLeadSheetTimelineSegments } from "@gcd/shared/lyricChordLayout";

describe("instrumental source timing", () => {
    it("keeps a chord at zero instead of skipping to a later stable chord", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 8, chord: "A", confidence: 0.81 },
            { start: 8, end: 10, chord: "C#m", confidence: 0.73 },
        ];
        const lines = [{ time: 0, text: "[Intro]" }];
        expect(buildLeadSheetTimelineSegments(lines, source, 10)).toEqual(source);
        expect(buildLeadSheetLyricChordMarkers(lines, 0, source, 10)).toEqual([
            { label: "A", left: 0 }, { label: "C#m", left: 80 },
        ]);
    });

    it("preserves a no-chord pickup and does not apply its offset twice", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 0.6, chord: "N", confidence: 0.9 },
            { start: 0.6, end: 1, chord: "Dm", confidence: 0.71 },
            { start: 1, end: 4, chord: "A", confidence: 0.83 },
        ];
        const lines = [{ time: 0, text: "[Intro]" }];
        expect(buildLeadSheetTimelineSegments(lines, source, 4)).toEqual(source);
        expect(buildLeadSheetLyricChordMarkers(lines, 0, source, 4).map((m) => m.left)).toEqual([0, 15, 25]);
    });

    it.each([1, 2, 3])("preserves %i detected repetitions without inventing a progression", (count) => {
        const labels: ChordSegment["chord"][] = ["Dm", "A#", "F", "A"];
        const source = Array.from({ length: count * labels.length }, (_, index) => ({
            start: index * 1.5, end: (index + 1) * 1.5,
            chord: labels[index % labels.length], confidence: 0.79,
        }));
        expect(buildLeadSheetTimelineSegments([{ time: 0, text: "[Instrumental]" }], source, count * 6)).toEqual(source);
    });

    it("preserves a fast change near the end of an instrumental window", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 9.8, chord: "Am", confidence: 0.78 },
            { start: 9.8, end: 10, chord: "E", confidence: 0.85 },
        ];
        const lines = [{ time: 0, text: "[Break]" }];
        const markers = buildLeadSheetLyricChordMarkers(lines, 0, source, 10);
        expect(markers[1].left).toBeCloseTo(98);
        expect(buildLeadSheetTimelineSegments(lines, source, 10)).toEqual(source);
    });

    it("retains instrumental audio before the first timestamped lyric", () => {
        const source: ChordSegment[] = [
            { start: 0, end: 2, chord: "Dm", confidence: 0.8 },
            { start: 2, end: 4, chord: "A", confidence: 0.77 },
            { start: 4, end: 6, chord: "C", confidence: 0.74 },
        ];
        const actual = buildLeadSheetTimelineSegments([{ time: 4, text: "[Outro]" }], source, 6);
        expect(actual).toEqual(source);
        expect(source[0].start).toBe(0);
    });
});
