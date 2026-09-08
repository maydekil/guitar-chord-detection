import { expect, it } from "vitest";
import type { SongLibraryRecord } from "@gcd/shared/library";
import { buildChordSheetExport, buildLrcExport } from "../../../../api/src/exportFormats";

it("API exports current corrected chords with the same timestamps and transpose", () => {
    const song: SongLibraryRecord = {
        id: "test", title: "Test", artist: "Artist", audioPath: "song.wav",
        fileHash: "test", algorithm: "test", contractVersion: "1", duration: 12,
        chordCount: 2, createdAt: "", updatedAt: "", lyrics: "[00:00]last lyric",
        analysis: {
            version: "1", source: { path: "song.wav", duration: 12, sampleRate: 22050 },
            analysis: {
                algorithm: "test",
                chords: [
                    { start: 0, end: 10, chord: "C", confidence: 0.8 },
                    { start: 10, end: 12, chord: "G", confidence: 0.8 },
                ],
                detectedChords: [{ start: 0, end: 12, chord: "F#m", confidence: 0.7 }],
                leadSheetChords: [{ start: 0, end: 12, chord: "A", confidence: 0.94 }],
            },
        },
    };
    const txt = buildChordSheetExport(song, 2);
    expect(txt).toContain("00:00.00 - 00:10.00  D");
    expect(txt).toContain("00:10.00 - 00:12.00  A");
    expect(buildLrcExport(song, 2)).toContain("[D][A]");
    song.lyrics = "";
    expect(buildLrcExport(song, 2)).toContain("[00:10.00]A");
    expect(song.analysis.analysis.chords[0].chord).toBe("C");
});
