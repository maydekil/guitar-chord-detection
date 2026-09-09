import { describe, expect, it } from "vitest";

import { buildAudioFileDialogOptions, buildGenreEvaluationManifestDialogOptions, toFileSelectionResult } from "./fileDialog";

describe("buildAudioFileDialogOptions", () => {
    it("limits file selection to mp3 and wav via native filter", () => {
        const options = buildAudioFileDialogOptions();

        expect(options.properties).toEqual(["openFile"]);
        expect(options.filters).toEqual([
            {
                name: "Audio Files",
                extensions: ["mp3", "wav"]
            }
        ]);
    });
});

describe("buildGenreEvaluationManifestDialogOptions", () => {
    it("limits genre evaluation manifest selection to json", () => {
        const options = buildGenreEvaluationManifestDialogOptions();

        expect(options.properties).toEqual(["openFile"]);
        expect(options.filters).toEqual([
            {
                name: "JSON Manifest",
                extensions: ["json"]
            }
        ]);
    });
});

describe("toFileSelectionResult", () => {
    it("returns canceled result for empty selection", () => {
        expect(toFileSelectionResult([])).toEqual({
            canceled: true,
            path: null,
            fileName: null
        });
    });

    it("maps selected path to typed payload", () => {
        const result = toFileSelectionResult(["/tmp/song.mp3"]);

        expect(result).toEqual({
            canceled: false,
            path: "/tmp/song.mp3",
            fileName: "song.mp3"
        });
    });
});
