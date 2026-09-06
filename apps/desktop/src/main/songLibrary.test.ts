import { access, mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import { describe, expect, it } from "vitest";
import type { ChordAnalysisSuccess } from "@gcd/shared/analysis";

import { SongLibraryStore, resolveSongLibraryPath } from "./songLibrary";

async function createTempAudio(content: string, filename = "Album Lama.mp3"): Promise<string> {
    const root = await mkdtemp(path.join(os.tmpdir(), "gcd-library-test-"));
    const filePath = path.join(root, filename);
    await writeFile(filePath, content, "utf-8");
    return filePath;
}

function successAnalysis(audioPath: string): ChordAnalysisSuccess {
    return {
        version: "1",
        source: {
            path: audioPath,
            duration: 42,
            sampleRate: 22050
        },
        analysis: {
            algorithm: "chroma-template-v3",
            chords: [{ start: 0, end: 42, chord: "A", confidence: 0.9 }]
        }
    };
}

describe("SongLibraryStore", () => {
    it("upserts successful analysis as a searchable song record", async () => {
        const audioPath = await createTempAudio("song-content");
        const libraryPath = resolveSongLibraryPath(path.dirname(audioPath));
        const library = new SongLibraryStore(libraryPath);

        const record = await library.upsertAnalysis({
            audioPath,
            algorithm: "chroma-template-v3",
            contractVersion: "1",
            analysis: successAnalysis(audioPath),
            metadata: {
                artist: "SR Banyak Cerita",
                title: "Album Lama"
            }
        });

        expect(record.title).toBe("Album Lama");
        expect(record.artist).toBe("SR Banyak Cerita");
        expect(record.audioPath).not.toBe(audioPath);
        await expect(access(record.audioPath)).resolves.toBeUndefined();
        expect(record.analysis.analysis.chords[0]?.chord).toBe("A");

        const byTitle = await library.listSongs({ query: "album" });
        expect(byTitle.map((song) => song.id)).toEqual([record.id]);

        const byPath = await library.listSongs({ query: ".mp3" });
        expect(byPath.map((song) => song.id)).toEqual([record.id]);
    });

    it("updates existing record for the same file and version scope", async () => {
        const audioPath = await createTempAudio("same-song", "track.wav");
        const libraryPath = resolveSongLibraryPath(path.dirname(audioPath));
        const library = new SongLibraryStore(libraryPath, {
            now: () => new Date("2026-01-01T00:00:00.000Z")
        });

        const first = await library.upsertAnalysis({
            audioPath,
            algorithm: "chroma-template-v3",
            contractVersion: "1",
            analysis: successAnalysis(audioPath),
            metadata: {
                artist: "First Artist",
                title: "First Title"
            }
        });

        const database = new DatabaseSync(libraryPath);
        database
            .prepare("UPDATE songs SET title = ?, artist = ? WHERE id = ?")
            .run("Edited Title", "Edited Artist", first.id);
        database.close();

        const updatedLibrary = new SongLibraryStore(libraryPath, {
            now: () => new Date("2026-01-02T00:00:00.000Z")
        });
        const updated = await updatedLibrary.upsertAnalysis({
            audioPath,
            algorithm: "chroma-template-v3",
            contractVersion: "1",
            analysis: {
                ...successAnalysis(audioPath),
                analysis: {
                    algorithm: "chroma-template-v3",
                    chords: [{ start: 0, end: 42, chord: "E", confidence: 0.8 }]
                }
            },
            metadata: {
                artist: "Updated Artist",
                title: "Updated Title"
            }
        });

        expect(updated.id).toBe(first.id);
        expect(updated.title).toBe("Updated Title");
        expect(updated.artist).toBe("Updated Artist");
        expect(updated.createdAt).toBe("2026-01-01T00:00:00.000Z");
        expect(updated.updatedAt).toBe("2026-01-02T00:00:00.000Z");
        expect(updated.analysis.analysis.chords[0]?.chord).toBe("E");
    });

    it("deletes song records and their managed audio file", async () => {
        const audioPath = await createTempAudio("delete-song", "delete-me.mp3");
        const libraryPath = resolveSongLibraryPath(path.dirname(audioPath));
        const library = new SongLibraryStore(libraryPath);

        const record = await library.upsertAnalysis({
            audioPath,
            algorithm: "chroma-template-v3",
            contractVersion: "1",
            analysis: successAnalysis(audioPath),
            metadata: {
                artist: "Delete Artist",
                title: "Delete Title"
            }
        });

        await expect(access(record.audioPath)).resolves.toBeUndefined();
        await expect(library.deleteSong(record.id)).resolves.toEqual({ deleted: true });
        await expect(access(record.audioPath)).rejects.toMatchObject({ code: "ENOENT" });
        expect(await library.listSongs()).toEqual([]);
    });
});
