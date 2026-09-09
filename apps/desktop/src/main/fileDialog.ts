import path from "node:path";
import type { OpenDialogOptions } from "electron";

export interface FileSelectionResult {
    canceled: boolean;
    path: string | null;
    fileName: string | null;
}

export function buildAudioFileDialogOptions(): OpenDialogOptions {
    return {
        title: "Open Audio",
        properties: ["openFile"],
        filters: [
            {
                name: "Audio Files",
                extensions: ["mp3", "wav"]
            }
        ]
    };
}

export function buildGenreEvaluationManifestDialogOptions(): OpenDialogOptions {
    return {
        title: "Open Genre Evaluation Manifest",
        properties: ["openFile"],
        filters: [
            {
                name: "JSON Manifest",
                extensions: ["json"]
            }
        ]
    };
}

export function toFileSelectionResult(filePaths: string[]): FileSelectionResult {
    const selected = filePaths[0];
    if (!selected) {
        return {
            canceled: true,
            path: null,
            fileName: null
        };
    }

    return {
        canceled: false,
        path: selected,
        fileName: path.basename(selected)
    };
}
