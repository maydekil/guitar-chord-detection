import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import http from "node:http";

import type { SongLibraryRecord } from "@gcd/shared/library";
import { parseRangeHeader, resolveAudioContentType, sendJson } from "./httpUtils.js";

export async function streamSongAudio(
    request: http.IncomingMessage,
    response: http.ServerResponse,
    id: string,
    kind: "original" | "instrumental",
    getSong: (id: string) => SongLibraryRecord | null
): Promise<void> {
    const song = getSong(id);
    if (!song) {
        sendJson(response, 404, { message: "Song not found" });
        return;
    }

    const audioPath = kind === "original" ? song.audioPath : song.instrumentalAudioPath;
    if (!audioPath) {
        sendJson(response, 404, { message: "Audio stream not found" });
        return;
    }

    await streamAudioFile(request, response, audioPath);
}

export async function streamAudioFile(request: http.IncomingMessage, response: http.ServerResponse, audioPath: string): Promise<void> {
    const fileStat = await stat(audioPath).catch(() => null);
    if (!fileStat) {
        sendJson(response, 404, { message: "Audio file not found" });
        return;
    }

    const range = parseRangeHeader(request.headers.range, fileStat.size);
    if (range) {
        response.writeHead(206, {
            "accept-ranges": "bytes",
            "content-length": range.end - range.start + 1,
            "content-range": `bytes ${range.start}-${range.end}/${fileStat.size}`,
            "content-type": resolveAudioContentType(audioPath)
        });
        createReadStream(audioPath, { start: range.start, end: range.end }).pipe(response);
        return;
    }

    response.writeHead(200, {
        "accept-ranges": "bytes",
        "content-length": fileStat.size,
        "content-type": resolveAudioContentType(audioPath)
    });
    createReadStream(audioPath).pipe(response);
}
