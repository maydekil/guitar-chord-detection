import http from "node:http";
import path from "node:path";

export async function readJsonBody<T>(request: http.IncomingMessage): Promise<T> {
    return JSON.parse((await readRawBody(request)).toString("utf-8")) as T;
}

export async function readRawBody(request: http.IncomingMessage): Promise<Buffer> {
    const chunks: Buffer[] = [];
    for await (const chunk of request) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
}

export function sendJson(response: http.ServerResponse, statusCode: number, payload: unknown): void {
    response.writeHead(statusCode, { "content-type": "application/json" });
    response.end(JSON.stringify(payload));
}

export function resolveAudioContentType(audioPath: string): string {
    const extension = path.extname(audioPath).toLowerCase();
    if (extension === ".mp3") {
        return "audio/mpeg";
    }
    if (extension === ".wav") {
        return "audio/wav";
    }
    return "application/octet-stream";
}

export function safeFileName(value: string): string {
    const safe = value.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    return safe || "song";
}

export function parseRangeHeader(rangeHeader: string | undefined, fileSize: number): { start: number; end: number } | null {
    if (!rangeHeader) {
        return null;
    }

    const match = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader.trim());
    if (!match) {
        return null;
    }

    const rawStart = match[1];
    const rawEnd = match[2];
    if (!rawStart && !rawEnd) {
        return null;
    }

    if (!rawStart) {
        const suffixLength = Number.parseInt(rawEnd, 10);
        if (!Number.isFinite(suffixLength) || suffixLength <= 0) {
            return null;
        }
        const start = Math.max(0, fileSize - suffixLength);
        return { start, end: fileSize - 1 };
    }

    const start = Number.parseInt(rawStart, 10);
    const end = rawEnd ? Number.parseInt(rawEnd, 10) : fileSize - 1;
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || start >= fileSize) {
        return null;
    }

    return {
        start,
        end: Math.min(end, fileSize - 1)
    };
}

export function resolveSafeAudioExtension(fileName: string): string {
    const extension = path.extname(fileName).toLowerCase();
    if (extension === ".mp3" || extension === ".wav") {
        return extension;
    }
    return ".audio";
}

export function isAlreadyExistsError(error: unknown): boolean {
    return error instanceof Error && "code" in error && (error as NodeJS.ErrnoException).code === "EEXIST";
}
