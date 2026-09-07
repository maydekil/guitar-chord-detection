export function toFileUrl(localPath: string): string {
    const normalizedPath = localPath.replace(/\\/g, "/");
    const withLeadingSlash = normalizedPath.startsWith("/") ? normalizedPath : `/${normalizedPath}`;
    return `file://${encodeURI(withLeadingSlash)}`;
}

export function toAudioSourceUrl(source: { kind?: string; url?: string; bytes?: Uint8Array; mimeType?: string }): string {
    if (source.kind === "url" && source.url) {
        return source.url;
    }
    if (source.bytes) {
        return createObjectUrl(source.bytes, source.mimeType);
    }
    return "";
}

export function createObjectUrl(bytes: Uint8Array, mimeType?: string): string {
    const blob = new Blob([bytes.slice().buffer], { type: mimeType ?? "audio/mpeg" });
    return URL.createObjectURL(blob);
}
