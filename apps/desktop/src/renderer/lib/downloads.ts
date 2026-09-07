export function safeDownloadName(value: string): string {
    const safe = value.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    return safe || "song";
}
