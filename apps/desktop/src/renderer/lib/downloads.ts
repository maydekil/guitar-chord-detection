export function safeDownloadName(value: string): string {
    const safe = value.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    return safe || "song";
}

export function triggerDownload(url: string, fileName?: string): void {
    const anchor = document.createElement("a");
    anchor.href = url;
    if (fileName) {
        anchor.download = fileName;
    }
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    if (url.startsWith("blob:")) {
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
}
