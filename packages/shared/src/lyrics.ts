export interface LyricsLine {
    start: number;
    end: number | null;
    text: string;
}

export interface LyricsTranscriptionSuccess {
    version: string;
    source: {
        path: string;
    };
    lyrics: {
        format: "lrc";
        language: string | null;
        text: string;
        lines: LyricsLine[];
    };
}

export interface LyricsTranscriptionError {
    version: string;
    error: {
        code: string;
        message: string;
    };
}

export type LyricsTranscriptionResult = LyricsTranscriptionSuccess | LyricsTranscriptionError;

export function normalizeLyricsText(lyrics: string): string {
    return lyrics
        .split(/\r?\n/)
        .map((line) => normalizeLrcLineText(line))
        .join("\n");
}

export function normalizeLyricDisplayText(text: string): string {
    const trimmed = text.trim();
    if (!trimmed || isSectionMarker(trimmed)) {
        return trimmed;
    }

    const leadingChordTags = trimmed.match(/^((?:\[[A-G](?:#|b)?m?\]\s*)+)/)?.[0] ?? "";
    const body = leadingChordTags ? trimmed.slice(leadingChordTags.length).trimStart() : trimmed;
    if (!body || isSectionMarker(body)) {
        return `${leadingChordTags}${body}`.trim();
    }

    const normalized = body
        .toLocaleLowerCase("id-ID")
        .replace(/(^|[.!?]\s+|["'([{]\s*)([a-zà-ÿ])/g, (_match, prefix: string, letter: string) => (
            `${prefix}${letter.toLocaleUpperCase("id-ID")}`
        ));

    return `${leadingChordTags}${normalized}`.trim();
}

function normalizeLrcLineText(line: string): string {
    const match = line.match(/^(\[\d{1,2}:\d{2}(?:\.\d{1,3})?\])(.*)$/);
    if (!match) {
        return normalizeLyricDisplayText(line);
    }
    return `${match[1]}${normalizeLyricDisplayText(match[2] ?? "")}`;
}

function isSectionMarker(text: string): boolean {
    return /^\[[^\]]+\]$/.test(text.trim());
}
