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
