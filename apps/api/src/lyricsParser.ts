export interface LyricLine {
    time: number | null;
    text: string;
}

export function parseLyrics(lyrics: string): LyricLine[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .filter(Boolean)
        .map((line) => {
            const match = line.match(/^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$/);
            if (!match) {
                return { time: null, text: line };
            }
            const minutes = Number.parseInt(match[1], 10);
            const seconds = Number.parseInt(match[2], 10);
            const fraction = match[3] ? Number.parseFloat(`0.${match[3].padEnd(3, "0")}`) : 0;
            const time = minutes * 60 + seconds + fraction;
            return { time: Number.isFinite(time) ? time : null, text: match[4].trim() };
        });
}
