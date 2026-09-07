import type { SongLibraryListResult, SongLibraryRecord, SongLibrarySummary, SongMetadataInput } from "@gcd/shared/library";

export function displaySongTitle(song: SongLibrarySummary): string {
    return song.title || song.audioPath.split(/[\\/]/).pop() || "Untitled";
}

export function buildSongMetadata(title: string, artist: string): SongMetadataInput | null {
    const trimmedTitle = title.trim();
    const trimmedArtist = artist.trim();
    if (!trimmedTitle || !trimmedArtist) {
        return null;
    }
    return { title: trimmedTitle, artist: trimmedArtist };
}

export function normalizeSongLibraryListResult(
    result: SongLibraryListResult | SongLibrarySummary[],
    fallbackPage: number,
    fallbackPageSize: number,
): SongLibraryListResult {
    if (Array.isArray(result)) {
        return {
            records: result,
            total: result.length,
            page: 1,
            pageSize: fallbackPageSize,
            totalPages: 1,
        };
    }
    return {
        records: result.records,
        total: result.total,
        page: result.page || fallbackPage,
        pageSize: result.pageSize || fallbackPageSize,
        totalPages: Math.max(1, result.totalPages || Math.ceil(result.total / Math.max(1, result.pageSize || fallbackPageSize))),
    };
}

export function buildCompactPageNumbers(currentPage: number, totalPages: number): Array<number | "ellipsis"> {
    if (totalPages <= 5) {
        return Array.from({ length: totalPages }, (_, index) => index + 1);
    }
    const pages = new Set([1, totalPages, currentPage - 1, currentPage, currentPage + 1]);
    const normalized = [...pages]
        .filter((page) => page >= 1 && page <= totalPages)
        .sort((left, right) => left - right);
    const compact: Array<number | "ellipsis"> = [];
    for (const page of normalized) {
        const previous = compact.at(-1);
        if (typeof previous === "number" && page - previous > 1) {
            compact.push("ellipsis");
        }
        compact.push(page);
    }
    return compact;
}

export function formatSongChordCount(song: SongLibrarySummary | SongLibraryRecord): number {
    if (Number.isFinite(song.chordCount)) {
        return song.chordCount;
    }
    return "analysis" in song ? song.analysis.analysis.chords.length : 0;
}

export function formatShortDate(value: string): string {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return "No date";
    }
    return date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
    });
}
