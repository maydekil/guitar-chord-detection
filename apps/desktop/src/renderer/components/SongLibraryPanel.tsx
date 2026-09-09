import type { SongLibrarySummary } from "@gcd/shared/library";

import { LibraryPageNumbers } from "./LibraryPageNumbers.js";
import { formatConfidenceValue } from "../lib/chords.js";
import { displaySongTitle, formatShortDate, formatSongChordCount } from "../lib/songLibrary.js";
import { formatTime } from "../lib/time.js";

type SongLibraryPanelProps = {
    songs: SongLibrarySummary[];
    total: number;
    query: string;
    page: number;
    pageSize: number;
    totalPages: number;
    selectedSongId: string | null;
    widthPx: number;
    canAddSong: boolean;
    onAddSong: () => void;
    onOpenEvaluation: () => void;
    onQueryChange: (value: string) => void;
    onPageChange: (page: number) => void;
    onPageSizeChange: (pageSize: number) => void;
    onOpenSong: (song: SongLibrarySummary) => void;
    onDeleteSong: (song: SongLibrarySummary) => void;
};

export function SongLibraryPanel({
    songs,
    total,
    query,
    page,
    pageSize,
    totalPages,
    selectedSongId,
    widthPx,
    canAddSong,
    onAddSong,
    onOpenEvaluation,
    onQueryChange,
    onPageChange,
    onPageSizeChange,
    onOpenSong,
    onDeleteSong,
}: SongLibraryPanelProps) {
    return (
        <section className="library-panel" aria-label="song library" style={{ width: `${widthPx}px` }}>
            <div className="library-toolbar">
                <div>
                    <h2>Song Library</h2>
                    <p>{total} analyzed song(s)</p>
                </div>
                <div className="library-toolbar-actions">
                    <button type="button" className="add-song-btn" onClick={onAddSong} disabled={!canAddSong}>
                        Add Song
                    </button>
                    <button type="button" className="add-song-btn secondary-btn" onClick={onOpenEvaluation} disabled={!canAddSong}>
                        Evaluate
                    </button>
                </div>
                <input
                    type="search"
                    value={query}
                    onChange={(event) => onQueryChange(event.currentTarget.value)}
                    placeholder="Search title, artist, path"
                    aria-label="Search library"
                />
            </div>
            {songs.length > 0 ? (
                <div className="library-list" data-testid="library-list">
                    {songs.map((song) => (
                        <div key={song.id} className="library-song" data-active={selectedSongId === song.id ? "true" : "false"}>
                            <button type="button" className="library-song-main" onClick={() => onOpenSong(song)}>
                                <span>{song.title}</span>
                                <small className="library-song-artist">{song.artist || "Unknown artist"}</small>
                                <small>{formatSongChordCount(song)} chords - {formatTime(song.duration)}</small>
                                <span className="library-song-meta">
                                    <small>Key {song.keyEstimate ?? "-"}</small>
                                    <small>Conf {formatConfidenceValue(song.averageConfidence)}</small>
                                    <small>{formatShortDate(song.updatedAt)}</small>
                                </span>
                            </button>
                            <button
                                type="button"
                                className="library-delete-btn"
                                aria-label={`Delete ${displaySongTitle(song)}`}
                                title="Delete from library"
                                onClick={() => onDeleteSong(song)}
                            >
                                Delete
                            </button>
                        </div>
                    ))}
                </div>
            ) : (
                <p className="library-empty">No analyzed songs yet.</p>
            )}
            <div className="library-pagination" aria-label="Song library pagination">
                <small>{total} song(s)</small>
                <label>
                    Page Rows
                    <select
                        aria-label="Songs per page"
                        value={pageSize}
                        onChange={(event) => onPageSizeChange(Number(event.currentTarget.value))}
                    >
                        <option value={5}>5</option>
                        <option value={10}>10</option>
                        <option value={20}>20</option>
                        <option value={50}>50</option>
                    </select>
                </label>
                <button type="button" aria-label="Previous page" onClick={() => onPageChange(Math.max(1, page - 1))} disabled={page <= 1}>
                    ‹
                </button>
                <span>
                    <LibraryPageNumbers currentPage={page} totalPages={totalPages} onSelectPage={onPageChange} />
                </span>
                <button
                    type="button"
                    aria-label="Next page"
                    onClick={() => onPageChange(Math.min(totalPages, page + 1))}
                    disabled={page >= totalPages}
                >
                    ›
                </button>
            </div>
        </section>
    );
}
