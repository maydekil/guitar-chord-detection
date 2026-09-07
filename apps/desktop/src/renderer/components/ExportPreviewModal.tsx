import type { ExportFormat } from "../appTypes.js";
import { formatTranspose } from "../lib/chords.js";

type ExportPreviewModalProps = {
    format: ExportFormat;
    artist: string;
    title: string;
    transposeSemitones: number;
    content: string;
    chordSegmentCount: number;
    hasLyrics: boolean;
    onDownload: (format: ExportFormat) => void;
    onClose: () => void;
};

export function ExportPreviewModal({
    format,
    artist,
    title,
    transposeSemitones,
    content,
    chordSegmentCount,
    hasLyrics,
    onDownload,
    onClose,
}: ExportPreviewModalProps) {
    return (
        <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
            <section
                className="export-preview-modal"
                role="dialog"
                aria-modal="true"
                aria-label="Export preview"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <div className="export-preview-head">
                    <div>
                        <strong>{format.toUpperCase()} Preview</strong>
                        <small>
                            {artist || "Unknown Artist"} - {title || "Untitled"} · Transpose {formatTranspose(transposeSemitones)}
                        </small>
                    </div>
                    <div>
                        <button type="button" onClick={() => onDownload(format)}>
                            Download
                        </button>
                        <button type="button" onClick={onClose}>
                            Close
                        </button>
                    </div>
                </div>
                <div className="export-preview-meta">
                    <span>{content.split(/\r?\n/).filter((line) => line.trim()).length} line(s)</span>
                    <span>{chordSegmentCount} chord segment(s)</span>
                    <span>{hasLyrics ? "Lyrics included" : "Timeline only"}</span>
                </div>
                {content.trim() ? (
                    <pre>{content}</pre>
                ) : (
                    <p className="export-preview-empty">Tidak ada konten export. Pastikan lagu sudah punya timeline chord atau lyric.</p>
                )}
            </section>
        </div>
    );
}
