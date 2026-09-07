import type { PendingConfirmation } from "../appTypes.js";
import { displaySongTitle } from "../lib/songLibrary.js";

type ConfirmationModalProps = {
    confirmation: PendingConfirmation;
    onCancel: () => void;
    onConfirm: () => void;
};

export function ConfirmationModal({ confirmation, onCancel, onConfirm }: ConfirmationModalProps) {
    return (
        <div className="modal-backdrop" role="presentation" onMouseDown={onCancel}>
            <section
                className="confirm-modal"
                role="dialog"
                aria-modal="true"
                aria-label={confirmation.kind === "save" ? "Confirm save" : "Confirm delete"}
                onMouseDown={(event) => event.stopPropagation()}
            >
                <strong>{confirmation.kind === "save" ? "Save chord analysis?" : "Delete song from library?"}</strong>
                {confirmation.kind === "save" ? (
                    <p>
                        Simpan chord, lyric, dan metadata untuk {confirmation.metadata.artist} - {confirmation.metadata.title}.
                    </p>
                ) : (
                    <p>
                        Hapus {displaySongTitle(confirmation.song)} dari Song Library. File audio yang dikelola API/local juga akan ikut dibersihkan jika tersedia.
                    </p>
                )}
                <div className="confirm-modal-actions">
                    <button type="button" onClick={onCancel}>
                        Cancel
                    </button>
                    <button type="button" className={confirmation.kind === "delete" ? "danger-btn" : ""} onClick={onConfirm}>
                        {confirmation.kind === "save" ? "Confirm Save" : "Confirm Delete"}
                    </button>
                </div>
            </section>
        </div>
    );
}
