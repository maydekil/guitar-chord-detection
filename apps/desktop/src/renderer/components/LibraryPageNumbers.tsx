import { buildCompactPageNumbers } from "../lib/songLibrary.js";

type LibraryPageNumbersProps = {
    currentPage: number;
    totalPages: number;
    onSelectPage: (page: number) => void;
};

export function LibraryPageNumbers({ currentPage, totalPages, onSelectPage }: LibraryPageNumbersProps) {
    return buildCompactPageNumbers(currentPage, totalPages).map((page, index) => {
        if (page === "ellipsis") {
            return <i key={`ellipsis-${index}`}>...</i>;
        }
        return (
            <button
                key={page}
                type="button"
                className="library-page-number"
                data-active={page === currentPage ? "true" : "false"}
                onClick={() => onSelectPage(page)}
                disabled={page === currentPage}
            >
                {page}
            </button>
        );
    });
}
