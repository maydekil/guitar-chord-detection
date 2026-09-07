import type { MouseEvent, RefObject } from "react";
import type { Dispatch, SetStateAction } from "react";

export function usePanelResize(
    workspaceRef: RefObject<HTMLDivElement | null>,
    setLibraryPanelWidthPx: Dispatch<SetStateAction<number>>,
) {
    return (event: MouseEvent<HTMLDivElement>): void => {
        event.preventDefault();
        const workspace = workspaceRef.current;
        if (!workspace) {
            return;
        }
        const bounds = workspace.getBoundingClientRect();
        const minLeftWidth = 260;
        const minRightWidth = 380;
        const gapWidth = 16;
        const maxLeftWidth = Math.max(minLeftWidth, bounds.width - minRightWidth - gapWidth);
        const handleMove = (moveEvent: globalThis.MouseEvent): void => {
            const nextWidth = Math.max(minLeftWidth, Math.min(moveEvent.clientX - bounds.left, maxLeftWidth));
            setLibraryPanelWidthPx(nextWidth);
        };
        const handleUp = (): void => {
            window.removeEventListener("mousemove", handleMove);
            window.removeEventListener("mouseup", handleUp);
            document.body.classList.remove("is-resizing-panels");
        };
        document.body.classList.add("is-resizing-panels");
        window.addEventListener("mousemove", handleMove);
        window.addEventListener("mouseup", handleUp);
    };
}
