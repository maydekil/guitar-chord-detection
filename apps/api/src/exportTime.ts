export function formatExportTime(totalSeconds: number): string {
    const safeSeconds = Math.max(0, Number.isFinite(totalSeconds) ? totalSeconds : 0);
    const minutes = Math.floor(safeSeconds / 60);
    const seconds = safeSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}
