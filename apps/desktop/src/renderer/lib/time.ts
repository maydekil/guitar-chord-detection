export function formatTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, Math.floor(totalSeconds)) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export function formatLrcTime(seconds: number): string {
    const safeSeconds = Math.max(0, seconds);
    const minutes = Math.floor(safeSeconds / 60);
    const wholeSeconds = Math.floor(safeSeconds % 60);
    const centiseconds = Math.floor((safeSeconds - Math.floor(safeSeconds)) * 100);
    return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(centiseconds).padStart(2, "0")}`;
}

export function formatPreciseTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}
