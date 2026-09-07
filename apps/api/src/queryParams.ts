export function normalizeSearch(value: string): string {
    return value.trim().toLocaleLowerCase();
}

export function normalizePage(value: number): number {
    return Number.isFinite(value) ? Math.max(1, Math.trunc(value)) : 1;
}

export function normalizePageSize(value: number): number {
    return Number.isFinite(value) ? Math.max(5, Math.min(100, Math.trunc(value))) : 10;
}

export function normalizeTransposeSemitones(semitones: number): number {
    if (!Number.isFinite(semitones)) {
        return 0;
    }
    return Math.max(-11, Math.min(11, Math.trunc(semitones)));
}

export function normalizeAssetCleanupMaxAgeMs(rawHours: string | undefined): number {
    const hours = rawHours ? Number.parseFloat(rawHours) : 24;
    if (!Number.isFinite(hours) || hours <= 0) {
        return 24 * 60 * 60_000;
    }
    return Math.max(1, hours) * 60 * 60_000;
}

export function normalizeLyricsModel(model: string | undefined): string {
    const allowedModels = new Set(["tiny", "base", "small", "medium", "large"]);
    return model && allowedModels.has(model) ? model : "small";
}
