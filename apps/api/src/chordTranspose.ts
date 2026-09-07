const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

export function formatTranspose(semitones: number): string {
    if (semitones === 0) {
        return "0";
    }
    return semitones > 0 ? `+${semitones}` : `${semitones}`;
}

export function transposeChordLabel(chord: string, semitones: number): string {
    if (chord === "N" || semitones === 0) {
        return chord;
    }
    const match = chord.match(/^([A-G]#?)(m?)$/);
    if (!match) {
        return chord;
    }
    const rootIndex = SHARP_ROOTS.indexOf(match[1] as (typeof SHARP_ROOTS)[number]);
    if (rootIndex < 0) {
        return chord;
    }
    return `${SHARP_ROOTS[modulo(rootIndex + semitones, SHARP_ROOTS.length)]}${match[2]}`;
}

function modulo(value: number, divisor: number): number {
    return ((value % divisor) + divisor) % divisor;
}
