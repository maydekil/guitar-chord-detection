import type { ChordLabel, ChordSegment } from "@gcd/shared/analysis";

export const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
export const EDITABLE_CHORD_OPTIONS = ["N", ...SHARP_ROOTS.flatMap((root) => [root, `${root}m`, `${root}dim`])];

const MAJOR_KEY_CHORDS: Record<string, Set<string>> = {
    C: new Set(["C", "Dm", "Em", "F", "G", "Am", "Bdim"]),
    "C#": new Set(["C#", "D#m", "Fm", "F#", "G#", "A#m", "Cdim"]),
    D: new Set(["D", "Em", "F#m", "G", "A", "Bm", "C#dim"]),
    "D#": new Set(["D#", "Fm", "Gm", "G#", "A#", "Cm", "Ddim"]),
    E: new Set(["E", "F#m", "G#m", "A", "B", "C#m", "D#dim"]),
    F: new Set(["F", "Gm", "Am", "A#", "C", "Dm", "Edim"]),
    "F#": new Set(["F#", "G#m", "A#m", "B", "C#", "D#m", "Fdim"]),
    G: new Set(["G", "Am", "Bm", "C", "D", "Em", "F#dim"]),
    "G#": new Set(["G#", "A#m", "Cm", "C#", "D#", "Fm", "Gdim"]),
    A: new Set(["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"]),
    "A#": new Set(["A#", "Cm", "Dm", "D#", "F", "Gm", "Adim"]),
    B: new Set(["B", "C#m", "D#m", "E", "F#", "G#m", "A#dim"]),
};

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
    const match = /^(C#|D#|F#|G#|A#|C|D|E|F|G|A|B)(.*)$/.exec(chord);
    if (!match) {
        return chord;
    }
    const rootIndex = SHARP_ROOTS.indexOf(match[1] as (typeof SHARP_ROOTS)[number]);
    if (rootIndex < 0) {
        return chord;
    }
    return `${SHARP_ROOTS[modulo(rootIndex + semitones, SHARP_ROOTS.length)]}${match[2]}`;
}

export function estimateMajorKey(segments: ChordSegment[]): string {
    if (segments.length === 0) {
        return "-";
    }
    let bestKey = "-";
    let bestScore = -1;
    for (const [key, chords] of Object.entries(MAJOR_KEY_CHORDS)) {
        const score = segments.reduce(
            (total, segment) => total + (chords.has(segment.chord) ? Math.max(0, segment.end - segment.start) : 0),
            0,
        );
        if (score > bestScore) {
            bestKey = key;
            bestScore = score;
        }
    }
    return bestKey;
}

export function formatAverageConfidence(segments: ChordSegment[]): string {
    if (segments.length === 0) {
        return "-";
    }
    const average = segments.reduce((total, segment) => total + segment.confidence, 0) / segments.length;
    return `${Math.round(average * 100)}%`;
}

export function formatConfidenceValue(confidence: number | undefined): string {
    if (confidence === undefined || !Number.isFinite(confidence)) {
        return "-";
    }
    return `${Math.round(confidence * 100)}%`;
}

export function normalizeEditableChord(value: string): ChordLabel | null {
    const trimmed = value.trim();
    if (trimmed.toUpperCase() === "N") {
        return "N";
    }
    const match = /^(c#|d#|f#|g#|a#|c|d|e|f|g|a|b)(m|dim)?$/i.exec(trimmed);
    if (!match) {
        return null;
    }
    const root = `${match[1][0].toUpperCase()}${match[1].slice(1)}`;
    const suffix = match[2]?.toLowerCase() ?? "";
    const normalized = `${root}${suffix}`;
    return EDITABLE_CHORD_OPTIONS.includes(normalized as ChordLabel) ? normalized as ChordLabel : null;
}

function modulo(value: number, divisor: number): number {
    return ((value % divisor) + divisor) % divisor;
}
