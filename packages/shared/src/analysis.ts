export type ChordLabel =
    | "C"
    | "C#"
    | "D"
    | "D#"
    | "E"
    | "F"
    | "F#"
    | "G"
    | "G#"
    | "A"
    | "A#"
    | "B"
    | "Cm"
    | "C#m"
    | "Dm"
    | "D#m"
    | "Em"
    | "Fm"
    | "F#m"
    | "Gm"
    | "G#m"
    | "Am"
    | "A#m"
    | "Bm"
    | "Cdim"
    | "C#dim"
    | "Ddim"
    | "D#dim"
    | "Edim"
    | "Fdim"
    | "F#dim"
    | "Gdim"
    | "G#dim"
    | "Adim"
    | "A#dim"
    | "Bdim"
    | "N";

export interface ChordSegment {
    start: number;
    end: number;
    chord: ChordLabel;
    confidence: number;
}

export interface AnalysisSource {
    path: string;
    duration: number;
    sampleRate: number;
}

export interface AnalysisMetadata {
    algorithm: string;
    chords: ChordSegment[];
    detectedChords?: ChordSegment[];
    leadSheetChords?: ChordSegment[];
    leadSheetSource?: "lyrics" | "audio";
}

export interface ChordAnalysisSuccess {
    version: string;
    source: AnalysisSource;
    analysis: AnalysisMetadata;
}

export interface ChordAnalysisError {
    version: string;
    error: {
        code: string;
        message: string;
    };
}

export type ChordAnalysisResult = ChordAnalysisSuccess | ChordAnalysisError;
