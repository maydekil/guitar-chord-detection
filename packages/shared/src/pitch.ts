export interface PitchShiftSuccess {
    version: string;
    source: {
        path: string;
        semitones: number;
    };
    audio: {
        path: string;
        format: "wav";
    };
}

export interface PitchShiftError {
    version: string;
    error: {
        code: string;
        message: string;
    };
}

export type PitchShiftResult = PitchShiftSuccess | PitchShiftError;
