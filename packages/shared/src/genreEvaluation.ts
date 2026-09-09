export interface GenreEvaluationErrorPayload {
    code: string;
    message: string;
}

export interface GenreEvaluationConfusionPair {
    pair: string;
    duration: number;
    percentageOfMismatchedTime: number;
}

export interface GenreEvaluationMetrics {
    itemCount: number;
    evaluatedDuration: number;
    timeWeightedChordAccuracy: number;
    exactChordMatchPercentage?: number;
    rootAccuracy: number;
    qualityAccuracy: number;
    falseTransitionCount: number;
    missedTransitionCount: number;
    groundTruthSegmentCount?: number;
    predictedSegmentCount?: number;
    boundaryTimingErrorSeconds: number | null;
    confusionPairs?: GenreEvaluationConfusionPair[];
}

export interface GenreEvaluationItemSuccess {
    id: string;
    genre: string;
    audioPath: string;
    annotationPath: string;
    status: "pass";
    clipStart?: number;
    clipEnd?: number;
    metrics: GenreEvaluationMetrics;
}

export interface GenreEvaluationItemError {
    id: string;
    genre: string;
    audioPath: string;
    annotationPath: string;
    status: "error";
    error: GenreEvaluationErrorPayload;
}

export type GenreEvaluationItem = GenreEvaluationItemSuccess | GenreEvaluationItemError;

export interface GenreEvaluationSuccess {
    version: string;
    manifestPath: string;
    itemCount: number;
    evaluatedItemCount: number;
    failedItemCount: number;
    status: "pass" | "fail";
    metrics: GenreEvaluationMetrics;
    genres: Record<string, GenreEvaluationMetrics>;
    items: GenreEvaluationItem[];
}

export interface GenreEvaluationFailure {
    version: string;
    error: GenreEvaluationErrorPayload;
}

export type GenreEvaluationResult = GenreEvaluationSuccess | GenreEvaluationFailure;
