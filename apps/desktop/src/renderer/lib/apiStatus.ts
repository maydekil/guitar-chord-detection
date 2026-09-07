import type { ApiHealthState, ApiJobProgressEvent } from "../appTypes.js";

export function formatApiHealthLabel(state: ApiHealthState): string {
    if (state === "connected") {
        return "API Connected";
    }
    if (state === "offline") {
        return "API Offline";
    }
    if (state === "checking") {
        return "Checking API";
    }
    return "Local Mode";
}

export function formatApiJobKind(kind: ApiJobProgressEvent["kind"]): string {
    if (kind === "analysis") {
        return "Analyzing";
    }
    if (kind === "lyrics") {
        return "Lyrics";
    }
    if (kind === "vocals") {
        return "Vocal Off";
    }
    return "Transpose";
}
