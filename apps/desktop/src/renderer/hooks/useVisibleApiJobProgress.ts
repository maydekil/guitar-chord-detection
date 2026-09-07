import { useEffect, useState } from "react";

import type { ApiJobProgressEvent } from "../appTypes.js";

export function useVisibleApiJobProgress(apiJobProgress: ApiJobProgressEvent | null): ApiJobProgressEvent | null {
    const [visibleProgress, setVisibleProgress] = useState<ApiJobProgressEvent | null>(apiJobProgress);

    useEffect(() => {
        if (!apiJobProgress) {
            setVisibleProgress(null);
            return;
        }

        setVisibleProgress((current) => {
            if (!current || current.id !== apiJobProgress.id) {
                return apiJobProgress;
            }
            return {
                ...apiJobProgress,
                progress: Math.max(current.progress, apiJobProgress.progress)
            };
        });
    }, [apiJobProgress]);

    useEffect(() => {
        if (!visibleProgress || (visibleProgress.status !== "queued" && visibleProgress.status !== "running")) {
            return;
        }

        const timerId = window.setInterval(() => {
            setVisibleProgress((current) => {
                if (!current || current.id !== visibleProgress.id || (current.status !== "queued" && current.status !== "running")) {
                    return current;
                }
                return {
                    ...current,
                    progress: Math.min(96, current.progress + getVisibleProgressStep(current.kind, current.progress))
                };
            });
        }, 850);

        return () => window.clearInterval(timerId);
    }, [visibleProgress?.id, visibleProgress?.status]);

    return visibleProgress;
}

function getVisibleProgressStep(kind: ApiJobProgressEvent["kind"], progress: number): number {
    if (progress >= 90) {
        return 0.25;
    }
    if (kind === "vocals") {
        return progress < 55 ? 1.25 : 0.7;
    }
    if (kind === "lyrics") {
        return progress < 70 ? 1.5 : 0.8;
    }
    if (kind === "pitch-shift") {
        return progress < 75 ? 2 : 1;
    }
    return progress < 70 ? 1 : 0.55;
}
