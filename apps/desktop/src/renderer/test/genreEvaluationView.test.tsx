import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { App } from "../App";

afterEach(cleanup);

it("runs genre evaluation from the desktop view without analyzing the active song", async () => {
    const analyzeAudio = vi.fn();
    const evaluateGenreCorpus = vi.fn().mockResolvedValue({
        version: "1",
        manifestPath: "/tmp/genre-manifest.json",
        itemCount: 2,
        evaluatedItemCount: 2,
        failedItemCount: 0,
        status: "pass",
        metrics: {
            itemCount: 2,
            evaluatedDuration: 20,
            timeWeightedChordAccuracy: 75,
            exactChordMatchPercentage: 70,
            rootAccuracy: 85,
            qualityAccuracy: 65,
            falseTransitionCount: 1,
            missedTransitionCount: 2,
            groundTruthSegmentCount: 8,
            predictedSegmentCount: 9,
            boundaryTimingErrorSeconds: 0.12,
        },
        genres: {
            pop: {
                itemCount: 1,
                evaluatedDuration: 10,
                timeWeightedChordAccuracy: 80,
                exactChordMatchPercentage: 80,
                rootAccuracy: 90,
                qualityAccuracy: 70,
                falseTransitionCount: 0,
                missedTransitionCount: 1,
                groundTruthSegmentCount: 4,
                predictedSegmentCount: 4,
                boundaryTimingErrorSeconds: 0.1,
            },
            reggae: {
                itemCount: 1,
                evaluatedDuration: 10,
                timeWeightedChordAccuracy: 70,
                exactChordMatchPercentage: 60,
                rootAccuracy: 80,
                qualityAccuracy: 60,
                falseTransitionCount: 1,
                missedTransitionCount: 1,
                groundTruthSegmentCount: 4,
                predictedSegmentCount: 5,
                boundaryTimingErrorSeconds: 0.14,
            },
        },
        items: [
            {
                id: "pop-one",
                genre: "pop",
                audioPath: "/tmp/pop.wav",
                annotationPath: "/tmp/pop.json",
                status: "pass",
                metrics: {
                    itemCount: 1,
                    evaluatedDuration: 10,
                    timeWeightedChordAccuracy: 80,
                    rootAccuracy: 90,
                    qualityAccuracy: 70,
                    falseTransitionCount: 0,
                    missedTransitionCount: 1,
                    boundaryTimingErrorSeconds: 0.1,
                    confusionPairs: [
                        { pair: "G->Em", duration: 2.4, percentageOfMismatchedTime: 30 },
                    ],
                },
            },
        ],
    });
    window.gcd = {
        getAppVersion: vi.fn().mockResolvedValue("test"),
        selectAudioFile: vi.fn(),
        analyzeAudio,
        selectGenreEvaluationManifest: vi.fn().mockResolvedValue({
            canceled: false,
            path: "/tmp/genre-manifest.json",
            fileName: "genre-manifest.json",
        }),
        evaluateGenreCorpus,
    };

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Evaluate" }));
    expect(screen.getByRole("heading", { name: "Genre Evaluation" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Open Manifest" }));
    await screen.findByText("Manifest ready.");

    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await screen.findByText("Evaluation complete: 2/2 item(s), 0 failed.");
    expect(screen.getByText("reggae")).toBeInTheDocument();
    expect(screen.getByText("pop-one")).toBeInTheDocument();
    expect(screen.getByText("G->Em")).toBeInTheDocument();
    expect(evaluateGenreCorpus).toHaveBeenCalledWith("/tmp/genre-manifest.json");
    await waitFor(() => expect(analyzeAudio).not.toHaveBeenCalled());
});
