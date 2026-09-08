import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { App } from "../App";
import { openAnalyzerForm } from "../testUtils";

afterEach(cleanup);

it("keeps timeline boundaries and lyric chords consistent through lyric edits and transpose", async () => {
    window.gcd = {
        getAppVersion: vi.fn().mockResolvedValue("test"),
        selectAudioFile: vi.fn().mockResolvedValue({ canceled: false, path: "/tmp/source.wav", fileName: "source.wav" }),
        analyzeAudio: vi.fn().mockResolvedValue({
            version: "1",
            source: { path: "/tmp/source.wav", duration: 12, sampleRate: 22050 },
            analysis: {
                algorithm: "test",
                chords: [
                    { start: 0, end: 3.5, chord: "C", confidence: 0.8 },
                    { start: 3.5, end: 3.75, chord: "D#", confidence: 0.7 },
                    { start: 3.75, end: 12, chord: "Am", confidence: 0.9 },
                ],
            },
        }),
    };
    const { container } = render(<App />);
    await openAnalyzerForm();
    fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
    await screen.findByText("State: ready");
    const boundaries = () => screen.getAllByTestId("timeline-segment").map((segment) => [
        segment.getAttribute("data-fragment-start"), segment.getAttribute("data-fragment-end"),
    ]);
    const original = boundaries();
    fireEvent.click(screen.getByRole("tab", { name: "Lyrics" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Song lyrics" }), {
        target: { value: "[00:00]first lyric\n[00:04]last lyric" },
    });
    const labels = () => Array.from(container.querySelectorAll(".lyrics-preview-chord"), (element) => element.textContent);
    expect(labels()).toEqual(["C", "D#", "Am", "Am"]);
    fireEvent.change(screen.getByRole("textbox", { name: "Song lyrics" }), {
        target: { value: "[00:00.5]new words\n[00:08]another line" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Transpose up" }));
    expect(labels()).toEqual(["C#", "E", "A#m", "A#m"]);
    fireEvent.click(screen.getByRole("tab", { name: "Timeline" }));
    expect(boundaries()).toEqual(original);
    const segments = screen.getAllByTestId("timeline-segment");
    expect(segments[0]).toHaveTextContent("C#");
    expect(segments[2]).toHaveTextContent("A#m");
});
