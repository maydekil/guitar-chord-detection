import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { activeChords, openAnalyzerForm, setMediaTiming } from "./testUtils";

afterEach(() => {
    cleanup();
});

describe("App", () => {
    it("supports play pause and time updates after analysis success", async () => {
        const playSpy = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
        const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => { });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.mp3",
                fileName: "demo.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/demo.mp3",
                    duration: 12,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 0, end: 1, chord: "C", confidence: 0.9 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();
        expect(screen.getByRole("slider", { name: "Seek" })).toBeDisabled();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 12, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        expect(screen.getByText("00:00 / 00:12")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();
        expect(screen.getByRole("slider", { name: "Seek" })).toBeEnabled();

        fireEvent.click(screen.getByRole("button", { name: "Play" }));
        fireEvent.play(audio);
        expect(await screen.findByText("State: playing")).toBeInTheDocument();
        expect(playSpy).toHaveBeenCalled();

        fireEvent.click(screen.getByRole("button", { name: "Pause" }));
        fireEvent.pause(audio);
        expect(await screen.findByText("State: paused")).toBeInTheDocument();
        expect(pauseSpy).toHaveBeenCalled();

        playSpy.mockRestore();
        pauseSpy.mockRestore();
    });

    it("invokes play on active audio element and switches to playing only after play event", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.mp3",
                fileName: "demo.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/demo.mp3",
                    duration: 8,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        const activePlay = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(audio, "play", { value: activePlay, configurable: true });

        setMediaTiming(audio, { duration: 8, currentTime: 0 });
        fireEvent.loadedMetadata(audio);
        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();

        fireEvent.click(screen.getByRole("button", { name: "Play" }));
        expect(activePlay).toHaveBeenCalledTimes(1);
        expect(screen.getByText("State: ready")).toBeInTheDocument();

        fireEvent.play(audio);
        expect(await screen.findByText("State: playing")).toBeInTheDocument();
    });

    it("keeps ready state when play promise rejects", async () => {
        const playSpy = vi.spyOn(HTMLMediaElement.prototype, "play").mockRejectedValue(new Error("play failed"));

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.mp3",
                fileName: "demo.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/demo.mp3",
                    duration: 12,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 12, currentTime: 0 });
        fireEvent.loadedMetadata(audio);
        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();

        fireEvent.click(screen.getByRole("button", { name: "Play" }));

        expect(playSpy).toHaveBeenCalled();
        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(screen.queryByText("State: error")).not.toBeInTheDocument();
        expect(screen.getByText("Could not play this audio file. (play failed)")).toBeInTheDocument();

        playSpy.mockRestore();
    });

    it("keeps playback available after analysis reaches ready state", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/ready.mp3",
                fileName: "ready.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/ready.mp3",
                    duration: 9,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));

        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(screen.getByText("00:00 / 00:09")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 9, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();
        expect(screen.getByRole("slider", { name: "Seek" })).toBeEnabled();
    });

    it("resets previous playback when changing to a different file", async () => {
        const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => { });

        let resolveSecond: ((value: unknown) => void) | undefined;
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi
                .fn()
                .mockResolvedValueOnce({
                    canceled: false,
                    path: "/tmp/first.mp3",
                    fileName: "first.mp3"
                })
                .mockResolvedValueOnce({
                    canceled: false,
                    path: "/tmp/second.mp3",
                    fileName: "second.mp3"
                }),
            analyzeAudio: vi.fn().mockImplementation((audioPath: string) => {
                if (audioPath === "/tmp/first.mp3") {
                    return Promise.resolve({
                        version: "1",
                        source: {
                            path: "/tmp/first.mp3",
                            duration: 10,
                            sampleRate: 22050
                        },
                        analysis: {
                            algorithm: "chroma-template-v1",
                            chords: []
                        }
                    });
                }

                return new Promise<unknown>((resolve) => {
                    resolveSecond = resolve;
                });
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 10, currentTime: 0 });
        fireEvent.loadedMetadata(audio);
        setMediaTiming(audio, { duration: 10, currentTime: 4 });
        fireEvent.play(audio);
        fireEvent.timeUpdate(audio);
        expect(await screen.findByText("State: playing")).toBeInTheDocument();
        expect(screen.getByText("00:04 / 00:10")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: analyzing")).toBeInTheDocument();
        expect(screen.getByText("00:00 / 00:00")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Play" })).toBeDisabled();
        expect(pauseSpy).toHaveBeenCalled();

        if (resolveSecond) {
            resolveSecond({
                version: "1",
                source: {
                    path: "/tmp/second.mp3",
                    duration: 8,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            });
        }

        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(screen.getByText("00:00 / 00:08")).toBeInTheDocument();
        pauseSpy.mockRestore();
    });

    it("keeps playback controls disabled for invalid audio analysis", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/bad.mp3",
                fileName: "bad.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                error: {
                    code: "AUDIO_DECODE_FAILED",
                    message: "Unable to decode audio file"
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));

        expect(await screen.findByText("State: error")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Play" })).toBeDisabled();
        expect(screen.getByRole("slider", { name: "Seek" })).toBeDisabled();
    });

    it("disables playback controls when audio element emits an error", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.mp3",
                fileName: "demo.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/demo.mp3",
                    duration: 20,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 20, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();
        fireEvent.error(audio);

        expect(await screen.findByText("State: error")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Play" })).toBeDisabled();
        expect(screen.getByRole("slider", { name: "Seek" })).toBeDisabled();
    });

    it("supports seek and updates current playback position", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.mp3",
                fileName: "demo.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/demo.mp3",
                    duration: 14,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 14, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        const slider = screen.getByRole("slider", { name: "Seek" });
        fireEvent.change(slider, { target: { value: "5" } });
        expect(audio.currentTime).toBe(5);
        expect(screen.getByText("00:05 / 00:14")).toBeInTheDocument();

        setMediaTiming(audio, { duration: 14, currentTime: 7 });
        fireEvent.timeUpdate(audio);
        expect(screen.getByText("00:07 / 00:14")).toBeInTheDocument();
    });

    it("seeks audio when timeline rows and segments are clicked", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/timeline-seek.mp3",
                fileName: "timeline-seek.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/timeline-seek.mp3",
                    duration: 30,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [
                        { start: 0, end: 15, chord: "C", confidence: 0.9 },
                        { start: 15, end: 30, chord: "G", confidence: 0.85 }
                    ]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 30, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        const tracks = screen.getAllByTestId("timeline-track");
        Object.defineProperty(tracks[0], "getBoundingClientRect", {
            value: () => ({ left: 100, width: 300 }),
            configurable: true
        });
        fireEvent.click(tracks[0], { clientX: 250 });
        expect(audio.currentTime).toBeCloseTo(7.5, 2);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: C");

        const secondRow = screen.getAllByTestId("timeline-row")[1];
        const secondRowSegment = within(secondRow).getByTestId("timeline-segment");
        Object.defineProperty(secondRowSegment, "getBoundingClientRect", {
            value: () => ({ left: 40, width: 200 }),
            configurable: true
        });
        fireEvent.click(secondRowSegment, { clientX: 140 });
        expect(audio.currentTime).toBeCloseTo(22.5, 2);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: G");
    });

    it("marks active chord with deterministic boundary rule during playback updates", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/active-boundary.mp3",
                fileName: "active-boundary.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/active-boundary.mp3",
                    duration: 12,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [
                        { start: 0, end: 5, chord: "C", confidence: 0.9 },
                        { start: 5, end: 10, chord: "G", confidence: 0.85 },
                        { start: 10, end: 12, chord: "N", confidence: 0.95 }
                    ]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 12, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        expect(activeChords()).toEqual(["C"]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: C");

        setMediaTiming(audio, { duration: 12, currentTime: 4.99 });
        fireEvent.timeUpdate(audio);
        expect(activeChords()).toEqual(["C"]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: C");

        setMediaTiming(audio, { duration: 12, currentTime: 5 });
        fireEvent.timeUpdate(audio);
        expect(activeChords()).toEqual(["G"]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: G");

        setMediaTiming(audio, { duration: 12, currentTime: 10 });
        fireEvent.timeUpdate(audio);
        expect(activeChords()).toEqual(["N"]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: N");

        setMediaTiming(audio, { duration: 12, currentTime: 12 });
        fireEvent.timeUpdate(audio);
        expect(activeChords()).toEqual([]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: None");
    });

    it("updates active chord immediately when seek changes current time", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/seek-active.mp3",
                fileName: "seek-active.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/seek-active.mp3",
                    duration: 12,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [
                        { start: 0, end: 5, chord: "C", confidence: 0.9 },
                        { start: 5, end: 10, chord: "G", confidence: 0.85 },
                        { start: 10, end: 12, chord: "N", confidence: 0.95 }
                    ]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        setMediaTiming(audio, { duration: 12, currentTime: 0 });
        fireEvent.loadedMetadata(audio);

        expect(activeChords()).toEqual(["C"]);

        const slider = screen.getByRole("slider", { name: "Seek" });
        fireEvent.change(slider, { target: { value: "10" } });
        expect(activeChords()).toEqual(["N"]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: N");

        fireEvent.change(slider, { target: { value: "5" } });
        expect(activeChords()).toEqual(["G"]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: G");

        fireEvent.change(slider, { target: { value: "12" } });
        expect(activeChords()).toEqual([]);
        expect(screen.getByTestId("active-chord")).toHaveTextContent("Active chord: None");
    });

    it("keeps the same audio element across analysis and playback state changes", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/stable.mp3",
                fileName: "stable.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/stable.mp3",
                    duration: 11,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            })
        };

        const playSpy = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
        const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => { });

        render(<App />);
        await openAnalyzerForm();
        const initialAudio = screen.getByTestId("audio-player") as HTMLAudioElement;

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const readyAudio = screen.getByTestId("audio-player") as HTMLAudioElement;
        expect(readyAudio).toBe(initialAudio);

        setMediaTiming(readyAudio, { duration: 11, currentTime: 0 });
        fireEvent.loadedMetadata(readyAudio);
        expect(screen.getByRole("button", { name: "Play" })).toBeEnabled();

        fireEvent.click(screen.getByRole("button", { name: "Play" }));
        fireEvent.play(readyAudio);
        expect(await screen.findByText("State: playing")).toBeInTheDocument();

        const playingAudio = screen.getByTestId("audio-player") as HTMLAudioElement;
        expect(playingAudio).toBe(initialAudio);

        fireEvent.click(screen.getByRole("button", { name: "Pause" }));
        fireEvent.pause(playingAudio);
        expect(await screen.findByText("State: paused")).toBeInTheDocument();

        const pausedAudio = screen.getByTestId("audio-player") as HTMLAudioElement;
        expect(pausedAudio).toBe(initialAudio);

        playSpy.mockRestore();
        pauseSpy.mockRestore();
    });
});
