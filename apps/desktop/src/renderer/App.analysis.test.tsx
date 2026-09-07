import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { activeChords, openAnalyzerForm, setMediaTiming } from "./testUtils";

afterEach(() => {
    cleanup();
});

describe("App", () => {
    it("renders without crashing when preload bridge is unavailable", () => {
        delete window.gcd;

        render(<App />);

        expect(screen.getByRole("heading", { name: "Guitar Chord Detector" })).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Add Song" })).toBeDisabled();
        expect(screen.queryByRole("button", { name: "Open Audio" })).not.toBeInTheDocument();
        expect(screen.getByText("Desktop Shell vbridge-unavailable")).toBeInTheDocument();
    });

    it("renders desktop shell header and idle state", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: true,
                path: null,
                fileName: null
            }),
            analyzeAudio: vi.fn()
        };

        render(<App />);

        expect(screen.getByRole("heading", { name: "Guitar Chord Detector" })).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Add Song" })).toBeEnabled();
        expect(screen.queryByRole("button", { name: "Open Audio" })).not.toBeInTheDocument();
        await openAnalyzerForm();
        expect(screen.getByText("State: idle")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Open Audio" })).toBeEnabled();
        expect(screen.getByText("No file selected")).toBeInTheDocument();
    });

    it("shows selected filename after bridge returns selection", async () => {
        const analysisResult = {
            version: "1",
            source: {
                path: "/tmp/demo.wav",
                duration: 1,
                sampleRate: 22050
            },
            analysis: {
                algorithm: "chroma-template-v1",
                chords: []
            }
        } as const;
        const analyzeAudio = vi.fn().mockResolvedValue({
            ...analysisResult
        });
        const saveSongAnalysis = vi.fn().mockResolvedValue({
            id: "saved-demo",
            title: "Demo Title",
            artist: "Demo Artist",
            audioPath: "/tmp/demo.wav",
            fileHash: "hash",
            algorithm: "chroma-template-v3",
            contractVersion: "1",
            duration: 1,
            analysis: analysisResult,
            createdAt: "2026-09-06T00:00:00.000Z",
            updatedAt: "2026-09-06T00:00:00.000Z"
        });
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.wav",
                fileName: "demo.wav"
            }),
            analyzeAudio,
            saveSongAnalysis
        };

        render(<App />);
        await openAnalyzerForm();
        expect(screen.queryByPlaceholderText("Contoh: SR Banyak Cerita")).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));

        expect(await screen.findByText("demo.wav")).toBeInTheDocument();
        expect(analyzeAudio).toHaveBeenCalledWith("/tmp/demo.wav", { forceRefresh: false });
        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(await screen.findByText("Analysis ready: 0 segment(s)")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Save" }));
        fireEvent.change(await screen.findByPlaceholderText("Contoh: SR Banyak Cerita"), {
            target: { value: "Demo Artist" }
        });
        fireEvent.change(screen.getByPlaceholderText("Contoh: Album Lama"), {
            target: { value: "Demo Title" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Save to Library" }));
        expect(await screen.findByText("Save chord analysis?")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Confirm Save" }));
        expect(saveSongAnalysis).toHaveBeenCalledWith({
            audioPath: "/tmp/demo.wav",
            analysis: analysisResult,
            lyrics: "",
            metadata: {
                artist: "Demo Artist",
                title: "Demo Title"
            }
        });
        expect(await screen.findByText("Saved to Song Library.")).toBeInTheDocument();
    });

    it("shows Re-analyze only when a valid selected file exists", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi
                .fn()
                .mockResolvedValueOnce({
                    canceled: true,
                    path: null,
                    fileName: null
                })
                .mockResolvedValueOnce({
                    canceled: false,
                    path: "/tmp/reanalyze.mp3",
                    fileName: "reanalyze.mp3"
                }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 1,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: []
                }
            })
        };

        render(<App />);
        expect(screen.queryByRole("button", { name: "Re-analyze" })).not.toBeInTheDocument();
        await openAnalyzerForm();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: idle")).toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Re-analyze" })).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Re-analyze" })).toBeEnabled();
    });

    it("re-analyze bypasses cache request path, reruns analysis, and clears old result immediately", async () => {
        let resolveRefresh: ((value: unknown) => void) | undefined;
        const analyzeAudio = vi.fn().mockImplementation((_audioPath: string, options?: { forceRefresh?: boolean }) => {
            if (options?.forceRefresh) {
                return new Promise<unknown>((resolve) => {
                    resolveRefresh = resolve;
                });
            }

            return Promise.resolve({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 4,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: [{ start: 0, end: 1, chord: "C", confidence: 0.8 }]
                }
            });
        });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/reanalyze.mp3",
                fileName: "reanalyze.mp3"
            }),
            analyzeAudio
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("Analysis ready: 1 segment(s)")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Re-analyze" }));
        expect(await screen.findByText("State: analyzing")).toBeInTheDocument();
        expect(screen.getByText("Analyzing...")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();

        expect(analyzeAudio).toHaveBeenNthCalledWith(1, "/tmp/reanalyze.mp3", { forceRefresh: false });
        expect(analyzeAudio).toHaveBeenNthCalledWith(2, "/tmp/reanalyze.mp3", { forceRefresh: true });

        if (resolveRefresh) {
            resolveRefresh({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 4,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: [
                        { start: 0, end: 2, chord: "Am", confidence: 0.85 },
                        { start: 2, end: 4, chord: "F", confidence: 0.82 }
                    ]
                }
            });
        }

        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(await screen.findByText("Analysis ready: 2 segment(s)")).toBeInTheDocument();
    });

    it("ignores stale older re-analysis response when a newer re-analysis finishes later", async () => {
        let resolveFirstRefresh: ((value: unknown) => void) | undefined;
        let resolveSecondRefresh: ((value: unknown) => void) | undefined;

        const analyzeAudio = vi
            .fn()
            .mockResolvedValueOnce({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 4,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: [{ start: 0, end: 4, chord: "C", confidence: 0.9 }]
                }
            })
            .mockImplementationOnce(() => {
                return new Promise<unknown>((resolve) => {
                    resolveFirstRefresh = resolve;
                });
            })
            .mockImplementationOnce(() => {
                return new Promise<unknown>((resolve) => {
                    resolveSecondRefresh = resolve;
                });
            });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/reanalyze.mp3",
                fileName: "reanalyze.mp3"
            }),
            analyzeAudio
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("Analysis ready: 1 segment(s)")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Re-analyze" }));
        fireEvent.click(screen.getByRole("button", { name: "Re-analyze" }));

        if (resolveSecondRefresh) {
            resolveSecondRefresh({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 4,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: [
                        { start: 0, end: 2, chord: "Am", confidence: 0.8 },
                        { start: 2, end: 4, chord: "F", confidence: 0.8 }
                    ]
                }
            });
        }

        expect(await screen.findByText("Analysis ready: 2 segment(s)")).toBeInTheDocument();

        if (resolveFirstRefresh) {
            resolveFirstRefresh({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 4,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: [{ start: 0, end: 4, chord: "G", confidence: 0.8 }]
                }
            });
        }

        expect(await screen.findByText("Analysis ready: 2 segment(s)")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();
    });

    it("keeps stale analysis cleared when re-analysis fails", async () => {
        let resolveRefresh: ((value: unknown) => void) | undefined;

        const analyzeAudio = vi.fn().mockImplementation((_audioPath: string, options?: { forceRefresh?: boolean }) => {
            if (options?.forceRefresh) {
                return new Promise<unknown>((resolve) => {
                    resolveRefresh = resolve;
                });
            }

            return Promise.resolve({
                version: "1",
                source: {
                    path: "/tmp/reanalyze.mp3",
                    duration: 4,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v2",
                    chords: [{ start: 0, end: 1, chord: "C", confidence: 0.8 }]
                }
            });
        });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/reanalyze.mp3",
                fileName: "reanalyze.mp3"
            }),
            analyzeAudio
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("Analysis ready: 1 segment(s)")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Re-analyze" }));
        expect(await screen.findByText("State: analyzing")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();

        if (resolveRefresh) {
            resolveRefresh({
                version: "1",
                error: {
                    code: "AUDIO_DECODE_FAILED",
                    message: "Unable to decode audio file"
                }
            });
        }

        expect(await screen.findByText("State: error")).toBeInTheDocument();
        expect(await screen.findByText("Could not analyze this audio file.")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();
    });

    it("converts selected local path into a renderer-usable file URL", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/Album #1?.mp3",
                fileName: "Album #1?.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/Album #1?.mp3",
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
        expect(audio.src).toContain("file:///tmp/Album%20%231%3F.mp3");
    });

    it("uses preload playback source bytes to build a blob URL", async () => {
        const originalCreateObjectUrl = URL.createObjectURL;
        const createObjectUrlSpy = vi.fn().mockReturnValue("blob:test-audio");
        Object.defineProperty(URL, "createObjectURL", {
            value: createObjectUrlSpy,
            configurable: true,
            writable: true
        });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/blob.mp3",
                fileName: "blob.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/blob.mp3",
                    duration: 6,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            }),
            getAudioPlaybackSource: vi.fn().mockResolvedValue({
                mimeType: "audio/mpeg",
                bytes: new Uint8Array([1, 2, 3, 4])
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const audio = screen.getByTestId("audio-player") as HTMLAudioElement;
        expect(window.gcd.getAudioPlaybackSource).toHaveBeenCalledWith("/tmp/blob.mp3");
        expect(createObjectUrlSpy).toHaveBeenCalled();
        expect(audio.src).toContain("blob:test-audio");

        Object.defineProperty(URL, "createObjectURL", {
            value: originalCreateObjectUrl,
            configurable: true,
            writable: true
        });
    });

    it("revokes previous blob URL when selecting a different file", async () => {
        const originalCreateObjectUrl = URL.createObjectURL;
        const originalRevokeObjectUrl = URL.revokeObjectURL;
        const createObjectUrlSpy = vi.fn()
            .mockReturnValueOnce("blob:first")
            .mockReturnValueOnce("blob:second");
        const revokeObjectUrlSpy = vi.fn();
        Object.defineProperty(URL, "createObjectURL", {
            value: createObjectUrlSpy,
            configurable: true,
            writable: true
        });
        Object.defineProperty(URL, "revokeObjectURL", {
            value: revokeObjectUrlSpy,
            configurable: true,
            writable: true
        });

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
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/first.mp3",
                    duration: 6,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: []
                }
            }),
            getAudioPlaybackSource: vi
                .fn()
                .mockResolvedValue({
                    mimeType: "audio/mpeg",
                    bytes: new Uint8Array([1, 2, 3])
                })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("second.mp3")).toBeInTheDocument();

        expect(createObjectUrlSpy).toHaveBeenCalledTimes(2);
        expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:first");

        Object.defineProperty(URL, "createObjectURL", {
            value: originalCreateObjectUrl,
            configurable: true,
            writable: true
        });
        Object.defineProperty(URL, "revokeObjectURL", {
            value: originalRevokeObjectUrl,
            configurable: true,
            writable: true
        });
    });

    it("keeps filename unchanged on dialog cancel", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: true,
                path: null,
                fileName: null
            }),
            analyzeAudio: vi.fn()
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));

        expect(await screen.findByText("No file selected")).toBeInTheDocument();
        expect(await screen.findByText("State: idle")).toBeInTheDocument();
    });

    it("clears previous analysis result immediately when a different file is selected", async () => {
        let resolveSecond: ((value: unknown) => void) | undefined;
        const analyzeAudio = vi.fn().mockImplementation((audioPath: string) => {
            if (audioPath === "/tmp/first.mp3") {
                return Promise.resolve({
                    version: "1",
                    source: {
                        path: audioPath,
                        duration: 3,
                        sampleRate: 22050
                    },
                    analysis: {
                        algorithm: "chroma-template-v1",
                        chords: [{ start: 0, end: 1, chord: "C", confidence: 0.9 }]
                    }
                });
            }

            return new Promise<unknown>((resolve) => {
                resolveSecond = resolve;
            });
        });

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
            analyzeAudio
        };

        render(<App />);
        await openAnalyzerForm();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("Analysis ready: 1 segment(s)")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: analyzing")).toBeInTheDocument();
        expect(await screen.findByText("Analyzing...")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();

        if (resolveSecond) {
            resolveSecond({
                version: "1",
                error: {
                    code: "AUDIO_DECODE_FAILED",
                    message: "Unable to decode audio file"
                }
            });
        }

        expect(await screen.findByText("State: error")).toBeInTheDocument();
        expect(await screen.findByText("Could not analyze this audio file.")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();
    });

    it("ignores stale older analysis response when a newer selection is active", async () => {
        let resolveFirst: ((value: unknown) => void) | undefined;
        let resolveSecond: ((value: unknown) => void) | undefined;

        const analyzeAudio = vi.fn().mockImplementation((audioPath: string) => {
            return new Promise<unknown>((resolve) => {
                if (audioPath === "/tmp/first.mp3") {
                    resolveFirst = resolve;
                    return;
                }
                resolveSecond = resolve;
            });
        });

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
            analyzeAudio
        };

        render(<App />);
        await openAnalyzerForm();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: analyzing")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("second.mp3")).toBeInTheDocument();

        if (resolveSecond) {
            resolveSecond({
                version: "1",
                source: {
                    path: "/tmp/second.mp3",
                    duration: 2,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [
                        { start: 0, end: 1, chord: "Am", confidence: 0.8 },
                        { start: 1, end: 2, chord: "F", confidence: 0.85 }
                    ]
                }
            });
        }

        expect(await screen.findByText("State: ready")).toBeInTheDocument();
        expect(await screen.findByText("Analysis ready: 2 segment(s)")).toBeInTheDocument();

        if (resolveFirst) {
            resolveFirst({
                version: "1",
                source: {
                    path: "/tmp/first.mp3",
                    duration: 1,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 0, end: 1, chord: "C", confidence: 0.9 }]
                }
            });
        }

        expect(await screen.findByText("Analysis ready: 2 segment(s)")).toBeInTheDocument();
        expect(screen.queryByText("Analysis ready: 1 segment(s)")).not.toBeInTheDocument();
    });

    it("shows analyzing state during pending analysis and error state on engine failure", async () => {
        let resolveAnalyze: ((value: unknown) => void) | undefined;
        const analyzeAudio = vi.fn().mockImplementation(() => {
            return new Promise<unknown>((resolve) => {
                resolveAnalyze = resolve;
            });
        });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/demo.mp3",
                fileName: "demo.mp3"
            }),
            analyzeAudio
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));

        expect(await screen.findByText("State: analyzing")).toBeInTheDocument();

        if (resolveAnalyze) {
            resolveAnalyze({
                version: "1",
                error: {
                    code: "AUDIO_DECODE_FAILED",
                    message: "Unable to decode audio file"
                }
            });
        }

        expect(await screen.findByText("State: error")).toBeInTheDocument();
        expect(await screen.findByText("Could not analyze this audio file.")).toBeInTheDocument();
    });

});
