import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
    cleanup();
});

function setMediaTiming(audio: HTMLAudioElement, values: { duration: number; currentTime: number }): void {
    const { duration, currentTime } = values;
    Object.defineProperty(audio, "duration", { value: duration, configurable: true });
    Object.defineProperty(audio, "currentTime", { value: currentTime, writable: true, configurable: true });
}

function activeChords(): string[] {
    return screen
        .getAllByTestId("timeline-segment")
        .filter((segment) => segment.getAttribute("data-active") === "true")
        .map((segment) => segment.getAttribute("data-chord") ?? "");
}

async function openAnalyzerForm(): Promise<void> {
    fireEvent.click(await screen.findByRole("button", { name: "Add Song" }));
}

describe("App", () => {
    it("renders searchable analyzed song library records", async () => {
        const listSongs = vi.fn().mockResolvedValue([
            {
                id: "song-1",
                title: "Album Lama",
                artist: "SR Banyak Cerita",
                audioPath: "/tmp/Album Lama.mp3",
                fileHash: "hash",
                algorithm: "chroma-template-v3",
                contractVersion: "1",
                duration: 282,
                analysis: {
                    version: "1",
                    source: {
                        path: "/tmp/Album Lama.mp3",
                        duration: 282,
                        sampleRate: 22050
                    },
                    analysis: {
                        algorithm: "chroma-template-v3",
                        chords: [{ start: 0, end: 282, chord: "A", confidence: 0.9 }]
                    }
                },
                createdAt: "2026-09-06T00:00:00.000Z",
                updatedAt: "2026-09-06T00:00:00.000Z"
            }
        ]);
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn(),
            analyzeAudio: vi.fn(),
            listSongs
        };

        render(<App />);

        expect(await screen.findByText("Album Lama")).toBeInTheDocument();
        expect(screen.getByText("SR Banyak Cerita")).toBeInTheDocument();
        expect(screen.getByText("1 chords - 04:42")).toBeInTheDocument();

        fireEvent.change(screen.getByRole("searchbox", { name: "Search library" }), { target: { value: "sr" } });
        expect(await screen.findByDisplayValue("sr")).toBeInTheDocument();
        await waitFor(() => expect(listSongs).toHaveBeenLastCalledWith({ query: "sr", page: 1, pageSize: 10 }));
    });

    it("keeps playback controls disabled when no audio is selected", async () => {
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

        expect(screen.getByRole("button", { name: "Play" })).toBeDisabled();
        expect(screen.getByRole("slider", { name: "Seek" })).toBeDisabled();
        expect(screen.getByText("No chord segments")).toBeInTheDocument();
    });

    it("renders timeline segments in start-time order with expected left and width percentages", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/timeline.mp3",
                fileName: "timeline.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/timeline.mp3",
                    duration: 10,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [
                        { start: 5, end: 10, chord: "G", confidence: 0.8 },
                        { start: 0, end: 5, chord: "C", confidence: 0.9 }
                    ]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const segments = screen.getAllByTestId("timeline-segment");
        expect(segments).toHaveLength(2);
        expect(segments[0]).toHaveTextContent("C");
        expect(segments[1]).toHaveTextContent("G");
        expect(segments[0]).toHaveStyle({ left: "0%", width: "50%" });
        expect(segments[1]).toHaveStyle({ left: "50%", width: "50%" });
    });

    it("splits timeline into deterministic 15-second rows and keeps final row partial", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/rows.wav",
                fileName: "rows.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/rows.wav",
                    duration: 40,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 0, end: 40, chord: "C", confidence: 0.9 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const rows = screen.getAllByTestId("timeline-row");
        expect(rows).toHaveLength(3);
        expect(rows[0]).toHaveAttribute("data-row-start", "0");
        expect(rows[0]).toHaveAttribute("data-row-end", "15");
        expect(rows[1]).toHaveAttribute("data-row-start", "15");
        expect(rows[1]).toHaveAttribute("data-row-end", "30");
        expect(rows[2]).toHaveAttribute("data-row-start", "30");
        expect(rows[2]).toHaveAttribute("data-row-end", "40");

        const ranges = screen.getAllByTestId("timeline-row-range");
        expect(ranges[0]).toHaveTextContent("00:00 - 00:15");
        expect(ranges[1]).toHaveTextContent("00:15 - 00:30");
        expect(ranges[2]).toHaveTextContent("00:30 - 00:40");
    });

    it("clips segment fragments at row boundaries without altering original segment identity", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/split.wav",
                fileName: "split.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/split.wav",
                    duration: 30,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 10, end: 20, chord: "C", confidence: 0.8 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const fragments = screen.getAllByTestId("timeline-segment");
        expect(fragments).toHaveLength(2);
        expect(fragments[0]).toHaveAttribute("data-chord", "C");
        expect(fragments[0]).toHaveAttribute("data-fragment-start", "10");
        expect(fragments[0]).toHaveAttribute("data-fragment-end", "15");
        expect(fragments[1]).toHaveAttribute("data-chord", "C");
        expect(fragments[1]).toHaveAttribute("data-fragment-start", "15");
        expect(fragments[1]).toHaveAttribute("data-fragment-end", "20");
    });

    it("keeps fragment left and width proportional to times within each row", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/proportion.wav",
                fileName: "proportion.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/proportion.wav",
                    duration: 45,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 3, end: 6, chord: "D", confidence: 0.8 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const fragment = screen.getByTestId("timeline-segment") as HTMLButtonElement;
        expect(Number.parseFloat(fragment.style.left)).toBeCloseTo(20, 4);
        expect(Number.parseFloat(fragment.style.width)).toBeCloseTo(20, 4);
    });

    it("renders final partial row fragment proportions using final row duration", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/final-row.wav",
                fileName: "final-row.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/final-row.wav",
                    duration: 33,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 30, end: 31.5, chord: "N", confidence: 0.95 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const rows = screen.getAllByTestId("timeline-row");
        expect(rows).toHaveLength(3);
        expect(rows[2]).toHaveAttribute("data-row-start", "30");
        expect(rows[2]).toHaveAttribute("data-row-end", "33");

        const finalRowSegments = within(rows[2]).getAllByTestId("timeline-segment");
        expect(finalRowSegments).toHaveLength(1);
        expect(Number.parseFloat(finalRowSegments[0].style.left)).toBeCloseTo(0, 4);
        expect(Number.parseFloat(finalRowSegments[0].style.width)).toBeCloseTo(50, 4);
    });

    it("clamps timeline segments to duration bounds", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/clamp.wav",
                fileName: "clamp.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/clamp.wav",
                    duration: 10,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [
                        { start: -2, end: 2, chord: "C", confidence: 0.9 },
                        { start: 9, end: 14, chord: "Am", confidence: 0.7 }
                    ]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const segments = screen.getAllByTestId("timeline-segment");
        expect(segments).toHaveLength(2);
        expect(segments[0]).toHaveStyle({ left: "0%", width: "20%" });
        expect(segments[1]).toHaveStyle({ left: "90%", width: "10%" });
    });

    it("renders N segments and supports empty timeline fallback", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi
                .fn()
                .mockResolvedValueOnce({
                    canceled: false,
                    path: "/tmp/nochord.wav",
                    fileName: "nochord.wav"
                })
                .mockResolvedValueOnce({
                    canceled: false,
                    path: "/tmp/empty.wav",
                    fileName: "empty.wav"
                }),
            analyzeAudio: vi
                .fn()
                .mockResolvedValueOnce({
                    version: "1",
                    source: {
                        path: "/tmp/nochord.wav",
                        duration: 4,
                        sampleRate: 22050
                    },
                    analysis: {
                        algorithm: "chroma-template-v1",
                        chords: [{ start: 0, end: 4, chord: "N", confidence: 0.98 }]
                    }
                })
                .mockResolvedValueOnce({
                    version: "1",
                    source: {
                        path: "/tmp/empty.wav",
                        duration: 4,
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

        const firstSegment = screen.getAllByTestId("timeline-segment")[0];
        expect(firstSegment).toHaveTextContent("N");
        expect(firstSegment).toHaveAttribute("data-chord", "N");

        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("Analysis ready: 0 segment(s)")).toBeInTheDocument();
        expect(screen.getByText("No chord segments")).toBeInTheDocument();
    });

    it("shows label text for wide timeline segments", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/wide.wav",
                fileName: "wide.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/wide.wav",
                    duration: 10,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 0, end: 6, chord: "Am", confidence: 0.88 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const segment = screen.getByTestId("timeline-segment");
        expect(segment).toHaveAttribute("data-show-label", "true");
        expect(segment).toHaveTextContent("Am");
    });

    it("visibly renders C text for a deterministic wide C segment", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/wide-c.wav",
                fileName: "wide-c.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/wide-c.wav",
                    duration: 20,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 0, end: 10, chord: "C", confidence: 0.8 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const segment = screen.getByTestId("timeline-segment");
        expect(segment).toHaveAttribute("data-chord", "C");
        expect(segment).toHaveAttribute("data-show-label", "true");

        const label = within(segment).getByText("C");
        expect(label).toBeInTheDocument();
        expect(label).toHaveClass("timeline-segment-label");
    });

    it("shows full minor label when fragment width is sufficient", async () => {
        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/compact-minor.wav",
                fileName: "compact-minor.wav"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/compact-minor.wav",
                    duration: 100,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: [{ start: 0, end: 1.2, chord: "Am", confidence: 0.8 }]
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("State: ready")).toBeInTheDocument();

        const segment = screen.getByTestId("timeline-segment");
        expect(segment).toHaveAttribute("data-chord", "Am");
        expect(segment).toHaveAttribute("data-show-label", "true");
        expect(within(segment).getByText("Am")).toBeInTheDocument();
    });

    it("hides labels for narrow segments and keeps timeline usable with many segments", async () => {
        const crowdedSegments = Array.from({ length: 273 }, (_, index) => {
            const start = index * 0.01;
            const end = start + 0.01;
            return {
                start,
                end,
                chord: index % 5 === 0 ? "N" : "C",
                confidence: 0.7
            };
        });

        window.gcd = {
            getAppVersion: vi.fn().mockResolvedValue("0.0.0"),
            selectAudioFile: vi.fn().mockResolvedValue({
                canceled: false,
                path: "/tmp/crowded.mp3",
                fileName: "crowded.mp3"
            }),
            analyzeAudio: vi.fn().mockResolvedValue({
                version: "1",
                source: {
                    path: "/tmp/crowded.mp3",
                    duration: 215,
                    sampleRate: 22050
                },
                analysis: {
                    algorithm: "chroma-template-v1",
                    chords: crowdedSegments
                }
            })
        };

        render(<App />);
        await openAnalyzerForm();
        fireEvent.click(screen.getByRole("button", { name: "Open Audio" }));
        expect(await screen.findByText("Analysis ready: 273 segment(s)")).toBeInTheDocument();

        const segments = screen.getAllByTestId("timeline-segment");
        expect(segments).toHaveLength(273);
        expect(segments[0]).toHaveAttribute("data-show-label", "false");
        expect(segments[0]).not.toHaveTextContent("C");
        expect(segments[0]).toHaveAttribute("data-chord", "N");
        expect(segments[1]).toHaveAttribute("data-show-label", "false");
    });

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
