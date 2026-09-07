import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { activeChords, openAnalyzerForm, setMediaTiming } from "./testUtils";

afterEach(() => {
    cleanup();
});

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

});
