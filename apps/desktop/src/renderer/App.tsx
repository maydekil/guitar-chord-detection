import { useEffect, useRef, useState } from "react";
import type { ChordSegment } from "@gcd/shared/analysis";

type ShellState = "idle" | "loading-file" | "analyzing" | "ready" | "playing" | "paused" | "error";

export function App() {
    const [version, setVersion] = useState<string>("0.0.0");
    const [state, setState] = useState<ShellState>("idle");
    const [selectedFileName, setSelectedFileName] = useState<string>("No file selected");
    const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
    const [audioSourceUrl, setAudioSourceUrl] = useState<string | null>(null);
    const [isMediaReady, setIsMediaReady] = useState<boolean>(false);
    const [durationSeconds, setDurationSeconds] = useState<number>(0);
    const [currentTimeSeconds, setCurrentTimeSeconds] = useState<number>(0);
    const [analysisSegmentCount, setAnalysisSegmentCount] = useState<number | null>(null);
    const [timelineSegments, setTimelineSegments] = useState<ChordSegment[]>([]);
    const [timelineWidthPx, setTimelineWidthPx] = useState<number>(0);
    const [analysisStatus, setAnalysisStatus] = useState<string>("No analysis yet");
    const activeRequestIdRef = useRef<number>(0);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const activeBlobUrlRef = useRef<string | null>(null);
    const timelineRef = useRef<HTMLElement | null>(null);
    const bridge = window.gcd;

    useEffect(() => {
        const element = timelineRef.current;
        if (!element) {
            setTimelineWidthPx(0);
            return;
        }

        const measureWidth = (): void => {
            const nextWidth = element.getBoundingClientRect().width;
            setTimelineWidthPx(Number.isFinite(nextWidth) ? Math.max(0, nextWidth) : 0);
        };

        measureWidth();

        if (typeof ResizeObserver === "undefined") {
            return;
        }

        const observer = new ResizeObserver(() => {
            measureWidth();
        });

        observer.observe(element);
        return () => {
            observer.disconnect();
        };
    }, []);

    useEffect(() => {
        return () => {
            if (activeBlobUrlRef.current) {
                if (typeof URL.revokeObjectURL === "function") {
                    URL.revokeObjectURL(activeBlobUrlRef.current);
                }
                activeBlobUrlRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (!bridge) {
            setVersion("bridge-unavailable");
            return;
        }
        void bridge.getAppVersion().then(setVersion).catch(() => setVersion("unknown"));
    }, [bridge]);

    const runAnalysis = async (audioPath: string, options?: { forceRefresh?: boolean }): Promise<void> => {
        if (!bridge?.analyzeAudio) {
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }

        setState("analyzing");
        setAnalysisSegmentCount(null);
        setTimelineSegments([]);
        setAnalysisStatus("Analyzing...");

        const requestId = activeRequestIdRef.current + 1;
        activeRequestIdRef.current = requestId;

        const analysis = await bridge.analyzeAudio(audioPath, {
            forceRefresh: options?.forceRefresh === true
        });

        if (requestId !== activeRequestIdRef.current) {
            return;
        }

        if ("error" in analysis) {
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }

        const normalizedTimelineSegments = normalizeTimelineSegments(analysis.analysis.chords, analysis.source.duration);
        const segmentCount = normalizedTimelineSegments.length;
        const analysisDuration = Number.isFinite(analysis.source.duration) ? Math.max(0, analysis.source.duration) : 0;
        setState("ready");
        setAnalysisSegmentCount(segmentCount);
        setTimelineSegments(normalizedTimelineSegments);
        setDurationSeconds(analysisDuration);
        setCurrentTimeSeconds(0);
        setAnalysisStatus(`Analysis ready: ${segmentCount} segment(s)`);
    };

    const handleOpenAudio = async (): Promise<void> => {
        if (!bridge) {
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }

        setState("loading-file");
        const result = await bridge.selectAudioFile();
        if (result.canceled || !result.fileName) {
            setState("idle");
            return;
        }

        setSelectedFileName(result.fileName);

        const nextPath = result.path;
        const isDifferentFile = Boolean(nextPath && nextPath !== selectedFilePath);
        if (isDifferentFile) {
            resetPlaybackState();
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setAnalysisStatus("No analysis yet");
        }

        if (!nextPath || !bridge.analyzeAudio) {
            resetPlaybackState();
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }

        let playbackSourceUrl: string;
        if (bridge.getAudioPlaybackSource) {
            try {
                const playbackSource = await bridge.getAudioPlaybackSource(nextPath);
                playbackSourceUrl = createObjectUrl(playbackSource.bytes);
            } catch {
                resetPlaybackState();
                setState("error");
                setAnalysisSegmentCount(null);
                setTimelineSegments([]);
                setAnalysisStatus("Could not play this audio file.");
                return;
            }
        } else {
            playbackSourceUrl = toFileUrl(nextPath);
        }

        if (activeBlobUrlRef.current) {
            if (typeof URL.revokeObjectURL === "function") {
                URL.revokeObjectURL(activeBlobUrlRef.current);
            }
            activeBlobUrlRef.current = null;
        }
        if (playbackSourceUrl.startsWith("blob:")) {
            activeBlobUrlRef.current = playbackSourceUrl;
        }

        setSelectedFilePath(nextPath);
        setAudioSourceUrl(playbackSourceUrl);
        setIsMediaReady(false);
        setDurationSeconds(0);
        setCurrentTimeSeconds(0);

        await runAnalysis(nextPath);
    };

    const handleReanalyze = async (): Promise<void> => {
        if (!selectedFilePath) {
            return;
        }
        await runAnalysis(selectedFilePath, { forceRefresh: true });
    };

    const resetPlaybackState = (): void => {
        const audio = audioRef.current;
        if (audio) {
            try {
                audio.pause();
            } catch {
                // Some test environments do not fully implement media pause.
            }

            try {
                audio.currentTime = 0;
            } catch {
                // Ignore media seek reset failures and still clear UI state.
            }
        }
        setIsMediaReady(false);
        setDurationSeconds(0);
        setCurrentTimeSeconds(0);
    };

    const handlePlay = async (): Promise<void> => {
        const audio = audioRef.current;
        if (!audio) {
            return;
        }

        if (!isMediaReady) {
            audio.load();
        }

        try {
            await audio.play();
        } catch (error) {
            const reason = error instanceof Error && error.message ? ` (${error.message})` : "";
            setAnalysisStatus(`Could not play this audio file.${reason}`);
            // play() can reject for transient media timing reasons; keep analysis state intact.
            setState((current) => (current === "playing" ? "paused" : current));
        }
    };

    const handlePause = (): void => {
        const audio = audioRef.current;
        if (!audio) {
            return;
        }
        audio.pause();
    };

    const handleSeek = (value: number): void => {
        const audio = audioRef.current;
        if (!audio) {
            return;
        }

        const safeValue = Math.max(0, Math.min(value, durationSeconds || value));
        audio.currentTime = safeValue;
        setCurrentTimeSeconds(safeValue);
    };

    const isPlayableState = state === "ready" || state === "playing" || state === "paused";
    const canUsePlayback = Boolean(audioSourceUrl) && Number.isFinite(durationSeconds) && durationSeconds > 0 && isPlayableState;
    const canSeek = canUsePlayback && isMediaReady;
    const isPlaying = state === "playing";
    const canReanalyze = Boolean(selectedFilePath) && Boolean(bridge?.analyzeAudio);

    return (
        <main className="shell" data-state={state}>
            <header className="shell-header">
                <h1>Guitar Chord Detector</h1>
                <p className="shell-meta">Desktop Shell v{version}</p>
            </header>

            <section className="shell-card" aria-label="audio controls shell">
                <button type="button" className="open-btn" onClick={() => void handleOpenAudio()} disabled={!bridge}>
                    Open Audio
                </button>
                {canReanalyze ? (
                    <button type="button" className="open-btn" onClick={() => void handleReanalyze()}>
                        Re-analyze
                    </button>
                ) : null}
                <audio
                    ref={audioRef}
                    src={audioSourceUrl ?? undefined}
                    data-testid="audio-player"
                    onLoadedMetadata={(event) => {
                        const nextDuration = Number.isFinite(event.currentTarget.duration)
                            ? Math.max(0, event.currentTarget.duration)
                            : 0;
                        if (nextDuration > 0) {
                            setDurationSeconds(nextDuration);
                        }
                        setIsMediaReady(nextDuration > 0);
                        setCurrentTimeSeconds(event.currentTarget.currentTime || 0);
                    }}
                    onTimeUpdate={(event) => {
                        if (!isMediaReady && event.currentTarget.currentTime > 0) {
                            setIsMediaReady(true);
                        }
                        setCurrentTimeSeconds(event.currentTarget.currentTime || 0);
                    }}
                    onPlay={() => {
                        setIsMediaReady(true);
                        setState("playing");
                    }}
                    onPause={() => {
                        setState((current) => (current === "playing" ? "paused" : current));
                    }}
                    onEnded={() => {
                        setState("paused");
                    }}
                    onError={() => {
                        setIsMediaReady(false);
                        setState("error");
                        setAnalysisStatus("Could not analyze this audio file.");
                    }}
                />
                <p className="file-name" aria-live="polite">{selectedFileName}</p>
                <p className="state-label">State: {state}</p>
                <p className="analysis-label" aria-live="polite">{analysisStatus}</p>
                <div className="playback-row" aria-label="playback controls">
                    <button type="button" onClick={() => void handlePlay()} disabled={!canUsePlayback || isPlaying}>
                        Play
                    </button>
                    <button type="button" onClick={handlePause} disabled={!canUsePlayback || !isPlaying}>
                        Pause
                    </button>
                    <p className="time-label" aria-live="polite">
                        {formatTime(currentTimeSeconds)} / {formatTime(durationSeconds)}
                    </p>
                </div>
                <input
                    type="range"
                    min={0}
                    max={durationSeconds || 0}
                    step={0.01}
                    value={Math.min(currentTimeSeconds, durationSeconds || currentTimeSeconds)}
                    onChange={(event) => handleSeek(Number(event.currentTarget.value))}
                    disabled={!canSeek}
                    aria-label="Seek"
                />
                <section ref={timelineRef} className="timeline" aria-label="Chord timeline">
                    {renderTimeline(timelineSegments, durationSeconds, timelineWidthPx, currentTimeSeconds)}
                </section>
                <p className="placeholder">Secure shell ready. Audio workflow is enabled in subsequent tasks.</p>
            </section>
        </main>
    );
}

function toFileUrl(localPath: string): string {
    const normalized = localPath.replace(/\\/g, "/");
    const url = new URL("file://");

    if (/^[A-Za-z]:\//.test(normalized)) {
        url.pathname = `/${normalized}`;
        return url.href;
    }

    url.pathname = normalized.startsWith("/") ? normalized : `/${normalized}`;
    return url.href;
}

function createObjectUrl(bytes: Uint8Array): string {
    const stableBytes = new Uint8Array(bytes.length);
    stableBytes.set(bytes);
    const blob = new Blob([stableBytes]);
    return URL.createObjectURL(blob);
}

function normalizeTimelineSegments(segments: ChordSegment[], durationSeconds: number): ChordSegment[] {
    const safeDuration = Number.isFinite(durationSeconds) ? Math.max(0, durationSeconds) : 0;
    return [...segments]
        .sort((a, b) => a.start - b.start)
        .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
        .map((segment) => {
            const start = Math.max(0, segment.start);
            const maxEnd = safeDuration > 0 ? safeDuration : segment.end;
            const end = Math.min(maxEnd, Math.max(start, segment.end));
            return {
                ...segment,
                start,
                end
            };
        })
        .filter((segment) => segment.end > segment.start);
}

type TimelineSegmentLayout = {
    segment: ChordSegment;
    fragmentStart: number;
    fragmentEnd: number;
    leftPercent: number;
    widthPercent: number;
    showLabel: boolean;
    labelText: string;
};

type TimelineRowLayout = {
    rowIndex: number;
    rowStart: number;
    rowEnd: number;
    rowDuration: number;
    fragments: TimelineSegmentLayout[];
};

const TIMELINE_ROW_WINDOW_SECONDS = 15;

function renderTimeline(segments: ChordSegment[], durationSeconds: number, timelineWidthPx: number, currentTimeSeconds: number) {
    const safeDuration = Number.isFinite(durationSeconds) ? Math.max(0, durationSeconds) : 0;
    const safeCurrentTime = Number.isFinite(currentTimeSeconds) ? Math.max(0, currentTimeSeconds) : 0;

    if (safeDuration <= 0 || segments.length === 0) {
        return <p className="timeline-empty">No chord segments</p>;
    }

    const rows = buildTimelineRows(segments, safeDuration, timelineWidthPx, TIMELINE_ROW_WINDOW_SECONDS);

    return (
        <div className="timeline-rows" data-testid="timeline-rows">
            {rows.map((row) => (
                <section
                    key={`timeline-row-${row.rowIndex}`}
                    className="timeline-row"
                    data-testid="timeline-row"
                    data-row-index={row.rowIndex}
                    data-row-start={row.rowStart}
                    data-row-end={row.rowEnd}
                >
                    <p className="timeline-row-range" data-testid="timeline-row-range">
                        {formatTime(row.rowStart)} - {formatTime(row.rowEnd)}
                    </p>
                    <div className="timeline-track" data-testid="timeline-track">
                        {row.fragments.map(({ segment, fragmentStart, fragmentEnd, leftPercent, widthPercent, showLabel, labelText }, index) => (
                            <button
                                key={`${row.rowIndex}-${segment.start}-${segment.end}-${segment.chord}-${fragmentStart}-${fragmentEnd}-${index}`}
                                type="button"
                                className="timeline-segment"
                                data-testid="timeline-segment"
                                data-chord={segment.chord}
                                data-show-label={showLabel ? "true" : "false"}
                                data-fragment-start={fragmentStart}
                                data-fragment-end={fragmentEnd}
                                data-active={fragmentStart <= safeCurrentTime && safeCurrentTime < fragmentEnd ? "true" : "false"}
                                aria-current={fragmentStart <= safeCurrentTime && safeCurrentTime < fragmentEnd ? "true" : "false"}
                                aria-label={`${segment.chord} from ${formatTime(fragmentStart)} to ${formatTime(fragmentEnd)}`}
                                title={`${segment.chord} (${Math.round(segment.confidence * 100)}%)`}
                                style={{
                                    left: `${leftPercent}%`,
                                    width: `${widthPercent}%`
                                }}
                            >
                                {showLabel ? <span className="timeline-segment-label">{labelText}</span> : null}
                            </button>
                        ))}
                    </div>
                </section>
            ))}
        </div>
    );
}

function buildTimelineRows(
    segments: ChordSegment[],
    safeDuration: number,
    timelineWidthPx: number,
    windowSeconds: number
): TimelineRowLayout[] {
    if (safeDuration <= 0) {
        return [];
    }

    const safeWindow = Number.isFinite(windowSeconds) ? Math.max(1, windowSeconds) : TIMELINE_ROW_WINDOW_SECONDS;
    const rowCount = Math.max(1, Math.ceil(safeDuration / safeWindow));

    return Array.from({ length: rowCount }, (_, rowIndex) => {
        const rowStart = rowIndex * safeWindow;
        const rowEnd = Math.min(safeDuration, rowStart + safeWindow);
        const rowDuration = Math.max(0, rowEnd - rowStart);
        const rowSegments = buildRowSegmentLayout(segments, rowStart, rowEnd, rowDuration);
        const fragments = computeVisibleLabels(rowSegments, timelineWidthPx > 0 ? timelineWidthPx : 960);
        return {
            rowIndex,
            rowStart,
            rowEnd,
            rowDuration,
            fragments
        };
    });
}

function buildRowSegmentLayout(
    segments: ChordSegment[],
    rowStart: number,
    rowEnd: number,
    rowDuration: number
): TimelineSegmentLayout[] {
    if (rowDuration <= 0) {
        return [];
    }

    const baseLayout = segments
        .map((segment) => {
            const fragmentStart = Math.max(segment.start, rowStart);
            const fragmentEnd = Math.min(segment.end, rowEnd);
            if (fragmentEnd <= fragmentStart) {
                return null;
            }

            const leftPercentRaw = ((fragmentStart - rowStart) / rowDuration) * 100;
            const widthPercentRaw = ((fragmentEnd - fragmentStart) / rowDuration) * 100;
            const leftPercent = Math.max(0, Math.min(leftPercentRaw, 100));
            const widthPercent = Math.max(0, Math.min(widthPercentRaw, 100 - leftPercent));
            return {
                segment,
                fragmentStart,
                fragmentEnd,
                leftPercent,
                widthPercent,
                showLabel: false,
                labelText: ""
            };
        })
        .filter((item): item is TimelineSegmentLayout => item !== null);

    return baseLayout;
}

function computeVisibleLabels(layout: TimelineSegmentLayout[], timelineWidthPx: number): TimelineSegmentLayout[] {
    let lastLabelRightEdgePx = -Infinity;

    return layout.map((item) => {
        const fullLabel = item.segment.chord.trim();
        if (!fullLabel) {
            return {
                ...item,
                showLabel: false,
                labelText: ""
            };
        }

        const compactLabel = toCompactChordLabel(fullLabel);
        const minimalLabel = toMinimalChordLabel(fullLabel);
        const candidateLabels = [fullLabel, compactLabel, minimalLabel].filter(
            (label, index, labels) => label.length > 0 && labels.indexOf(label) === index
        );
        const leftPx = (item.leftPercent / 100) * timelineWidthPx;
        const widthPx = (item.widthPercent / 100) * timelineWidthPx;

        for (const labelText of candidateLabels) {
            const labelWidthPx = estimateLabelWidthPx(labelText);
            const innerPaddingPx = 2;
            if (widthPx < labelWidthPx + innerPaddingPx) {
                continue;
            }

            const segmentLeftBoundPx = leftPx + 1;
            const segmentRightBoundPx = leftPx + widthPx - 1;
            const centeredLeftPx = leftPx + (widthPx - labelWidthPx) / 2;
            const labelLeftPx = Math.max(segmentLeftBoundPx, Math.min(centeredLeftPx, segmentRightBoundPx - labelWidthPx));
            const labelRightPx = labelLeftPx + labelWidthPx;

            if (labelLeftPx < lastLabelRightEdgePx + 1) {
                continue;
            }

            lastLabelRightEdgePx = labelRightPx;
            return {
                ...item,
                showLabel: true,
                labelText
            };
        }

        return {
            ...item,
            showLabel: false,
            labelText: ""
        };
    });
}

function estimateLabelWidthPx(chord: string): number {
    const charCount = chord.length;
    const compactPaddingPx = 1;
    const averageCharWidthPx = 5;
    return compactPaddingPx + charCount * averageCharWidthPx;
}

function toCompactChordLabel(chord: string): string {
    if (chord === "N") {
        return chord;
    }

    if (chord.endsWith("m") && chord.length > 1) {
        return chord.slice(0, -1);
    }

    return chord;
}

function toMinimalChordLabel(chord: string): string {
    if (chord === "N") {
        return chord;
    }

    const [root] = chord;
    return root ?? "";
}

function formatTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, Math.floor(totalSeconds)) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}
