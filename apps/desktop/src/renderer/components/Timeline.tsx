import type { MouseEvent } from "react";
import type { ChordSegment } from "@gcd/shared/analysis";

import { transposeChordLabel } from "../lib/chords.js";
import { formatTime } from "../lib/time.js";

const TIMELINE_ROW_WINDOW_SECONDS = 15;

type TimelineSegmentLayout = {
    segmentIndex: number;
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

type TimelineProps = {
    segments: ChordSegment[];
    durationSeconds: number;
    timelineWidthPx: number;
    currentTimeSeconds: number;
    transposeSemitones: number;
    selectedSegmentIndex: number | null;
    onSeek: (value: number) => void;
    onSelectSegment: (segmentIndex: number) => void;
};

export function Timeline({
    segments,
    durationSeconds,
    timelineWidthPx,
    currentTimeSeconds,
    transposeSemitones,
    selectedSegmentIndex,
    onSeek,
    onSelectSegment,
}: TimelineProps) {
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
                    <div
                        className="timeline-track"
                        data-testid="timeline-track"
                        onClick={(event) => {
                            onSeek(timelineClickTime(event, row.rowStart, row.rowDuration, safeDuration));
                        }}
                    >
                        {row.fragments.map(({ segmentIndex, segment, fragmentStart, fragmentEnd, leftPercent, widthPercent, showLabel, labelText }, index) => {
                            const displayedChord = transposeChordLabel(segment.chord, transposeSemitones);
                            const displayedLabel = transposeChordLabel(labelText, transposeSemitones);
                            const confidencePercent = Math.round(segment.confidence * 100);
                            const confidenceLevel = chordConfidenceLevel(segment.confidence);
                            const isActive = fragmentStart <= safeCurrentTime && safeCurrentTime < fragmentEnd;
                            return (
                                <button
                                    key={`${row.rowIndex}-${segment.start}-${segment.end}-${segment.chord}-${fragmentStart}-${fragmentEnd}-${index}`}
                                    type="button"
                                    className="timeline-segment"
                                    data-testid="timeline-segment"
                                    data-chord={displayedChord}
                                    data-show-label={showLabel ? "true" : "false"}
                                    data-fragment-start={fragmentStart}
                                    data-fragment-end={fragmentEnd}
                                    data-confidence={confidenceLevel}
                                    data-selected={selectedSegmentIndex === segmentIndex ? "true" : "false"}
                                    data-active={isActive ? "true" : "false"}
                                    aria-current={isActive ? "true" : "false"}
                                    aria-label={`${displayedChord} from ${formatTime(fragmentStart)} to ${formatTime(fragmentEnd)}, confidence ${confidencePercent}%`}
                                    title={`${displayedChord} confidence ${confidencePercent}%`}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        onSelectSegment(segmentIndex);
                                        onSeek(timelineClickTime(event, row.rowStart, row.rowDuration, safeDuration));
                                    }}
                                    style={{
                                        left: `${leftPercent}%`,
                                        width: `${widthPercent}%`,
                                    }}
                                >
                                    {showLabel ? (
                                        <span className="timeline-segment-label-wrap">
                                            <span className="timeline-segment-label">{displayedLabel}</span>
                                            <small>{confidencePercent}%</small>
                                        </span>
                                    ) : null}
                                </button>
                            );
                        })}
                    </div>
                </section>
            ))}
        </div>
    );
}

function timelineClickTime(
    event: MouseEvent<HTMLElement>,
    rowStart: number,
    rowDuration: number,
    durationSeconds: number,
): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const offset = Math.max(0, Math.min(event.clientX - rect.left, width));
    const ratio = offset / width;
    const target = rowStart + ratio * rowDuration;
    return Math.max(0, Math.min(target, durationSeconds));
}

function chordConfidenceLevel(confidence: number): "high" | "medium" | "low" {
    if (!Number.isFinite(confidence)) {
        return "low";
    }
    if (confidence >= 0.78) {
        return "high";
    }
    if (confidence >= 0.58) {
        return "medium";
    }
    return "low";
}

function buildTimelineRows(
    segments: ChordSegment[],
    safeDuration: number,
    timelineWidthPx: number,
    windowSeconds: number,
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
        return { rowIndex, rowStart, rowEnd, rowDuration, fragments };
    });
}

function buildRowSegmentLayout(
    segments: ChordSegment[],
    rowStart: number,
    rowEnd: number,
    rowDuration: number,
): TimelineSegmentLayout[] {
    if (rowDuration <= 0) {
        return [];
    }
    return segments
        .map((segment, segmentIndex) => {
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
                segmentIndex,
                fragmentStart,
                fragmentEnd,
                leftPercent,
                widthPercent,
                showLabel: false,
                labelText: "",
            };
        })
        .filter((item): item is TimelineSegmentLayout => item !== null);
}

function computeVisibleLabels(layout: TimelineSegmentLayout[], timelineWidthPx: number): TimelineSegmentLayout[] {
    let lastLabelRightEdgePx = -Infinity;

    return layout.map((item) => {
        const fullLabel = item.segment.chord.trim();
        if (!fullLabel) {
            return { ...item, showLabel: false, labelText: "" };
        }
        const candidateLabels = [fullLabel, toCompactChordLabel(fullLabel), toMinimalChordLabel(fullLabel)].filter(
            (label, index, labels) => label.length > 0 && labels.indexOf(label) === index,
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
            return { ...item, showLabel: true, labelText };
        }

        return { ...item, showLabel: false, labelText: "" };
    });
}

function estimateLabelWidthPx(chord: string): number {
    return 1 + chord.length * 5;
}

function toCompactChordLabel(chord: string): string {
    if (chord === "N") {
        return chord;
    }
    if (chord.endsWith("dim") && chord.length > 3) {
        return `${chord.slice(0, -3)}°`;
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
    if (chord.endsWith("dim") && chord.length > 3) {
        return chord.slice(0, -3)[0] ?? "";
    }
    return chord[0] ?? "";
}
