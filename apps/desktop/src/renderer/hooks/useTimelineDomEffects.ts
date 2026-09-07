import { useEffect } from "react";
import type { RefObject } from "react";
import type { ChordSegment } from "@gcd/shared/analysis";

import type { DetectorTab, ViewMode } from "../appTypes.js";

type TimelineDomEffectsOptions = {
    timelineRef: RefObject<HTMLElement | null>;
    chordEditorRef: RefObject<HTMLDivElement | null>;
    lyricsPreviewRef: RefObject<HTMLDivElement | null>;
    viewMode: ViewMode;
    detectorTab: DetectorTab;
    currentTimeSeconds: number;
    timelineSegments: ChordSegment[];
    lyricsText: string;
    selectedSegmentIndex: number | null;
    onClearSelectedSegment: () => void;
    onUndoTimelineEdit: () => void;
    onRedoTimelineEdit: () => void;
    onTimelineWidthChange: (widthPx: number) => void;
};

export function useTimelineDomEffects(options: TimelineDomEffectsOptions): void {
    useTimelineWidthMeasurement(options);
    useTimelineAutoScroll(options);
    useChordEditorClickAway(options);
    useTimelineKeyboardShortcuts(options);
    useLyricsAutoScroll(options);
}

function useTimelineWidthMeasurement({ timelineRef, viewMode, onTimelineWidthChange }: TimelineDomEffectsOptions): void {
    useEffect(() => {
        const element = timelineRef.current;
        if (!element) {
            onTimelineWidthChange(0);
            return;
        }
        const measureWidth = (): void => {
            const nextWidth = element.getBoundingClientRect().width;
            onTimelineWidthChange(Number.isFinite(nextWidth) ? Math.max(0, nextWidth) : 0);
        };
        measureWidth();
        if (typeof ResizeObserver === "undefined") {
            return;
        }
        const observer = new ResizeObserver(measureWidth);
        observer.observe(element);
        return () => observer.disconnect();
    }, [timelineRef, viewMode, onTimelineWidthChange]);
}

function useTimelineAutoScroll({ timelineRef, viewMode, currentTimeSeconds, timelineSegments }: TimelineDomEffectsOptions): void {
    useEffect(() => {
        const timeline = timelineRef.current;
        if (!timeline || viewMode !== "detail") {
            return;
        }
        const activeSegment = timeline.querySelector<HTMLElement>('[data-testid="timeline-segment"][data-active="true"]');
        if (typeof activeSegment?.scrollIntoView === "function") {
            activeSegment.scrollIntoView({ block: "nearest", inline: "nearest" });
        }
    }, [currentTimeSeconds, timelineRef, timelineSegments, viewMode]);
}

function useChordEditorClickAway(options: TimelineDomEffectsOptions): void {
    useEffect(() => {
        if (options.selectedSegmentIndex === null) {
            return;
        }
        const handlePointerDown = (event: PointerEvent): void => {
            const target = event.target;
            if (!(target instanceof Node)) {
                return;
            }
            if (options.chordEditorRef.current?.contains(target)) {
                return;
            }
            if (options.timelineRef.current?.contains(target)) {
                const segment = target instanceof Element ? target.closest('[data-testid="timeline-segment"]') : null;
                if (segment) {
                    return;
                }
            }
            options.onClearSelectedSegment();
        };
        document.addEventListener("pointerdown", handlePointerDown);
        return () => document.removeEventListener("pointerdown", handlePointerDown);
    }, [options]);
}

function useTimelineKeyboardShortcuts(options: TimelineDomEffectsOptions): void {
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent): void => {
            if (options.viewMode !== "detail" || options.detectorTab !== "timeline") {
                return;
            }
            const target = event.target;
            const isTextEditing = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement;
            const isUndo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" && !event.shiftKey;
            const isRedo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" && event.shiftKey;
            if (isUndo && !isTextEditing) {
                event.preventDefault();
                options.onUndoTimelineEdit();
            }
            if (isRedo && !isTextEditing) {
                event.preventDefault();
                options.onRedoTimelineEdit();
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [options]);
}

function useLyricsAutoScroll({ lyricsPreviewRef, viewMode, detectorTab, currentTimeSeconds, lyricsText }: TimelineDomEffectsOptions): void {
    useEffect(() => {
        const lyricsPreview = lyricsPreviewRef.current;
        if (!lyricsPreview || viewMode !== "detail" || detectorTab !== "lyrics") {
            return;
        }
        const activeLine = lyricsPreview.querySelector<HTMLElement>('[data-active="true"]');
        if (typeof activeLine?.scrollIntoView === "function") {
            activeLine.scrollIntoView({ block: "center", inline: "nearest" });
        }
    }, [currentTimeSeconds, detectorTab, lyricsPreviewRef, lyricsText, viewMode]);
}
