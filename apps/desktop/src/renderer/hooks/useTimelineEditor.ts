import type { Dispatch, SetStateAction } from "react";
import type { ChordSegment } from "@gcd/shared/analysis";

import type { TimelineEditSnapshot } from "../appTypes.js";
import { normalizeEditableChord } from "../lib/chords.js";
import { formatTime } from "../lib/time.js";
import { cloneTimelineSegments, snapSplitTime } from "../lib/timelineSegments.js";

type TimelineEditorOptions = {
    timelineSegments: ChordSegment[];
    selectedSegmentIndex: number | null;
    editingChord: string;
    currentTimeSeconds: number;
    timelineUndoStack: TimelineEditSnapshot[];
    timelineRedoStack: TimelineEditSnapshot[];
    commitTimelineSegments: (segments: ChordSegment[], options?: { markDirty?: boolean; pushHistory?: boolean }) => void;
    restoreTimelineSnapshot: (snapshot: TimelineEditSnapshot) => void;
    setSelectedSegmentIndex: Dispatch<SetStateAction<number | null>>;
    setEditingChord: Dispatch<SetStateAction<string>>;
    setTimelineUndoStack: Dispatch<SetStateAction<TimelineEditSnapshot[]>>;
    setTimelineRedoStack: Dispatch<SetStateAction<TimelineEditSnapshot[]>>;
    setAnalysisStatus: Dispatch<SetStateAction<string>>;
};

export function useTimelineEditor(options: TimelineEditorOptions) {
    const handleSelectTimelineSegment = (segmentIndex: number): void => {
        const segment = options.timelineSegments[segmentIndex];
        if (!segment) {
            return;
        }
        options.setSelectedSegmentIndex(segmentIndex);
        options.setEditingChord(segment.chord);
    };

    const handleApplyEditedChord = (): void => {
        if (options.selectedSegmentIndex === null) {
            return;
        }
        const normalizedChord = normalizeEditableChord(options.editingChord);
        if (!normalizedChord) {
            options.setAnalysisStatus("Chord tidak valid. Contoh: A, Bm, C#m, Ddim, atau N.");
            return;
        }
        const nextSegments = options.timelineSegments.map((segment, index) => index === options.selectedSegmentIndex
            ? { ...segment, chord: normalizedChord, confidence: Math.max(segment.confidence, 0.96) }
            : segment);
        options.commitTimelineSegments(nextSegments, { markDirty: true, pushHistory: true });
        options.setEditingChord(normalizedChord);
        options.setAnalysisStatus(`Chord corrected to ${normalizedChord}. Save to persist.`);
    };

    const handleSplitSelectedSegment = (): void => {
        if (options.selectedSegmentIndex === null) {
            return;
        }
        const segment = options.timelineSegments[options.selectedSegmentIndex];
        if (!segment) {
            return;
        }
        const rawSplitTime = options.currentTimeSeconds > segment.start + 0.2 && options.currentTimeSeconds < segment.end - 0.2
            ? options.currentTimeSeconds
            : segment.start + ((segment.end - segment.start) / 2);
        const splitTime = snapSplitTime(rawSplitTime, segment);
        if (splitTime <= segment.start || splitTime >= segment.end) {
            return;
        }
        const nextSegments = [
            ...options.timelineSegments.slice(0, options.selectedSegmentIndex),
            { ...segment, end: splitTime },
            { ...segment, start: splitTime },
            ...options.timelineSegments.slice(options.selectedSegmentIndex + 1),
        ];
        options.commitTimelineSegments(nextSegments, { markDirty: true, pushHistory: true });
        options.setSelectedSegmentIndex(options.selectedSegmentIndex + 1);
        options.setEditingChord(segment.chord);
        options.setAnalysisStatus(`Segment split at ${formatTime(splitTime)} with snap. Save to persist.`);
    };

    const handleMergeSelectedSegment = (direction: "left" | "right"): void => {
        if (options.selectedSegmentIndex === null) {
            return;
        }
        const neighborIndex = direction === "left" ? options.selectedSegmentIndex - 1 : options.selectedSegmentIndex + 1;
        const segment = options.timelineSegments[options.selectedSegmentIndex];
        const neighbor = options.timelineSegments[neighborIndex];
        if (!segment || !neighbor) {
            return;
        }
        const merged: ChordSegment = {
            start: Math.min(segment.start, neighbor.start),
            end: Math.max(segment.end, neighbor.end),
            chord: segment.chord,
            confidence: Math.max(segment.confidence, neighbor.confidence, 0.96),
        };
        const leftIndex = Math.min(options.selectedSegmentIndex, neighborIndex);
        const rightIndex = Math.max(options.selectedSegmentIndex, neighborIndex);
        const nextSegments = [
            ...options.timelineSegments.slice(0, leftIndex),
            merged,
            ...options.timelineSegments.slice(rightIndex + 1),
        ];
        options.commitTimelineSegments(nextSegments, { markDirty: true, pushHistory: true });
        options.setSelectedSegmentIndex(leftIndex);
        options.setEditingChord(merged.chord);
        options.setAnalysisStatus(`Merged segment ${direction}. Save to persist.`);
    };

    const handleUndoTimelineEdit = (): void => {
        const previous = options.timelineUndoStack[options.timelineUndoStack.length - 1];
        if (!previous) {
            return;
        }
        options.setTimelineUndoStack((current) => current.slice(0, -1));
        options.setTimelineRedoStack((current) => [
            ...current.slice(-24),
            { segments: cloneTimelineSegments(options.timelineSegments), selectedSegmentIndex: options.selectedSegmentIndex },
        ]);
        options.restoreTimelineSnapshot(previous);
        options.setAnalysisStatus("Chord edit undone. Save to persist.");
    };

    const handleRedoTimelineEdit = (): void => {
        const next = options.timelineRedoStack[options.timelineRedoStack.length - 1];
        if (!next) {
            return;
        }
        options.setTimelineRedoStack((current) => current.slice(0, -1));
        options.setTimelineUndoStack((current) => [
            ...current.slice(-24),
            { segments: cloneTimelineSegments(options.timelineSegments), selectedSegmentIndex: options.selectedSegmentIndex },
        ]);
        options.restoreTimelineSnapshot(next);
        options.setAnalysisStatus("Chord edit redone. Save to persist.");
    };

    return {
        handleSelectTimelineSegment,
        handleApplyEditedChord,
        handleSplitSelectedSegment,
        handleMergeSelectedSegment,
        handleUndoTimelineEdit,
        handleRedoTimelineEdit,
    };
}
