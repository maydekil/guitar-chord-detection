import type { ChordSegment } from "@gcd/shared/analysis";
import type { SongLibrarySummary, SongMetadataInput } from "@gcd/shared/library";

export type ShellState = "idle" | "loading-file" | "analyzing" | "ready" | "playing" | "paused" | "error";
export type ViewMode = "library" | "detail" | "evaluation";
export type DetectorTab = "timeline" | "lyrics";
export type LyricsModel = "tiny" | "base" | "small" | "medium" | "large";
export type ApiHealthState = "local" | "checking" | "connected" | "offline";
export type ExportFormat = "txt" | "lrc";
export type PendingConfirmation =
    | { kind: "save"; metadata: SongMetadataInput }
    | { kind: "delete"; song: SongLibrarySummary };
export type ApiJobProgressEvent = {
    id: string;
    kind: "analysis" | "lyrics" | "vocals" | "pitch-shift";
    status: "queued" | "running" | "succeeded" | "failed";
    progress: number;
};
export type TimelineEditSnapshot = {
    segments: ChordSegment[];
    selectedSegmentIndex: number | null;
};
