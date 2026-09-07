import type { RefObject } from "react";
import type { ChordSegment } from "@gcd/shared/analysis";

import type { ApiJobProgressEvent, DetectorTab, LyricsModel, ShellState } from "../appTypes.js";
import { Timeline } from "./Timeline.js";
import { formatApiJobKind } from "../lib/apiStatus.js";
import { EDITABLE_CHORD_OPTIONS, formatTranspose } from "../lib/chords.js";
import { renderLyricsPreview, stripLyricsTiming } from "../lib/lyrics.js";
import { formatTime } from "../lib/time.js";

type DetectorPanelProps = {
    viewMode: "library" | "detail";
    state: ShellState;
    selectedFileName: string;
    audioSourceUrl: string | null;
    durationSeconds: number;
    currentTimeSeconds: number;
    analysisStatus: string;
    activeChordLabel: string;
    hasUnsavedChordEdits: boolean;
    apiJobProgress: ApiJobProgressEvent | null;
    canCancelApiJob: boolean;
    canReanalyze: boolean;
    canSaveAnalysis: boolean;
    canUsePlayback: boolean;
    canSeek: boolean;
    isPlaying: boolean;
    selectedSongId: string | null;
    isSaveFormOpen: boolean;
    songArtist: string;
    songTitle: string;
    transposeSemitones: number;
    isTransposeKeyOnly: boolean;
    isPitchShiftingAudio: boolean;
    detectorTab: DetectorTab;
    timelineSegments: ChordSegment[];
    timelineWidthPx: number;
    selectedSegmentIndex: number | null;
    selectedTimelineSegment: ChordSegment | null;
    editingChord: string;
    canUndoTimelineEdit: boolean;
    canRedoTimelineEdit: boolean;
    lyricsText: string;
    lyricsModel: LyricsModel;
    isGeneratingLyrics: boolean;
    selectedFilePath: string | null;
    isRemovingVocals: boolean;
    isUsingInstrumentalAudio: boolean;
    audioRef: RefObject<HTMLAudioElement | null>;
    timelineRef: RefObject<HTMLElement | null>;
    chordEditorRef: RefObject<HTMLDivElement | null>;
    lyricsPreviewRef: RefObject<HTMLDivElement | null>;
    onOpenAudio: () => void;
    onReanalyze: () => void;
    onSaveAnalysis: () => void;
    onOpenSaveForm: () => void;
    onOpenExportPreview: (format: "txt" | "lrc") => void;
    onLoadedMetadata: (event: React.SyntheticEvent<HTMLAudioElement>) => void;
    onTimeUpdate: (event: React.SyntheticEvent<HTMLAudioElement>) => void;
    onSeeked: (event: React.SyntheticEvent<HTMLAudioElement>) => void;
    onPlay: () => void;
    onPause: () => void;
    onEnded: () => void;
    onAudioError: () => void;
    onCancelApiJob: () => void;
    onTogglePlayback: () => void;
    onToggleVocalHide: () => void;
    onSetTranspose: (value: number) => void;
    onSetTransposeKeyOnly: (value: boolean) => void;
    onSeek: (value: number) => void;
    onTabChange: (tab: DetectorTab) => void;
    onSongArtistChange: (value: string) => void;
    onSongTitleChange: (value: string) => void;
    onEditingChordChange: (value: string) => void;
    onApplyEditedChord: () => void;
    onSplitSelectedSegment: () => void;
    onMergeSelectedSegment: (direction: "left" | "right") => void;
    onUndoTimelineEdit: () => void;
    onRedoTimelineEdit: () => void;
    onSelectTimelineSegment: (segmentIndex: number) => void;
    onClearSelectedSegment: () => void;
    onGenerateLyricsFromAudio: () => void;
    onAutoSyncLyrics: () => void;
    onLyricsTextChange: (value: string) => void;
    onLyricsModelChange: (value: LyricsModel) => void;
};

export function DetectorPanel(props: DetectorPanelProps) {
    if (props.viewMode !== "detail") {
        return (
            <section className="shell-card detector-panel" aria-label="song analyzer detail">
                <div className="detector-empty">
                    <h2>Chord Detector</h2>
                    <p>Pilih lagu dari Song Library, atau klik Add Song untuk menganalisa audio baru.</p>
                </div>
            </section>
        );
    }

    return (
        <section className="shell-card detector-panel" aria-label="song analyzer detail">
            <DetectorTopbar {...props} />
            <SaveMetadataForm {...props} />
            <AudioElement {...props} />
            <DetectorSummary {...props} />
            <TransportStrip {...props} />
            <input
                type="range"
                min={0}
                max={props.durationSeconds || 0}
                step={0.01}
                value={Math.min(props.currentTimeSeconds, props.durationSeconds || props.currentTimeSeconds)}
                onChange={(event) => props.onSeek(Number(event.currentTarget.value))}
                disabled={!props.canSeek}
                aria-label="Seek"
            />
            <DetectorTabs detectorTab={props.detectorTab} onTabChange={props.onTabChange} />
            {props.detectorTab === "timeline" ? <TimelineFrame {...props} /> : <LyricsFrame {...props} />}
        </section>
    );
}

function DetectorTopbar(props: DetectorPanelProps) {
    return (
        <div className="detector-topbar">
            <div className="detector-actions">
                <button type="button" className="open-btn" onClick={props.onOpenAudio}>
                    Open Audio
                </button>
                {props.canReanalyze ? (
                    <button type="button" className="open-btn" onClick={props.onReanalyze}>
                        Re-analyze
                    </button>
                ) : null}
                <button
                    type="button"
                    className="open-btn"
                    onClick={() => {
                        if (props.selectedSongId) {
                            props.onSaveAnalysis();
                        } else {
                            props.onOpenSaveForm();
                        }
                    }}
                    disabled={!props.canSaveAnalysis}
                    title={props.canSaveAnalysis ? "Save analysis changes to Song Library" : "Analyze audio first before saving"}
                >
                    {props.hasUnsavedChordEdits ? "Save Changes" : "Save"}
                </button>
                <button type="button" className="open-btn secondary-btn" onClick={() => props.onOpenExportPreview("txt")} disabled={!props.selectedSongId}>
                    Preview TXT
                </button>
                <button type="button" className="open-btn secondary-btn" onClick={() => props.onOpenExportPreview("lrc")} disabled={!props.selectedSongId}>
                    Preview LRC
                </button>
            </div>
            <p className="state-label">State: {props.state}</p>
        </div>
    );
}

function SaveMetadataForm(props: DetectorPanelProps) {
    if (!props.isSaveFormOpen) {
        return null;
    }
    return (
        <form
            className="song-metadata-form"
            aria-label="song metadata"
            onSubmit={(event) => {
                event.preventDefault();
                props.onSaveAnalysis();
            }}
        >
            <label>
                <span>Artist</span>
                <input
                    type="text"
                    value={props.songArtist}
                    onChange={(event) => props.onSongArtistChange(event.currentTarget.value)}
                    placeholder="Contoh: SR Banyak Cerita"
                    required
                />
            </label>
            <label>
                <span>Judul</span>
                <input
                    type="text"
                    value={props.songTitle}
                    onChange={(event) => props.onSongTitleChange(event.currentTarget.value)}
                    placeholder="Contoh: Album Lama"
                    required
                />
            </label>
            <button type="submit" className="open-btn">
                Save to Library
            </button>
        </form>
    );
}

function AudioElement(props: DetectorPanelProps) {
    return (
        <audio
            ref={props.audioRef}
            src={props.audioSourceUrl ?? undefined}
            data-testid="audio-player"
            onLoadedMetadata={props.onLoadedMetadata}
            onTimeUpdate={props.onTimeUpdate}
            onSeeked={props.onSeeked}
            onPlay={props.onPlay}
            onPause={props.onPause}
            onEnded={props.onEnded}
            onError={props.onAudioError}
        />
    );
}

function DetectorSummary(props: DetectorPanelProps) {
    return (
        <div className="detector-summary">
            <div>
                <p className="file-name" aria-live="polite">{props.selectedFileName}</p>
                <p className="analysis-label" aria-live="polite">
                    {props.analysisStatus}
                    {props.hasUnsavedChordEdits ? <span className="unsaved-edits-badge">Unsaved chord edits</span> : null}
                </p>
                {props.apiJobProgress && (props.apiJobProgress.status === "queued" || props.apiJobProgress.status === "running") ? (
                    <div className="job-progress" aria-label="API job progress">
                        <span>{formatApiJobKind(props.apiJobProgress.kind)}</span>
                        <div className="job-progress-track">
                            <span style={{ width: `${Math.max(5, Math.min(100, props.apiJobProgress.progress))}%` }} />
                        </div>
                        <strong>{Math.round(props.apiJobProgress.progress)}%</strong>
                        <button type="button" onClick={props.onCancelApiJob} disabled={!props.canCancelApiJob}>
                            Cancel
                        </button>
                    </div>
                ) : null}
            </div>
            <p className="active-chord" aria-live="polite" data-testid="active-chord">
                Active chord: <span>{props.activeChordLabel}</span>
            </p>
        </div>
    );
}

function TransportStrip(props: DetectorPanelProps) {
    return (
        <div className="transport-strip">
            <div className="playback-row" aria-label="playback controls">
                <button type="button" onClick={props.onTogglePlayback} disabled={!props.canUsePlayback}>
                    <span aria-hidden="true">{props.isPlaying ? "II" : "▶"}</span>
                    {props.isPlaying ? "Pause" : "Play"}
                </button>
                <button
                    type="button"
                    onClick={props.onToggleVocalHide}
                    disabled={!props.canUsePlayback || props.isRemovingVocals}
                    aria-pressed={props.isUsingInstrumentalAudio}
                >
                    <span aria-hidden="true">{props.isUsingInstrumentalAudio ? "◼" : "♪"}</span>
                    {props.isRemovingVocals ? "Preparing..." : props.isUsingInstrumentalAudio ? "Vocal: Off" : "Vocal: On"}
                </button>
                <p className="time-label" aria-live="polite">
                    {formatTime(props.currentTimeSeconds)} / {formatTime(props.durationSeconds)}
                </p>
            </div>
            <div className="transpose-controls" aria-label="transpose controls">
                <label className="transpose-key-only">
                    <input
                        type="checkbox"
                        checked={props.isTransposeKeyOnly}
                        onChange={(event) => props.onSetTransposeKeyOnly(event.currentTarget.checked)}
                        disabled={props.isPitchShiftingAudio}
                    />
                    Only Key
                </label>
                <span className="transpose-stepper">
                    <button type="button" onClick={() => props.onSetTranspose(props.transposeSemitones - 1)} disabled={props.isPitchShiftingAudio} aria-label="Transpose down">
                        -
                    </button>
                    <strong>{formatTranspose(props.transposeSemitones)}</strong>
                    <button type="button" onClick={() => props.onSetTranspose(props.transposeSemitones + 1)} disabled={props.isPitchShiftingAudio} aria-label="Transpose up">
                        +
                    </button>
                </span>
                <button type="button" onClick={() => props.onSetTranspose(0)} disabled={props.isPitchShiftingAudio}>
                    <span aria-hidden="true">↺</span>
                    Reset
                </button>
                {props.isPitchShiftingAudio ? <small>Rendering audio...</small> : null}
            </div>
        </div>
    );
}

function DetectorTabs({ detectorTab, onTabChange }: Pick<DetectorPanelProps, "detectorTab" | "onTabChange">) {
    return (
        <div className="detector-tabs" role="tablist" aria-label="Chord detector views">
            <button type="button" role="tab" aria-selected={detectorTab === "timeline"} onClick={() => onTabChange("timeline")}>
                Timeline
            </button>
            <button type="button" role="tab" aria-selected={detectorTab === "lyrics"} onClick={() => onTabChange("lyrics")}>
                Lyrics
            </button>
        </div>
    );
}

function TimelineFrame(props: DetectorPanelProps) {
    return (
        <section className="timeline-frame" aria-label="Chord timeline frame">
            <div className="chord-editor" ref={props.chordEditorRef} aria-label="Chord correction editor">
                {props.selectedTimelineSegment ? <ChordEditor {...props} /> : <span>Click chord segment untuk edit, split, atau merge.</span>}
            </div>
            <section ref={props.timelineRef} className="timeline" aria-label="Chord timeline">
                <Timeline
                    segments={props.timelineSegments}
                    durationSeconds={props.durationSeconds}
                    timelineWidthPx={props.timelineWidthPx}
                    currentTimeSeconds={props.currentTimeSeconds}
                    transposeSemitones={props.transposeSemitones}
                    selectedSegmentIndex={props.selectedSegmentIndex}
                    onSeek={props.onSeek}
                    onSelectSegment={props.onSelectTimelineSegment}
                />
            </section>
        </section>
    );
}

function ChordEditor(props: DetectorPanelProps) {
    const segment = props.selectedTimelineSegment;
    if (!segment) {
        return null;
    }
    return (
        <>
            <span>{formatTime(segment.start)} - {formatTime(segment.end)}</span>
            <input
                type="text"
                value={props.editingChord}
                onChange={(event) => props.onEditingChordChange(event.currentTarget.value)}
                onKeyDown={(event) => {
                    if (event.key === "Enter") {
                        event.preventDefault();
                        props.onApplyEditedChord();
                    }
                    if (event.key === "Escape") {
                        event.preventDefault();
                        props.onClearSelectedSegment();
                    }
                }}
                aria-label="Edit selected chord"
                list="chord-editor-options"
            />
            <datalist id="chord-editor-options">
                {EDITABLE_CHORD_OPTIONS.map((chord) => (
                    <option key={chord} value={chord} />
                ))}
            </datalist>
            <button type="button" onClick={props.onApplyEditedChord}>Apply</button>
            <button type="button" onClick={props.onSplitSelectedSegment}>Split</button>
            <button type="button" onClick={() => props.onMergeSelectedSegment("left")} disabled={props.selectedSegmentIndex === 0}>
                Merge ‹
            </button>
            <button type="button" onClick={() => props.onMergeSelectedSegment("right")} disabled={props.selectedSegmentIndex === props.timelineSegments.length - 1}>
                Merge ›
            </button>
            <button type="button" onClick={props.onUndoTimelineEdit} disabled={!props.canUndoTimelineEdit}>
                Undo
            </button>
            <button type="button" onClick={props.onRedoTimelineEdit} disabled={!props.canRedoTimelineEdit}>
                Redo
            </button>
            <small className="chord-editor-shortcuts">Enter apply, Esc cancel, Cmd/Ctrl+Z undo</small>
        </>
    );
}

function LyricsFrame(props: DetectorPanelProps) {
    return (
        <section className="lyrics-frame" aria-label="Lyrics editor">
            <div className="lyrics-editor">
                <div className="lyrics-tools">
                    <div className="lyrics-actions">
                        <button type="button" onClick={props.onGenerateLyricsFromAudio} disabled={!props.selectedFilePath || props.isGeneratingLyrics}>
                            {props.isGeneratingLyrics ? "Generating..." : "Generate Lyrics"}
                        </button>
                        <label className="lyrics-model">
                            <span>Model</span>
                            <select value={props.lyricsModel} onChange={(event) => props.onLyricsModelChange(event.currentTarget.value as LyricsModel)} disabled={props.isGeneratingLyrics}>
                                <option value="tiny">tiny</option>
                                <option value="base">base</option>
                                <option value="small">small</option>
                                <option value="medium">medium</option>
                                <option value="large">large</option>
                            </select>
                        </label>
                        <button type="button" onClick={props.onAutoSyncLyrics} disabled={!props.lyricsText.trim() || props.durationSeconds <= 0 || props.isGeneratingLyrics}>
                            Auto Sync
                        </button>
                        <button type="button" onClick={() => props.onLyricsTextChange(stripLyricsTiming(props.lyricsText))} disabled={!props.lyricsText.trim() || props.isGeneratingLyrics}>
                            Clear
                        </button>
                    </div>
                    <p>Generate dari audio, atau paste lyric polos lalu Auto Sync.</p>
                </div>
                <textarea
                    value={props.lyricsText}
                    onChange={(event) => props.onLyricsTextChange(event.currentTarget.value)}
                    placeholder={"Paste lyric polos di sini...\nbaris lyric pertama\nbaris lyric kedua"}
                    aria-label="Song lyrics"
                />
            </div>
            <div ref={props.lyricsPreviewRef} className="lyrics-preview" aria-label="Lyrics preview">
                {renderLyricsPreview(
                    props.lyricsText,
                    props.timelineSegments,
                    props.durationSeconds,
                    props.currentTimeSeconds,
                    props.transposeSemitones,
                )}
            </div>
        </section>
    );
}
