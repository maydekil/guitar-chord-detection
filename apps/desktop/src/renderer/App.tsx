import { useEffect, useMemo, useRef, useState } from "react";
import type { ChordAnalysisResult, ChordAnalysisSuccess, ChordLabel, ChordSegment } from "@gcd/shared/analysis";
import {
    buildLeadSheetAnalysis,
    extractLeadSheetLearningProfile,
    mergeLeadSheetLearningProfiles,
    selectPlayableChordSegments,
    type LeadSheetLearningProfile,
} from "@gcd/shared/lyricChordLayout";
import type { SongLibraryListResult, SongLibraryRecord, SongLibrarySummary, SongMetadataInput } from "@gcd/shared/library";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";
import { formatApiJobKind } from "./lib/apiStatus.js";
import type { ApiHealthState, ApiJobProgressEvent, DetectorTab, ExportFormat, LyricsModel, PendingConfirmation, ShellState, TimelineEditSnapshot, ViewMode } from "./appTypes.js";
import { ConfirmationModal } from "./components/ConfirmationModal.js";
import { DetectorPanel } from "./components/DetectorPanel.js";
import { ExportPreviewModal } from "./components/ExportPreviewModal.js";
import { ShellHeader } from "./components/ShellHeader.js";
import { SongLibraryPanel } from "./components/SongLibraryPanel.js";
import { toAudioSourceUrl, toFileUrl } from "./lib/audioSources.js";
import { estimateMajorKey, formatAverageConfidence, formatTranspose, normalizeEditableChord, transposeChordLabel } from "./lib/chords.js";
import { safeDownloadName, triggerDownload } from "./lib/downloads.js";
import { buildLocalChordSheetExport, buildLocalLrcExport } from "./lib/exports.js";
import { autoSyncLyrics, parseLyrics } from "./lib/lyrics.js";
import { buildSongMetadata, displaySongTitle, normalizeSongLibraryListResult } from "./lib/songLibrary.js";
import { formatTime } from "./lib/time.js";
import { usePanelResize } from "./hooks/usePanelResize.js";
import { usePlaybackController } from "./hooks/usePlaybackController.js";
import { useTimelineEditor } from "./hooks/useTimelineEditor.js";
import { useTimelineDomEffects } from "./hooks/useTimelineDomEffects.js";
import { cloneTimelineSegments, findActiveChord, normalizeTimelineSegments, snapSplitTime } from "./lib/timelineSegments.js";
const LEAD_SHEET_LEARNING_PROFILE_KEY = "gcd.leadSheetLearningProfile.v1";

function loadLeadSheetLearningProfile(): LeadSheetLearningProfile {
    try {
        const raw = window.localStorage.getItem(LEAD_SHEET_LEARNING_PROFILE_KEY);
        if (!raw) {
            return { phrasePatterns: {} };
        }
        const parsed = JSON.parse(raw) as Partial<LeadSheetLearningProfile>;
        return parsed && typeof parsed === "object" && parsed.phrasePatterns && typeof parsed.phrasePatterns === "object"
            ? { phrasePatterns: parsed.phrasePatterns as LeadSheetLearningProfile["phrasePatterns"] }
            : { phrasePatterns: {} };
    } catch {
        return { phrasePatterns: {} };
    }
}

function saveLeadSheetLearningProfile(next: LeadSheetLearningProfile): void {
    try {
        const merged = mergeLeadSheetLearningProfiles(loadLeadSheetLearningProfile(), next);
        window.localStorage.setItem(LEAD_SHEET_LEARNING_PROFILE_KEY, JSON.stringify(merged));
    } catch {
        // Learning is opportunistic; analysis/save must still succeed if storage is unavailable.
    }
}

export function App() {
    const [version, setVersion] = useState<string>("0.0.0");
    const [isApiMode, setIsApiMode] = useState<boolean>(false);
    const [apiHealthState, setApiHealthState] = useState<ApiHealthState>("checking");
    const [apiBaseUrl, setApiBaseUrl] = useState<string>("");
    const [draftApiBaseUrl, setDraftApiBaseUrl] = useState<string>("");
    const [isApiSettingsOpen, setIsApiSettingsOpen] = useState<boolean>(false);
    const [apiSettingsStatus, setApiSettingsStatus] = useState<string>("");
    const [isTestingApiConfig, setIsTestingApiConfig] = useState<boolean>(false);
    const [isSavingApiConfig, setIsSavingApiConfig] = useState<boolean>(false);
    const [state, setState] = useState<ShellState>("idle");
    const [selectedFileName, setSelectedFileName] = useState<string>("No file selected");
    const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
    const [audioSourceUrl, setAudioSourceUrl] = useState<string | null>(null);
    const [originalAudioStreamUrl, setOriginalAudioStreamUrl] = useState<string | null>(null);
    const [isMediaReady, setIsMediaReady] = useState<boolean>(false);
    const [durationSeconds, setDurationSeconds] = useState<number>(0);
    const [currentTimeSeconds, setCurrentTimeSeconds] = useState<number>(0);
    const [analysisSegmentCount, setAnalysisSegmentCount] = useState<number | null>(null);
    const [timelineSegments, setTimelineSegments] = useState<ChordSegment[]>([]);
    const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | null>(null);
    const [editingChord, setEditingChord] = useState<string>("");
    const [timelineUndoStack, setTimelineUndoStack] = useState<TimelineEditSnapshot[]>([]);
    const [timelineRedoStack, setTimelineRedoStack] = useState<TimelineEditSnapshot[]>([]);
    const [timelineWidthPx, setTimelineWidthPx] = useState<number>(0);
    const [analysisStatus, setAnalysisStatus] = useState<string>("No analysis yet");
    const [librarySongs, setLibrarySongs] = useState<SongLibrarySummary[]>([]);
    const [libraryQuery, setLibraryQuery] = useState<string>("");
    const [debouncedLibraryQuery, setDebouncedLibraryQuery] = useState<string>("");
    const [libraryPage, setLibraryPage] = useState<number>(1);
    const [libraryPageSize, setLibraryPageSize] = useState<number>(10);
    const [libraryTotal, setLibraryTotal] = useState<number>(0);
    const [libraryTotalPages, setLibraryTotalPages] = useState<number>(1);
    const [selectedSongId, setSelectedSongId] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<ViewMode>("library");
    const [libraryPanelWidthPx, setLibraryPanelWidthPx] = useState<number>(320);
    const [songTitle, setSongTitle] = useState<string>("");
    const [songArtist, setSongArtist] = useState<string>("");
    const [latestAnalysis, setLatestAnalysis] = useState<ChordAnalysisSuccess | null>(null);
    const [isSaveFormOpen, setIsSaveFormOpen] = useState<boolean>(false);
    const [hasUnsavedChordEdits, setHasUnsavedChordEdits] = useState<boolean>(false);
    const [exportPreviewFormat, setExportPreviewFormat] = useState<ExportFormat | null>(null);
    const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
    const [transposeSemitones, setTransposeSemitones] = useState<number>(0);
    const [isTransposeKeyOnly, setIsTransposeKeyOnly] = useState<boolean>(true);
    const [detectorTab, setDetectorTab] = useState<DetectorTab>("timeline");
    const [lyricsText, setLyricsText] = useState<string>("");
    const [isGeneratingLyrics, setIsGeneratingLyrics] = useState<boolean>(false);
    const [lyricsModel, setLyricsModel] = useState<LyricsModel>("small");
    const [isRemovingVocals, setIsRemovingVocals] = useState<boolean>(false);
    const [instrumentalAudioPath, setInstrumentalAudioPath] = useState<string | null>(null);
    const [instrumentalAudioStreamUrl, setInstrumentalAudioStreamUrl] = useState<string | null>(null);
    const [isUsingInstrumentalAudio, setIsUsingInstrumentalAudio] = useState<boolean>(false);
    const [isPitchShiftingAudio, setIsPitchShiftingAudio] = useState<boolean>(false);
    const [apiJobProgress, setApiJobProgress] = useState<ApiJobProgressEvent | null>(null);
    const activeRequestIdRef = useRef<number>(0);
    const timelineRef = useRef<HTMLElement | null>(null);
    const chordEditorRef = useRef<HTMLDivElement | null>(null);
    const lyricsPreviewRef = useRef<HTMLDivElement | null>(null);
    const workspaceRef = useRef<HTMLDivElement | null>(null);
    const latestApiJobProgressRef = useRef<ApiJobProgressEvent | null>(null);
    const cancelledApiJobIdsRef = useRef<Set<string>>(new Set());
    const bridge = window.gcd;
    const {
        audioRef,
        activeBlobUrlRef,
        resetPlaybackState,
        handleTogglePlayback,
        handleSeek,
        handleSetTranspose,
        handleSetTransposeKeyOnly,
        handleToggleVocalHide,
        cleanupPlaybackResources,
        cancelPitchShiftRequest,
        audioHandlers,
    } = usePlaybackController({
        bridge,
        state,
        isApiMode,
        isMediaReady,
        selectedFilePath,
        originalAudioStreamUrl,
        currentTimeSeconds,
        durationSeconds,
        transposeSemitones,
        isTransposeKeyOnly,
        instrumentalAudioPath,
        instrumentalAudioStreamUrl,
        isUsingInstrumentalAudio,
        latestApiJobProgressRef,
        cancelledApiJobIdsRef,
        setState,
        setAnalysisStatus,
        setAudioSourceUrl,
        setIsMediaReady,
        setDurationSeconds,
        setCurrentTimeSeconds,
        setTransposeSemitones,
        setIsTransposeKeyOnly,
        setIsPitchShiftingAudio,
        setIsRemovingVocals,
        setInstrumentalAudioPath,
        setInstrumentalAudioStreamUrl,
        setIsUsingInstrumentalAudio,
    });
    useEffect(() => {
        return () => {
            cleanupPlaybackResources();
        };
    }, [cleanupPlaybackResources]);
    useEffect(() => {
        if (!bridge) {
            setVersion("bridge-unavailable");
            return;
        }
        void bridge.getAppVersion().then(setVersion).catch(() => setVersion("unknown"));
        void bridge.isApiMode?.()
            .then((apiMode) => {
                setIsApiMode(apiMode);
                if (!bridge.getApiStatus) {
                    setApiHealthState(apiMode ? "checking" : "local");
                }
            })
            .catch(() => {
                setIsApiMode(false);
                setApiHealthState("local");
            });
        if (bridge.getApiStatus) {
            void bridge.getApiStatus()
                .then((status) => {
                setIsApiMode(status.mode === "api");
                setApiHealthState(status.mode === "local" ? "local" : status.healthy ? "connected" : "offline");
                setApiBaseUrl(status.baseUrl ?? "");
                setDraftApiBaseUrl(status.baseUrl ?? "");
            })
            .catch(() => setApiHealthState("offline"));
        } else if (!bridge.isApiMode) {
            setApiHealthState("local");
        }
    }, [bridge]);
    useEffect(() => {
        if (!bridge?.onApiJobProgress) {
            return;
        }
        return bridge.onApiJobProgress((event) => {
            latestApiJobProgressRef.current = event;
            setApiJobProgress(event);
            if (event.status === "queued" || event.status === "running") {
                setAnalysisStatus(`${formatApiJobKind(event.kind)} ${event.status}: ${Math.round(event.progress)}%`);
            } else if (event.status === "succeeded") {
                setAnalysisStatus(`${formatApiJobKind(event.kind)} complete.`);
            } else if (event.status === "failed") {
                const wasCancelled = cancelledApiJobIdsRef.current.has(event.id);
                setAnalysisStatus(wasCancelled ? `${formatApiJobKind(event.kind)} cancelled.` : `${formatApiJobKind(event.kind)} stopped.`);
                cancelledApiJobIdsRef.current.delete(event.id);
            }
        });
    }, [bridge]);
    useEffect(() => {
        const timeoutId = window.setTimeout(() => {
            setDebouncedLibraryQuery(libraryQuery);
        }, 180);
        return () => window.clearTimeout(timeoutId);
    }, [libraryQuery]);
    useEffect(() => {
        void refreshLibrary(debouncedLibraryQuery, libraryPage, libraryPageSize);
    }, [bridge, debouncedLibraryQuery, libraryPage, libraryPageSize]);
    const refreshLibrary = async (
        query = libraryQuery,
        page = libraryPage,
        pageSize = libraryPageSize
    ): Promise<void> => {
        if (!bridge?.listSongs) {
            setLibrarySongs([]);
            setLibraryTotal(0);
            setLibraryTotalPages(1);
            return;
        }
        const result = await bridge.listSongs({ query, page, pageSize });
        const normalized = normalizeSongLibraryListResult(result, page, pageSize);
        setLibrarySongs(normalized.records);
        setLibraryTotal(normalized.total);
        setLibraryTotalPages(normalized.totalPages);
        if (normalized.page !== page) {
            setLibraryPage(normalized.page);
        }
    };
    const commitTimelineSegments = (segments: ChordSegment[], options?: { markDirty?: boolean; pushHistory?: boolean }): void => {
        if (options?.pushHistory && timelineSegments.length > 0) {
            setTimelineUndoStack((current) => [
                ...current.slice(-24),
                {
                    segments: cloneTimelineSegments(timelineSegments),
                    selectedSegmentIndex
                }
            ]);
            setTimelineRedoStack([]);
        }
        const normalized = normalizeTimelineSegments(segments, durationSeconds);
        setTimelineSegments(normalized);
        setAnalysisSegmentCount(normalized.length);
        setLatestAnalysis((current) => current
            ? {
                ...current,
                analysis: {
                    ...current.analysis,
                    chords: normalized,
                    leadSheetChords: normalized,
                }
            }
            : current);
        if (options?.markDirty) {
            setHasUnsavedChordEdits(true);
        }
    };
    const restoreTimelineSnapshot = (snapshot: TimelineEditSnapshot): void => {
        const normalized = normalizeTimelineSegments(snapshot.segments, durationSeconds);
        setTimelineSegments(normalized);
        setAnalysisSegmentCount(normalized.length);
        setLatestAnalysis((current) => current
            ? {
                ...current,
                analysis: {
                    ...current.analysis,
                    chords: normalized,
                    leadSheetChords: normalized,
                }
            }
            : current);
        const nextSelectedIndex = snapshot.selectedSegmentIndex !== null && snapshot.selectedSegmentIndex < normalized.length
            ? snapshot.selectedSegmentIndex
            : null;
        setSelectedSegmentIndex(nextSelectedIndex);
        setEditingChord(nextSelectedIndex === null ? "" : normalized[nextSelectedIndex]?.chord ?? "");
        setHasUnsavedChordEdits(true);
    };
    const effectiveTimelineSegments = timelineSegments;
    const {
        handleSelectTimelineSegment,
        handleApplyEditedChord,
        handleSplitSelectedSegment,
        handleMergeSelectedSegment,
        handleUndoTimelineEdit,
        handleRedoTimelineEdit,
    } = useTimelineEditor({
        timelineSegments: effectiveTimelineSegments,
        selectedSegmentIndex,
        editingChord,
        currentTimeSeconds,
        timelineUndoStack,
        timelineRedoStack,
        commitTimelineSegments,
        restoreTimelineSnapshot,
        setSelectedSegmentIndex,
        setEditingChord,
        setTimelineUndoStack,
        setTimelineRedoStack,
        setAnalysisStatus,
    });
    useTimelineDomEffects({
        timelineRef,
        chordEditorRef,
        lyricsPreviewRef,
        viewMode,
        detectorTab,
        currentTimeSeconds,
        timelineSegments: effectiveTimelineSegments,
        lyricsText,
        selectedSegmentIndex,
        onClearSelectedSegment: () => {
            setSelectedSegmentIndex(null);
            setEditingChord("");
        },
        onUndoTimelineEdit: handleUndoTimelineEdit,
        onRedoTimelineEdit: handleRedoTimelineEdit,
        onTimelineWidthChange: setTimelineWidthPx,
    });
    const runAnalysis = async (audioPath: string, options?: { forceRefresh?: boolean }): Promise<void> => {
        if (!bridge?.analyzeAudio) {
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }
        setState("analyzing");
        setAnalysisSegmentCount(null);
        setTimelineSegments([]);
        setLatestAnalysis(null);
        setIsSaveFormOpen(false);
        setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
        setAnalysisStatus(isApiMode ? "Uploading audio and analyzing on API server..." : "Analyzing...");
        const requestId = activeRequestIdRef.current + 1;
        activeRequestIdRef.current = requestId;
        let analysis: ChordAnalysisResult;
        try {
            analysis = await bridge.analyzeAudio(audioPath, {
                forceRefresh: options?.forceRefresh === true
            });
        } catch {
            if (requestId !== activeRequestIdRef.current) {
                return;
            }
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }
        if (requestId !== activeRequestIdRef.current) {
            return;
        }
        if ("error" in analysis) {
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }
        const leadSheetAnalysis = buildLeadSheetAnalysis(analysis, lyricsText, loadLeadSheetLearningProfile());
        const normalizedTimelineSegments = normalizeTimelineSegments(leadSheetAnalysis.analysis.chords, analysis.source.duration);
        const segmentCount = normalizedTimelineSegments.length;
        const analysisDuration = Number.isFinite(analysis.source.duration) ? Math.max(0, analysis.source.duration) : 0;
        setState("ready");
        setLatestAnalysis(leadSheetAnalysis);
        setAnalysisSegmentCount(segmentCount);
        setTimelineSegments(normalizedTimelineSegments);
        setSelectedSegmentIndex(null);
        setEditingChord("");
        setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
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
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
            setLyricsText("");
            setInstrumentalAudioPath(null);
            setInstrumentalAudioStreamUrl(null);
            setAnalysisStatus("No analysis yet");
        }
        if (!nextPath || !bridge.analyzeAudio) {
            resetPlaybackState();
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }
        let playbackSourceUrl: string;
        if (bridge.getAudioPlaybackSource) {
            try {
                const playbackSource = await bridge.getAudioPlaybackSource(nextPath);
                playbackSourceUrl = toAudioSourceUrl(playbackSource);
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
        setOriginalAudioStreamUrl(null);
        setSelectedSongId(null);
        setAudioSourceUrl(playbackSourceUrl);
        setIsMediaReady(false);
        setDurationSeconds(0);
        setCurrentTimeSeconds(0);
        await runAnalysis(nextPath);
    };
    const handleTestApiConfig = async (): Promise<void> => {
        if (!bridge?.testApiConfig) {
            setApiSettingsStatus("API settings belum tersedia. Restart aplikasi lalu coba lagi.");
            return;
        }
        setIsTestingApiConfig(true);
        try {
            const status = await bridge.testApiConfig({ baseUrl: draftApiBaseUrl.trim() || null });
            setApiSettingsStatus(status.mode === "local"
                ? "Local mode selected."
                : status.healthy
                    ? "API connected."
                    : "API offline.");
        } catch {
            setApiSettingsStatus("Tidak bisa test API URL.");
        } finally {
            setIsTestingApiConfig(false);
        }
    };
    const handleSaveApiConfig = async (): Promise<void> => {
        if (!bridge?.saveApiConfig) {
            setApiSettingsStatus("API settings belum tersedia. Restart aplikasi lalu coba lagi.");
            return;
        }
        setIsSavingApiConfig(true);
        try {
            const status = await bridge.saveApiConfig({ baseUrl: draftApiBaseUrl.trim() || null });
            setIsApiMode(status.mode === "api");
            setApiHealthState(status.mode === "local" ? "local" : status.healthy ? "connected" : "offline");
            setApiBaseUrl(status.baseUrl ?? "");
            setDraftApiBaseUrl(status.baseUrl ?? "");
            setApiSettingsStatus(status.mode === "local"
                ? "Saved: Local mode."
                : status.healthy
                    ? "Saved: API connected."
                    : "Saved, but API offline.");
            void refreshLibrary(libraryQuery, 1, libraryPageSize);
            setLibraryPage(1);
        } catch {
            setApiSettingsStatus("Gagal menyimpan API setting.");
        } finally {
            setIsSavingApiConfig(false);
        }
    };
    const handleAddSong = (): void => {
        resetPlaybackState();
        setState("idle");
        setSelectedSongId(null);
        setSelectedFileName("No file selected");
        setSelectedFilePath(null);
        setOriginalAudioStreamUrl(null);
        setAudioSourceUrl(null);
        setAnalysisSegmentCount(null);
        setTimelineSegments([]);
        setLatestAnalysis(null);
        setIsSaveFormOpen(false);
        setAnalysisStatus("No analysis yet");
        setSongTitle("");
        setSongArtist("");
        setTransposeSemitones(0);
        setDetectorTab("timeline");
        setLyricsText("");
        setInstrumentalAudioPath(null);
        setInstrumentalAudioStreamUrl(null);
        setViewMode("detail");
    };
    const handleOpenLibrarySong = async (summary: SongLibrarySummary): Promise<void> => {
        resetPlaybackState();
        const song = bridge?.getSong ? await bridge.getSong(summary.id) : null;
        if (!song) {
            setState("error");
            setAnalysisStatus("Could not load this song detail.");
            return;
        }
        setSelectedSongId(song.id);
        setViewMode("detail");
        setSelectedFileName(displaySongTitle(song));
        setSelectedFilePath(song.audioPath);
        setOriginalAudioStreamUrl(song.audioStreamUrl ?? null);
        setSongTitle(song.title);
        setSongArtist(song.artist);
        setTransposeSemitones(0);
        setDetectorTab("timeline");
        setLyricsText(song.lyrics ?? "");
        setInstrumentalAudioPath(song.instrumentalAudioPath ?? null);
        setInstrumentalAudioStreamUrl(song.instrumentalAudioStreamUrl ?? null);
        let playbackSourceUrl: string;
        if (bridge?.getAudioPlaybackSource) {
            try {
                const playbackSource = await bridge.getAudioPlaybackSource(song.audioStreamUrl ?? song.audioPath);
                playbackSourceUrl = toAudioSourceUrl(playbackSource);
            } catch {
                setState("error");
                setAnalysisStatus("Could not play this audio file.");
                return;
            }
        } else {
            playbackSourceUrl = toFileUrl(song.audioStreamUrl ?? song.audioPath);
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
        const normalizedTimelineSegments = normalizeTimelineSegments(song.analysis.analysis.chords, song.duration);
        setAudioSourceUrl(playbackSourceUrl);
        setLatestAnalysis(buildLeadSheetAnalysis(song.analysis, song.lyrics ?? ""));
        setIsSaveFormOpen(false);
        setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
        setIsMediaReady(false);
        setDurationSeconds(song.duration);
        setCurrentTimeSeconds(0);
        setTimelineSegments(normalizedTimelineSegments);
        setSelectedSegmentIndex(null);
        setEditingChord("");
        setAnalysisSegmentCount(normalizedTimelineSegments.length);
        setAnalysisStatus(`Analysis ready: ${normalizedTimelineSegments.length} segment(s)`);
        setState("ready");
    };
    const handleReanalyze = async (): Promise<void> => {
        if (!selectedFilePath) {
            return;
        }
        await runAnalysis(selectedFilePath, { forceRefresh: true });
    };
    const handleSaveAnalysis = async (): Promise<void> => {
        if (!selectedFilePath || !latestAnalysis) {
            return;
        }
        if (!bridge?.saveSongAnalysis) {
            setAnalysisStatus("Save API belum tersedia. Restart aplikasi lalu coba Save lagi.");
            return;
        }
        const metadata = buildSongMetadata(songTitle, songArtist);
        if (!metadata) {
            setAnalysisStatus("Artist dan judul wajib diisi sebelum save.");
            return;
        }
        setPendingConfirmation({ kind: "save", metadata });
    };
    const performSaveAnalysis = async (metadata: SongMetadataInput): Promise<void> => {
        if (!selectedFilePath || !latestAnalysis || !bridge?.saveSongAnalysis) {
            setPendingConfirmation(null);
            return;
        }
        const analysisToSave: ChordAnalysisSuccess = buildLeadSheetAnalysis({
            ...latestAnalysis,
            analysis: {
                ...latestAnalysis.analysis,
                detectedChords: latestAnalysis.analysis.detectedChords ?? timelineSegments,
                chords: effectiveTimelineSegments,
            },
        }, lyricsText, loadLeadSheetLearningProfile());
        const savedSong = await bridge.saveSongAnalysis({
            audioPath: selectedFilePath,
            analysis: analysisToSave,
            metadata,
            lyrics: lyricsText,
            instrumentalAudioPath: instrumentalAudioPath ?? undefined
        });
        saveLeadSheetLearningProfile(extractLeadSheetLearningProfile(analysisToSave, lyricsText));
        setSelectedSongId(savedSong.id);
        setSongTitle(savedSong.title);
        setSongArtist(savedSong.artist);
        setSelectedFilePath(savedSong.audioPath);
        setOriginalAudioStreamUrl(savedSong.audioStreamUrl ?? null);
        const savedPlaybackPath = savedSong.audioStreamUrl ?? savedSong.audioPath;
        let savedPlaybackSourceUrl = savedSong.audioStreamUrl ?? null;
        if (bridge.getAudioPlaybackSource) {
            const playbackSource = await bridge.getAudioPlaybackSource(savedPlaybackPath);
            savedPlaybackSourceUrl = toAudioSourceUrl(playbackSource);
        } else {
            savedPlaybackSourceUrl = toFileUrl(savedPlaybackPath);
        }
        if (savedPlaybackSourceUrl) {
            if (activeBlobUrlRef.current && typeof URL.revokeObjectURL === "function") {
                URL.revokeObjectURL(activeBlobUrlRef.current);
                activeBlobUrlRef.current = null;
            }
            if (savedPlaybackSourceUrl.startsWith("blob:")) {
                activeBlobUrlRef.current = savedPlaybackSourceUrl;
            }
            setAudioSourceUrl(savedPlaybackSourceUrl);
            setIsMediaReady(false);
            window.setTimeout(() => audioRef.current?.load(), 0);
        }
        setInstrumentalAudioPath(savedSong.instrumentalAudioPath ?? null);
        setInstrumentalAudioStreamUrl(savedSong.instrumentalAudioStreamUrl ?? null);
        setLatestAnalysis(savedSong.analysis);
        const savedTimelineSegments = normalizeTimelineSegments(selectPlayableChordSegments(savedSong.analysis), savedSong.duration);
        setTimelineSegments(savedTimelineSegments);
        setAnalysisSegmentCount(savedTimelineSegments.length);
        setSelectedFileName(displaySongTitle(savedSong));
        setIsSaveFormOpen(false);
        setPendingConfirmation(null);
        setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
        setAnalysisStatus(selectedSongId ? "Saved changes to Song Library." : "Saved to Song Library.");
        await refreshLibrary(libraryQuery, libraryPage, libraryPageSize);
    };
    const handleAutoSyncLyrics = (): void => {
        const syncedLyrics = autoSyncLyrics(lyricsText, durationSeconds, timelineSegments);
        if (!syncedLyrics) {
            setAnalysisStatus("Paste lyric polos dulu, lalu klik Auto Sync Lyrics.");
            return;
        }
        setLyricsText(syncedLyrics);
        setAnalysisStatus("Lyrics auto-synced from chord timeline.");
    };
    const handleGenerateLyricsFromAudio = async (): Promise<void> => {
        if (!selectedFilePath) {
            setAnalysisStatus("Open audio dulu sebelum generate lyrics.");
            return;
        }
        if (!bridge?.generateLyricsFromAudio) {
            setAnalysisStatus("Generate lyrics API belum tersedia. Restart aplikasi lalu coba lagi.");
            return;
        }
        setIsGeneratingLyrics(true);
        setAnalysisStatus(isApiMode ? "Uploading audio and generating lyrics on API server..." : "Generating lyrics from audio...");
        let result: LyricsTranscriptionResult;
        try {
            result = await bridge.generateLyricsFromAudio(selectedFilePath, { model: lyricsModel });
        } catch {
            const latestJob = latestApiJobProgressRef.current;
            if (latestJob && cancelledApiJobIdsRef.current.has(latestJob.id)) {
                return;
            }
            setIsGeneratingLyrics(false);
            setAnalysisStatus("Gagal generate lyrics dari audio.");
            return;
        }
        setIsGeneratingLyrics(false);
        if ("error" in result) {
            setAnalysisStatus(result.error.message);
            return;
        }
        setLyricsText(result.lyrics.text);
        setDetectorTab("lyrics");
        setAnalysisStatus(`Lyrics generated${result.lyrics.language ? ` (${result.lyrics.language})` : ""}.`);
    };
    const handleDeleteLibrarySong = async (song: SongLibrarySummary): Promise<void> => {
        if (!bridge?.deleteSong) {
            setAnalysisStatus("Delete API belum tersedia. Restart aplikasi lalu coba Delete lagi.");
            return;
        }
        setPendingConfirmation({ kind: "delete", song });
    };
    const performDeleteLibrarySong = async (song: SongLibrarySummary): Promise<void> => {
        if (!bridge?.deleteSong) {
            setPendingConfirmation(null);
            return;
        }
        await bridge.deleteSong(song.id);
        setPendingConfirmation(null);
        if (selectedSongId === song.id) {
            resetPlaybackState();
            setSelectedSongId(null);
            setSelectedFileName("No file selected");
            setSelectedFilePath(null);
            setOriginalAudioStreamUrl(null);
            setAudioSourceUrl(null);
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setHasUnsavedChordEdits(false);
            setTimelineUndoStack([]);
            setTimelineRedoStack([]);
            setSongTitle("");
            setSongArtist("");
            setLyricsText("");
            setInstrumentalAudioPath(null);
            setInstrumentalAudioStreamUrl(null);
            setAnalysisStatus("Deleted from Song Library.");
            setState("idle");
            setViewMode("library");
        }
        await refreshLibrary(libraryQuery, libraryPage, libraryPageSize);
    };
    const handleConfirmAction = async (): Promise<void> => {
        if (!pendingConfirmation) {
            return;
        }
        if (pendingConfirmation.kind === "save") {
            await performSaveAnalysis(pendingConfirmation.metadata);
            return;
        }
        await performDeleteLibrarySong(pendingConfirmation.song);
    };
    const handleOpenExportPreview = (format: ExportFormat): void => {
        if (!selectedSongId || !latestAnalysis) {
            setAnalysisStatus("Save song dulu sebelum export.");
            return;
        }
        setExportPreviewFormat(format);
    };
    const handleExportSong = async (format: ExportFormat): Promise<void> => {
        if (!selectedSongId || !latestAnalysis) {
            setAnalysisStatus("Save song dulu sebelum export.");
            return;
        }
        if (isApiMode && bridge?.getSongExportUrl) {
            const result = await bridge.getSongExportUrl(selectedSongId, format, transposeSemitones);
            if (result.url) {
                triggerDownload(result.url);
                setAnalysisStatus(`Export ${format.toUpperCase()} ready.`);
                return;
            }
        }
        const content = format === "lrc"
            ? buildLocalLrcExport(lyricsText, timelineSegments, transposeSemitones)
            : buildLocalChordSheetExport(songArtist, songTitle, durationSeconds, timelineSegments, lyricsText, transposeSemitones);
        triggerDownload(
            URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" })),
            `${safeDownloadName(songArtist)}-${safeDownloadName(songTitle)}.${format}`
        );
        setAnalysisStatus(`Export ${format.toUpperCase()} ready.`);
    };
    const handleCancelApiJob = async (): Promise<void> => {
        if (!apiJobProgress || !bridge?.cancelApiJob) {
            return;
        }
        try {
            cancelledApiJobIdsRef.current.add(apiJobProgress.id);
            await bridge.cancelApiJob(apiJobProgress.id);
            setAnalysisStatus(`${formatApiJobKind(apiJobProgress.kind)} cancelled.`);
            setApiJobProgress({
                ...apiJobProgress,
                status: "failed",
                progress: 100
            });
            if (apiJobProgress.kind === "lyrics") {
                setIsGeneratingLyrics(false);
            }
            if (apiJobProgress.kind === "vocals") {
                setIsRemovingVocals(false);
            }
            if (apiJobProgress.kind === "pitch-shift") {
                cancelPitchShiftRequest();
                setIsPitchShiftingAudio(false);
            }
            if (apiJobProgress.kind === "analysis") {
                activeRequestIdRef.current += 1;
                setState("ready");
            }
        } catch {
            cancelledApiJobIdsRef.current.delete(apiJobProgress.id);
            setAnalysisStatus("Gagal cancel job API.");
        }
    };
    const handlePanelResizeStart = usePanelResize(workspaceRef, setLibraryPanelWidthPx);
    const isPlayableState = state === "ready" || state === "playing" || state === "paused";
    const canUsePlayback = Boolean(audioSourceUrl) && Number.isFinite(durationSeconds) && durationSeconds > 0 && isPlayableState;
    const canSeek = canUsePlayback && isMediaReady;
    const isPlaying = state === "playing";
    const canReanalyze = Boolean(selectedFilePath) && Boolean(bridge?.analyzeAudio);
    const canSaveAnalysis = Boolean(selectedFilePath) && Boolean(latestAnalysis);
    const canCancelApiJob = Boolean(apiJobProgress && (apiJobProgress.status === "queued" || apiJobProgress.status === "running"));
    const selectedTimelineSegment = selectedSegmentIndex === null ? null : effectiveTimelineSegments[selectedSegmentIndex] ?? null;
    const canUndoTimelineEdit = timelineUndoStack.length > 0;
    const canRedoTimelineEdit = timelineRedoStack.length > 0;
    const exportPreviewContent = exportPreviewFormat === null
        ? ""
        : exportPreviewFormat === "lrc"
            ? buildLocalLrcExport(lyricsText, effectiveTimelineSegments, transposeSemitones)
            : buildLocalChordSheetExport(songArtist, songTitle, durationSeconds, effectiveTimelineSegments, lyricsText, transposeSemitones);
    const activeChord = findActiveChord(effectiveTimelineSegments, currentTimeSeconds);
    const activeChordLabel = activeChord ? transposeChordLabel(activeChord, transposeSemitones) : "None";
    return (
        <main className="shell" data-state={state}>
            <ShellHeader
                version={version}
                apiHealthState={apiHealthState}
                apiBaseUrl={apiBaseUrl}
                draftApiBaseUrl={draftApiBaseUrl}
                isApiSettingsOpen={isApiSettingsOpen}
                apiSettingsStatus={apiSettingsStatus}
                isTestingApiConfig={isTestingApiConfig}
                isSavingApiConfig={isSavingApiConfig}
                onToggleApiSettings={() => setIsApiSettingsOpen((current) => !current)}
                onDraftApiBaseUrlChange={setDraftApiBaseUrl}
                onTestApiConfig={() => void handleTestApiConfig()}
                onSaveApiConfig={() => void handleSaveApiConfig()}
            />
            <div className="workspace-panels" ref={workspaceRef}>
                <SongLibraryPanel
                    songs={librarySongs}
                    total={libraryTotal}
                    query={libraryQuery}
                    page={libraryPage}
                    pageSize={libraryPageSize}
                    totalPages={libraryTotalPages}
                    selectedSongId={selectedSongId}
                    widthPx={libraryPanelWidthPx}
                    canAddSong={Boolean(bridge)}
                    onAddSong={handleAddSong}
                    onQueryChange={(value) => {
                        setLibraryQuery(value);
                        setLibraryPage(1);
                    }}
                    onPageChange={setLibraryPage}
                    onPageSizeChange={(value) => {
                        setLibraryPageSize(value);
                        setLibraryPage(1);
                    }}
                    onOpenSong={(song) => void handleOpenLibrarySong(song)}
                    onDeleteSong={(song) => void handleDeleteLibrarySong(song)}
                />
                <div
                    className="panel-resizer"
                    role="separator"
                    aria-label="Resize panels"
                    aria-orientation="vertical"
                    onMouseDown={handlePanelResizeStart}
                />
                <DetectorPanel
                    viewMode={viewMode}
                    state={state}
                    selectedFileName={selectedFileName}
                    selectedFilePath={selectedFilePath}
                    audioSourceUrl={audioSourceUrl}
                    durationSeconds={durationSeconds}
                    currentTimeSeconds={currentTimeSeconds}
                    analysisStatus={analysisStatus}
                    activeChordLabel={activeChordLabel}
                    hasUnsavedChordEdits={hasUnsavedChordEdits}
                    apiJobProgress={apiJobProgress}
                    canCancelApiJob={canCancelApiJob}
                    canReanalyze={canReanalyze}
                    canSaveAnalysis={canSaveAnalysis}
                    canUsePlayback={canUsePlayback}
                    canSeek={canSeek}
                    isPlaying={isPlaying}
                    selectedSongId={selectedSongId}
                    isSaveFormOpen={isSaveFormOpen}
                    songArtist={songArtist}
                    songTitle={songTitle}
                    transposeSemitones={transposeSemitones}
                    isTransposeKeyOnly={isTransposeKeyOnly}
                    isPitchShiftingAudio={isPitchShiftingAudio}
                    detectorTab={detectorTab}
                    timelineSegments={effectiveTimelineSegments}
                    timelineWidthPx={timelineWidthPx}
                    selectedSegmentIndex={selectedSegmentIndex}
                    selectedTimelineSegment={selectedTimelineSegment}
                    editingChord={editingChord}
                    canUndoTimelineEdit={canUndoTimelineEdit}
                    canRedoTimelineEdit={canRedoTimelineEdit}
                    lyricsText={lyricsText}
                    lyricsModel={lyricsModel}
                    isGeneratingLyrics={isGeneratingLyrics}
                    isRemovingVocals={isRemovingVocals}
                    isUsingInstrumentalAudio={isUsingInstrumentalAudio}
                    audioRef={audioRef}
                    timelineRef={timelineRef}
                    chordEditorRef={chordEditorRef}
                    lyricsPreviewRef={lyricsPreviewRef}
                    onOpenAudio={() => void handleOpenAudio()}
                    onReanalyze={() => void handleReanalyze()}
                    onSaveAnalysis={() => void handleSaveAnalysis()}
                    onOpenSaveForm={() => setIsSaveFormOpen(true)}
                    onOpenExportPreview={handleOpenExportPreview}
                    onLoadedMetadata={audioHandlers.onLoadedMetadata}
                    onCanPlay={audioHandlers.onCanPlay}
                    onTimeUpdate={audioHandlers.onTimeUpdate}
                    onSeeked={audioHandlers.onSeeked}
                    onPlay={audioHandlers.onPlay}
                    onPause={audioHandlers.onPause}
                    onEnded={audioHandlers.onEnded}
                    onAudioError={audioHandlers.onAudioError}
                    onCancelApiJob={() => void handleCancelApiJob()}
                    onTogglePlayback={() => void handleTogglePlayback()}
                    onToggleVocalHide={() => void handleToggleVocalHide()}
                    onSetTranspose={(value) => void handleSetTranspose(value)}
                    onSetTransposeKeyOnly={(value) => void handleSetTransposeKeyOnly(value)}
                    onSeek={handleSeek}
                    onTabChange={setDetectorTab}
                    onSongArtistChange={setSongArtist}
                    onSongTitleChange={setSongTitle}
                    onEditingChordChange={setEditingChord}
                    onApplyEditedChord={handleApplyEditedChord}
                    onSplitSelectedSegment={handleSplitSelectedSegment}
                    onMergeSelectedSegment={handleMergeSelectedSegment}
                    onUndoTimelineEdit={handleUndoTimelineEdit}
                    onRedoTimelineEdit={handleRedoTimelineEdit}
                    onSelectTimelineSegment={handleSelectTimelineSegment}
                    onClearSelectedSegment={() => {
                        setSelectedSegmentIndex(null);
                        setEditingChord("");
                    }}
                    onGenerateLyricsFromAudio={() => void handleGenerateLyricsFromAudio()}
                    onAutoSyncLyrics={handleAutoSyncLyrics}
                    onLyricsTextChange={setLyricsText}
                    onLyricsModelChange={setLyricsModel}
                />
            </div>
            {exportPreviewFormat ? (
                <ExportPreviewModal
                    format={exportPreviewFormat}
                    artist={songArtist}
                    title={songTitle}
                    transposeSemitones={transposeSemitones}
                    content={exportPreviewContent}
                    chordSegmentCount={effectiveTimelineSegments.length}
                    hasLyrics={Boolean(lyricsText.trim())}
                    onDownload={(format) => void handleExportSong(format)}
                    onClose={() => setExportPreviewFormat(null)}
                />
            ) : null}
            {pendingConfirmation ? (
                <ConfirmationModal
                    confirmation={pendingConfirmation}
                    onCancel={() => setPendingConfirmation(null)}
                    onConfirm={() => void handleConfirmAction()}
                />
            ) : null}
        </main>
    );
}
