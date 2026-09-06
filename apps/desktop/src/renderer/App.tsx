import { useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import type { ChordAnalysisResult, ChordAnalysisSuccess, ChordSegment } from "@gcd/shared/analysis";
import type { SongLibraryListResult, SongLibraryRecord, SongMetadataInput } from "@gcd/shared/library";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

type ShellState = "idle" | "loading-file" | "analyzing" | "ready" | "playing" | "paused" | "error";
type ViewMode = "library" | "detail";
type DetectorTab = "timeline" | "lyrics";
type LyricsModel = "tiny" | "base" | "small" | "medium" | "large";
type ApiHealthState = "local" | "checking" | "connected" | "offline";
type ApiJobProgressEvent = {
    id: string;
    kind: "analysis" | "lyrics" | "vocals" | "pitch-shift";
    status: "queued" | "running" | "succeeded" | "failed";
    progress: number;
};

export function App() {
    const [version, setVersion] = useState<string>("0.0.0");
    const [isApiMode, setIsApiMode] = useState<boolean>(false);
    const [apiHealthState, setApiHealthState] = useState<ApiHealthState>("checking");
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
    const [librarySongs, setLibrarySongs] = useState<SongLibraryRecord[]>([]);
    const [libraryQuery, setLibraryQuery] = useState<string>("");
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
    const [transposeSemitones, setTransposeSemitones] = useState<number>(0);
    const [detectorTab, setDetectorTab] = useState<DetectorTab>("timeline");
    const [lyricsText, setLyricsText] = useState<string>("");
    const [isGeneratingLyrics, setIsGeneratingLyrics] = useState<boolean>(false);
    const [lyricsModel, setLyricsModel] = useState<LyricsModel>("small");
    const [isVocalHidden, setIsVocalHidden] = useState<boolean>(false);
    const [isRemovingVocals, setIsRemovingVocals] = useState<boolean>(false);
    const [instrumentalAudioPath, setInstrumentalAudioPath] = useState<string | null>(null);
    const [instrumentalAudioStreamUrl, setInstrumentalAudioStreamUrl] = useState<string | null>(null);
    const [isUsingInstrumentalAudio, setIsUsingInstrumentalAudio] = useState<boolean>(false);
    const [isPitchShiftingAudio, setIsPitchShiftingAudio] = useState<boolean>(false);
    const [apiJobProgress, setApiJobProgress] = useState<ApiJobProgressEvent | null>(null);
    const activeRequestIdRef = useRef<number>(0);
    const activePitchShiftRequestIdRef = useRef<number>(0);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const activeBlobUrlRef = useRef<string | null>(null);
    const timelineRef = useRef<HTMLElement | null>(null);
    const lyricsPreviewRef = useRef<HTMLDivElement | null>(null);
    const workspaceRef = useRef<HTMLDivElement | null>(null);
    const pendingAudioSwitchRef = useRef<{ time: number; autoplay: boolean; fadeIn: boolean } | null>(null);
    const audioFadeTimerRef = useRef<number | null>(null);
    const pendingSeekTimeRef = useRef<number | null>(null);
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
    }, [viewMode]);

    useEffect(() => {
        const timeline = timelineRef.current;
        if (!timeline || viewMode !== "detail") {
            return;
        }

        const activeSegment = timeline.querySelector<HTMLElement>('[data-testid="timeline-segment"][data-active="true"]');
        if (typeof activeSegment?.scrollIntoView === "function") {
            activeSegment.scrollIntoView({
                block: "nearest",
                inline: "nearest"
            });
        }
    }, [currentTimeSeconds, timelineSegments, viewMode]);

    useEffect(() => {
        const lyricsPreview = lyricsPreviewRef.current;
        if (!lyricsPreview || viewMode !== "detail" || detectorTab !== "lyrics") {
            return;
        }

        const activeLine = lyricsPreview.querySelector<HTMLElement>('[data-active="true"]');
        if (typeof activeLine?.scrollIntoView === "function") {
            activeLine.scrollIntoView({
                block: "center",
                inline: "nearest"
            });
        }
    }, [currentTimeSeconds, detectorTab, lyricsText, viewMode]);

    useEffect(() => {
        return () => {
            if (audioFadeTimerRef.current !== null) {
                window.clearInterval(audioFadeTimerRef.current);
                audioFadeTimerRef.current = null;
            }
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
            setApiJobProgress(event);
            if (event.status === "queued" || event.status === "running") {
                setAnalysisStatus(`${formatApiJobKind(event.kind)} ${event.status}: ${Math.round(event.progress)}%`);
            }
        });
    }, [bridge]);

    useEffect(() => {
        void refreshLibrary(libraryQuery, libraryPage, libraryPageSize);
    }, [bridge, libraryQuery, libraryPage, libraryPageSize]);

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

    const runAnalysis = async (audioPath: string, options?: { forceRefresh?: boolean }): Promise<void> => {
        if (!bridge?.analyzeAudio) {
            setState("error");
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }

        setState("analyzing");
        setAnalysisSegmentCount(null);
        setTimelineSegments([]);
        setLatestAnalysis(null);
        setIsSaveFormOpen(false);
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
            setAnalysisStatus("Could not analyze this audio file.");
            return;
        }

        const normalizedTimelineSegments = normalizeTimelineSegments(analysis.analysis.chords, analysis.source.duration);
        const segmentCount = normalizedTimelineSegments.length;
        const analysisDuration = Number.isFinite(analysis.source.duration) ? Math.max(0, analysis.source.duration) : 0;
        setState("ready");
        setLatestAnalysis(analysis);
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
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
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
        setSelectedSongId(null);
        setAudioSourceUrl(playbackSourceUrl);
        setIsMediaReady(false);
        setDurationSeconds(0);
        setCurrentTimeSeconds(0);

        await runAnalysis(nextPath);
    };

    const handleAddSong = (): void => {
        resetPlaybackState();
        setState("idle");
        setSelectedSongId(null);
        setSelectedFileName("No file selected");
        setSelectedFilePath(null);
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

    const handleOpenLibrarySong = async (song: SongLibraryRecord): Promise<void> => {
        resetPlaybackState();
        setSelectedSongId(song.id);
        setViewMode("detail");
        setSelectedFileName(displaySongTitle(song));
        setSelectedFilePath(song.audioPath);
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
        setLatestAnalysis(song.analysis);
        setIsSaveFormOpen(false);
        setIsMediaReady(false);
        setDurationSeconds(song.duration);
        setCurrentTimeSeconds(0);
        setTimelineSegments(normalizedTimelineSegments);
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

        const savedSong = await bridge.saveSongAnalysis({
            audioPath: selectedFilePath,
            analysis: latestAnalysis,
            metadata,
            lyrics: lyricsText,
            instrumentalAudioPath: instrumentalAudioPath ?? undefined
        });
        setSelectedSongId(savedSong.id);
        setSongTitle(savedSong.title);
        setSongArtist(savedSong.artist);
        setSelectedFilePath(savedSong.audioPath);
        setInstrumentalAudioPath(savedSong.instrumentalAudioPath ?? null);
        setInstrumentalAudioStreamUrl(savedSong.instrumentalAudioStreamUrl ?? null);
        setSelectedFileName(displaySongTitle(savedSong));
        setIsSaveFormOpen(false);
        setAnalysisStatus("Saved to Song Library.");
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

    const handleDeleteLibrarySong = async (song: SongLibraryRecord): Promise<void> => {
        if (!bridge?.deleteSong) {
            setAnalysisStatus("Delete API belum tersedia. Restart aplikasi lalu coba Delete lagi.");
            return;
        }

        await bridge.deleteSong(song.id);
        if (selectedSongId === song.id) {
            resetPlaybackState();
            setSelectedSongId(null);
            setSelectedFileName("No file selected");
            setSelectedFilePath(null);
            setAudioSourceUrl(null);
            setAnalysisSegmentCount(null);
            setTimelineSegments([]);
            setLatestAnalysis(null);
            setIsSaveFormOpen(false);
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

    const handleExportSong = async (format: "txt" | "lrc"): Promise<void> => {
        if (!selectedSongId || !latestAnalysis) {
            setAnalysisStatus("Save song dulu sebelum export.");
            return;
        }

        if (isApiMode && bridge?.getSongExportUrl) {
            const result = await bridge.getSongExportUrl(selectedSongId, format);
            if (result.url) {
                triggerDownload(result.url);
                setAnalysisStatus(`Export ${format.toUpperCase()} ready.`);
                return;
            }
        }

        const content = format === "lrc"
            ? buildLocalLrcExport(lyricsText, timelineSegments)
            : buildLocalChordSheetExport(songArtist, songTitle, durationSeconds, timelineSegments, lyricsText);
        triggerDownload(
            URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" })),
            `${safeDownloadName(songArtist)}-${safeDownloadName(songTitle)}.${format}`
        );
        setAnalysisStatus(`Export ${format.toUpperCase()} ready.`);
    };

    const triggerDownload = (url: string, fileName?: string): void => {
        const anchor = document.createElement("a");
        anchor.href = url;
        if (fileName) {
            anchor.download = fileName;
        }
        document.body.append(anchor);
        anchor.click();
        anchor.remove();
        if (url.startsWith("blob:")) {
            window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        }
    };

    const resetPlaybackState = (): void => {
        pendingSeekTimeRef.current = null;
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
        setIsVocalHidden(false);
        setIsUsingInstrumentalAudio(false);
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

    const handleTogglePlayback = async (): Promise<void> => {
        if (isPlaying) {
            handlePause();
            return;
        }
        await handlePlay();
    };

    const handleSeek = (value: number): void => {
        const audio = audioRef.current;
        if (!audio) {
            return;
        }

        const safeValue = Math.max(0, Math.min(value, durationSeconds || value));
        pendingSeekTimeRef.current = safeValue;
        audio.currentTime = safeValue;
        setCurrentTimeSeconds(safeValue);
    };

    const handleSetTranspose = async (nextSemitones: number): Promise<void> => {
        const safeSemitones = Math.max(-11, Math.min(11, Math.trunc(nextSemitones)));
        setTransposeSemitones(safeSemitones);
        await applyTransposeAudio(safeSemitones);
    };

    const applyTransposeAudio = async (semitones: number): Promise<void> => {
        const baseAudioPath = isUsingInstrumentalAudio ? instrumentalAudioPath : selectedFilePath;
        if (!baseAudioPath || !bridge?.getAudioPlaybackSource) {
            return;
        }

        const requestId = activePitchShiftRequestIdRef.current + 1;
        activePitchShiftRequestIdRef.current = requestId;
        setIsPitchShiftingAudio(semitones !== 0);
        setAnalysisStatus(semitones === 0
            ? "Transpose reset."
            : isApiMode
                ? `Rendering transpose audio ${formatTranspose(semitones)} on API server...`
                : `Rendering transpose audio ${formatTranspose(semitones)}...`);

        let playbackPath = baseAudioPath;
        if (semitones !== 0) {
            if (!bridge.pitchShiftAudio) {
                setIsPitchShiftingAudio(false);
                setAnalysisStatus("Pitch shift API belum tersedia. Restart aplikasi lalu coba lagi.");
                return;
            }

            let result: PitchShiftResult;
            try {
                result = await bridge.pitchShiftAudio(baseAudioPath, { semitones });
            } catch {
                setIsPitchShiftingAudio(false);
                setAnalysisStatus("Gagal menyesuaikan transpose audio.");
                return;
            }
            if (requestId !== activePitchShiftRequestIdRef.current) {
                return;
            }
            if ("error" in result) {
                setIsPitchShiftingAudio(false);
                setAnalysisStatus(result.error.message);
                return;
            }
            playbackPath = result.audio.streamUrl ?? result.audio.path;
        }

        try {
            const playbackSource = await bridge.getAudioPlaybackSource(playbackPath);
            if (requestId !== activePitchShiftRequestIdRef.current) {
                return;
            }
            const audio = audioRef.current;
            const previousTime = audio?.currentTime ?? currentTimeSeconds;
            const wasPlaying = state === "playing";
            await replaceAudioSource(toAudioSourceUrl(playbackSource), previousTime, wasPlaying);
            setAnalysisStatus(semitones === 0 ? "Transpose reset." : `Audio transposed ${formatTranspose(semitones)}.`);
        } catch {
            setAnalysisStatus("Gagal memuat audio transpose.");
        } finally {
            if (requestId === activePitchShiftRequestIdRef.current) {
                setIsPitchShiftingAudio(false);
            }
        }
    };

    const handleToggleVocalHide = async (): Promise<void> => {
        if (isUsingInstrumentalAudio) {
            await handleUseOriginalAudio();
            return;
        }

        if (instrumentalAudioPath) {
            await handleUseInstrumentalAudio(instrumentalAudioPath, instrumentalAudioStreamUrl, "Karaoke audio enabled.");
            return;
        }
        if (!selectedFilePath) {
            setAnalysisStatus("Open audio dulu sebelum Vocal Off.");
            return;
        }
        if (!bridge?.removeVocals || !bridge?.getAudioPlaybackSource) {
            setAnalysisStatus("Vocal Off API belum tersedia. Restart aplikasi lalu coba lagi.");
            return;
        }

        setIsRemovingVocals(true);
        setAnalysisStatus(isApiMode ? "Uploading audio and preparing Vocal Off on API server..." : "Preparing Vocal Off...");
        let result: VocalRemovalResult;
        try {
            result = await bridge.removeVocals(selectedFilePath);
        } catch {
            setIsRemovingVocals(false);
            setAnalysisStatus("Vocal Off gagal dibuat.");
            return;
        }
        setIsRemovingVocals(false);

        if ("error" in result) {
            setAnalysisStatus(result.error.message);
            return;
        }

        setInstrumentalAudioPath(result.audio.path);
        setInstrumentalAudioStreamUrl(result.audio.streamUrl ?? null);
        await handleUseInstrumentalAudio(result.audio.path, result.audio.streamUrl ?? null, "Vocal Off ready.");
    };

    const handleUseInstrumentalAudio = async (audioPath: string, audioStreamUrl: string | null, statusMessage: string): Promise<void> => {
        if (!bridge?.getAudioPlaybackSource) {
            return;
        }

        const audio = audioRef.current;
        const previousTime = audio?.currentTime ?? currentTimeSeconds;
        const wasPlaying = state === "playing";
        let playbackPath = audioPath;
        if (transposeSemitones !== 0 && bridge.pitchShiftAudio) {
            const shifted = await bridge.pitchShiftAudio(audioPath, { semitones: transposeSemitones });
            if (!("error" in shifted)) {
                playbackPath = shifted.audio.streamUrl ?? shifted.audio.path;
            }
        }
        const playbackSource = await bridge.getAudioPlaybackSource(transposeSemitones === 0 ? audioStreamUrl ?? playbackPath : playbackPath);
        await replaceAudioSource(toAudioSourceUrl(playbackSource), previousTime, wasPlaying);
        setIsUsingInstrumentalAudio(true);
        setIsVocalHidden(true);
        setAnalysisStatus(statusMessage);
    };

    const handleUseOriginalAudio = async (): Promise<void> => {
        if (!selectedFilePath || !bridge?.getAudioPlaybackSource) {
            return;
        }

        const audio = audioRef.current;
        const previousTime = audio?.currentTime ?? currentTimeSeconds;
        const wasPlaying = state === "playing";
        let playbackPath = selectedFilePath;
        if (transposeSemitones !== 0 && bridge.pitchShiftAudio) {
            const shifted = await bridge.pitchShiftAudio(selectedFilePath, { semitones: transposeSemitones });
            if (!("error" in shifted)) {
                playbackPath = shifted.audio.streamUrl ?? shifted.audio.path;
            }
        }
        const playbackSource = await bridge.getAudioPlaybackSource(playbackPath);
        await replaceAudioSource(toAudioSourceUrl(playbackSource), previousTime, wasPlaying);
        setIsUsingInstrumentalAudio(false);
        setIsVocalHidden(false);
        setAnalysisStatus("Original audio restored.");
    };

    const replaceAudioSource = async (nextSourceUrl: string, seekTime: number, autoplay: boolean): Promise<void> => {
        const audio = audioRef.current;
        const previousBlobUrl = activeBlobUrlRef.current;
        pendingSeekTimeRef.current = null;
        if (audio) {
            await fadeAudioVolume(audio, 0, 150);
        }
        pendingAudioSwitchRef.current = {
            time: Math.max(0, seekTime),
            autoplay,
            fadeIn: true
        };
        setIsMediaReady(false);
        setState("paused");
        setAudioSourceUrl(nextSourceUrl);
        activeBlobUrlRef.current = nextSourceUrl.startsWith("blob:") ? nextSourceUrl : null;
        if (previousBlobUrl && previousBlobUrl !== nextSourceUrl && typeof URL.revokeObjectURL === "function") {
            URL.revokeObjectURL(previousBlobUrl);
        }
    };

    const fadeAudioVolume = (audio: HTMLAudioElement, targetVolume: number, durationMs: number): Promise<void> => {
        if (audioFadeTimerRef.current !== null) {
            window.clearInterval(audioFadeTimerRef.current);
            audioFadeTimerRef.current = null;
        }

        const startVolume = Number.isFinite(audio.volume) ? audio.volume : 1;
        const safeTargetVolume = Math.max(0, Math.min(1, targetVolume));
        const startedAt = window.performance.now();

        return new Promise((resolve) => {
            audioFadeTimerRef.current = window.setInterval(() => {
                const progress = Math.min(1, (window.performance.now() - startedAt) / Math.max(1, durationMs));
                audio.volume = startVolume + (safeTargetVolume - startVolume) * progress;
                if (progress >= 1) {
                    if (audioFadeTimerRef.current !== null) {
                        window.clearInterval(audioFadeTimerRef.current);
                        audioFadeTimerRef.current = null;
                    }
                    audio.volume = safeTargetVolume;
                    resolve();
                }
            }, 16);
        });
    };

    const handlePanelResizeStart = (event: MouseEvent<HTMLDivElement>): void => {
        event.preventDefault();
        const workspace = workspaceRef.current;
        if (!workspace) {
            return;
        }

        const bounds = workspace.getBoundingClientRect();
        const minLeftWidth = 260;
        const minRightWidth = 380;
        const gapWidth = 16;
        const maxLeftWidth = Math.max(minLeftWidth, bounds.width - minRightWidth - gapWidth);

        const handleMove = (moveEvent: globalThis.MouseEvent): void => {
            const nextWidth = Math.max(minLeftWidth, Math.min(moveEvent.clientX - bounds.left, maxLeftWidth));
            setLibraryPanelWidthPx(nextWidth);
        };

        const handleUp = (): void => {
            window.removeEventListener("mousemove", handleMove);
            window.removeEventListener("mouseup", handleUp);
            document.body.classList.remove("is-resizing-panels");
        };

        document.body.classList.add("is-resizing-panels");
        window.addEventListener("mousemove", handleMove);
        window.addEventListener("mouseup", handleUp);
    };

    const isPlayableState = state === "ready" || state === "playing" || state === "paused";
    const canUsePlayback = Boolean(audioSourceUrl) && Number.isFinite(durationSeconds) && durationSeconds > 0 && isPlayableState;
    const canSeek = canUsePlayback && isMediaReady;
    const isPlaying = state === "playing";
    const canReanalyze = Boolean(selectedFilePath) && Boolean(bridge?.analyzeAudio);
    const canSaveAnalysis = Boolean(selectedFilePath) && Boolean(latestAnalysis);
    const activeChord = findActiveChord(timelineSegments, currentTimeSeconds);
    const activeChordLabel = activeChord ? transposeChordLabel(activeChord, transposeSemitones) : "None";

    return (
        <main className="shell" data-state={state}>
            <header className="shell-header">
                <div className="shell-title-row">
                    <h1>Guitar Chord Detector</h1>
                    <span className={`mode-badge mode-badge-${apiHealthState}`}>{formatApiHealthLabel(apiHealthState)}</span>
                </div>
                <p className="shell-meta">Desktop Shell v{version}</p>
            </header>

            <div className="workspace-panels" ref={workspaceRef}>
                <section
                    className="library-panel"
                    aria-label="song library"
                    style={{ width: `${libraryPanelWidthPx}px` }}
                >
                    <div className="library-toolbar">
                        <div>
                            <h2>Song Library</h2>
                            <p>{libraryTotal} analyzed song(s)</p>
                        </div>
                        <button type="button" className="add-song-btn" onClick={handleAddSong} disabled={!bridge}>
                            Add Song
                        </button>
                        <input
                            type="search"
                            value={libraryQuery}
                            onChange={(event) => {
                                setLibraryQuery(event.currentTarget.value);
                                setLibraryPage(1);
                            }}
                            placeholder="Search title, artist, path"
                            aria-label="Search library"
                        />
                    </div>
                    {librarySongs.length > 0 ? (
                        <div className="library-list" data-testid="library-list">
                            {librarySongs.map((song) => (
                                <div
                                    key={song.id}
                                    className="library-song"
                                    data-active={selectedSongId === song.id ? "true" : "false"}
                                >
                                    <button
                                        type="button"
                                        className="library-song-main"
                                        onClick={() => void handleOpenLibrarySong(song)}
                                    >
                                        <span>{song.title}</span>
                                        <small className="library-song-artist">{song.artist || "Unknown artist"}</small>
                                        <small>{song.analysis.analysis.chords.length} chords - {formatTime(song.duration)}</small>
                                    </button>
                                    <button
                                        type="button"
                                        className="library-delete-btn"
                                        aria-label={`Delete ${displaySongTitle(song)}`}
                                        title="Delete from library"
                                        onClick={() => void handleDeleteLibrarySong(song)}
                                    >
                                        Delete
                                    </button>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="library-empty">No analyzed songs yet.</p>
                    )}
                    <div className="library-pagination" aria-label="Song library pagination">
                        <small>{libraryTotal} song(s)</small>
                        <label>
                            Page Rows
                            <select
                                aria-label="Songs per page"
                                value={libraryPageSize}
                                onChange={(event) => {
                                    setLibraryPageSize(Number(event.currentTarget.value));
                                    setLibraryPage(1);
                                }}
                            >
                                <option value={5}>5</option>
                                <option value={10}>10</option>
                                <option value={20}>20</option>
                                <option value={50}>50</option>
                            </select>
                        </label>
                        <button
                            type="button"
                            aria-label="Previous page"
                            onClick={() => setLibraryPage((current) => Math.max(1, current - 1))}
                            disabled={libraryPage <= 1}
                        >
                            ‹
                        </button>
                        <span>
                            {renderLibraryPageNumbers(libraryPage, libraryTotalPages, setLibraryPage)}
                        </span>
                        <button
                            type="button"
                            aria-label="Next page"
                            onClick={() => setLibraryPage((current) => Math.min(libraryTotalPages, current + 1))}
                            disabled={libraryPage >= libraryTotalPages}
                        >
                            ›
                        </button>
                    </div>
                </section>

                <div
                    className="panel-resizer"
                    role="separator"
                    aria-label="Resize panels"
                    aria-orientation="vertical"
                    onMouseDown={handlePanelResizeStart}
                />

                <section className="shell-card detector-panel" aria-label="song analyzer detail">
                    {viewMode === "detail" ? (
                        <>
                            <div className="detector-topbar">
                                <div className="detector-actions">
                                    <button type="button" className="open-btn" onClick={() => void handleOpenAudio()} disabled={!bridge}>
                                        Open Audio
                                    </button>
                                    {canReanalyze ? (
                                        <button type="button" className="open-btn" onClick={() => void handleReanalyze()}>
                                            Re-analyze
                                        </button>
                                    ) : null}
                                    <button
                                        type="button"
                                        className="open-btn"
                                        onClick={() => setIsSaveFormOpen(true)}
                                        disabled={!canSaveAnalysis}
                                        title={canSaveAnalysis ? "Save analysis to Song Library" : "Analyze audio first before saving"}
                                    >
                                        Save
                                    </button>
                                    <button
                                        type="button"
                                        className="open-btn secondary-btn"
                                        onClick={() => void handleExportSong("txt")}
                                        disabled={!selectedSongId}
                                    >
                                        Export TXT
                                    </button>
                                    <button
                                        type="button"
                                        className="open-btn secondary-btn"
                                        onClick={() => void handleExportSong("lrc")}
                                        disabled={!selectedSongId}
                                    >
                                        Export LRC
                                    </button>
                                </div>
                                <p className="state-label">State: {state}</p>
                            </div>
                            {isSaveFormOpen ? (
                                <form
                                    className="song-metadata-form"
                                    aria-label="song metadata"
                                    onSubmit={(event) => {
                                        event.preventDefault();
                                        void handleSaveAnalysis();
                                    }}
                                >
                                    <label>
                                        <span>Artist</span>
                                        <input
                                            type="text"
                                            value={songArtist}
                                            onChange={(event) => setSongArtist(event.currentTarget.value)}
                                            placeholder="Contoh: SR Banyak Cerita"
                                            required
                                        />
                                    </label>
                                    <label>
                                        <span>Judul</span>
                                        <input
                                            type="text"
                                            value={songTitle}
                                            onChange={(event) => setSongTitle(event.currentTarget.value)}
                                            placeholder="Contoh: Album Lama"
                                            required
                                        />
                                    </label>
                                    <button type="submit" className="open-btn">
                                        Save to Library
                                    </button>
                                </form>
                            ) : null}
                            <audio
                                ref={audioRef}
                                src={audioSourceUrl ?? undefined}
                                data-testid="audio-player"
                                onLoadedMetadata={(event) => {
                                    const nextDuration = Number.isFinite(event.currentTarget.duration)
                                        ? Math.max(0, event.currentTarget.duration)
                                        : 0;
                                    const pendingSwitch = pendingAudioSwitchRef.current;
                                    if (pendingSwitch) {
                                        event.currentTarget.volume = pendingSwitch.fadeIn ? 0 : 1;
                                        event.currentTarget.currentTime = Math.min(pendingSwitch.time, nextDuration || pendingSwitch.time);
                                        pendingAudioSwitchRef.current = null;
                                    }
                                    if (nextDuration > 0) {
                                        setDurationSeconds(nextDuration);
                                    }
                                    setIsMediaReady(nextDuration > 0);
                                    setCurrentTimeSeconds(event.currentTarget.currentTime || 0);
                                    if (pendingSwitch?.autoplay) {
                                        void event.currentTarget.play();
                                    }
                                    if (pendingSwitch?.fadeIn) {
                                        void fadeAudioVolume(event.currentTarget, 1, 220);
                                    }
                                }}
                                onTimeUpdate={(event) => {
                                    const nextTime = event.currentTarget.currentTime || 0;
                                    const pendingSeekTime = pendingSeekTimeRef.current;
                                    if (pendingSeekTime !== null && nextTime < pendingSeekTime - 0.35) {
                                        return;
                                    }
                                    if (pendingSeekTime !== null) {
                                        pendingSeekTimeRef.current = null;
                                    }
                                    if (!isMediaReady && event.currentTarget.currentTime > 0) {
                                        setIsMediaReady(true);
                                    }
                                    setCurrentTimeSeconds(nextTime);
                                }}
                                onSeeked={(event) => {
                                    pendingSeekTimeRef.current = null;
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
                            <div className="detector-summary">
                                <div>
                                    <p className="file-name" aria-live="polite">{selectedFileName}</p>
                                    <p className="analysis-label" aria-live="polite">{analysisStatus}</p>
                                    {apiJobProgress && (apiJobProgress.status === "queued" || apiJobProgress.status === "running") ? (
                                        <div className="job-progress" aria-label="API job progress">
                                            <span>{formatApiJobKind(apiJobProgress.kind)}</span>
                                            <div className="job-progress-track">
                                                <span style={{ width: `${Math.max(5, Math.min(100, apiJobProgress.progress))}%` }} />
                                            </div>
                                            <strong>{Math.round(apiJobProgress.progress)}%</strong>
                                        </div>
                                    ) : null}
                                </div>
                                <p className="active-chord" aria-live="polite" data-testid="active-chord">
                                    Active chord: <span>{activeChordLabel}</span>
                                </p>
                            </div>
                            <div className="transport-strip">
                                <div className="playback-row" aria-label="playback controls">
                                    <button type="button" onClick={() => void handleTogglePlayback()} disabled={!canUsePlayback}>
                                        <span aria-hidden="true">{isPlaying ? "II" : "▶"}</span>
                                        {isPlaying ? "Pause" : "Play"}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => void handleToggleVocalHide()}
                                        disabled={!canUsePlayback || isRemovingVocals}
                                        aria-pressed={isUsingInstrumentalAudio}
                                    >
                                        <span aria-hidden="true">{isUsingInstrumentalAudio ? "◼" : "♪"}</span>
                                        {isRemovingVocals ? "Preparing..." : isUsingInstrumentalAudio ? "Vocal: Off" : "Vocal: On"}
                                    </button>
                                    <p className="time-label" aria-live="polite">
                                        {formatTime(currentTimeSeconds)} / {formatTime(durationSeconds)}
                                    </p>
                                </div>
                                <div className="transpose-controls" aria-label="transpose controls">
                                    <span>Transpose</span>
                                    <button
                                        type="button"
                                        onClick={() => void handleSetTranspose(transposeSemitones - 1)}
                                        disabled={isPitchShiftingAudio}
                                        aria-label="Transpose down"
                                    >
                                        -
                                    </button>
                                    <strong>{formatTranspose(transposeSemitones)}</strong>
                                    <button
                                        type="button"
                                        onClick={() => void handleSetTranspose(transposeSemitones + 1)}
                                        disabled={isPitchShiftingAudio}
                                        aria-label="Transpose up"
                                    >
                                        +
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => void handleSetTranspose(0)}
                                        disabled={isPitchShiftingAudio}
                                    >
                                        <span aria-hidden="true">↺</span>
                                        Reset
                                    </button>
                                    {isPitchShiftingAudio ? <small>Rendering audio...</small> : null}
                                </div>
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
                            <div className="detector-tabs" role="tablist" aria-label="Chord detector views">
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={detectorTab === "timeline"}
                                    onClick={() => setDetectorTab("timeline")}
                                >
                                    Timeline
                                </button>
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={detectorTab === "lyrics"}
                                    onClick={() => setDetectorTab("lyrics")}
                                >
                                    Lyrics
                                </button>
                            </div>
                            {detectorTab === "timeline" ? (
                                <section className="timeline-frame" aria-label="Chord timeline frame">
                                    <section ref={timelineRef} className="timeline" aria-label="Chord timeline">
                                        {renderTimeline(
                                            timelineSegments,
                                            durationSeconds,
                                            timelineWidthPx,
                                            currentTimeSeconds,
                                            transposeSemitones,
                                            handleSeek
                                        )}
                                    </section>
                                </section>
                            ) : (
                                <section className="lyrics-frame" aria-label="Lyrics editor">
                                    <div className="lyrics-editor">
                                        <div className="lyrics-tools">
                                            <div className="lyrics-actions">
                                                <button
                                                    type="button"
                                                    onClick={() => void handleGenerateLyricsFromAudio()}
                                                    disabled={!selectedFilePath || isGeneratingLyrics}
                                                >
                                                    {isGeneratingLyrics ? "Generating..." : "Generate Lyrics"}
                                                </button>
                                                <label className="lyrics-model">
                                                    <span>Model</span>
                                                    <select
                                                        value={lyricsModel}
                                                        onChange={(event) => setLyricsModel(event.currentTarget.value as LyricsModel)}
                                                        disabled={isGeneratingLyrics}
                                                    >
                                                        <option value="tiny">tiny</option>
                                                        <option value="base">base</option>
                                                        <option value="small">small</option>
                                                        <option value="medium">medium</option>
                                                        <option value="large">large</option>
                                                    </select>
                                                </label>
                                                <button
                                                    type="button"
                                                    onClick={handleAutoSyncLyrics}
                                                    disabled={!lyricsText.trim() || durationSeconds <= 0 || isGeneratingLyrics}
                                                >
                                                    Auto Sync
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => setLyricsText(stripLyricsTiming(lyricsText))}
                                                    disabled={!lyricsText.trim() || isGeneratingLyrics}
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                            <p>Generate dari audio, atau paste lyric polos lalu Auto Sync.</p>
                                        </div>
                                        <textarea
                                            value={lyricsText}
                                            onChange={(event) => setLyricsText(event.currentTarget.value)}
                                            placeholder={"Paste lyric polos di sini...\nbaris lyric pertama\nbaris lyric kedua"}
                                            aria-label="Song lyrics"
                                        />
                                    </div>
                                    <div ref={lyricsPreviewRef} className="lyrics-preview" aria-label="Lyrics preview">
                                        {renderLyricsPreview(
                                            lyricsText,
                                            currentTimeSeconds,
                                            timelineSegments,
                                            transposeSemitones
                                        )}
                                    </div>
                                </section>
                            )}
                        </>
                    ) : (
                        <div className="detector-empty">
                            <h2>Chord Detector</h2>
                            <p>Pilih lagu dari Song Library, atau klik Add Song untuk menganalisa audio baru.</p>
                        </div>
                    )}
                </section>
            </div>
        </main>
    );
}

function toFileUrl(localPath: string): string {
    if (/^https?:\/\//i.test(localPath)) {
        return localPath;
    }

    const normalized = localPath.replace(/\\/g, "/");
    const url = new URL("file://");

    if (/^[A-Za-z]:\//.test(normalized)) {
        url.pathname = `/${normalized}`;
        return url.href;
    }

    url.pathname = normalized.startsWith("/") ? normalized : `/${normalized}`;
    return url.href;
}

function toAudioSourceUrl(source: { kind?: string; url?: string; bytes?: Uint8Array; mimeType?: string }): string {
    if (source.kind === "url" && source.url) {
        return source.url;
    }
    if (source.bytes) {
        return createObjectUrl(source.bytes, source.mimeType);
    }
    throw new Error("Unsupported audio playback source");
}

function createObjectUrl(bytes: Uint8Array, mimeType?: string): string {
    const stableBytes = new Uint8Array(bytes.length);
    stableBytes.set(bytes);
    const blob = new Blob([stableBytes], mimeType ? { type: mimeType } : undefined);
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

function findActiveChord(segments: ChordSegment[], currentTimeSeconds: number): string | null {
    if (!Number.isFinite(currentTimeSeconds)) {
        return null;
    }

    return segments.find((segment) => segment.start <= currentTimeSeconds && currentTimeSeconds < segment.end)?.chord ?? null;
}

function displaySongTitle(song: SongLibraryRecord): string {
    return song.artist.trim().length > 0 ? `${song.artist} - ${song.title}` : song.title;
}

function buildSongMetadata(title: string, artist: string): SongMetadataInput | null {
    const normalizedTitle = title.trim();
    const normalizedArtist = artist.trim();
    if (!normalizedTitle || !normalizedArtist) {
        return null;
    }
    return {
        title: normalizedTitle,
        artist: normalizedArtist
    };
}

function normalizeSongLibraryListResult(
    result: SongLibraryListResult | SongLibraryRecord[],
    fallbackPage: number,
    fallbackPageSize: number
): SongLibraryListResult {
    if (Array.isArray(result)) {
        return {
            records: result,
            total: result.length,
            page: fallbackPage,
            pageSize: fallbackPageSize,
            totalPages: Math.max(1, Math.ceil(result.length / fallbackPageSize))
        };
    }

    return {
        records: result.records,
        total: result.total,
        page: result.page,
        pageSize: result.pageSize,
        totalPages: result.totalPages
    };
}

function renderLibraryPageNumbers(
    currentPage: number,
    totalPages: number,
    onSelectPage: (page: number) => void
) {
    const pages = buildCompactPageNumbers(currentPage, totalPages);
    return pages.map((page, index) => {
        if (page === "ellipsis") {
            return <i key={`ellipsis-${index}`}>...</i>;
        }

        return (
            <button
                key={page}
                type="button"
                className="library-page-number"
                data-active={page === currentPage ? "true" : "false"}
                onClick={() => onSelectPage(page)}
                disabled={page === currentPage}
            >
                {page}
            </button>
        );
    });
}

function buildCompactPageNumbers(currentPage: number, totalPages: number): Array<number | "ellipsis"> {
    if (totalPages <= 5) {
        return Array.from({ length: totalPages }, (_, index) => index + 1);
    }

    const pages = new Set([1, totalPages, currentPage - 1, currentPage, currentPage + 1]);
    const sortedPages = [...pages]
        .filter((page) => page >= 1 && page <= totalPages)
        .sort((a, b) => a - b);
    const result: Array<number | "ellipsis"> = [];
    for (const page of sortedPages) {
        const previous = result[result.length - 1];
        if (typeof previous === "number" && page - previous > 1) {
            result.push("ellipsis");
        }
        result.push(page);
    }
    return result;
}

interface LyricLine {
    time: number | null;
    text: string;
}

function renderLyricsPreview(
    lyrics: string,
    currentTimeSeconds: number,
    segments: ChordSegment[],
    transposeSemitones: number
) {
    const lines = parseLyrics(lyrics);
    if (lines.length === 0) {
        return <p className="lyrics-empty">Belum ada lyric. Paste text biasa atau format LRC untuk sinkron timestamp.</p>;
    }

    return (
        <div className="lyrics-lines" data-testid="lyrics-lines">
            {lines.map((line, index) => {
                const nextTimedLine = lines.slice(index + 1).find((candidate) => candidate.time !== null);
                const isActive = line.time !== null
                    && line.time <= currentTimeSeconds
                    && (nextTimedLine?.time == null || currentTimeSeconds < nextTimedLine.time);
                const chordMarkers = line.time === null
                    ? []
                    : buildLyricChordMarkers(
                        segments,
                        line.time,
                        nextTimedLine?.time ?? line.time + 5,
                        transposeSemitones
                    );
                return (
                    <div
                        key={`${line.time ?? "plain"}-${index}-${line.text}`}
                        className="lyrics-line"
                        data-active={isActive ? "true" : "false"}
                    >
                        <time>{line.time === null ? "--:--" : formatTime(line.time)}</time>
                        <div className="lyrics-chord-sheet">
                            <div className="lyrics-chords" aria-label="Line chords">
                                {chordMarkers.length > 0 ? chordMarkers.map((marker) => (
                                    <strong key={`${marker.label}-${marker.left}`} style={{ left: `${marker.left}%` }}>
                                        {marker.label}
                                    </strong>
                                )) : <strong style={{ left: "0%" }}>-</strong>}
                            </div>
                            <span>{line.text}</span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

interface LyricChordMarker {
    label: string;
    left: number;
}

function buildLyricChordMarkers(
    segments: ChordSegment[],
    startTime: number,
    endTime: number,
    transposeSemitones: number
): LyricChordMarker[] {
    const safeEndTime = Math.max(startTime + 0.25, endTime);
    const windowDuration = safeEndTime - startTime;
    const markers: LyricChordMarker[] = [];
    const openingChord = findActiveChord(segments, startTime);

    if (openingChord) {
        markers.push({
            label: transposeChordLabel(openingChord, transposeSemitones),
            left: 0
        });
    }

    for (const segment of segments) {
        if (segment.start <= startTime || segment.start >= safeEndTime) {
            continue;
        }
        const label = transposeChordLabel(segment.chord, transposeSemitones);
        const previous = markers[markers.length - 1];
        if (previous?.label === label) {
            continue;
        }
        markers.push({
            label,
            left: Math.min(92, Math.max(0, ((segment.start - startTime) / windowDuration) * 100))
        });
    }

    return markers;
}

function parseLyrics(lyrics: string): LyricLine[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .filter((line) => line.length > 0)
        .map((line) => {
            const match = /^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$/.exec(line);
            if (!match) {
                return { time: null, text: line };
            }

            const minutes = Number.parseInt(match[1], 10);
            const seconds = Number.parseInt(match[2], 10);
            const fraction = match[3] ? Number.parseFloat(`0.${match[3]}`) : 0;
            const time = minutes * 60 + seconds + fraction;
            if (!Number.isFinite(time) || seconds >= 60) {
                return { time: null, text: line };
            }

            return {
                time,
                text: match[4].trim()
            };
        });
}

function stripLyricsTiming(lyrics: string): string {
    return extractPlainLyricLines(lyrics).join("\n");
}

function autoSyncLyrics(lyrics: string, durationSeconds: number, segments: ChordSegment[]): string {
    const plainLines = extractPlainLyricLines(lyrics);
    if (plainLines.length === 0 || durationSeconds <= 0) {
        return "";
    }

    const syncTimes = buildLyricSyncTimes(plainLines.length, durationSeconds, segments);
    return plainLines
        .map((line, index) => `[${formatLrcTime(syncTimes[index] ?? 0)}]${line}`)
        .join("\n");
}

function extractPlainLyricLines(lyrics: string): string[] {
    return lyrics
        .split(/\r?\n/)
        .map((rawLine) => rawLine.trim())
        .map((line) => line.replace(/^(?:\[\d{1,2}:\d{2}(?:\.\d{1,3})?\])+/, "").trim())
        .filter((line) => line.length > 0);
}

function buildLyricSyncTimes(lineCount: number, durationSeconds: number, segments: ChordSegment[]): number[] {
    if (lineCount === 1) {
        return [Math.min(Math.max(durationSeconds * 0.08, 0), Math.max(durationSeconds - 1, 0))];
    }

    const introOffset = Math.min(12, Math.max(2, durationSeconds * 0.045));
    const outroPadding = Math.min(10, Math.max(2, durationSeconds * 0.035));
    const startTime = Math.min(introOffset, Math.max(durationSeconds - 1, 0));
    const endTime = Math.max(startTime + 1, durationSeconds - outroPadding);
    const musicalAnchors = selectMusicalLyricAnchors(segments, startTime, endTime, lineCount);

    if (musicalAnchors.length >= lineCount) {
        return spreadAnchors(musicalAnchors, lineCount);
    }

    return Array.from({ length: lineCount }, (_, index) => {
        const ratio = index / (lineCount - 1);
        return startTime + (endTime - startTime) * ratio;
    });
}

function selectMusicalLyricAnchors(
    segments: ChordSegment[],
    startTime: number,
    endTime: number,
    lineCount: number
): number[] {
    const minGap = Math.max(1.8, Math.min(4.5, (endTime - startTime) / Math.max(lineCount * 1.35, 1)));
    const anchors: number[] = [];
    for (const segment of segments) {
        if (segment.start < startTime || segment.start > endTime) {
            continue;
        }
        const previous = anchors[anchors.length - 1];
        if (previous === undefined || segment.start - previous >= minGap) {
            anchors.push(segment.start);
        }
    }
    return anchors;
}

function spreadAnchors(anchors: number[], lineCount: number): number[] {
    if (anchors.length === lineCount) {
        return anchors;
    }
    return Array.from({ length: lineCount }, (_, index) => {
        const anchorIndex = Math.round((index / (lineCount - 1)) * (anchors.length - 1));
        return anchors[anchorIndex] ?? anchors[anchors.length - 1] ?? 0;
    });
}

function formatLrcTime(seconds: number): string {
    const safeSeconds = Math.max(0, seconds);
    const minutes = Math.floor(safeSeconds / 60);
    const wholeSeconds = Math.floor(safeSeconds % 60);
    const centiseconds = Math.floor((safeSeconds - Math.floor(safeSeconds)) * 100);
    return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(centiseconds).padStart(2, "0")}`;
}

const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

function formatTranspose(semitones: number): string {
    if (semitones === 0) {
        return "0";
    }
    return semitones > 0 ? `+${semitones}` : `${semitones}`;
}

function transposeChordLabel(chord: string, semitones: number): string {
    if (chord === "N" || semitones === 0) {
        return chord;
    }

    const match = /^(C#|D#|F#|G#|A#|C|D|E|F|G|A|B)(.*)$/.exec(chord);
    if (!match) {
        return chord;
    }

    const rootIndex = SHARP_ROOTS.indexOf(match[1] as (typeof SHARP_ROOTS)[number]);
    if (rootIndex < 0) {
        return chord;
    }

    const nextIndex = modulo(rootIndex + semitones, SHARP_ROOTS.length);
    return `${SHARP_ROOTS[nextIndex]}${match[2]}`;
}

function modulo(value: number, divisor: number): number {
    return ((value % divisor) + divisor) % divisor;
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

function renderTimeline(
    segments: ChordSegment[],
    durationSeconds: number,
    timelineWidthPx: number,
    currentTimeSeconds: number,
    transposeSemitones: number,
    onSeek: (value: number) => void
) {
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
                        {row.fragments.map(({ segment, fragmentStart, fragmentEnd, leftPercent, widthPercent, showLabel, labelText }, index) => {
                            const displayedChord = transposeChordLabel(segment.chord, transposeSemitones);
                            const displayedLabel = transposeChordLabel(labelText, transposeSemitones);
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
                                    data-active={fragmentStart <= safeCurrentTime && safeCurrentTime < fragmentEnd ? "true" : "false"}
                                    aria-current={fragmentStart <= safeCurrentTime && safeCurrentTime < fragmentEnd ? "true" : "false"}
                                    aria-label={`${displayedChord} from ${formatTime(fragmentStart)} to ${formatTime(fragmentEnd)}`}
                                    title={`${displayedChord} (${Math.round(segment.confidence * 100)}%)`}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        onSeek(timelineClickTime(event, row.rowStart, row.rowDuration, safeDuration));
                                    }}
                                    style={{
                                        left: `${leftPercent}%`,
                                        width: `${widthPercent}%`
                                    }}
                                >
                                    {showLabel ? <span className="timeline-segment-label">{displayedLabel}</span> : null}
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
    durationSeconds: number
): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const offset = Math.max(0, Math.min(event.clientX - rect.left, width));
    const ratio = offset / width;
    const target = rowStart + ratio * rowDuration;
    return Math.max(0, Math.min(target, durationSeconds));
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

	const [root] = chord;
	return root ?? "";
}

function formatTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, Math.floor(totalSeconds)) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatApiHealthLabel(state: ApiHealthState): string {
    if (state === "connected") {
        return "API Connected";
    }
    if (state === "offline") {
        return "API Offline";
    }
    if (state === "checking") {
        return "Checking API";
    }
    return "Local Mode";
}

function formatApiJobKind(kind: ApiJobProgressEvent["kind"]): string {
    if (kind === "analysis") {
        return "Analyzing";
    }
    if (kind === "lyrics") {
        return "Lyrics";
    }
    if (kind === "vocals") {
        return "Vocal Off";
    }
    return "Transpose";
}

function buildLocalChordSheetExport(
    artist: string,
    title: string,
    durationSeconds: number,
    segments: ChordSegment[],
    lyrics: string
): string {
    const lines = [
        `${artist || "Unknown Artist"} - ${title || "Untitled"}`,
        `Duration: ${formatPreciseTime(durationSeconds)}`,
        "",
        "Chords:",
        ...segments.map((segment) => `${formatPreciseTime(segment.start)} - ${formatPreciseTime(segment.end)}  ${segment.chord}`)
    ];

    if (lyrics.trim()) {
        lines.push("", "Lyrics:", lyrics.trim());
    }

    return `${lines.join("\n")}\n`;
}

function buildLocalLrcExport(lyrics: string, segments: ChordSegment[]): string {
    if (lyrics.trim()) {
        return `${lyrics.trim()}\n`;
    }
    return `${segments.map((segment) => `[${formatPreciseTime(segment.start)}]${segment.chord}`).join("\n")}\n`;
}

function safeDownloadName(value: string): string {
    const safe = value.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    return safe || "song";
}

function formatPreciseTime(totalSeconds: number): string {
    const safe = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0;
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}
