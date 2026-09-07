import { useRef } from "react";
import type { Dispatch, RefObject, SetStateAction, SyntheticEvent } from "react";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

import type { ApiJobProgressEvent, ShellState } from "../appTypes.js";
import { toAudioSourceUrl } from "../lib/audioSources.js";
import { formatTranspose } from "../lib/chords.js";

type PlaybackControllerOptions = {
    bridge: typeof window.gcd;
    state: ShellState;
    isApiMode: boolean;
    isMediaReady: boolean;
    selectedFilePath: string | null;
    originalAudioStreamUrl: string | null;
    currentTimeSeconds: number;
    durationSeconds: number;
    transposeSemitones: number;
    isTransposeKeyOnly: boolean;
    instrumentalAudioPath: string | null;
    instrumentalAudioStreamUrl: string | null;
    isUsingInstrumentalAudio: boolean;
    latestApiJobProgressRef: RefObject<ApiJobProgressEvent | null>;
    cancelledApiJobIdsRef: RefObject<Set<string>>;
    setState: Dispatch<SetStateAction<ShellState>>;
    setAnalysisStatus: Dispatch<SetStateAction<string>>;
    setAudioSourceUrl: Dispatch<SetStateAction<string | null>>;
    setIsMediaReady: Dispatch<SetStateAction<boolean>>;
    setDurationSeconds: Dispatch<SetStateAction<number>>;
    setCurrentTimeSeconds: Dispatch<SetStateAction<number>>;
    setTransposeSemitones: Dispatch<SetStateAction<number>>;
    setIsTransposeKeyOnly: Dispatch<SetStateAction<boolean>>;
    setIsPitchShiftingAudio: Dispatch<SetStateAction<boolean>>;
    setIsRemovingVocals: Dispatch<SetStateAction<boolean>>;
    setInstrumentalAudioPath: Dispatch<SetStateAction<string | null>>;
    setInstrumentalAudioStreamUrl: Dispatch<SetStateAction<string | null>>;
    setIsUsingInstrumentalAudio: Dispatch<SetStateAction<boolean>>;
};

export function usePlaybackController(options: PlaybackControllerOptions) {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const activeBlobUrlRef = useRef<string | null>(null);
    const pendingAudioSwitchRef = useRef<{ time: number; autoplay: boolean; fadeIn: boolean } | null>(null);
    const audioFadeTimerRef = useRef<number | null>(null);
    const audioSwitchFallbackTimerRef = useRef<number | null>(null);
    const pendingSeekTimeRef = useRef<number | null>(null);
    const activePitchShiftRequestIdRef = useRef<number>(0);

    const resetPlaybackState = (): void => {
        pendingSeekTimeRef.current = null;
        const audio = audioRef.current;
        if (audio) {
            try {
                audio.pause();
                audio.currentTime = 0;
            } catch {
                // Test environments may not fully implement media controls.
            }
        }
        options.setIsMediaReady(false);
        options.setDurationSeconds(0);
        options.setCurrentTimeSeconds(0);
        options.setIsUsingInstrumentalAudio(false);
    };

    const handlePlay = async (): Promise<void> => {
        const audio = audioRef.current;
        if (!audio) {
            return;
        }
        if (!options.isMediaReady) {
            audio.load();
        }
        try {
            await audio.play();
        } catch (error) {
            const reason = error instanceof Error && error.message ? ` (${error.message})` : "";
            options.setAnalysisStatus(`Could not play this audio file.${reason}`);
            options.setState((current) => (current === "playing" ? "paused" : current));
        }
    };

    const handlePause = (): void => {
        audioRef.current?.pause();
    };

    const handleTogglePlayback = async (): Promise<void> => {
        if (options.state === "playing") {
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
        const safeValue = Math.max(0, Math.min(value, options.durationSeconds || value));
        pendingSeekTimeRef.current = safeValue;
        audio.currentTime = safeValue;
        options.setCurrentTimeSeconds(safeValue);
    };

    const handleSetTranspose = async (nextSemitones: number): Promise<void> => {
        const safeSemitones = Math.max(-11, Math.min(11, Math.trunc(nextSemitones)));
        options.setTransposeSemitones(safeSemitones);
        if (options.isTransposeKeyOnly) {
            options.setAnalysisStatus(safeSemitones === 0
                ? "Transpose reset."
                : `Key transposed ${formatTranspose(safeSemitones)}. Audio unchanged.`);
            return;
        }
        await applyTransposeAudio(safeSemitones);
    };

    const handleSetTransposeKeyOnly = async (nextKeyOnly: boolean): Promise<void> => {
        options.setIsTransposeKeyOnly(nextKeyOnly);
        if (nextKeyOnly) {
            await applyTransposeAudio(0);
            options.setAnalysisStatus(options.transposeSemitones === 0
                ? "Only Key enabled. Audio unchanged."
                : `Only Key enabled. Key remains ${formatTranspose(options.transposeSemitones)}, audio restored.`);
            return;
        }
        if (options.transposeSemitones !== 0) {
            await applyTransposeAudio(options.transposeSemitones);
        } else {
            options.setAnalysisStatus("Only Key disabled. Next transpose will also render audio.");
        }
    };

    const handleToggleVocalHide = async (): Promise<void> => {
        if (options.isUsingInstrumentalAudio) {
            await handleUseOriginalAudio();
            return;
        }
        if (options.instrumentalAudioPath) {
            await handleUseInstrumentalAudio(options.instrumentalAudioPath, options.instrumentalAudioStreamUrl, "Karaoke audio enabled.");
            return;
        }
        if (!options.selectedFilePath) {
            options.setAnalysisStatus("Open audio dulu sebelum Vocal Off.");
            return;
        }
        if (!options.bridge?.removeVocals || !options.bridge?.getAudioPlaybackSource) {
            options.setAnalysisStatus("Vocal Off API belum tersedia. Restart aplikasi lalu coba lagi.");
            return;
        }

        options.setIsRemovingVocals(true);
        options.setAnalysisStatus(options.isApiMode ? "Uploading audio and preparing Vocal Off on API server..." : "Preparing Vocal Off...");
        let result: VocalRemovalResult;
        try {
            result = await options.bridge.removeVocals(options.selectedFilePath);
        } catch {
            const latestJob = options.latestApiJobProgressRef.current;
            if (latestJob && options.cancelledApiJobIdsRef.current.has(latestJob.id)) {
                return;
            }
            options.setIsRemovingVocals(false);
            options.setAnalysisStatus("Vocal Off gagal dibuat.");
            return;
        }
        options.setIsRemovingVocals(false);

        if ("error" in result) {
            options.setAnalysisStatus(result.error.message);
            return;
        }

        options.setInstrumentalAudioPath(result.audio.path);
        options.setInstrumentalAudioStreamUrl(result.audio.streamUrl ?? null);
        await handleUseInstrumentalAudio(result.audio.path, result.audio.streamUrl ?? null, "Vocal Off ready.");
    };

    const cleanupPlaybackResources = (): void => {
        if (audioFadeTimerRef.current !== null) {
            window.clearInterval(audioFadeTimerRef.current);
            audioFadeTimerRef.current = null;
        }
        if (audioSwitchFallbackTimerRef.current !== null) {
            window.clearTimeout(audioSwitchFallbackTimerRef.current);
            audioSwitchFallbackTimerRef.current = null;
        }
        if (activeBlobUrlRef.current && typeof URL.revokeObjectURL === "function") {
            URL.revokeObjectURL(activeBlobUrlRef.current);
            activeBlobUrlRef.current = null;
        }
    };

    const audioHandlers = {
        onLoadedMetadata: (event: SyntheticEvent<HTMLAudioElement>) => {
            const nextDuration = Number.isFinite(event.currentTarget.duration) ? Math.max(0, event.currentTarget.duration) : 0;
            const pendingSwitch = pendingAudioSwitchRef.current;
            if (pendingSwitch) {
                preparePendingAudioSwitch(event.currentTarget, pendingSwitch, nextDuration);
                event.currentTarget.currentTime = Math.min(pendingSwitch.time, nextDuration || pendingSwitch.time);
                pendingAudioSwitchRef.current = null;
            }
            if (nextDuration > 0) {
                options.setDurationSeconds(nextDuration);
            }
            options.setIsMediaReady(nextDuration > 0);
            options.setCurrentTimeSeconds(event.currentTarget.currentTime || 0);
            if (pendingSwitch?.autoplay) {
                void event.currentTarget.play();
            }
            if (pendingSwitch?.fadeIn) {
                void fadeAudioVolume(event.currentTarget, 1, 220);
            }
        },
        onCanPlay: (event: SyntheticEvent<HTMLAudioElement>) => {
            const pendingSwitch = pendingAudioSwitchRef.current;
            if (pendingSwitch) {
                const nextDuration = Number.isFinite(event.currentTarget.duration) ? Math.max(0, event.currentTarget.duration) : 0;
                preparePendingAudioSwitch(event.currentTarget, pendingSwitch, nextDuration);
                pendingAudioSwitchRef.current = null;
                if (pendingSwitch.autoplay) {
                    void event.currentTarget.play();
                }
            }
            if (event.currentTarget.volume === 0) {
                void fadeAudioVolume(event.currentTarget, 1, 220);
            }
        },
        onTimeUpdate: (event: SyntheticEvent<HTMLAudioElement>) => {
            const nextTime = event.currentTarget.currentTime || 0;
            const pendingSeekTime = pendingSeekTimeRef.current;
            if (pendingSeekTime !== null && nextTime < pendingSeekTime - 0.35) {
                return;
            }
            if (pendingSeekTime !== null) {
                pendingSeekTimeRef.current = null;
            }
            if (!options.isMediaReady && event.currentTarget.currentTime > 0) {
                options.setIsMediaReady(true);
            }
            options.setCurrentTimeSeconds(nextTime);
        },
        onSeeked: (event: SyntheticEvent<HTMLAudioElement>) => {
            pendingSeekTimeRef.current = null;
            options.setCurrentTimeSeconds(event.currentTarget.currentTime || 0);
        },
        onPlay: () => {
            options.setIsMediaReady(true);
            options.setState("playing");
        },
        onPause: () => {
            options.setState((current) => (current === "playing" ? "paused" : current));
        },
        onEnded: () => {
            options.setState("paused");
        },
        onAudioError: () => {
            options.setIsMediaReady(false);
            options.setState("error");
            options.setAnalysisStatus("Could not analyze this audio file.");
        },
    };

    const applyTransposeAudio = async (semitones: number): Promise<void> => {
        const baseAudioPath = options.isUsingInstrumentalAudio ? options.instrumentalAudioPath : options.selectedFilePath;
        if (!baseAudioPath || !options.bridge?.getAudioPlaybackSource) {
            return;
        }

        const requestId = activePitchShiftRequestIdRef.current + 1;
        activePitchShiftRequestIdRef.current = requestId;
        options.setIsPitchShiftingAudio(semitones !== 0);
        options.setAnalysisStatus(semitones === 0
            ? "Transpose reset."
            : options.isApiMode
                ? `Rendering transpose audio ${formatTranspose(semitones)} on API server...`
                : `Rendering transpose audio ${formatTranspose(semitones)}...`);

        let playbackPath = baseAudioPath;
        if (semitones !== 0) {
            if (!options.bridge.pitchShiftAudio) {
                options.setIsPitchShiftingAudio(false);
                options.setAnalysisStatus("Pitch shift API belum tersedia. Restart aplikasi lalu coba lagi.");
                return;
            }
            let result: PitchShiftResult;
            try {
                result = await options.bridge.pitchShiftAudio(baseAudioPath, { semitones });
            } catch {
                const latestJob = options.latestApiJobProgressRef.current;
                if (latestJob && options.cancelledApiJobIdsRef.current.has(latestJob.id)) {
                    return;
                }
                options.setIsPitchShiftingAudio(false);
                options.setAnalysisStatus("Gagal menyesuaikan transpose audio.");
                return;
            }
            if (requestId !== activePitchShiftRequestIdRef.current) {
                return;
            }
            if ("error" in result) {
                options.setIsPitchShiftingAudio(false);
                options.setAnalysisStatus(result.error.message);
                return;
            }
            playbackPath = result.audio.streamUrl ?? result.audio.path;
        }

        try {
            const playbackSource = await options.bridge.getAudioPlaybackSource(playbackPath);
            if (requestId !== activePitchShiftRequestIdRef.current) {
                return;
            }
            const audio = audioRef.current;
            const previousTime = audio?.currentTime ?? options.currentTimeSeconds;
            const wasPlaying = options.state === "playing";
            await replaceAudioSource(toAudioSourceUrl(playbackSource), previousTime, wasPlaying);
            options.setAnalysisStatus(semitones === 0 ? "Transpose reset." : `Audio transposed ${formatTranspose(semitones)}.`);
        } catch {
            options.setAnalysisStatus("Gagal memuat audio transpose.");
        } finally {
            if (requestId === activePitchShiftRequestIdRef.current) {
                options.setIsPitchShiftingAudio(false);
            }
        }
    };

    const handleUseInstrumentalAudio = async (audioPath: string, audioStreamUrl: string | null, statusMessage: string): Promise<void> => {
        if (!options.bridge?.getAudioPlaybackSource) {
            return;
        }
        const audio = audioRef.current;
        const previousTime = audio?.currentTime ?? options.currentTimeSeconds;
        const wasPlaying = options.state === "playing";
        let playbackPath = audioPath;
        if (!options.isTransposeKeyOnly && options.transposeSemitones !== 0 && options.bridge.pitchShiftAudio) {
            const shifted = await options.bridge.pitchShiftAudio(audioPath, { semitones: options.transposeSemitones });
            if (!("error" in shifted)) {
                playbackPath = shifted.audio.streamUrl ?? shifted.audio.path;
            }
        }
        try {
            const playbackSourceUrl = await loadPlaybackSourceUrl(
                options.isTransposeKeyOnly || options.transposeSemitones === 0 ? audioStreamUrl : null,
                playbackPath,
            );
            await replaceAudioSource(playbackSourceUrl, previousTime, wasPlaying);
            options.setIsUsingInstrumentalAudio(true);
            options.setAnalysisStatus(statusMessage);
        } catch {
            options.setAnalysisStatus("Gagal memuat Vocal Off audio.");
        }
    };

    const handleUseOriginalAudio = async (): Promise<void> => {
        if (!options.selectedFilePath || !options.bridge?.getAudioPlaybackSource) {
            return;
        }
        const audio = audioRef.current;
        const previousTime = audio?.currentTime ?? options.currentTimeSeconds;
        const wasPlaying = options.state === "playing";
        let playbackPath = options.selectedFilePath;
        if (!options.isTransposeKeyOnly && options.transposeSemitones !== 0 && options.bridge.pitchShiftAudio) {
            const shifted = await options.bridge.pitchShiftAudio(options.selectedFilePath, { semitones: options.transposeSemitones });
            if (!("error" in shifted)) {
                playbackPath = shifted.audio.streamUrl ?? shifted.audio.path;
            }
        }
        try {
            const playbackSourceUrl = await loadPlaybackSourceUrl(
                options.isTransposeKeyOnly || options.transposeSemitones === 0 ? options.originalAudioStreamUrl : null,
                playbackPath,
            );
            await replaceAudioSource(playbackSourceUrl, previousTime, wasPlaying);
            options.setIsUsingInstrumentalAudio(false);
            options.setAnalysisStatus("Original audio restored.");
        } catch {
            options.setAnalysisStatus("Gagal memuat original audio.");
        }
    };

    const loadPlaybackSourceUrl = async (preferredStreamUrl: string | null | undefined, fallbackPath: string): Promise<string> => {
        if (!options.bridge?.getAudioPlaybackSource) {
            return preferredStreamUrl ?? fallbackPath;
        }
        const source = await options.bridge.getAudioPlaybackSource(preferredStreamUrl ?? fallbackPath);
        const sourceUrl = toAudioSourceUrl(source);
        if (!sourceUrl) {
            throw new Error("Audio playback source is empty");
        }
        return sourceUrl;
    };

    const replaceAudioSource = async (nextSourceUrl: string, seekTime: number, autoplay: boolean): Promise<void> => {
        const audio = audioRef.current;
        const previousBlobUrl = activeBlobUrlRef.current;
        pendingSeekTimeRef.current = null;
        if (audio) {
            audio.volume = 1;
        }
        pendingAudioSwitchRef.current = { time: Math.max(0, seekTime), autoplay, fadeIn: false };
        options.setIsMediaReady(false);
        options.setState("paused");
        options.setAudioSourceUrl(nextSourceUrl);
        activeBlobUrlRef.current = nextSourceUrl.startsWith("blob:") ? nextSourceUrl : null;
        if (previousBlobUrl && previousBlobUrl !== nextSourceUrl && typeof URL.revokeObjectURL === "function") {
            URL.revokeObjectURL(previousBlobUrl);
        }
        scheduleAudioSwitchFallback();
    };

    const preparePendingAudioSwitch = (
        audio: HTMLAudioElement,
        pendingSwitch: { time: number; autoplay: boolean; fadeIn: boolean },
        durationSeconds: number
    ): void => {
        audio.volume = pendingSwitch.fadeIn ? 0 : 1;
        audio.currentTime = Math.min(pendingSwitch.time, durationSeconds || pendingSwitch.time);
    };

    const scheduleAudioSwitchFallback = (): void => {
        if (audioSwitchFallbackTimerRef.current !== null) {
            window.clearTimeout(audioSwitchFallbackTimerRef.current);
        }
        audioSwitchFallbackTimerRef.current = window.setTimeout(() => {
            audioSwitchFallbackTimerRef.current = null;
            const audio = audioRef.current;
            const pendingSwitch = pendingAudioSwitchRef.current;
            if (!audio || !pendingSwitch) {
                return;
            }
            try {
                audio.load();
                const safeDuration = Number.isFinite(audio.duration) ? Math.max(0, audio.duration) : 0;
                audio.currentTime = Math.min(pendingSwitch.time, safeDuration || pendingSwitch.time);
            } catch {
                // Media metadata may still be pending; volume recovery must not depend on it.
            }
            pendingAudioSwitchRef.current = null;
            audio.volume = 1;
            if (pendingSwitch.autoplay) {
                void audio.play();
            }
        }, 420);
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

    return {
        audioRef,
        activeBlobUrlRef,
        resetPlaybackState,
        handleTogglePlayback,
        handleSeek,
        handleSetTranspose,
        handleSetTransposeKeyOnly,
        handleToggleVocalHide,
        cleanupPlaybackResources,
        cancelPitchShiftRequest: () => {
            activePitchShiftRequestIdRef.current += 1;
        },
        audioHandlers,
    };
}
