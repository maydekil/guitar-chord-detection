import { spawn } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";

import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";
import { materializeAudioAsset } from "./audioAssets.js";
import { normalizeLyricsModel, normalizeTransposeSemitones } from "./queryParams.js";

export interface AnalyzeAudioRequest {
    audioPath: string;
    forceRefresh?: boolean;
}

export interface GenerateLyricsRequest {
    audioPath: string;
    model?: string;
}

export interface RemoveVocalsRequest {
    audioPath: string;
}

export interface PitchShiftAudioRequest {
    audioPath: string;
    semitones: number;
}

export type EngineJobKind = "analysis" | "lyrics" | "vocals" | "pitch-shift";

export interface EngineCommandOptions {
    timeoutMs: number;
    env?: NodeJS.ProcessEnv;
    jobId?: string;
}

export interface EngineProcessRegistry {
    register(jobId: string, child: ChildProcessWithoutNullStreams): void;
    unregister(jobId: string): void;
    isCancelled(jobId: string): boolean;
}

export interface EngineCommandConfig {
    pythonExecutable: string;
    engineCwd: string;
    engineOutputDir: string;
    audioDir: string;
    publicBaseUrl: string;
    processRegistry: EngineProcessRegistry;
}

export interface EngineCommands {
    analyzeAudio(request: AnalyzeAudioRequest, jobId?: string): Promise<ChordAnalysisResult>;
    transcribeLyrics(request: GenerateLyricsRequest, jobId?: string): Promise<LyricsTranscriptionResult>;
    removeVocals(request: RemoveVocalsRequest, jobId?: string): Promise<VocalRemovalResult>;
    pitchShiftAudio(request: PitchShiftAudioRequest, jobId?: string): Promise<PitchShiftResult>;
}

export function createEngineCommands(config: EngineCommandConfig): EngineCommands {
    async function analyzeAudio(request: AnalyzeAudioRequest, jobId?: string): Promise<ChordAnalysisResult> {
        if (!request.audioPath) {
            return enginePlaceholder("ENGINE_INVALID_INPUT", "Audio path is required") satisfies ChordAnalysisResult;
        }

        return await runEngineJson<ChordAnalysisResult>(
            ["-m", "chord_engine.cli", "analyze", request.audioPath],
            "ENGINE",
            { timeoutMs: 5 * 60_000, jobId }
        );
    }

    async function transcribeLyrics(request: GenerateLyricsRequest, jobId?: string): Promise<LyricsTranscriptionResult> {
        if (!request.audioPath) {
            return enginePlaceholder("LYRICS_INVALID_INPUT", "Audio path is required") satisfies LyricsTranscriptionResult;
        }

        return await runEngineJson<LyricsTranscriptionResult>(
            ["-m", "chord_engine.cli", "transcribe-lyrics", request.audioPath, "--model", normalizeLyricsModel(request.model)],
            "LYRICS",
            { timeoutMs: 10 * 60_000, jobId }
        );
    }

    async function removeVocals(request: RemoveVocalsRequest, jobId?: string): Promise<VocalRemovalResult> {
        if (!request.audioPath) {
            return enginePlaceholder("VOCAL_REMOVAL_INVALID_INPUT", "Audio path is required") satisfies VocalRemovalResult;
        }

        const vocalOutputRoot = path.join(config.engineOutputDir, "vocal-removal");
        const result = await runEngineJson<VocalRemovalResult>(
            ["-m", "chord_engine.cli", "remove-vocals", request.audioPath, "--output-root", vocalOutputRoot, "--model", "htdemucs"],
            "VOCAL_REMOVAL",
            {
                timeoutMs: 20 * 60_000,
                jobId,
                env: {
                    ...process.env,
                    DEMUCS_CACHE: path.join(vocalOutputRoot, "model-cache", "demucs"),
                    HF_HOME: path.join(vocalOutputRoot, "model-cache", "huggingface"),
                    TORCH_HOME: path.join(vocalOutputRoot, "model-cache", "torch")
                }
            }
        );
        return await withStreamableAudioResult(result);
    }

    async function pitchShiftAudio(request: PitchShiftAudioRequest, jobId?: string): Promise<PitchShiftResult> {
        if (!request.audioPath) {
            return enginePlaceholder("PITCH_SHIFT_INVALID_INPUT", "Audio path is required") satisfies PitchShiftResult;
        }

        const result = await runEngineJson<PitchShiftResult>(
            [
                "-m",
                "chord_engine.cli",
                "pitch-shift-audio",
                request.audioPath,
                "--output-root",
                path.join(config.engineOutputDir, "pitch-shift"),
                "--semitones",
                String(normalizeTransposeSemitones(request.semitones))
            ],
            "PITCH_SHIFT",
            { timeoutMs: 5 * 60_000, jobId }
        );
        return await withStreamableAudioResult(result);
    }

    async function withStreamableAudioResult<T extends { audio?: { path: string; streamUrl?: string }; error?: unknown }>(result: T): Promise<T> {
        if (result.error || !result.audio?.path) {
            return result;
        }

        const uploaded = await materializeAudioAsset(result.audio.path, config.audioDir, config.publicBaseUrl);
        return {
            ...result,
            audio: {
                ...result.audio,
                path: uploaded.audioPath,
                streamUrl: uploaded.audioStreamUrl
            }
        };
    }

    async function runEngineJson<T extends { version: string; error?: { code: string; message: string } }>(
        args: string[],
        errorPrefix: string,
        options: EngineCommandOptions
    ): Promise<T> {
        return await new Promise<T>((resolve) => {
            let settled = false;
            let stdout = "";
            let stderr = "";
            const child = spawn(config.pythonExecutable, args, {
                cwd: config.engineCwd,
                env: options.env ?? process.env
            });
            if (options.jobId) {
                config.processRegistry.register(options.jobId, child);
            }

            const finish = (result: T): void => {
                if (settled) {
                    return;
                }
                settled = true;
                clearTimeout(timeoutId);
                if (options.jobId) {
                    config.processRegistry.unregister(options.jobId);
                }
                if (stderr.trim()) {
                    console.error(`[api engine stderr] ${stderr.trim()}`);
                }
                resolve(result);
            };

            const timeoutId = setTimeout(() => {
                child.kill("SIGKILL");
                finish(enginePlaceholder(`${errorPrefix}_TIMEOUT`, "Engine request timed out") as T);
            }, options.timeoutMs);

            child.stdout.on("data", (chunk: Buffer | string) => {
                stdout += String(chunk);
            });

            child.stderr.on("data", (chunk: Buffer | string) => {
                stderr += String(chunk);
            });

            child.on("error", () => {
                finish(enginePlaceholder(`${errorPrefix}_SPAWN_FAILED`, "Could not start engine process") as T);
            });

            child.on("close", (code) => {
                if (options.jobId && config.processRegistry.isCancelled(options.jobId)) {
                    finish(enginePlaceholder(`${errorPrefix}_CANCELLED`, "Engine job cancelled") as T);
                    return;
                }
                const parsed = parseEngineJson<T>(stdout);
                if (!parsed) {
                    finish(enginePlaceholder(`${errorPrefix}_INVALID_JSON`, "Engine returned invalid JSON output") as T);
                    return;
                }
                if (code !== 0 && !parsed.error) {
                    finish(enginePlaceholder(`${errorPrefix}_EXIT_NONZERO`, "Engine failed with non-zero exit code") as T);
                    return;
                }
                finish(parsed);
            });
        });
    }

    return { analyzeAudio, transcribeLyrics, removeVocals, pitchShiftAudio };
}

export function enginePlaceholder(code: string, message: string) {
    return {
        version: "1",
        error: {
            code,
            message
        }
    };
}

function parseEngineJson<T>(stdout: string): T | null {
    const trimmed = stdout.trim();
    if (!trimmed) {
        return null;
    }

    try {
        return JSON.parse(trimmed) as T;
    } catch {
        return null;
    }
}
