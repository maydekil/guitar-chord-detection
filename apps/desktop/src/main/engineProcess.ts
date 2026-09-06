import { spawn } from "node:child_process";
import path from "node:path";

import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";
import type { PitchShiftResult } from "@gcd/shared/pitch";
import type { VocalRemovalResult } from "@gcd/shared/vocals";

const DEFAULT_TIMEOUT_MS = 30_000;
const LYRICS_TIMEOUT_MS = 10 * 60_000;
const VOCAL_REMOVAL_TIMEOUT_MS = 20 * 60_000;
const PITCH_SHIFT_TIMEOUT_MS = 5 * 60_000;

export interface EngineProcessConfig {
    appPath: string;
    timeoutMs?: number;
}

export interface EngineCommand {
    pythonExecutable: string;
    cwd: string;
    args: string[];
}

export interface EngineChildProcess {
    stdout: NodeJS.ReadableStream;
    stderr: NodeJS.ReadableStream;
    kill(signal?: NodeJS.Signals | number): boolean;
    on(event: "error", listener: (error: Error) => void): this;
    on(event: "close", listener: (code: number | null) => void): this;
}

export type SpawnEngineProcess = (
    command: string,
    args: string[],
    options: { cwd: string; env: NodeJS.ProcessEnv }
) => EngineChildProcess;

export function resolveDevelopmentEngineCommand(appPath: string, audioPath: string): EngineCommand {
    const repositoryRoot = path.resolve(appPath, "../..");
    const engineCwd = path.resolve(repositoryRoot, "engine");
    const pythonExecutable = path.resolve(repositoryRoot, ".venv", "bin", "python");

    return {
        pythonExecutable,
        cwd: engineCwd,
        args: ["-m", "chord_engine.cli", "analyze", audioPath]
    };
}

export function resolveDevelopmentLyricsCommand(appPath: string, audioPath: string, modelName = "small"): EngineCommand {
    const repositoryRoot = path.resolve(appPath, "../..");
    const engineCwd = path.resolve(repositoryRoot, "engine");
    const pythonExecutable = path.resolve(repositoryRoot, ".venv", "bin", "python");

    return {
        pythonExecutable,
        cwd: engineCwd,
        args: ["-m", "chord_engine.cli", "transcribe-lyrics", audioPath, "--model", modelName]
    };
}

export function resolveDevelopmentVocalRemovalCommand(
    appPath: string,
    audioPath: string,
    outputRoot: string,
    modelName = "htdemucs"
): EngineCommand {
    const repositoryRoot = path.resolve(appPath, "../..");
    const engineCwd = path.resolve(repositoryRoot, "engine");
    const pythonExecutable = path.resolve(repositoryRoot, ".venv", "bin", "python");

    return {
        pythonExecutable,
        cwd: engineCwd,
        args: ["-m", "chord_engine.cli", "remove-vocals", audioPath, "--output-root", outputRoot, "--model", modelName]
    };
}

export function resolveDevelopmentPitchShiftCommand(
    appPath: string,
    audioPath: string,
    outputRoot: string,
    semitones: number
): EngineCommand {
    const repositoryRoot = path.resolve(appPath, "../..");
    const engineCwd = path.resolve(repositoryRoot, "engine");
    const pythonExecutable = path.resolve(repositoryRoot, ".venv", "bin", "python");

    return {
        pythonExecutable,
        cwd: engineCwd,
        args: [
            "-m",
            "chord_engine.cli",
            "pitch-shift-audio",
            audioPath,
            "--output-root",
            outputRoot,
            "--semitones",
            String(semitones)
        ]
    };
}

export async function analyzeAudioInEngine(
    audioPath: string,
    config: EngineProcessConfig,
    spawnProcess: SpawnEngineProcess = defaultSpawnProcess
): Promise<ChordAnalysisResult> {
    if (!audioPath || audioPath.trim().length === 0) {
        return toProcessError("ENGINE_INVALID_INPUT", "Audio path is required");
    }

    const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const command = resolveDevelopmentEngineCommand(config.appPath, audioPath);

    return await new Promise<ChordAnalysisResult>((resolve) => {
        let settled = false;
        let stdout = "";
        let stderr = "";

        const child = spawnProcess(command.pythonExecutable, command.args, {
            cwd: command.cwd,
            env: process.env
        });

        const finish = (result: ChordAnalysisResult): void => {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timeoutId);
            if (stderr.trim().length > 0) {
                console.error(`[engine stderr] ${stderr.trim()}`);
            }
            resolve(result);
        };

        const timeoutId = setTimeout(() => {
            child.kill("SIGKILL");
            finish(toProcessError("ENGINE_TIMEOUT", "Engine analysis timed out"));
        }, timeoutMs);

        child.stdout.on("data", (chunk: Buffer | string) => {
            stdout += String(chunk);
        });

        child.stderr.on("data", (chunk: Buffer | string) => {
            stderr += String(chunk);
        });

        child.on("error", () => {
            finish(toProcessError("ENGINE_SPAWN_FAILED", "Could not start analysis engine"));
        });

        child.on("close", (code) => {
            const parsed = parseEngineStdout(stdout);
            if (!parsed) {
                finish(toProcessError("ENGINE_INVALID_JSON", "Engine returned invalid JSON output"));
                return;
            }

            if (code !== 0) {
                if ("error" in parsed) {
                    finish(parsed);
                    return;
                }
                finish(toProcessError("ENGINE_EXIT_NONZERO", "Engine failed with non-zero exit code"));
                return;
            }

            finish(parsed);
        });
    });
}

export async function transcribeLyricsInEngine(
    audioPath: string,
    config: EngineProcessConfig,
    modelName = "small",
    spawnProcess: SpawnEngineProcess = defaultSpawnProcess
): Promise<LyricsTranscriptionResult> {
    if (!audioPath || audioPath.trim().length === 0) {
        return toLyricsProcessError("LYRICS_INVALID_INPUT", "Audio path is required");
    }

    const timeoutMs = config.timeoutMs ?? LYRICS_TIMEOUT_MS;
    const command = resolveDevelopmentLyricsCommand(config.appPath, audioPath, modelName);

    return await new Promise<LyricsTranscriptionResult>((resolve) => {
        let settled = false;
        let stdout = "";
        let stderr = "";

        const child = spawnProcess(command.pythonExecutable, command.args, {
            cwd: command.cwd,
            env: process.env
        });

        const finish = (result: LyricsTranscriptionResult): void => {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timeoutId);
            if (stderr.trim().length > 0) {
                console.error(`[lyrics engine stderr] ${stderr.trim()}`);
            }
            resolve(result);
        };

        const timeoutId = setTimeout(() => {
            child.kill("SIGKILL");
            finish(toLyricsProcessError("LYRICS_TIMEOUT", "Lyrics transcription timed out"));
        }, timeoutMs);

        child.stdout.on("data", (chunk: Buffer | string) => {
            stdout += String(chunk);
        });

        child.stderr.on("data", (chunk: Buffer | string) => {
            stderr += String(chunk);
        });

        child.on("error", () => {
            finish(toLyricsProcessError("LYRICS_SPAWN_FAILED", "Could not start lyrics engine"));
        });

        child.on("close", (code) => {
            const parsed = parseLyricsEngineStdout(stdout);
            if (!parsed) {
                finish(toLyricsProcessError("LYRICS_INVALID_JSON", "Lyrics engine returned invalid JSON output"));
                return;
            }

            if (code !== 0) {
                if ("error" in parsed) {
                    finish(parsed);
                    return;
                }
                finish(toLyricsProcessError("LYRICS_EXIT_NONZERO", "Lyrics engine failed with non-zero exit code"));
                return;
            }

            finish(parsed);
        });
    });
}

export async function removeVocalsInEngine(
    audioPath: string,
    outputRoot: string,
    config: EngineProcessConfig,
    spawnProcess: SpawnEngineProcess = defaultSpawnProcess
): Promise<VocalRemovalResult> {
    if (!audioPath || audioPath.trim().length === 0) {
        return toVocalRemovalProcessError("VOCAL_REMOVAL_INVALID_INPUT", "Audio path is required");
    }

    const timeoutMs = config.timeoutMs ?? VOCAL_REMOVAL_TIMEOUT_MS;
    const command = resolveDevelopmentVocalRemovalCommand(config.appPath, audioPath, outputRoot);

    return await new Promise<VocalRemovalResult>((resolve) => {
        let settled = false;
        let stdout = "";
        let stderr = "";

        const child = spawnProcess(command.pythonExecutable, command.args, {
            cwd: command.cwd,
            env: {
                ...process.env,
                DEMUCS_CACHE: path.join(outputRoot, "model-cache", "demucs"),
                HF_HOME: path.join(outputRoot, "model-cache", "huggingface"),
                TORCH_HOME: path.join(outputRoot, "model-cache", "torch")
            }
        });

        const finish = (result: VocalRemovalResult): void => {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timeoutId);
            if (stderr.trim().length > 0) {
                console.error(`[vocal removal engine stderr] ${stderr.trim()}`);
            }
            resolve(result);
        };

        const timeoutId = setTimeout(() => {
            child.kill("SIGKILL");
            finish(toVocalRemovalProcessError("VOCAL_REMOVAL_TIMEOUT", "AI Vocal Remove timed out"));
        }, timeoutMs);

        child.stdout.on("data", (chunk: Buffer | string) => {
            stdout += String(chunk);
        });

        child.stderr.on("data", (chunk: Buffer | string) => {
            stderr += String(chunk);
        });

        child.on("error", () => {
            finish(toVocalRemovalProcessError("VOCAL_REMOVAL_SPAWN_FAILED", "Could not start AI Vocal Remove"));
        });

        child.on("close", (code) => {
            const parsed = parseVocalRemovalStdout(stdout);
            if (!parsed) {
                finish(toVocalRemovalProcessError("VOCAL_REMOVAL_INVALID_JSON", "AI Vocal Remove returned invalid JSON output"));
                return;
            }

            if (code !== 0) {
                if ("error" in parsed) {
                    finish(parsed);
                    return;
                }
                finish(toVocalRemovalProcessError("VOCAL_REMOVAL_EXIT_NONZERO", "AI Vocal Remove failed with non-zero exit code"));
                return;
            }

            finish(parsed);
        });
    });
}

export async function pitchShiftAudioInEngine(
    audioPath: string,
    outputRoot: string,
    semitones: number,
    config: EngineProcessConfig,
    spawnProcess: SpawnEngineProcess = defaultSpawnProcess
): Promise<PitchShiftResult> {
    if (!audioPath || audioPath.trim().length === 0) {
        return toPitchShiftProcessError("PITCH_SHIFT_INVALID_INPUT", "Audio path is required");
    }

    const timeoutMs = config.timeoutMs ?? PITCH_SHIFT_TIMEOUT_MS;
    const command = resolveDevelopmentPitchShiftCommand(config.appPath, audioPath, outputRoot, semitones);

    return await new Promise<PitchShiftResult>((resolve) => {
        let settled = false;
        let stdout = "";
        let stderr = "";

        const child = spawnProcess(command.pythonExecutable, command.args, {
            cwd: command.cwd,
            env: process.env
        });

        const finish = (result: PitchShiftResult): void => {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timeoutId);
            if (stderr.trim().length > 0) {
                console.error(`[pitch shift engine stderr] ${stderr.trim()}`);
            }
            resolve(result);
        };

        const timeoutId = setTimeout(() => {
            child.kill("SIGKILL");
            finish(toPitchShiftProcessError("PITCH_SHIFT_TIMEOUT", "Pitch shift timed out"));
        }, timeoutMs);

        child.stdout.on("data", (chunk: Buffer | string) => {
            stdout += String(chunk);
        });

        child.stderr.on("data", (chunk: Buffer | string) => {
            stderr += String(chunk);
        });

        child.on("error", () => {
            finish(toPitchShiftProcessError("PITCH_SHIFT_SPAWN_FAILED", "Could not start pitch shift engine"));
        });

        child.on("close", (code) => {
            const parsed = parsePitchShiftStdout(stdout);
            if (!parsed) {
                finish(toPitchShiftProcessError("PITCH_SHIFT_INVALID_JSON", "Pitch shift engine returned invalid JSON output"));
                return;
            }

            if (code !== 0) {
                if ("error" in parsed) {
                    finish(parsed);
                    return;
                }
                finish(toPitchShiftProcessError("PITCH_SHIFT_EXIT_NONZERO", "Pitch shift failed with non-zero exit code"));
                return;
            }

            finish(parsed);
        });
    });
}

function defaultSpawnProcess(
    command: string,
    args: string[],
    options: { cwd: string; env: NodeJS.ProcessEnv }
): EngineChildProcess {
    return spawn(command, args, options);
}

function parseEngineStdout(stdout: string): ChordAnalysisResult | null {
    const trimmed = stdout.trim();
    if (trimmed.length === 0) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (!isContractResult(parsed)) {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function parseLyricsEngineStdout(stdout: string): LyricsTranscriptionResult | null {
    const trimmed = stdout.trim();
    if (trimmed.length === 0) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (!isLyricsContractResult(parsed)) {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function parseVocalRemovalStdout(stdout: string): VocalRemovalResult | null {
    const trimmed = stdout.trim();
    if (trimmed.length === 0) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (!isVocalRemovalContractResult(parsed)) {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function parsePitchShiftStdout(stdout: string): PitchShiftResult | null {
    const trimmed = stdout.trim();
    if (trimmed.length === 0) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (!isPitchShiftContractResult(parsed)) {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function isContractResult(value: unknown): value is ChordAnalysisResult {
    if (!value || typeof value !== "object") {
        return false;
    }

    const candidate = value as { version?: unknown; source?: unknown; analysis?: unknown; error?: unknown };
    if (candidate.version !== "1") {
        return false;
    }
    if (candidate.error && typeof candidate.error === "object") {
        return true;
    }
    return Boolean(candidate.source && candidate.analysis);
}

function isLyricsContractResult(value: unknown): value is LyricsTranscriptionResult {
    if (!value || typeof value !== "object") {
        return false;
    }

    const candidate = value as { version?: unknown; source?: unknown; lyrics?: unknown; error?: unknown };
    if (candidate.version !== "1") {
        return false;
    }
    if (candidate.error && typeof candidate.error === "object") {
        return true;
    }
    return Boolean(candidate.source && candidate.lyrics);
}

function isVocalRemovalContractResult(value: unknown): value is VocalRemovalResult {
    if (!value || typeof value !== "object") {
        return false;
    }

    const candidate = value as { version?: unknown; source?: unknown; audio?: unknown; error?: unknown };
    if (candidate.version !== "1") {
        return false;
    }
    if (candidate.error && typeof candidate.error === "object") {
        return true;
    }
    return Boolean(candidate.source && candidate.audio);
}

function isPitchShiftContractResult(value: unknown): value is PitchShiftResult {
    if (!value || typeof value !== "object") {
        return false;
    }

    const candidate = value as { version?: unknown; source?: unknown; audio?: unknown; error?: unknown };
    if (candidate.version !== "1") {
        return false;
    }
    if (candidate.error && typeof candidate.error === "object") {
        return true;
    }
    return Boolean(candidate.source && candidate.audio);
}

function toProcessError(code: string, message: string): ChordAnalysisResult {
    return {
        version: "1",
        error: {
            code,
            message
        }
    };
}

function toLyricsProcessError(code: string, message: string): LyricsTranscriptionResult {
    return {
        version: "1",
        error: {
            code,
            message
        }
    };
}

function toVocalRemovalProcessError(code: string, message: string): VocalRemovalResult {
    return {
        version: "1",
        error: {
            code,
            message
        }
    };
}

function toPitchShiftProcessError(code: string, message: string): PitchShiftResult {
    return {
        version: "1",
        error: {
            code,
            message
        }
    };
}
