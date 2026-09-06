import { spawn } from "node:child_process";
import path from "node:path";

import type { ChordAnalysisResult } from "@gcd/shared/analysis";
import type { LyricsTranscriptionResult } from "@gcd/shared/lyrics";

const DEFAULT_TIMEOUT_MS = 30_000;
const LYRICS_TIMEOUT_MS = 10 * 60_000;

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
