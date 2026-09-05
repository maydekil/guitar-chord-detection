import { PassThrough } from "node:stream";
import { describe, expect, it, vi } from "vitest";

import {
    analyzeAudioInEngine,
    resolveDevelopmentEngineCommand,
    type EngineChildProcess,
    type SpawnEngineProcess
} from "./engineProcess";

function createFakeChild(): EngineChildProcess & {
    emitError: (error: Error) => void;
    emitClose: (code: number | null) => void;
    emitStdout: (text: string) => void;
    emitStderr: (text: string) => void;
} {
    const stdout = new PassThrough();
    const stderr = new PassThrough();
    const errorListeners: Array<(error: Error) => void> = [];
    const closeListeners: Array<(code: number | null) => void> = [];

    return {
        stdout,
        stderr,
        kill: vi.fn(() => true),
        on(event: "error" | "close", listener: ((error: Error) => void) | ((code: number | null) => void)) {
            if (event === "error") {
                errorListeners.push(listener as (error: Error) => void);
            } else {
                closeListeners.push(listener as (code: number | null) => void);
            }
            return this;
        },
        emitError(error: Error) {
            for (const listener of errorListeners) {
                listener(error);
            }
        },
        emitClose(code: number | null) {
            for (const listener of closeListeners) {
                listener(code);
            }
        },
        emitStdout(text: string) {
            stdout.write(text);
        },
        emitStderr(text: string) {
            stderr.write(text);
        }
    };
}

describe("resolveDevelopmentEngineCommand", () => {
    it("resolves repository .venv and engine cwd from app path", () => {
        const command = resolveDevelopmentEngineCommand("/repo/apps/desktop", "/tmp/song.wav");

        expect(command.pythonExecutable).toBe("/repo/.venv/bin/python");
        expect(command.cwd).toBe("/repo/engine");
        expect(command.args).toEqual(["-m", "chord_engine.cli", "analyze", "/tmp/song.wav"]);
    });
});

describe("analyzeAudioInEngine", () => {
    it("returns success JSON when process exits zero with valid stdout", async () => {
        const child = createFakeChild();
        const spawnProcess: SpawnEngineProcess = vi.fn(() => {
            queueMicrotask(() => {
                child.emitStdout('{"version":"1","source":{"path":"/tmp/song.wav","duration":1.0,"sampleRate":22050},"analysis":{"algorithm":"chroma-template-v1","chords":[]}}');
                child.emitClose(0);
            });
            return child;
        });

        const result = await analyzeAudioInEngine("/tmp/song.wav", { appPath: "/repo/apps/desktop" }, spawnProcess);

        expect("analysis" in result).toBe(true);
        expect("error" in result).toBe(false);
    });

    it("returns engine JSON error when process exits non-zero with valid error payload", async () => {
        const child = createFakeChild();
        const spawnProcess: SpawnEngineProcess = vi.fn(() => {
            queueMicrotask(() => {
                child.emitStdout('{"version":"1","error":{"code":"AUDIO_DECODE_FAILED","message":"Unable to decode audio file"}}');
                child.emitStderr("decoder warning");
                child.emitClose(1);
            });
            return child;
        });

        const stderrSpy = vi.spyOn(console, "error").mockImplementation(() => { });
        const result = await analyzeAudioInEngine("/tmp/missing.wav", { appPath: "/repo/apps/desktop" }, spawnProcess);

        expect("error" in result).toBe(true);
        if ("error" in result) {
            expect(result.error.code).toBe("AUDIO_DECODE_FAILED");
        }
        expect(stderrSpy).toHaveBeenCalled();
        stderrSpy.mockRestore();
    });

    it("returns controlled error when process cannot be spawned", async () => {
        const child = createFakeChild();
        const spawnProcess: SpawnEngineProcess = vi.fn(() => {
            queueMicrotask(() => child.emitError(new Error("spawn failed")));
            return child;
        });

        const result = await analyzeAudioInEngine("/tmp/song.wav", { appPath: "/repo/apps/desktop" }, spawnProcess);

        expect("error" in result).toBe(true);
        if ("error" in result) {
            expect(result.error.code).toBe("ENGINE_SPAWN_FAILED");
        }
    });

    it("returns controlled error when stdout is malformed JSON", async () => {
        const child = createFakeChild();
        const spawnProcess: SpawnEngineProcess = vi.fn(() => {
            queueMicrotask(() => {
                child.emitStdout("not-json");
                child.emitClose(0);
            });
            return child;
        });

        const result = await analyzeAudioInEngine("/tmp/song.wav", { appPath: "/repo/apps/desktop" }, spawnProcess);

        expect("error" in result).toBe(true);
        if ("error" in result) {
            expect(result.error.code).toBe("ENGINE_INVALID_JSON");
        }
    });

    it("returns timeout error and kills process when timeout elapses", async () => {
        vi.useFakeTimers();
        const child = createFakeChild();
        const spawnProcess: SpawnEngineProcess = vi.fn(() => child);

        const promise = analyzeAudioInEngine("/tmp/song.wav", { appPath: "/repo/apps/desktop", timeoutMs: 5 }, spawnProcess);
        await vi.advanceTimersByTimeAsync(5);

        const result = await promise;
        expect(child.kill).toHaveBeenCalledWith("SIGKILL");
        expect("error" in result).toBe(true);
        if ("error" in result) {
            expect(result.error.code).toBe("ENGINE_TIMEOUT");
        }
        vi.useRealTimers();
    });

    it("returns input validation error for empty audio path", async () => {
        const spawnProcess: SpawnEngineProcess = vi.fn(() => createFakeChild());

        const result = await analyzeAudioInEngine(" ", { appPath: "/repo/apps/desktop" }, spawnProcess);

        expect(spawnProcess).not.toHaveBeenCalled();
        expect("error" in result).toBe(true);
        if ("error" in result) {
            expect(result.error.code).toBe("ENGINE_INVALID_INPUT");
        }
    });
});
