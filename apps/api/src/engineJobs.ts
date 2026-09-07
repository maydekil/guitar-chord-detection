import { randomUUID } from "node:crypto";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import type { DatabaseSync } from "node:sqlite";

import type { EngineCommands, EngineJobKind } from "./engineCommands.js";
import { enginePlaceholder } from "./engineCommands.js";

export type EngineJobStatus = "queued" | "running" | "succeeded" | "failed";

export interface EngineJobRequest {
    kind: EngineJobKind;
    payload: unknown;
}

export interface EngineJob {
    id: string;
    kind: EngineJobKind;
    status: EngineJobStatus;
    progress: number;
    result: unknown;
    error: { code: string; message: string } | null;
    createdAt: string;
    updatedAt: string;
}

interface EngineJobRow {
    id: string;
    kind: string;
    status: string;
    progress: number;
    result_json: string | null;
    error_json: string | null;
    created_at: string;
    updated_at: string;
}

export interface EngineJobStore {
    createEngineJob(request: EngineJobRequest): EngineJob;
    getEngineJob(id: string): EngineJob | null;
    cancelEngineJob(id: string): EngineJob | null;
    toEngineJobSnapshot(job: EngineJob): EngineJob;
    registerProcess(jobId: string, child: ChildProcessWithoutNullStreams): void;
    unregisterProcess(jobId: string): void;
    isCancelled(jobId: string): boolean;
}

export function initializeEngineJobTables(db: DatabaseSync): void {
    db.exec(`
        CREATE TABLE IF NOT EXISTS engine_jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL,
            result_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_engine_jobs_updated_at ON engine_jobs(updated_at);
    `);
}

export function markInterruptedJobs(db: DatabaseSync): void {
    const now = new Date().toISOString();
    db.prepare(`
        UPDATE engine_jobs
        SET status = 'failed',
            progress = 100,
            error_json = ?,
            updated_at = ?
        WHERE status IN ('queued', 'running')
    `).run(
        JSON.stringify({
            code: "ENGINE_JOB_INTERRUPTED",
            message: "API server restarted before this job finished"
        }),
        now
    );
}

export function createEngineJobStore(db: DatabaseSync, commands: EngineCommands): EngineJobStore {
    const engineJobs = new Map<string, EngineJob>();
    const activeEngineProcesses = new Map<string, ChildProcessWithoutNullStreams>();
    const cancelledEngineJobs = new Set<string>();

    function createEngineJob(request: EngineJobRequest): EngineJob {
        const now = new Date().toISOString();
        const job: EngineJob = {
            id: randomUUID(),
            kind: request.kind,
            status: "queued",
            progress: 0,
            result: null,
            error: null,
            createdAt: now,
            updatedAt: now
        };

        engineJobs.set(job.id, job);
        persistEngineJob(job);
        void runEngineJob(job, request.payload);
        return toEngineJobSnapshot(job);
    }

    async function runEngineJob(job: EngineJob, payload: unknown): Promise<void> {
        updateEngineJob(job, { status: "running", progress: 5 });
        const progressTimer = startEstimatedProgress(job);

        try {
            const result = await executeEngineJob(job, payload);
            if (result && typeof result === "object" && "error" in result) {
                const error = (result as { error: { code?: unknown; message?: unknown } }).error;
                updateEngineJob(job, {
                    status: "failed",
                    progress: 100,
                    result,
                    error: {
                        code: String(error.code ?? "ENGINE_JOB_FAILED"),
                        message: String(error.message ?? "Engine job failed")
                    }
                });
                return;
            }

            updateEngineJob(job, {
                status: "succeeded",
                progress: 100,
                result,
                error: null
            });
        } catch (error) {
            updateEngineJob(job, {
                status: "failed",
                progress: 100,
                result: null,
                error: {
                    code: "ENGINE_JOB_EXCEPTION",
                    message: error instanceof Error ? error.message : "Engine job failed"
                }
            });
        } finally {
            clearInterval(progressTimer);
            activeEngineProcesses.delete(job.id);
            cancelledEngineJobs.delete(job.id);
        }
    }

    async function executeEngineJob(job: EngineJob, payload: unknown): Promise<unknown> {
        if (job.kind === "analysis") {
            return await commands.analyzeAudio(payload as Parameters<EngineCommands["analyzeAudio"]>[0], job.id);
        }
        if (job.kind === "lyrics") {
            return await commands.transcribeLyrics(payload as Parameters<EngineCommands["transcribeLyrics"]>[0], job.id);
        }
        if (job.kind === "vocals") {
            return await commands.removeVocals(payload as Parameters<EngineCommands["removeVocals"]>[0], job.id);
        }
        if (job.kind === "pitch-shift") {
            return await commands.pitchShiftAudio(payload as Parameters<EngineCommands["pitchShiftAudio"]>[0], job.id);
        }
        return enginePlaceholder("ENGINE_JOB_UNKNOWN_KIND", "Unknown engine job kind");
    }

    function cancelEngineJob(id: string): EngineJob | null {
        const job = getEngineJob(id);
        if (!job) {
            return null;
        }
        if (job.status !== "queued" && job.status !== "running") {
            return job;
        }

        cancelledEngineJobs.add(id);
        const child = activeEngineProcesses.get(id);
        if (child) {
            child.kill("SIGTERM");
            setTimeout(() => {
                if (child.exitCode === null && child.signalCode === null) {
                    child.kill("SIGKILL");
                }
            }, 1200);
        }
        updateEngineJob(job, {
            status: "failed",
            progress: 100,
            error: {
                code: "ENGINE_JOB_CANCELLED",
                message: "Engine job cancelled"
            }
        });
        engineJobs.set(job.id, job);
        return job;
    }

    function startEstimatedProgress(job: EngineJob): NodeJS.Timeout {
        const startedAt = Date.now();
        return setInterval(() => {
            if (job.status !== "running") {
                return;
            }
            const elapsed = Date.now() - startedAt;
            const estimated = estimateJobProgress(job.kind, elapsed);
            if (estimated > job.progress && estimated < 96) {
                updateEngineJob(job, { progress: estimated });
            }
        }, 650);
    }

    function estimateJobProgress(kind: EngineJobKind, elapsedMs: number): number {
        const expectedMs = getExpectedJobDurationMs(kind);
        const ratio = Math.max(0, Math.min(1, elapsedMs / expectedMs));
        const easedRatio = 1 - ((1 - ratio) ** 2);
        return 5 + Math.min(90, Math.floor(easedRatio * 90));
    }

    function getExpectedJobDurationMs(kind: EngineJobKind): number {
        if (kind === "lyrics") {
            return 90_000;
        }
        if (kind === "vocals") {
            return 180_000;
        }
        if (kind === "pitch-shift") {
            return 45_000;
        }
        return 90_000;
    }

    function updateEngineJob(job: EngineJob, patch: Partial<EngineJob>): void {
        Object.assign(job, patch, { updatedAt: new Date().toISOString() });
        persistEngineJob(job);
    }

    function toEngineJobSnapshot(job: EngineJob): EngineJob {
        return { ...job };
    }

    function getEngineJob(id: string): EngineJob | null {
        const memoryJob = engineJobs.get(id);
        if (memoryJob) {
            return memoryJob;
        }

        const row = db.prepare("SELECT * FROM engine_jobs WHERE id = ?").get(id) as EngineJobRow | undefined;
        return row ? rowToEngineJob(row) : null;
    }

    function persistEngineJob(job: EngineJob): void {
        db.prepare(`
            INSERT INTO engine_jobs (
                id, kind, status, progress, result_json, error_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                status = excluded.status,
                progress = excluded.progress,
                result_json = excluded.result_json,
                error_json = excluded.error_json,
                updated_at = excluded.updated_at
        `).run(
            job.id,
            job.kind,
            job.status,
            job.progress,
            JSON.stringify(job.result),
            JSON.stringify(job.error),
            job.createdAt,
            job.updatedAt
        );
    }

    function rowToEngineJob(row: EngineJobRow): EngineJob {
        return {
            id: row.id,
            kind: row.kind as EngineJobKind,
            status: row.status as EngineJobStatus,
            progress: row.progress,
            result: row.result_json ? JSON.parse(row.result_json) as unknown : null,
            error: row.error_json ? JSON.parse(row.error_json) as EngineJob["error"] : null,
            createdAt: row.created_at,
            updatedAt: row.updated_at
        };
    }

    return {
        createEngineJob,
        getEngineJob,
        cancelEngineJob,
        toEngineJobSnapshot,
        registerProcess(jobId, child) {
            activeEngineProcesses.set(jobId, child);
        },
        unregisterProcess(jobId) {
            activeEngineProcesses.delete(jobId);
        },
        isCancelled(jobId) {
            return cancelledEngineJobs.has(jobId);
        }
    };
}
