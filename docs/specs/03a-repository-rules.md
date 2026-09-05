# 03a — Repository Rules

## Allowed Top-Level Directories
Only `.vscode/`, `apps/`, `docs/`, `engine/`, `fixtures/`, `packages/`, and `resources/`.

Allowed root files: `.gitignore`, `README.md`, `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`.

Do not create top-level `backend/`, `frontend/`, `client/`, `server/`, `api/`, `python/`, `services/`, `src/`, `lib/`, `common/`, `shared/`, `tests/`, or `scripts/`.

## Folder Creation Rule
Create a subfolder only when it exists in `03-project-structure.md` or the active task explicitly allows it. Otherwise STOP, record the need in `docs/implementation-notes.md`, and request a human decision.

## Scope Lock
For every task: identify task → read required specs → inspect allowed paths → implement only there → test → validate relevant AC → STOP. Never automatically continue to the next task.

## Architecture Guardrails
Do not add database, HTTP backend, Docker, authentication, cloud processing, source separation, ML/deep-learning runtime, telemetry, or unrelated frameworks unless the specification is explicitly revised.

Do not create generic dumping-ground files such as `utils.py`, `helpers.py`, `common.py`, `utils.ts`, or `helpers.ts` without explicit approval.

## Paths
Never hardcode developer-specific absolute paths. Development paths must be repository-relative; production paths must resolve packaged resources.

## Architecture Changes
If implementation requires changing architecture, do not silently change the spec. Record the blocker and STOP.
