# Guitar Chord Detector — AI Agent Build Specification

## Goal
Build an offline desktop application that opens WAV/MP3, analyzes major/minor chords, renders a chord timeline, plays audio, highlights the active chord, and caches analysis locally.

## Mandatory First Read
Before modifying the repository read:

```text
00-development-environment.md
01-product-scope.md
02-architecture.md
03-project-structure.md
03a-repository-rules.md
03b-file-ownership.md
03c-dependency-rules.md
11-agent-task-plan.md
```

Then read only feature specs relevant to the active task.

## Specification Precedence
If instructions conflict:

```text
1. 03a-repository-rules.md
2. 02-architecture.md
3. 03-project-structure.md
4. 03b-file-ownership.md
5. 03c-dependency-rules.md
6. 11-agent-task-plan.md
7. feature-specific spec
8. implementation-notes.md
```

Acceptance criteria remain mandatory. If a conflict cannot be resolved: STOP, document it, and request a human decision.

## Agent Protocol

```text
IDENTIFY ONE TASK
→ READ REQUIRED SPECS
→ CHECK ALLOWED FILES/PATHS
→ IMPLEMENT
→ TEST
→ VALIDATE RELEVANT ACCEPTANCE CRITERIA
→ RECORD DEVIATIONS IF ANY
→ STOP
```

Never automatically continue to another task or phase.

## Technology
VS Code; Node.js 22 LTS; pnpm; Electron + React + TypeScript + Vite; Python 3.11; librosa + NumPy + SciPy + SoundFile; FFmpeg; pytest; Vitest; electron-builder; PyInstaller.

Production must not require end-users to install Python or FFmpeg.

## Definition of Done
Functional MVP: `AC-01` through `AC-20` PASS.
Distribution-ready macOS Apple Silicon build: `AC-01` through `AC-25` PASS.
