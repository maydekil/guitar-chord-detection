# GitHub Copilot Agent Instructions

This repository is governed by explicit project specifications located under:

```text
docs/specs/
```

GitHub Copilot Agent MUST treat these specifications as the source of truth for repository structure, architecture, implementation scope, task boundaries, testing, and acceptance criteria.

---

# 1. Primary Objective

Build an offline desktop application that can:

1. Open local WAV and MP3 audio files.
2. Analyze the harmonic content of a song.
3. Detect major and minor chords.
4. Detect `N` for no-chord / silence.
5. Display detected chords on a song timeline.
6. Play, pause, and seek audio.
7. Highlight the active chord according to the current playback position.
8. Cache completed analysis locally.
9. Work offline after dependencies are installed.
10. Be packaged as a macOS Apple Silicon desktop application.

The MVP supports:

```text
12 major chords
12 minor chords
N
```

Do not expand the chord vocabulary unless the specification is explicitly changed.

---

# 2. Mandatory Specification Location

All project specifications are located under:

```text
docs/specs/
```

The primary specification entry point is:

```text
docs/specs/README.md
```

Before making repository changes, Copilot MUST inspect the specification files required by the active task.

---

# 3. Mandatory First Read

Before the first implementation task in a new session, read:

```text
docs/specs/README.md
docs/specs/00-development-environment.md
docs/specs/01-product-scope.md
docs/specs/02-architecture.md
docs/specs/03-project-structure.md
docs/specs/03a-repository-rules.md
docs/specs/03b-file-ownership.md
docs/specs/03c-dependency-rules.md
docs/specs/11-agent-task-plan.md
```

After the repository architecture is understood, do not repeatedly load every specification for every task.

For later tasks, read only:

1. the repository governance specifications;
2. the active task definition;
3. the feature-specific specification;
4. the relevant acceptance criteria.

This is intended to keep agent context small and focused.

---

# 4. Specification Priority

If instructions appear to conflict, use this priority order:

```text
1. docs/specs/03a-repository-rules.md
2. docs/specs/02-architecture.md
3. docs/specs/03-project-structure.md
4. docs/specs/03b-file-ownership.md
5. docs/specs/03c-dependency-rules.md
6. docs/specs/11-agent-task-plan.md
7. feature-specific specification
8. docs/implementation-notes.md
```

Acceptance criteria remain mandatory.

If the conflict cannot be resolved using this priority:

```text
STOP
```

Do not silently redesign the architecture.

Explain the conflict and wait for human direction.

---

# 5. One Task at a Time

Copilot Agent MUST work on only one task at a time unless the user explicitly requests multiple tasks.

The active task must come from:

```text
docs/specs/11-agent-task-plan.md
```

For every task, follow this process:

```text
IDENTIFY ACTIVE PHASE
→ IDENTIFY ACTIVE TASK
→ READ REQUIRED SPECS
→ CHECK ALLOWED FILES/PATHS
→ INSPECT EXISTING CODE
→ IMPLEMENT ONLY THE ACTIVE TASK
→ ADD OR UPDATE REQUIRED TESTS
→ RUN VERIFICATION
→ CHECK RELEVANT ACCEPTANCE CRITERIA
→ RECORD DEVIATIONS OR BLOCKERS
→ STOP
```

Never automatically continue to the next task or phase.

---

# 6. Repository Structure Is Locked

The canonical repository structure is defined by:

```text
docs/specs/03-project-structure.md
```

Copilot MUST follow that structure.

Allowed top-level directories are:

```text
.github/
.vscode/
apps/
docs/
engine/
fixtures/
packages/
resources/
```

Do not create new top-level architecture directories such as:

```text
backend/
frontend/
client/
server/
api/
python/
services/
src/
lib/
common/
shared/
tests/
scripts/
```

unless the specification is explicitly changed by the user.

---

# 7. Folder Creation Rule

Copilot may create a subdirectory only when:

1. it already exists in `docs/specs/03-project-structure.md`; or
2. the active task explicitly allows it.

If implementation appears to require a new directory that is not defined:

```text
STOP
```

Record the requirement in:

```text
docs/implementation-notes.md
```

Do not invent a new project structure.

---

# 8. File Ownership Rules

File responsibilities are defined in:

```text
docs/specs/03b-file-ownership.md
```

Copilot MUST respect ownership boundaries.

Examples:

```text
engine/chord_engine/audio.py
```

must handle:

```text
audio decode
mono conversion
resampling
audio validation
```

and must not handle:

```text
chord inference
Electron IPC
UI
cache
```

Similarly:

```text
engine/chord_engine/detector.py
```

owns chord scoring and prediction.

```text
engine/chord_engine/segmentation.py
```

owns chord segment generation.

```text
apps/desktop/src/main/
```

owns Electron main-process behavior.

```text
apps/desktop/src/preload/
```

owns the safe renderer bridge.

```text
apps/desktop/src/renderer/
```

owns React UI and playback interaction.

Do not move responsibilities between layers for convenience.

---

# 9. Dependency Direction

The desktop dependency direction is:

```text
React Renderer
      ↓
Preload API
      ↓
Electron Main
      ↓
Engine Process Adapter
      ↓
Python / Packaged Engine
```

Do not create dependencies in the opposite direction.

The renderer MUST NOT directly use:

```text
fs
path access to arbitrary local files
child_process
Python
FFmpeg executable paths
engine executable paths
```

The renderer must interact through the preload bridge.

---

# 10. Python Dependency Direction

The preferred Python architecture is:

```text
cli.py
  ↓
analyze.py

analyze.py
 ├── audio.py
 ├── features.py
 ├── detector.py
 ├── smoothing.py
 └── segmentation.py

detector.py
  ↓
templates.py
```

Lower-level Python modules must not depend on:

```text
cli.py
Electron
React
desktop application code
```

Avoid circular imports.

---

# 11. Shared TypeScript Package

Shared contracts belong in:

```text
packages/shared/
```

This package may contain:

```text
TypeScript interfaces
analysis JSON contract
pure shared types
contract validation helpers
```

It must not contain:

```text
Electron application logic
React components
filesystem logic
Python implementation details
```

---

# 12. No Generic Dumping Ground

Do not create generic files such as:

```text
utils.py
helpers.py
common.py
misc.py
utils.ts
helpers.ts
common.ts
misc.ts
```

unless the specification explicitly requires them.

Prefer files with clear ownership and specific responsibilities.

---

# 13. No Unrequested Technology

Do not add technologies that are outside the current specification.

Do not add:

```text
database
HTTP backend server
Docker
authentication
user accounts
cloud processing
remote audio service
source separation
stem extraction
deep-learning runtime
ML model
telemetry
analytics
Redux
large UI framework
external API integration
```

unless the user explicitly approves a specification change.

---

# 14. Development Technology Stack

The approved stack is:

```text
IDE:
Visual Studio Code

Desktop:
Electron
React
TypeScript
Vite

Node runtime:
Node.js 24
Minimum supported: Node.js 24.x
Development target: Node.js 24.x

Package manager:
pnpm

Audio engine:
Python 3.11

DSP libraries:
librosa
NumPy
SciPy
SoundFile

Audio decoder:
FFmpeg

Python testing:
pytest

TypeScript testing:
Vitest

Desktop packaging:
electron-builder

Python production packaging:
PyInstaller
```

Do not replace this stack without a documented technical blocker and user approval.

---

# 15. Development Environment vs Production Environment

Development machines may require:

```text
Node.js
pnpm
Python 3.11
FFmpeg
```

The final packaged desktop application MUST NOT require the end-user to manually install:

```text
Node.js
pnpm
Python
pip
FFmpeg
```

Production runtime dependencies must be bundled with the application where required.

---

# 16. Python Virtual Environment

Python dependencies must use a project virtual environment.

Expected location:

```text
.venv/
```

Do not install project Python dependencies into the macOS system Python.

Typical setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

The `.venv/` directory must not be committed.

---

# 17. Absolute Paths Are Forbidden

Do not hardcode developer-specific paths such as:

```text
/Users/someone/project
/opt/homebrew/bin/ffmpeg
/usr/local/bin/python
```

Development paths must be resolved relative to the repository.

Production resources must be resolved from the packaged Electron application.

For packaged resources, prefer appropriate Electron runtime path resolution such as:

```text
process.resourcesPath
```

where required by the implementation.

---

# 18. Audio Engine Rules

The first chord-detection implementation must use the baseline defined in the specifications:

```text
Audio
→ preprocessing
→ CQT/chroma
→ major/minor template comparison
→ confidence
→ no-chord detection
→ temporal smoothing
→ segment merging
```

Do not introduce deep learning for the initial MVP.

---

# 19. Supported Chords for MVP

Only these chord classes are required:

Major:

```text
C
C#
D
D#
E
F
F#
G
G#
A
A#
B
```

Minor:

```text
Cm
C#m
Dm
D#m
Em
Fm
F#m
Gm
G#m
Am
A#m
Bm
```

Special:

```text
N
```

Do not implement chord extensions such as:

```text
7
maj7
m7
sus2
sus4
dim
aug
slash chords
```

during MVP unless the user changes the scope.

---

# 20. Audio Analysis Contract

The Python engine and desktop application communicate using the contract defined in:

```text
docs/specs/06-analysis-api-contract.md
```

Important rules:

```text
stdout = machine-readable JSON only
stderr = logs and diagnostics
```

Do not mix logs into JSON stdout.

The output contract must remain versioned.

---

# 21. CLI Independence

The Python engine must work independently of Electron.

The following workflow must remain possible:

```bash
python -m chord_engine.cli analyze path/to/audio.wav
```

Electron must not be required to test the chord engine.

---

# 22. Cache Responsibility

Analysis cache belongs to the Electron main-process layer.

The Python engine must not require cache state to operate.

Cache keys should include:

```text
file content fingerprint
analysis algorithm/version
```

Do not rely only on filename for cache identity.

---

# 23. Electron Security

Electron security rules are mandatory.

Renderer must not receive unrestricted Node access.

Use a safe preload bridge.

Do not enable:

```text
nodeIntegration: true
```

unless the architecture is explicitly revised and approved.

Expose only the minimum required functions.

---

# 24. Testing Is Mandatory

Every implementation task must include relevant tests where specified.

Python tests belong under:

```text
engine/tests/
```

TypeScript/Desktop tests belong inside the appropriate desktop/shared package structure.

Do not declare a task complete without running the required verification.

---

# 25. Test Audio Rules

Do not commit full commercial copyrighted songs.

Fixtures should use:

```text
synthetic generated tones
synthetic chord progressions
self-recorded samples
public-domain audio
```

Prefer short test fixtures.

---

# 26. Acceptance Criteria

Acceptance criteria are defined in:

```text
docs/specs/10-acceptance-criteria.md
```

A task is not complete merely because the code compiles.

Relevant acceptance criteria must pass.

Functional MVP requires the functional acceptance criteria to pass.

Repository governance and dependency-boundary acceptance criteria must also be respected throughout implementation.

---

# 27. Implementation Notes

Use:

```text
docs/implementation-notes.md
```

to record:

```text
technical decisions
deviations from specification
blockers
algorithm tuning
performance observations
temporary compromises
```

Do not modify the authoritative specification merely to make an implementation mismatch appear valid.

---

# 28. Error Handling

Do not expose internal stack traces to the desktop user.

User-facing errors should be concise and understandable.

Diagnostic detail may be logged separately.

Python CLI failures must use controlled non-zero exit codes where appropriate.

---

# 29. Change Discipline

Before editing code, inspect the existing implementation.

Prefer the smallest coherent change required by the active task.

Do not perform unrelated refactors.

Do not rename files or reorganize directories unless the active task requires it.

Do not improve unrelated code while implementing a scoped task.

---

# 30. Do Not Auto-Continue

This rule is critical.

After completing the active task:

```text
STOP
```

Report:

1. files created or modified;
2. tests or verification executed;
3. acceptance criteria checked;
4. PASS / FAIL / BLOCKED status;
5. any deviations or blockers.

Do not begin the next task automatically.

---

# 31. When a Task Is Blocked

If a task cannot be completed without violating specifications:

1. do not work around the rule silently;
2. do not change architecture;
3. do not create a new subsystem;
4. record the blocker;
5. explain exactly why the task is blocked;
6. request human direction.

---

# 32. Initial Project State

At the very beginning, the repository may contain only:

```text
.github/
└── copilot-instructions.md

docs/
└── specs/
    └── specification files
```

This is valid.

Do not assume missing application directories are errors.

They should be created only when the appropriate bootstrap task authorizes them.

---

# 33. First Execution

The first Copilot Agent execution should normally be:

```text
Phase -1 — Environment Validation
```

from:

```text
docs/specs/11-agent-task-plan.md
```

Do not bootstrap the application during Phase -1.

Do not implement application code.

Do not automatically proceed to Phase 0.

---

# 34. Project Philosophy

Prefer:

```text
clear boundaries
small modules
deterministic behavior
testable components
offline operation
explicit contracts
incremental implementation
```

Avoid:

```text
premature abstraction
architecture drift
large single-file implementations
hidden dependencies
implicit global state
unnecessary frameworks
unrequested features
```

The goal is not to create the largest implementation.

The goal is to create the smallest correct implementation that satisfies the active task and its acceptance criteria.
