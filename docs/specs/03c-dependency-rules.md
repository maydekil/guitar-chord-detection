# 03c — Dependency Rules

## Desktop Direction

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

Forbidden: Renderer → filesystem, child_process, Python directly, or engine executable path. Python must not depend on Electron or React.

## Python Direction

```text
cli.py → analyze.py

analyze.py
 ├─ audio.py
 ├─ features.py
 ├─ detector.py
 ├─ smoothing.py
 └─ segmentation.py

detector.py → templates.py
```

Lower-level modules must not depend on CLI.

## Shared Package
Allowed: `desktop → packages/shared`.
Forbidden: `packages/shared → desktop` or Electron.

## Process Contract
`stdout` = machine-readable analysis JSON only.
`stderr` = diagnostic logs.

Cache belongs to Electron main. Python engine remains independently runnable.

Circular dependencies are forbidden. Do not solve them by moving unrelated logic into a generic utility module.
