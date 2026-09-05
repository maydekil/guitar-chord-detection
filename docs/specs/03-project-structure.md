# 03 — Canonical Project Structure

This structure is authoritative.

```text
guitar-chord-detector/
├── .vscode/
│   └── settings.json
├── apps/
│   └── desktop/
│       ├── src/
│       │   ├── main/
│       │   ├── preload/
│       │   └── renderer/
│       ├── package.json
│       ├── tsconfig.json
│       └── vite.config.ts
├── docs/
│   ├── specs/
│   │   ├── README.md
│   │   ├── 00-development-environment.md
│   │   ├── 01-product-scope.md
│   │   ├── 02-architecture.md
│   │   ├── 03-project-structure.md
│   │   ├── 03a-repository-rules.md
│   │   ├── 03b-file-ownership.md
│   │   ├── 03c-dependency-rules.md
│   │   ├── 04-audio-engine.md
│   │   ├── 05-chord-detection.md
│   │   ├── 06-analysis-api-contract.md
│   │   ├── 07-desktop-app.md
│   │   ├── 08-waveform-timeline.md
│   │   ├── 09-testing-strategy.md
│   │   ├── 10-acceptance-criteria.md
│   │   ├── 11-agent-task-plan.md
│   │   └── 12-implementation-notes-template.md
│   └── implementation-notes.md
├── engine/
│   ├── pyproject.toml
│   ├── chord_engine/
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   ├── features.py
│   │   ├── templates.py
│   │   ├── detector.py
│   │   ├── smoothing.py
│   │   ├── segmentation.py
│   │   ├── models.py
│   │   ├── analyze.py
│   │   └── cli.py
│   └── tests/
├── fixtures/
│   └── audio/
├── packages/
│   └── shared/
│       ├── src/
│       │   └── analysis.ts
│       ├── package.json
│       └── tsconfig.json
├── resources/
│   ├── engine/
│   └── ffmpeg/
├── .gitignore
├── package.json
├── pnpm-workspace.yaml
└── README.md
```

Do not put subsystem implementation in repository root. Python uses `snake_case`; TypeScript uses `camelCase` and `PascalCase`.

See `03a-repository-rules.md`, `03b-file-ownership.md`, and `03c-dependency-rules.md`.
