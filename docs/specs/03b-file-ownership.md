# 03b — File Ownership

## Python Engine
- `audio.py`: decode, mono conversion, resampling, audio validation only.
- `features.py`: CQT/chroma extraction.
- `templates.py`: pitch classes and generated major/minor templates.
- `detector.py`: template scoring, frame prediction, confidence, no-chord decision.
- `smoothing.py`: temporal stabilization.
- `segmentation.py`: predictions to non-overlapping time segments, merge, minimum duration.
- `models.py`: typed data models/validation; no algorithm dumping.
- `analyze.py`: orchestration only: audio → features → detector → smoothing → segmentation.
- `cli.py`: CLI parsing, JSON stdout, stderr diagnostics, exit codes; no DSP logic.
- `engine/tests/`: Python engine tests only.

## Desktop
- `apps/desktop/src/main/`: Electron lifecycle, file dialog, child process, cache, engine path resolution.
- `apps/desktop/src/preload/`: minimal safe `contextBridge`.
- `apps/desktop/src/renderer/`: React UI, playback, timeline and interaction. Must not access filesystem, child_process, Python, or engine executable paths directly.

## Shared
`packages/shared/`: TypeScript analysis contract and pure shared types only.

## Fixtures
`fixtures/audio/`: generated, self-recorded, or public-domain short test audio only.

## Resources
- `resources/engine/`: generated standalone production engine.
- `resources/ffmpeg/`: packaged FFmpeg runtime.

## Documentation
- `docs/specs/`: authoritative build specifications.
- `docs/implementation-notes.md`: deviations, blockers, decisions, performance and tuning notes.
