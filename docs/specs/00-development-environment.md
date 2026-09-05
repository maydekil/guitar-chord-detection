# 00 — Development Environment & Toolchain

Dokumen ini WAJIB dibaca sebelum Phase 0.

Tujuannya agar AI agent memahami dengan jelas program, runtime, dependency, dan tool build yang digunakan.

---

# 1. Development Machine

Target development awal:

```text
macOS
Apple Silicon (M1/M2/M3/M4)
16 GB RAM atau lebih direkomendasikan
```

Primary IDE:

```text
Visual Studio Code
```

AI coding agent diasumsikan bekerja di workspace VS Code atau workspace repository yang sama.

---

# 2. Main Technology Stack

Desktop application:

```text
Electron
React
TypeScript
Vite
```

Package manager:

```text
pnpm
```

Audio/chord engine:

```text
Python 3.11
```

DSP/audio libraries:

```text
numpy
scipy
librosa
soundfile
pydantic
```

Audio decoder:

```text
FFmpeg
```

Testing:

```text
pytest
Vitest
```

Desktop packaging:

```text
electron-builder
```

---

# 3. Required Development Programs

Developer machine harus memiliki:

```text
Visual Studio Code
Git
Node.js
pnpm
Python 3.11
FFmpeg
```

Recommended Node.js:

```text
Node.js 22 LTS
```

Jangan menggunakan Node version yang sudah end-of-life.

---

# 4. Verify Development Environment

AI agent harus mengecek environment sebelum bootstrap.

Commands:

```bash
git --version
node --version
pnpm --version
python3.11 --version
ffmpeg -version
```

Jika salah satu dependency belum tersedia, agent harus berhenti pada environment setup dan mencatat dependency yang kurang.

Jangan diam-diam mengganti runtime version.

---

# 5. Install Tooling on macOS

Jika Homebrew tersedia:

```bash
brew install node
brew install pnpm
brew install python@3.11
brew install ffmpeg
```

Verifikasi:

```bash
node --version
pnpm --version
python3.11 --version
ffmpeg -version
```

Catatan:

Node dapat juga dikelola menggunakan `nvm`, `fnm`, atau tool version manager lain jika repository sudah memiliki standard tersendiri.

Jangan install runtime global secara paksa jika repository sudah memakai version manager.

---

# 6. Python Virtual Environment

Python dependency tidak boleh diinstall langsung ke system Python.

Dari root repository:

```bash
python3.11 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Upgrade tooling:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install engine:

```bash
pip install -e ./engine
```

Development dependencies:

```bash
pip install -e "./engine[dev]"
```

Jika optional dependency syntax belum tersedia, gunakan dependency group yang didefinisikan di `engine/pyproject.toml`.

---

# 7. Node Workspace Setup

Repository menggunakan pnpm workspace.

Install:

```bash
pnpm install
```

Expected workspace:

```text
apps/desktop
packages/shared
```

Python engine tetap berada di:

```text
engine/
```

dan tidak dikelola oleh pnpm.

---

# 8. Development Run Modes

## Python Engine Only

Activate venv:

```bash
source .venv/bin/activate
```

Run:

```bash
python -m chord_engine.cli analyze fixtures/audio/sample.wav
```

Output JSON harus muncul di stdout.

---

## Desktop Development

Recommended root command:

```bash
pnpm dev
```

Jika belum tersedia, AI agent harus membuat root script yang menjalankan desktop development.

Expected flow:

```text
Vite dev server
+
Electron main process
```

---

# 9. Test Commands

Python:

```bash
source .venv/bin/activate
pytest engine/tests
```

TypeScript:

```bash
pnpm test
```

Lint:

```bash
pnpm lint
```

Build:

```bash
pnpm build
```

Agent harus membuat scripts tersebut konsisten di root repository.

---

# 10. FFmpeg Usage

FFmpeg digunakan untuk decoding audio ketika diperlukan.

Selama development, engine boleh memakai FFmpeg dari development machine.

Tetapi production application TIDAK BOLEH bergantung pada FFmpeg global milik end-user.

Path FFmpeg tidak boleh di-hardcode seperti:

```text
/usr/local/bin/ffmpeg
/opt/homebrew/bin/ffmpeg
```

Gunakan resolver/configuration layer.

---

# 11. Development vs Production Runtime

Ini dua environment berbeda.

## Development

Boleh membutuhkan:

```text
Node.js
pnpm
Python 3.11
FFmpeg
```

terinstall di developer machine.

## Production

End-user TIDAK BOLEH diwajibkan menginstall:

```text
Node.js
pnpm
Python
pip
FFmpeg
```

Aplikasi desktop yang sudah dipackage harus membawa runtime/dependency yang dibutuhkan.

---

# 12. Production Packaging Strategy

Target awal:

```text
macOS Apple Silicon
```

Distribution artifact:

```text
.app
.dmg
```

Desktop app dipackage menggunakan:

```text
electron-builder
```

---

# 13. Python Engine Bundling

MVP development memakai Python process.

Untuk production distribution, Python system installation tidak boleh menjadi requirement.

Recommended strategy:

```text
Python source
   ↓
PyInstaller
   ↓
Standalone chord-engine executable
   ↓
Electron resources
```

Contoh artifact:

```text
resources/
└── engine/
    └── chord-engine
```

Electron menjalankan executable tersebut menggunakan:

```text
child_process.spawn
```

bukan:

```text
python script.py
```

pada production package.

---

# 14. FFmpeg Bundling

FFmpeg binary harus dibundle bersama app jika engine membutuhkannya.

Contoh:

```text
resources/
└── ffmpeg/
    └── ffmpeg
```

Pada runtime:

```text
Electron
  ↓
resolve process.resourcesPath
  ↓
bundled ffmpeg
```

Jangan mengasumsikan FFmpeg tersedia di PATH end-user.

---

# 15. Development Engine Invocation

Development mode boleh menggunakan:

```text
.venv/bin/python
```

Example:

```text
.venv/bin/python -m chord_engine.cli analyze input.wav
```

Path harus dihitung dari repository root.

Jangan menggunakan absolute path milik developer.

---

# 16. Production Engine Invocation

Production:

```text
Electron
  ↓
process.resourcesPath
  ↓
engine/chord-engine
```

Electron harus mendeteksi:

```text
app.isPackaged
```

untuk memilih development atau production engine path.

Concept:

```ts
if (app.isPackaged) {
  // bundled standalone engine
} else {
  // development Python venv
}
```

---

# 17. Platform Architecture

Pada fase pertama build hanya wajib:

```text
darwin-arm64
```

Future support:

```text
darwin-x64
win32-x64
```

Jangan menggabungkan multi-platform packaging ke MVP sebelum Mac Apple Silicon build stabil.

---

# 18. Repository Runtime Files

Recommended:

```text
guitar-chord-detector/
│
├── .venv/
│
├── apps/
│
├── engine/
│
├── packages/
│
├── fixtures/
│
├── resources/
│   ├── engine/
│   └── ffmpeg/
│
└── docs/
```

`.venv` tidak boleh di-commit.

---

# 19. Git Ignore Minimum

Pastikan `.gitignore` mencakup:

```text
.venv/
node_modules/
dist/
build/
coverage/
__pycache__/
.pytest_cache/
*.pyc
.DS_Store
```

Build resource hasil generate juga tidak boleh di-commit jika dapat direproduce.

---

# 20. Environment Acceptance Criteria

Environment phase dianggap selesai jika:

- Node dapat dijalankan.
- pnpm dapat dijalankan.
- Python 3.11 tersedia.
- `.venv` berhasil dibuat.
- engine dependencies berhasil diinstall.
- FFmpeg tersedia selama development.
- `pnpm install` berhasil.
- Python CLI minimal dapat dijalankan.
- tidak ada absolute local path di source code.
- production design tidak bergantung pada Python/FFmpeg global milik end-user.

---

# Important Rule for AI Agent

Jangan mencampur:

```text
development prerequisite
```

dengan:

```text
end-user prerequisite
```

Developer boleh membutuhkan Python dan FFmpeg.

User aplikasi final tidak boleh diwajibkan menginstall keduanya.
