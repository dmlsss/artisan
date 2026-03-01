# Artisan Project Profile

## What Is This?

**Artisan** is a cross-platform desktop application (Python 3.12+ / PyQt6) for coffee roasters. It records, analyzes, and controls roast profiles by integrating with thermocouple data loggers, PID controllers, and 40+ commercial roasting machines. It serves both home enthusiasts and professional specialty coffee businesses.

**License:** GPL-3.0
**Distribution:** Standalone executables for Windows, macOS (Intel + Apple Silicon), Linux, Raspberry Pi

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ (targeting 3.13/3.14) |
| GUI | PyQt6 + Matplotlib (for plotting) |
| Scientific | NumPy, SciPy (ODE solvers, optimization), Matplotlib |
| Device I/O | PySerial, PyModbus, Phidgets22, Bleak (BLE), python-snap7 (S7 PLC), WebSockets |
| Cloud | artisan.plus SaaS integration (REST API, JWT auth) |
| Build | PyInstaller (executables), AppVeyor (multi-platform CI) |
| Package Mgr | pip with requirements.txt + pyproject.toml |
| Quality | Mypy (strict), Pylint, Ruff, Black, Codespell, Pyright, pre-commit hooks |
| Testing | pytest with sanity/smoke/unit/UAT suites |

---

## Architecture

**Monolithic desktop GUI application** with tightly coupled domain logic and UI.

### Entry Point
`src/artisan.py` -> `src/artisanlib/main.py` (29K LOC god-class)

### Key Modules (src/artisanlib/)
| Module | LOC | Responsibility |
|--------|-----|---------------|
| `main.py` | 29,063 | Main app window, state management, orchestration |
| `canvas.py` | 20,144 | Roast profile visualization (Matplotlib) |
| `comm.py` | 7,452 | Device communication abstraction |
| `roast_properties.py` | 5,900 | Profile metadata and analysis UI |
| `devices.py` | 4,907 | 40+ roaster machine drivers |
| `curves.py` | 3,464 | Mathematical curve fitting, RoR smoothing |
| `events.py` | ~4,000 | Roast event handling (CHARGE, DRY, FC, SC, DROP) |
| `thermal_model*.py` | ~2,048 | Physics-based ODE thermal modeling |
| `pid_control.py` / `pid_dialogs.py` | ~2,000 | PID temperature control |

### Subpackage: `src/plus/`
14 modules for artisan.plus cloud integration (login, sync, schedule, stock, blend management).

### Codebase Size
- **195 Python files**, ~185K LOC total
- **89 core library modules** in artisanlib (~121K LOC)
- **61 test files** with ~2,261 test functions
- **30+ translation files** (Arabic through Vietnamese)

---

## Directory Structure

```
artisan/
├── src/
│   ├── artisan.py              # Entry point
│   ├── artisanlib/             # Core library (89 modules)
│   ├── plus/                   # Cloud/SaaS integration (14 modules)
│   ├── test/                   # Test suites (sanity/smoke/unit/UAT)
│   ├── translations/           # i18n files (.ts/.qm)
│   ├── includes/               # Resources (fonts, icons, themes, machine configs)
│   ├── ui/ / uic/              # Qt UI definitions / generated code
│   ├── proto/                  # Protocol buffers
│   ├── build/ / dist/          # Build artifacts (~711MB)
│   └── pyproject.toml          # Project config
├── .github/workflows/          # 5 CI workflows
├── doc/                        # Help dialog generation
├── wiki/                       # Documentation
└── README.md
```

---

## Development Workflow

### Commands
- **Setup:** `./setup-ubuntu.sh` (or `--dev` / `--build` modes)
- **Run:** `python src/artisan.py` (from venv)
- **Test:** `pytest` (from src/)
- **Lint:** Ruff, Pylint, Mypy, Codespell (all via CI or pre-commit)

### CI/CD (GitHub Actions)
5 workflows on push/PR to master: pytest, pylint, mypy, ruff, codespell

### Pre-commit Hooks
- check-yaml, check-xml, check-ast
- end-of-file-fixer, trailing-whitespace, double-quote-string-fixer
- pyupgrade (Python 3.12+)

### Git Workflow
Single `master` branch with descriptive commit messages. No formal branching strategy evident.

---

## Domain Complexity

### Core Concepts
- **Roast Profiles** - Temperature curves (ET/BT) over time
- **Events** - CHARGE, DRY, FIRST CRACK, SECOND CRACK, DROP, COOL
- **Phases** - Dry -> Maillard -> Development with statistical analysis
- **Devices** - 15+ hardware drivers (serial, Modbus, BLE, WebSocket, S7)
- **Thermal Model** - Physics-based ODE with 15 fittable parameters
- **PID Control** - Automated heat/fan/drum management
- **Scales** - BLE weight tracking (Acaia integration)

### Complexity Hotspots
1. **main.py** (29K LOC) - God-class with entangled state
2. **canvas.py** (20K LOC) - Real-time Matplotlib rendering
3. **Thermal modeling** - ODE solvers, differential evolution fitting, numerical stability
4. **Device ecosystem** - 15+ protocols, each with quirks
5. **State management** - Background/foreground profiles, no clear transaction boundaries

### Known Pain Points (30 TODO/FIXME/HACK)
- 8 items flagged for v4.0 removal (deprecated features)
- eval()/exec() in plotterprogram (security risk)
- Matplotlib annotation rendering hacks
- RTL language support incomplete
- Negative time value handling unclear

### Security Concerns
- Roaster remote control via Modbus/serial with no command signing
- PID tuning parameters without safety bounds
- eval()/exec() in user-facing feature
- Cloud auth token handling

---

## Existing Infrastructure

- **No CLAUDE.md, .claude/, or AI automation setup**
- **.gitignore** exists (standard Python patterns)
- **No .claudeignore**
- **Pre-commit hooks** configured
- **5 GitHub Actions workflows** for quality gates

---

## Large Files/Dirs to Exclude

| Path | Size | Reason |
|------|------|--------|
| `src/dist/` | ~711 MB | PyInstaller output |
| `src/build/` | Large | Build intermediates |
| `src/includes/` | ~60 MB | Fonts, icons, themes |
| `src/translations/` | Large | Generated .qm/.ts files |
| `src/uic/` | Generated | Auto-generated Qt UI code |
| `generated-linux.zip` | 9.1 MB | Build artifact |
