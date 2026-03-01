# Infrastructure Plan: Artisan

## 2A: CLAUDE.md Design

```markdown
# Artisan - Coffee Roasting Software

## Project Overview

Artisan is a cross-platform PyQt6 desktop application for coffee roasters. It records, analyzes, and
controls roast profiles, integrating with 40+ commercial roasting machines via serial, Modbus, BLE,
WebSocket, and S7 protocols. Core library lives in `src/artisanlib/` (89 modules, ~121K LOC).
Entry point: `src/artisan.py` -> `src/artisanlib/main.py`.

## Commands

- **Run app:** `cd src && python artisan.py`
- **Run tests:** `cd src && pytest`
- **Run single test:** `cd src && pytest test/unitary/artisanlib/test_<module>.py -v`
- **Type check:** `cd src && mypy artisanlib plus`
- **Lint:** `cd src && ruff check .`
- **Spell check:** `cd src && codespell src/`
- **Pre-commit:** `pre-commit run --all-files`

## Architecture

Two dominant files: `main.py` (29K LOC, app orchestration + state) and `canvas.py` (20K LOC,
Matplotlib rendering). Device drivers in `comm.py` + `devices.py` + machine-specific modules
(aillio_r1.py, kaleido.py, loring.py, etc.). Thermal modeling in `thermal_model*.py` uses SciPy
ODE solvers and differential evolution fitting. Cloud integration in `src/plus/` subpackage.

## Code Conventions

- **Python 3.12+** target. Use `pyupgrade --py312-plus` idioms.
- **Line length:** 100 (Black), 80 (Ruff). Black wins for formatting; Ruff for lint rules.
- **Type annotations required** — mypy strict mode is enforced.
- **String quotes:** Double quotes enforced by pre-commit hook.
- **Imports:** Standard library -> third-party -> local. No wildcard imports.
- **Test layout:** `src/test/{sanity,smoke,unitary,uat}/` mirrors `src/artisanlib/` and `src/plus/`.
- **Translations:** Never edit `.qm` files (compiled). `.ts` files are the source but managed externally.

## Critical Rules

- **Never modify `src/uic/`** — these are auto-generated from Qt Designer `.ui` files.
- **Never commit `src/build/` or `src/dist/`** — PyInstaller artifacts.
- **The `eval()`/`exec()` in `curves.py` plotterprogram is a known security risk** — do not extend
  its use. Any new dynamic execution must be sandboxed.
- **Thermal model changes require numerical stability testing** — check for overflow in sigmoid
  functions, validate parameter bounds, test with `solve_ivp` edge cases.
- **Device drivers must handle serial timeouts gracefully** — never block the UI thread on I/O.
- **All roast events (CHARGE, DRY, FCs, FCe, SCs, SCe, DROP) have strict ordering** — validate
  event sequencing when modifying event logic.
- **Platform markers in tests:** Use `@pytest.mark.darwin`, `@pytest.mark.linux`,
  `@pytest.mark.win32` for platform-specific tests.

## Task Management

- `tasks/todo.md` — active task tracking
- `tasks/lessons.md` — accumulated learnings and gotchas

## Working with This Codebase

- `main.py` and `canvas.py` are god-classes. Changes to them should be surgical and well-tested.
  Prefer extracting logic to focused modules over adding to these files.
- When adding device support, follow the pattern in existing drivers (e.g., `kaleido.py`).
- The `plus/` subpackage is the cloud integration layer — it has its own test suite under
  `test/unitary/plus/`.
- Pre-commit hooks run automatically. If they fail, fix the issues rather than bypassing.
```

**Estimated tokens: ~2,200** (well under 3,000 limit)

---

## 2B: Sub-Agents

### Agent 1: "The Sentinel" — Security Auditor

**Role:** Scans for security vulnerabilities specific to Artisan's attack surface: eval/exec usage,
device control safety, authentication handling, and injection risks.

**What it checks:**
- `eval()` / `exec()` usage and any expansion of plotterprogram dynamic execution
- Serial/Modbus command injection vectors in device drivers
- PID parameter bounds — values that could cause thermal runaway
- Authentication token storage and transmission in `plus/` modules
- User input sanitization in UI dialogs that feed into device commands
- Hardcoded credentials, API keys, or secrets
- File path traversal in profile import/export

**Files/directories it primarily operates on:**
- `src/artisanlib/curves.py` (eval/exec)
- `src/artisanlib/comm.py`, `src/artisanlib/devices.py` (device I/O)
- `src/artisanlib/pid_control.py`, `src/artisanlib/pid_dialogs.py` (PID safety)
- `src/plus/` (authentication, cloud sync)
- `src/artisanlib/main.py` (file operations, user input paths)

**MCP tools:** None needed — file search and grep are sufficient.

---

### Agent 2: "The Roast Scientist" — Domain & Numerical Correctness

**Role:** Validates thermal modeling, scientific computations, and domain logic correctness.
This is the trickiest code in the project and benefits from focused review.

**What it checks:**
- Thermal model ODE stability (overflow in exp/sigmoid, parameter bounds)
- `solve_ivp` configuration (tolerances, max_step, event handling)
- `differential_evolution` and `minimize` fitting convergence and bounds
- Unit conversions (Fahrenheit/Celsius, kg/lb, time units)
- RoR (Rate of Rise) smoothing algorithm correctness
- Roast event ordering constraints (CHARGE < DRY < FCs < FCe < SCs < SCe < DROP)
- Phase calculation logic (dry/maillard/development percentages)
- Calibration parameter physical plausibility (temperature ranges, heat transfer coefficients)

**Files/directories it primarily operates on:**
- `src/artisanlib/thermal_model.py`, `thermal_model_fitting.py`, `thermal_model_parameters.py`
- `src/artisanlib/thermal_planner*.py`
- `src/artisanlib/curves.py` (mathematical functions)
- `src/artisanlib/canvas.py` (phase calculations, RoR rendering)
- `src/artisanlib/pid_control.py` (control algorithm correctness)

**MCP tools:** None needed.

---

### Agent 3: "The Test Warden" — Test Quality Guardian

**Role:** Identifies test coverage gaps and validates test quality for this complex domain.
With 185K LOC and only 62 test files, there are guaranteed coverage gaps in critical paths.

**What it checks:**
- Missing test coverage for thermal modeling edge cases
- Device driver test coverage (serial timeout handling, malformed responses)
- State management test gaps in main.py (profile load/save, background/foreground)
- Event ordering validation in tests
- Test assertions that are too weak (e.g., just checking no exception vs. checking values)
- Flaky test patterns (timing-dependent, platform-dependent without markers)
- Test data adequacy (are edge case profiles included?)

**Files/directories it primarily operates on:**
- `src/test/` (all test suites)
- `src/artisanlib/` (identifying untested modules)
- `src/conftest.py` (test configuration)

**MCP tools:** None needed.

---

## 2C: MCP Server

**Decision: Not recommended.**

Rationale:
- The codebase is well-structured Python — Grep and Glob are efficient for lookups.
- No repeated reference data patterns that would benefit from externalization.
- Device driver specs vary too much per manufacturer to usefully index.
- The project doesn't have a knowledge base that agents repeatedly query.
- Adding an MCP server would increase maintenance burden without clear ROI.

---

## 2D: .claudeignore

```
# Build artifacts
src/build/
src/dist/
generated-linux.zip

# Generated code
src/uic/

# Translation files (managed externally)
src/translations/

# Large resource files (fonts, icons, themes)
src/includes/Themes/
src/includes/Icons/
src/includes/Fonts/

# Documentation assets (screenshots, images)
wiki/

# Python artifacts
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/

# IDE and OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# CI/packaging
.appveyor.yml
src/debian/
src/pyinstaller_hooks/
src/Wheels/

# Other
.augment/
```

---

## 2E: Skills

**Decision: No custom skills recommended at this time.**

Rationale:
- Quality checks are individual CLI commands already documented in CLAUDE.md.
- No complex multi-step workflow is repeated often enough to justify a skill.
- The pre-commit hooks already handle the standard formatting/validation pipeline.
- If a "run all quality checks" workflow becomes common, it can be added later as a simple skill.

---

## Summary

| Component | Action | Rationale |
|-----------|--------|-----------|
| CLAUDE.md | Create | Essential — project-specific commands, conventions, critical rules |
| .claudeignore | Create | ~770MB of build/resource/generated files to exclude |
| 3 Agents | Create | Security, domain science, and test quality — the real gaps |
| MCP Server | Skip | Grep/Glob sufficient for this codebase |
| Skills | Skip | No repeated complex workflows identified |
| tasks/ | Create todo.md + lessons.md | Task and learning tracking |
