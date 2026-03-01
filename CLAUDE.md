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
