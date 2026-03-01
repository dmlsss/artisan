# Lessons Learned

<!-- Accumulate gotchas, debugging insights, and decisions here. -->

## Codebase Navigation

- `main.py` (29K LOC) and `canvas.py` (20K LOC) are god-classes — changes should be surgical.
- Thermal modeling code (`thermal_model*.py`) uses SciPy ODE solvers with tight numerical constraints.
- Device drivers each have unique protocols — follow existing patterns (e.g., `kaleido.py`) when adding new ones.
- Pre-commit hooks enforce double quotes, trailing whitespace, and pyupgrade (3.12+).

## Testing

- Tests run on macOS in CI (Ubuntu disabled due to missing Qt libs).
- Platform-specific tests use `@pytest.mark.darwin`, `@pytest.mark.linux`, `@pytest.mark.win32`.
- Test command: `cd src && pytest` (pytest config in `src/pyproject.toml`).

## Known Risks

- `eval()`/`exec()` exists in `curves.py` plotterprogram — known security risk, do not extend.
- PID parameters lack safety bounds checking — potential runaway risk.
- Serial device control has no command signing or encryption.
