# Fork Feature Guide

Updated: 2026-02-19

This page is the single entry point for features added in this fork.
Use it as a map to the right workflow and detailed documentation.

## 1) Roast Planning and Thermal Automation

Primary workflow docs:

- [RoastPlanner.md](./RoastPlanner.md)
- [CompetitiveFeatureHarvest.md](./CompetitiveFeatureHarvest.md)

What is included:

- Unified `Tools >> Roast Planner...` entry point
- `Template/Event Planner` for fast feedforward profile adaptation
- `Thermal Model Planner` for model-based schedule generation
- thermal planner batch presets (`50/75/100/125/150 g`) with goal-based defaults
- target-curve inspection summary in planner drawer (duration/points/BT range)
- pre-export alarm review table with editable rows and optional flavor-impact notes
- learnable flavor-impact suggestions (review edits are persisted and reused)
- Optional fan + drum joint optimization
- regime-aware bean heat uptake in thermal simulation/inversion (drying/maillard/development)
- Time-trigger or BT-trigger schedule emission
- Milestone and safety popup generation
- Dry-run schedule safety validation
- Quality scoring report
- Interop export/import (`artisan-thermal-plan-v1` JSON, HiBean-style CSV)

## 2) Kaleido Workflow Enhancements

Kaleido-specific UX and control updates in this fork:

- Connection status indicator for Kaleido sessions
- `PID` button for Kaleido Auto-Heating toggle
- Live DTR display in phases area
- RoR trend color coding (decline/flat/flick segments)
- selectable RoR smoothing modes (`Classic`, `Savitzky-Golay`, `EMA`, `Hybrid`)
- Background BT tracking HUD (current roast vs background delta)
- Roast defect notifications:
  - Baking (sustained low RoR after FC)
  - Crash (sharp RoR drop)
  - Scorching (high BT before DRY)
  - Underdeveloped (low DTR at DROP)
- Quick Cupping prompt after DROP for rapid quality notes

Bundled assets:

- Kaleido profile templates:
  - `src/includes/Profiles/Kaleido/Light_City.alog`
  - `src/includes/Profiles/Kaleido/Full_City.alog`
  - `src/includes/Profiles/Kaleido/Vienna.alog`
- Kaleido dark theme:
  - `src/includes/Themes/Artisan/KaleidoDark.athm`

## 3) CLI and Automation Interfaces

Command line interface:

- `python -m artisanlib.thermal_model_cli fit ...`
- `python -m artisanlib.thermal_model_cli generate ...`
- `python -m artisanlib.thermal_model_cli interop-convert ...`

See examples in:

- [RoastPlanner.md](./RoastPlanner.md)
- [HowToRunFromSource.md](./HowToRunFromSource.md)

## 4) Developer Setup and Validation

Ubuntu one-step helper script:

- `./setup-ubuntu.sh`

Recommended validation commands:

- `pytest -q -s src/test/unitary`
- `ruff check src/artisanlib`

Thermal/planner-focused smoke set:

- `pytest -q -s src/test/unitary/artisanlib/test_roast_planner.py`
- `pytest -q -s src/test/unitary/artisanlib/test_thermal_integration.py`
- `pytest -q -s src/test/unitary/artisanlib/test_thermal_interop.py`

## 5) Feature Index

| Feature | Main code path | Main docs |
| --- | --- | --- |
| Template/Event planning | `src/artisanlib/roast_planner.py` | `wiki/RoastPlanner.md` |
| Thermal model planning | `src/artisanlib/thermal_control_dlg.py` | `wiki/RoastPlanner.md` |
| Thermal inversion/fitting | `src/artisanlib/thermal_model_inversion.py`, `src/artisanlib/thermal_model_fitting.py` | `wiki/RoastPlanner.md` |
| Safety alarm generation | `src/artisanlib/thermal_alarm_generator.py` | `wiki/RoastPlanner.md` |
| Schedule safety validator | `src/artisanlib/thermal_schedule_validator.py` | `wiki/RoastPlanner.md` |
| Quality scoring | `src/artisanlib/thermal_planner_quality.py` | `wiki/RoastPlanner.md` |
| Interop JSON/CSV | `src/artisanlib/thermal_interop.py` | `wiki/RoastPlanner.md` |
| Thermal CLI | `src/artisanlib/thermal_model_cli.py` | `wiki/HowToRunFromSource.md` |
| Kaleido communication/control | `src/artisanlib/kaleido.py` | This guide + release notes |
| Quick Cupping dialog | `src/artisanlib/quick_cupping.py` | This guide + release notes |
| Kaleido UI/HUD/defect alerts | `src/artisanlib/canvas.py`, `src/artisanlib/main.py` | This guide + release notes |
