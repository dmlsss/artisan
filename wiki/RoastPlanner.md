# Roast Planner (Integrated Template + Thermal Modes)

## Overview

`Tools >> Roast Planner...` now provides one unified entry point with two planning engines:

1. `Template/Event Planner`
2. `Thermal Model Planner`

Both modes target the same goal: generate repeatable, executable roast control schedules for Kaleido workflows.

## Planner Modes

### 1. Template/Event Planner

Use this mode when you already have a reference roast and want a fast feedforward adaptation.

It:

- extracts and normalizes control events (`Heat`, `Fan`, `Drum`) from a source `.alog`
- scales timing and channel bias for a new batch size
- exports:
  - planned playback profile (`*_planned.alog`)
  - optional safety alarm file (`*_safety.alrm`)

### 2. Thermal Model Planner

Use this mode when you want model-based schedule generation from BT curve targets.

It supports:

- model calibration from Kaleido calibration roasts
- target curve generation from:
  - loaded background profile
  - file-based profile (`.alog`)
- target inspection summary (sample count, duration, BT range, start/end BT)
- batch planning presets (`50g`, `75g`, `100g`, `125g`, `150g`)
- planner goals:
  - `Safety-first`
  - `Balanced`
  - `Precision tracking`
- schedule inversion into heater/fan trajectories
- regime-aware bean heat uptake in the thermal model (drying/maillard/development multipliers)
- optional joint fan+drum optimisation during inversion
- trigger modes:
  - time-triggered (`CHARGE + seconds`)
  - BT-triggered (control action on BT threshold crossings)
- BT trigger hardening:
  - hysteresis (`BT hysteresis`)
  - minimum trigger spacing (`BT min gap`)
- optional drum control schedule (`Off`, `Constant`, `Ramp`)
- deadband filtering via minimum control-change threshold
- optional milestone popups (`Yellowing`, `First Crack`, `Drop`)
- optional BT/ET safety-ceiling popup alarms
- dry-run schedule safety validation (BT / ET / RoR) before export/apply/store
- quality scoring and milestone delta reporting (target vs predicted)
- interoperability exports:
  - Artisan thermal-plan JSON
  - HiBean-style replay CSV
- alarm schedule export (`.alrm`) and direct apply/store in the live alarm table
- optional pre-finalization alarm-table review/edit step before save/apply/store
- optional per-row `Flavor Impact` notes in the review table (collaboration notes only; not exported into `.alrm`)
- learnable flavor-impact suggestions:
  - planner seeds a first-pass flavor impact guess per alarm row
  - your review edits are learned and reused as future defaults for similar actions/stages
- advanced RoR smoothing modes in `Config >> Curves`:
  - `Classic`, `Savitzky-Golay`, `Exponential (EMA)`, `Hybrid (SG+EMA)`

## Prerequisites

Before using either mode:

1. Confirm machine slider control is configured.
2. Confirm Kaleido slider mapping:
   - slider 1: `kaleido(HP,{})` (`Heat`)
   - slider 2: `kaleido(FC,{})` (`Fan`)
   - slider 3: `kaleido(RC,{})` (`Drum`)
3. Validate automation in Simulator before live roasting.
4. Keep independent machine-level safety limits active.

## How To Use

### Template/Event Planner Workflow

1. Open `Tools >> Roast Planner...`.
2. Select `Template/Event Planner`.
3. Choose a source `.alog` profile.
4. Set target batch size in grams (`0` keeps source scaling).
5. Set time scale (`0` uses automatic scaling).
6. Save planned profile (`*_planned.alog`).
7. Optionally export matching safety alarms (`*_safety.alrm`).
8. Optionally load planned profile as background immediately.

### Thermal Model Planner Workflow

1. Open `Tools >> Roast Planner...`.
2. Select `Thermal Model Planner`.
3. `Calibrate` tab:
   - load 1–3 calibration profiles
   - fit model
   - optionally save fitted model JSON
4. `Generate Schedule` tab:
   - choose target source (`Background Profile` or `Load from file...`)
   - click `Inspect` to verify target summary
   - choose batch preset (`50/75/100/125/150 g`) and planner goal
   - click `Apply Preset` for recommended fan/drum/interval/safety defaults
   - adjust fan strategy and optional drum strategy as needed
   - optionally enable `Jointly optimize fan + drum` and set `Passes`
   - select trigger mode (`Time from CHARGE` or `BT temperature`)
   - for BT mode, tune `BT hysteresis` and `BT min gap` to reduce chatter
   - optionally change RoR smoothing mode in `Config >> Curves` when RoR is still noisy
   - set control deadband (`Min control change`)
   - optionally enable milestone and safety popup alarms
   - set dry-run safety limits (`BT max`, `ET max`, `RoR max`)
   - keep `Validate schedule before export` enabled for enforcement
   - choose control interval
   - generate schedule
5. `Export` tab:
   - optionally click `Review/Edit Alarms` to adjust rows before finalizing
   - adjust `Flavor Impact` notes during review so the planner can learn your sensory mapping over time
   - save `.alrm`
   - apply alarms directly
   - store schedule into an alarm set slot
   - save interop JSON / HiBean CSV exports

### Thermal CLI (optional)

Use `thermal_model_cli generate` when you want to build schedules outside the GUI.

Example:

```bash
python -m artisanlib.thermal_model_cli generate model.json target.alog \
  --mass 120 --fan 35 --drum 60 --optimize-actuators \
  --trigger-mode bt --min-delta 3 --bt-hysteresis 1.0 --bt-min-gap 2.0 \
  --bt-max 225 --et-max 255 --max-ror 30 \
  --interop-json thermal_schedule.json --hibean-csv thermal_schedule.csv \
  -o thermal_schedule.alrm
```

Convert an external interop schedule into `.alrm`:

```bash
python -m artisanlib.thermal_model_cli interop-convert \
  incoming_schedule.json converted_schedule.alrm --format auto
```

## Safety Alarm Behavior

Template/Event safety export creates two popup+beep ceiling alarms:

- ET ceiling
- BT ceiling

The generated `alarmactions` now use the correct Artisan popup action code (`0`).

Thermal planner export/apply/store actions can be blocked by dry-run validation if limits are exceeded.

## File Compatibility

The integrated planning stack accepts both profile formats:

- legacy Python-literal `.alog`
- JSON `.alog`

Thermal target profile parsing no longer requires Kaleido extra-device channels; only `timex` + `temp2` are required for target generation.
Thermal target and calibration profile temperatures are normalized to Celsius when source profiles are stored in Fahrenheit mode.
Background-profile targets in Fahrenheit mode are converted to Celsius before inversion.
Interop adapter formats:

- `artisan-thermal-plan-v1` JSON
- HiBean-style replay CSV (`time_s,bt_trigger_c,heater_pct,fan_pct,drum_pct`)

## Troubleshooting

- `No usable control events found in profile`
  - source profile lacks usable `specialevents*` control entries
- `Failed to parse target profile`
  - ensure target profile contains valid `timex` and `temp2`
- Thermal fit behaves inconsistently between profiles
  - verify profile temperature unit (`C`/`F`) is set correctly in source logs; the parser uses it for normalization
- Planned profile does not move roaster controls
  - verify slider action mappings in `Config >> Events`
- Alarms imported but do not execute
  - verify alarm enable flags and action mappings in `Config >> Alarms`
- BT-trigger schedule behaves late or early
  - confirm BT probe lag and use `Time from CHARGE` mode if your probe dynamics are unstable
- BT-trigger schedule fires too often
  - increase `BT hysteresis` and `BT min gap`
- Too many control alarms
  - increase `Min control change` to suppress tiny step updates
- Safety validation fails
  - reduce heat/fan/drum aggressiveness, increase batch size realism, or adjust target curve slope
