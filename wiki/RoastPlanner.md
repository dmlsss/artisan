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
- schedule inversion into heater/fan trajectories
- alarm schedule export (`.alrm`) and direct apply/store in the live alarm table

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
   - set batch mass, fan strategy, and control interval
   - generate schedule
5. `Export` tab:
   - save `.alrm`
   - apply alarms directly
   - store schedule into an alarm set slot

## Safety Alarm Behavior

Template/Event safety export creates two popup+beep ceiling alarms:

- ET ceiling
- BT ceiling

The generated `alarmactions` now use the correct Artisan popup action code (`0`).

## File Compatibility

The integrated planning stack accepts both profile formats:

- legacy Python-literal `.alog`
- JSON `.alog`

Thermal target profile parsing no longer requires Kaleido extra-device channels; only `timex` + `temp2` are required for target generation.

## Troubleshooting

- `No usable control events found in profile`
  - source profile lacks usable `specialevents*` control entries
- `Failed to parse target profile`
  - ensure target profile contains valid `timex` and `temp2`
- Planned profile does not move roaster controls
  - verify slider action mappings in `Config >> Events`
- Alarms imported but do not execute
  - verify alarm enable flags and action mappings in `Config >> Alarms`
