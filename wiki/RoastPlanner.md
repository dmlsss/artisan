# Roast Planner (Kaleido Feedforward)

## Overview

The Roast Planner adds a feedforward planning flow for Kaleido-style control profiles.

It takes a source Artisan profile (`.alog`) containing control events, normalizes those events to the Artisan slider model (`Heat`, `Fan`, `Drum`), applies optional scaling, and exports:

- a planned profile for playback (`*.alog`)
- an optional safety alarm set (`*.alrm`)

The planner is available from:

- `Tools >> Roast Planner`

## What Changed

The planner integration includes:

- a new planner module (`src/artisanlib/roast_planner.py`)
- a new menu action (`Tools >> Roast Planner`)
- export of planned profiles plus optional safety alarms
- compatibility in profile loading for both legacy Python-literal `.alog` and JSON-formatted `.alog`

## Prerequisites

Before using the planner on Kaleido:

1. Confirm your machine setup is configured for slider control.
2. Confirm event sliders are mapped to Kaleido commands:
   - slider 1: `kaleido(HP,{})` (`Heat`)
   - slider 2: `kaleido(FC,{})` (`Fan`)
   - slider 3: `kaleido(RC,{})` (`Drum`)
3. Validate all control actions in Simulator mode before running a live batch.
4. Keep independent machine safety limits active.

## How To Use

1. Open `Tools >> Roast Planner`.
2. Select a source `.alog` profile.
3. Enter target batch size in grams.
   - enter `0` to keep source batch scaling
4. Enter time scale.
   - enter `0` to let Artisan compute automatic scaling
5. Save the planned profile (`*_planned.alog` by default).
6. Choose whether to generate matching safety alarms.
   - when enabled, a `*_safety.alrm` file is written next to the planned profile
7. Choose whether to load the planned profile as background immediately.

## Running A Planned Roast

1. Load the planned profile as background (if not already loaded).
2. Ensure `Playback Events` is enabled.
3. Select replay mode (time, BT, or ET) as appropriate for your workflow.
4. Optionally enable event ramping for smoother transitions.
5. Import the generated alarm set in `Config >> Alarms` (`Load Alarms`) if you exported safety alarms.
6. Run one simulation pass, then roast live.

## Safety Alarm Output

When alarm export is enabled, the generated `.alrm` contains:

- ET ceiling alarm
- BT ceiling alarm
- popup + beep actions for both alarms

Ceilings are estimated from profile maxima with a margin unless explicit overrides are supplied in code.

## Notes On Planning Behavior

Current planning is deterministic and event-based:

- source control events are extracted from `specialevents*`
- legacy Kaleido event mappings are normalized to `Heat/Fan/Drum`
- batch-size scaling adjusts channel bias and time scaling
- closely spaced duplicate events are compressed

This is intended as a practical feedforward baseline that integrates with existing Artisan playback/alarm tooling.

## Troubleshooting

- `No usable control events found in profile`
  - ensure the source profile contains slider control events in `specialevents*`
- Planned profile loads but does not move machine controls
  - verify slider actions and Kaleido command mappings in `Config >> Events`
- Alarm file imports but no alarms trigger
  - verify alarm enable flags and source channels (`ET`/`BT`) in `Config >> Alarms`
