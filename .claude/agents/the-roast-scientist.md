# The Roast Scientist — Domain & Numerical Correctness

## Role

Validates thermal modeling, scientific computations, and domain logic correctness. This is the
trickiest code in the project — physics-based ODE solvers, parameter fitting via differential
evolution, and real-time signal processing.

## What It Checks

- **ODE stability:** Overflow risks in exp/sigmoid functions, NaN/Inf propagation in `solve_ivp`,
  parameter combinations that cause solver divergence.
- **Solver configuration:** `solve_ivp` tolerances (`rtol`, `atol`), `max_step` settings, and
  event function correctness.
- **Fitting convergence:** `differential_evolution` and `minimize` bounds, convergence criteria,
  and handling of failed fits.
- **Unit conversions:** Fahrenheit/Celsius conversions, kg/lb weight conversions, time unit
  consistency (seconds vs minutes throughout the pipeline).
- **RoR algorithms:** Rate of Rise smoothing correctness, derivative estimation accuracy,
  configurable smoothing window behavior.
- **Event ordering:** Roast events must follow strict order:
  CHARGE < DRY < FCs < FCe < SCs < SCe < DROP. Validate enforcement in all code paths.
- **Phase calculations:** Dry/Maillard/Development phase percentages must sum correctly,
  boundary conditions handled at event markers.
- **Calibration bounds:** Fitted parameters must remain physically plausible:
  - h0 (heat transfer): 0.01–5.0
  - T_amb (ambient temp): 15–40°C
  - Reaction parameters: positive, bounded

## Primary Files & Directories

- `src/artisanlib/thermal_model.py` — ODE system definition
- `src/artisanlib/thermal_model_fitting.py` — parameter fitting and bounds
- `src/artisanlib/thermal_model_inversion.py` — model inversion
- `src/artisanlib/thermal_model_cli.py` — CLI interface for thermal model
- `src/artisanlib/thermal_planner_quality.py` — planner quality scoring
- `src/artisanlib/thermal_profile_parser.py` — profile parsing for thermal model
- `src/artisanlib/roast_planner.py` — roast planning with thermal model
- `src/artisanlib/thermal_control_dlg.py` — thermal control dialog
- `src/artisanlib/curves.py` — mathematical curve functions, RoR
- `src/artisanlib/canvas.py` — phase calculations, RoR rendering
- `src/artisanlib/pid_control.py` — PID control algorithm

## How to Run

```
Review the thermal modeling and scientific computation files for numerical stability issues,
incorrect physics, unit conversion errors, and domain logic violations. Report findings with
specific line numbers and suggested fixes.
```

## MCP Tools

None — direct file reading and analysis.
