# Competitive Feature Harvest (Roasting Apps)

Updated: 2026-02-17

## Sources Reviewed

- Cropster Roast features: `https://www.cropster.com/products/roast/features/`
- RoastPATH features: `https://roastpath.com/pages/data-logging-features-for-coffee-roasters`
- HiBean docs: `https://docs.hibean.fun/`
- HiBean site: `https://www.hibean.fun/en/`
- HiBean app listings:
  - iOS: `https://apps.apple.com/us/app/hibean/id6739746285`
  - Android: `https://play.google.com/store/apps/details?id=com.hibean.hib_roast`

## Recurrent High-Value Features

1. Automated recipe replay with robust triggers
2. Milestone prediction and proactive operator alerts
3. Full actuator scheduling (heater/fan/drum), not heater-only
4. Noise-resistant schedule execution
5. Built-in safety ceiling alarms
6. Pre-flight schedule validation before execution
7. Cross-platform/interop profile exchange

## Implemented In This Fork

1. Thermal trigger mode selection:
   - `Time from CHARGE`
   - `BT temperature`
2. Optional drum schedule generation:
   - `Off`
   - `Constant`
   - `Ramp`
3. Deadband control emission (`Min control change %`) to suppress tiny actuator chatter
4. Milestone popup generation:
   - `Yellowing`
   - `First Crack`
   - `Drop`
5. Safety popup generation:
   - BT ceiling
   - ET ceiling
6. Dry-run safety validator (BT/ET/RoR) with export/apply/store guardrails
7. Thermal planner milestone estimates in UI:
   - yellowing time
   - first crack time
   - drop time
   - DTR estimate
8. Thermal planner quality scoring:
   - tracking RMSE/max error
   - milestone deltas (predicted vs target)
   - control change count
   - safety pass/fail impact
9. Joint fan+drum optimization mode during inversion
10. BT-trigger hardening:
    - hysteresis (`BT hysteresis`)
    - minimum trigger spacing (`BT min gap`)
11. Interoperability adapters:
    - export `artisan-thermal-plan-v1` JSON
    - export HiBean-style replay CSV
    - import/convert interop JSON/CSV to `.alrm` via CLI
12. Extended CLI parity (`thermal_model_cli generate` + `interop-convert`)

## Remaining Backlog

1. Cloud sync / team collaboration workflows
2. Production/inventory coupling to roast plans
3. ML residual model for adaptive online correction
