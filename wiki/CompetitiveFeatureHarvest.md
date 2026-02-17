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
6. Thermal planner milestone estimates in UI:
   - yellowing time
   - first crack time
   - drop time
   - DTR estimate
7. CLI parity (`thermal_model_cli generate`) for all of the above controls

## Remaining Backlog

1. Cloud sync / team collaboration workflows
2. Production/inventory coupling to roast plans
3. ML residual model for adaptive online correction
