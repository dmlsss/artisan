from pathlib import Path

import numpy as np

from artisanlib.thermal_interop import (
    export_artisan_plan_json,
    export_hibean_csv,
    import_interop_schedule,
    interop_to_alarm_table,
    schedule_from_inversion,
)
from artisanlib.thermal_model import KaleidoThermalModel, ThermalModelParams
from artisanlib.thermal_model_inversion import invert_model


def _sample_inversion():
    model = KaleidoThermalModel(ThermalModelParams())
    t = np.linspace(0.0, 240.0, 13, dtype=np.float64)
    bt = np.linspace(95.0, 210.0, 13, dtype=np.float64)
    return invert_model(
        model=model,
        target_time=t,
        target_bt=bt,
        mass_kg=0.1,
        fan_schedule=30.0,
        drum_schedule=60.0,
    )


def test_interop_json_round_trip(tmp_path: Path) -> None:
    inversion = _sample_inversion()
    schedule = schedule_from_inversion(inversion, trigger_mode='bt', label='RoundTrip')
    path = tmp_path / 'plan.json'

    export_artisan_plan_json(str(path), schedule)
    loaded = import_interop_schedule(str(path), fmt='json')

    assert loaded.label == 'RoundTrip'
    assert loaded.trigger_mode == 'bt'
    assert len(loaded.time_s) == len(schedule.time_s)
    assert np.allclose(loaded.heater_pct, np.rint(schedule.heater_pct))


def test_hibean_csv_round_trip_and_alarm_conversion(tmp_path: Path) -> None:
    inversion = _sample_inversion()
    schedule = schedule_from_inversion(inversion, trigger_mode='time', label='CSVRoundTrip')
    path = tmp_path / 'plan.csv'

    export_hibean_csv(str(path), schedule)
    loaded = import_interop_schedule(str(path), fmt='csv')
    alarms = interop_to_alarm_table(loaded, min_delta_pct=2)

    assert loaded.trigger_mode == 'time'
    assert alarms.alarm_count() > 0
