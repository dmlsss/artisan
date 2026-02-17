import numpy as np

from artisanlib.thermal_model import KaleidoThermalModel, ThermalModelParams
from artisanlib.thermal_model_inversion import invert_model
from artisanlib.thermal_schedule_validator import validate_schedule


def test_validate_schedule_passes_with_relaxed_limits() -> None:
    model = KaleidoThermalModel(ThermalModelParams())
    t = np.array([0.0, 60.0, 120.0, 180.0], dtype=np.float64)
    bt = np.array([95.0, 140.0, 180.0, 205.0], dtype=np.float64)
    inv = invert_model(model=model, target_time=t, target_bt=bt, mass_kg=0.1, fan_schedule=30.0)

    result = validate_schedule(
        model,
        inv,
        bt_limit_c=240.0,
        et_limit_c=300.0,
        max_ror_limit_c_per_min=50.0,
    )

    assert result.is_safe is True
    assert result.failures == []
    assert result.bt_peak_c <= 240.0


def test_validate_schedule_fails_with_tight_limits() -> None:
    model = KaleidoThermalModel(ThermalModelParams())
    t = np.array([0.0, 30.0, 60.0, 90.0], dtype=np.float64)
    bt = np.array([100.0, 150.0, 200.0, 220.0], dtype=np.float64)
    inv = invert_model(model=model, target_time=t, target_bt=bt, mass_kg=0.1, fan_schedule=20.0)

    result = validate_schedule(
        model,
        inv,
        bt_limit_c=150.0,
        et_limit_c=180.0,
        max_ror_limit_c_per_min=10.0,
    )

    assert result.is_safe is False
    assert len(result.failures) >= 1
