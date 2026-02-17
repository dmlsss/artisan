import numpy as np

from artisanlib.thermal_model import KaleidoThermalModel, ThermalModelParams
from artisanlib.thermal_model_inversion import invert_model


def test_invert_model_exposes_milestone_fields_and_resample_updates_drop_time() -> None:
    model = KaleidoThermalModel(ThermalModelParams())
    target_time = np.array([0.0, 60.0, 120.0, 180.0, 240.0], dtype=np.float64)
    target_bt = np.array([90.0, 140.0, 175.0, 205.0, 225.0], dtype=np.float64)

    result = invert_model(
        model=model,
        target_time=target_time,
        target_bt=target_bt,
        mass_kg=0.10,
        fan_schedule=30.0,
    )

    assert result.drop_time == float(target_time[-1])
    assert result.yellowing_time is not None
    if np.max(result.predicted_bt) >= 196.0:
        assert result.first_crack_time is not None
    assert result.dtr_percent is None or result.dtr_percent >= 0.0

    resampled = result.resample_to_interval(30.0)
    assert resampled.drop_time == float(resampled.time[-1])
    assert resampled.dtr_percent is None or resampled.dtr_percent >= 0.0
