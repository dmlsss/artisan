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
    assert len(result.drum_pct) == len(target_time)
    if np.max(result.predicted_bt) >= 196.0:
        assert result.first_crack_time is not None
    assert result.dtr_percent is None or result.dtr_percent >= 0.0

    resampled = result.resample_to_interval(30.0)
    assert resampled.drop_time == float(resampled.time[-1])
    assert resampled.dtr_percent is None or resampled.dtr_percent >= 0.0
    assert len(resampled.drum_pct) == len(resampled.time)


def test_invert_model_can_optimize_fan_and_drum() -> None:
    model = KaleidoThermalModel(ThermalModelParams())
    target_time = np.linspace(0.0, 360.0, 25, dtype=np.float64)
    target_bt = np.linspace(90.0, 215.0, 25, dtype=np.float64)

    result = invert_model(
        model=model,
        target_time=target_time,
        target_bt=target_bt,
        mass_kg=0.10,
        fan_schedule=30.0,
        drum_schedule=60.0,
        optimize_actuators=True,
        optimizer_iterations=2,
        optimizer_segments=6,
        optimizer_step_pct=6,
    )

    assert result.objective_score is not None
    assert np.all(result.fan_pct >= 0.0)
    assert np.all(result.fan_pct <= 100.0)
    assert np.all(result.drum_pct >= 0.0)
    assert np.all(result.drum_pct <= 100.0)


def test_inversion_accounts_for_regime_heat_uptake_in_development() -> None:
    target_time = np.linspace(0.0, 360.0, 31, dtype=np.float64)
    target_bt = np.linspace(95.0, 208.0, 31, dtype=np.float64)

    higher_dev_transfer = KaleidoThermalModel(
        ThermalModelParams(h_dry_mult=1.0, h_maillard_mult=1.0, h_dev_mult=1.15)
    )
    lower_dev_transfer = KaleidoThermalModel(
        ThermalModelParams(h_dry_mult=1.0, h_maillard_mult=1.0, h_dev_mult=0.80)
    )

    high_transfer_result = invert_model(
        model=higher_dev_transfer,
        target_time=target_time,
        target_bt=target_bt,
        mass_kg=0.10,
        fan_schedule=35.0,
        drum_schedule=60.0,
    )
    low_transfer_result = invert_model(
        model=lower_dev_transfer,
        target_time=target_time,
        target_bt=target_bt,
        mass_kg=0.10,
        fan_schedule=35.0,
        drum_schedule=60.0,
    )

    dev_mask = target_bt >= 196.0
    assert np.any(dev_mask)
    mean_hp_high_transfer = float(np.mean(high_transfer_result.heater_pct[dev_mask]))
    mean_hp_low_transfer = float(np.mean(low_transfer_result.heater_pct[dev_mask]))
    assert abs(mean_hp_low_transfer - mean_hp_high_transfer) > 0.05
