import numpy as np

from artisanlib.thermal_model import (
    PARAM_NAMES,
    KaleidoThermalModel,
    ThermalModelParams,
    regime_heat_multiplier,
)


def test_thermal_model_vector_includes_regime_parameters() -> None:
    vec = ThermalModelParams().to_vector()
    assert len(vec) == len(PARAM_NAMES)
    assert {'h_dry_mult', 'h_maillard_mult', 'h_dev_mult'}.issubset(set(PARAM_NAMES))


def test_regime_heat_multiplier_transitions_by_temperature_stage() -> None:
    params = ThermalModelParams(
        h_dry_mult=1.10,
        h_maillard_mult=1.00,
        h_dev_mult=0.85,
        T_regime_yellowing=150.0,
        T_regime_first_crack=196.0,
        T_regime_width=4.0,
    )
    dry = regime_heat_multiplier(params, 100.0)
    maillard = regime_heat_multiplier(params, 172.0)
    development = regime_heat_multiplier(params, 220.0)

    assert dry > maillard > development
    assert 1.04 <= dry <= 1.12
    assert 0.94 <= maillard <= 1.03
    assert 0.80 <= development <= 0.90


def test_ode_rhs_reflects_regime_dependent_heat_uptake() -> None:
    high_uptake_model = KaleidoThermalModel(
        ThermalModelParams(h_dry_mult=1.0, h_maillard_mult=1.0, h_dev_mult=1.20)
    )
    low_uptake_model = KaleidoThermalModel(
        ThermalModelParams(h_dry_mult=1.0, h_maillard_mult=1.0, h_dev_mult=0.80)
    )

    def hp(_t: float) -> float:
        return 60.0

    def fan(_t: float) -> float:
        return 30.0

    def drum(_t: float) -> float:
        return 60.0

    dtdt_high = high_uptake_model.ode_rhs(
        t=240.0,
        T=210.0,
        hp_func=hp,
        fan_func=fan,
        drum_func=drum,
        mass_kg=0.10,
    )
    dtdt_low = low_uptake_model.ode_rhs(
        t=240.0,
        T=210.0,
        hp_func=hp,
        fan_func=fan,
        drum_func=drum,
        mass_kg=0.10,
    )

    assert np.isfinite(dtdt_high)
    assert np.isfinite(dtdt_low)
    assert dtdt_high > dtdt_low
