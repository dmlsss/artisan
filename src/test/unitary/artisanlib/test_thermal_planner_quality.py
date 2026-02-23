import numpy as np

from artisanlib.thermal_model import KaleidoThermalModel, ThermalModelParams
from artisanlib.thermal_model_inversion import invert_model
from artisanlib.thermal_planner_quality import build_quality_report


def test_build_quality_report_returns_score_and_milestone_deltas() -> None:
    model = KaleidoThermalModel(ThermalModelParams())
    target_time = np.linspace(0.0, 420.0, 22, dtype=np.float64)
    target_bt = np.linspace(90.0, 220.0, 22, dtype=np.float64)

    inversion = invert_model(
        model=model,
        target_time=target_time,
        target_bt=target_bt,
        mass_kg=0.1,
        fan_schedule=30.0,
        drum_schedule=55.0,
    )

    report = build_quality_report(
        target_time=target_time,
        target_bt=target_bt,
        inversion=inversion,
        control_change_count=18,
    )

    assert 0.0 <= report.score <= 100.0
    assert report.grade in {'A', 'B', 'C', 'D', 'F'}
    assert 'Drop' in report.milestone_deltas_s
    assert len(report.summary_lines()) >= 4
