#
# ABOUT
# Quality scoring and milestone comparison for thermal planner schedules.

# LICENSE
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from artisanlib.thermal_model_inversion import InversionResult
    from artisanlib.thermal_schedule_validator import SafetyValidationResult

_EPS = 1e-9
_MILESTONE_THRESHOLDS: dict[str, float] = {
    'Yellowing': 150.0,
    'First Crack': 196.0,
    'Second Crack': 224.0,
}


def _crossing_time(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
    threshold: float,
) -> float | None:
    if len(time) == 0 or len(values) == 0:
        return None
    above = values >= threshold
    if not np.any(above):
        return None
    idx = int(np.argmax(above))
    if idx <= 0:
        return float(time[0])
    t0, t1 = float(time[idx - 1]), float(time[idx])
    v0, v1 = float(values[idx - 1]), float(values[idx])
    dv = v1 - v0
    if abs(dv) <= _EPS:
        return t1
    frac = (threshold - v0) / dv
    frac = min(1.0, max(0.0, frac))
    return t0 + frac * (t1 - t0)


def _grade(score: float) -> str:
    if score >= 90:
        return 'A'
    if score >= 80:
        return 'B'
    if score >= 70:
        return 'C'
    if score >= 60:
        return 'D'
    return 'F'


def _fmt_delta(seconds: float | None) -> str:
    if seconds is None:
        return '--'
    sign = '+' if seconds >= 0 else '-'
    s = int(round(abs(seconds)))
    return f'{sign}{s // 60}:{s % 60:02d}'


@dataclass
class QualityReport:
    score: float
    grade: str
    rmse_c: float
    max_error_c: float
    milestone_deltas_s: dict[str, float | None]
    control_change_count: int
    notes: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [
            f'Quality score: {self.score:.1f}/100 ({self.grade})',
            f'Error metrics: RMSE={self.rmse_c:.2f}C, max={self.max_error_c:.2f}C',
            f'Control changes: {self.control_change_count}',
        ]
        for name in ['Yellowing', 'First Crack', 'Second Crack', 'Drop']:
            if name in self.milestone_deltas_s:
                lines.append(f'{name} delta (pred-target): {_fmt_delta(self.milestone_deltas_s[name])}')
        lines.extend(self.notes)
        return lines


def build_quality_report(
    target_time: NDArray[np.float64],
    target_bt: NDArray[np.float64],
    inversion: InversionResult,
    *,
    control_change_count: int,
    safety: SafetyValidationResult | None = None,
) -> QualityReport:
    target_time = np.asarray(target_time, dtype=np.float64)
    target_bt = np.asarray(target_bt, dtype=np.float64)
    pred_time = np.asarray(inversion.time, dtype=np.float64)

    target_milestones: dict[str, float | None] = {
        name: _crossing_time(target_time, target_bt, threshold)
        for name, threshold in _MILESTONE_THRESHOLDS.items()
    }
    target_milestones['Drop'] = float(target_time[-1]) if len(target_time) else None

    pred_milestones: dict[str, float | None] = {
        'Yellowing': inversion.yellowing_time,
        'First Crack': inversion.first_crack_time,
        'Second Crack': inversion.second_crack_time,
        'Drop': float(pred_time[-1]) if len(pred_time) else None,
    }

    milestone_deltas: dict[str, float | None] = {}
    milestone_errors_abs: list[float] = []
    for name in ['Yellowing', 'First Crack', 'Second Crack', 'Drop']:
        t_ref = target_milestones.get(name)
        t_pred = pred_milestones.get(name)
        if t_ref is None or t_pred is None:
            milestone_deltas[name] = None
            continue
        delta = float(t_pred - t_ref)
        milestone_deltas[name] = delta
        milestone_errors_abs.append(abs(delta))

    phase_error_penalty = 0.0
    if milestone_errors_abs:
        mean_abs_min = float(np.mean(milestone_errors_abs) / 60.0)
        phase_error_penalty = min(20.0, mean_abs_min * 6.0)

    rmse_penalty = min(40.0, inversion.rmse * 4.0)
    max_penalty = min(20.0, inversion.max_tracking_error * 2.0)
    churn_penalty = min(10.0, max(0, control_change_count - 20) * 0.25)
    safety_penalty = 0.0 if (safety is None or safety.is_safe) else 12.0

    score = 100.0 - (rmse_penalty + max_penalty + phase_error_penalty + churn_penalty + safety_penalty)
    score = float(max(0.0, min(100.0, score)))

    notes: list[str] = []
    if safety is not None and not safety.is_safe:
        notes.append('Safety penalty applied due to failed dry-run limits.')
    if safety is not None and safety.is_safe:
        notes.append('Dry-run safety validation passed.')

    return QualityReport(
        score=score,
        grade=_grade(score),
        rmse_c=float(inversion.rmse),
        max_error_c=float(inversion.max_tracking_error),
        milestone_deltas_s=milestone_deltas,
        control_change_count=int(control_change_count),
        notes=notes,
    )
