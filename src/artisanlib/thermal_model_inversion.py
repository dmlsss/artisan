#
# ABOUT
# Invert the lumped-parameter thermal ODE model to compute heater%/fan%/drum%
# schedules that reproduce a target bean-temperature (BT) curve on the
# Kaleido M1 Lite coffee roaster.

# LICENSE
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.

# AUTHOR
# Derek Kwan, 2025


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray  # pylint: disable=unused-import

from artisanlib.thermal_model import KaleidoThermalModel, _safe_sigmoid, regime_heat_multiplier


_log: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numeric safety floor — avoids division-by-zero in the inversion algebra
# ---------------------------------------------------------------------------

_EPS: Final[float] = 1e-9
_YELLOWING_BT_C: Final[float] = 150.0
_FIRST_CRACK_BT_C: Final[float] = 196.0
_SECOND_CRACK_BT_C: Final[float] = 224.0


def _crossing_time(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
    threshold: float,
) -> float | None:
    """Return the first timestamp where *values* crosses *threshold*."""
    if len(time) == 0 or len(values) == 0:
        return None

    above = values >= threshold
    if not np.any(above):
        return None

    idx = int(np.argmax(above))
    if idx <= 0:
        return float(time[0])

    t0 = float(time[idx - 1])
    t1 = float(time[idx])
    v0 = float(values[idx - 1])
    v1 = float(values[idx])

    dv = v1 - v0
    if abs(dv) <= _EPS:
        return t1

    frac = (threshold - v0) / dv
    frac = min(1.0, max(0.0, frac))
    return t0 + frac * (t1 - t0)


def _compute_dtr_percent(first_crack_time: float | None, drop_time: float) -> float | None:
    if first_crack_time is None or first_crack_time >= drop_time or drop_time <= 0:
        return None
    return 100.0 * (drop_time - first_crack_time) / drop_time


def _as_schedule_array(
    value: NDArray[np.float64] | float | None,
    length: int,
    default: float,
) -> NDArray[np.float64]:
    """Normalize scalar/array optional schedule inputs to a numpy array."""
    if value is None:
        return np.full(length, float(default), dtype=np.float64)
    if isinstance(value, (int, float)):
        return np.full(length, float(value), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InversionResult:
    """Result of inverting the thermal model for a target BT curve."""

    time: NDArray[np.float64]
    heater_pct: NDArray[np.float64]
    fan_pct: NDArray[np.float64]
    drum_pct: NDArray[np.float64]
    predicted_bt: NDArray[np.float64]
    tracking_error: NDArray[np.float64]
    max_tracking_error: float
    rmse: float
    objective_score: float | None
    exo_warning_time: float | None
    yellowing_time: float | None
    first_crack_time: float | None
    second_crack_time: float | None
    drop_time: float
    dtr_percent: float | None

    # references kept for resample forward-simulation
    _model: KaleidoThermalModel | None = None
    _mass_kg: float | None = None
    _T0: float | None = None
    _target_bt: NDArray[np.float64] | None = None

    def resample_to_interval(self, interval_s: float) -> InversionResult:
        """Resample schedules to a fixed time interval."""
        if self._model is None or self._mass_kg is None or self._T0 is None:
            msg = (
                'Cannot resample: InversionResult was created without '
                'an internal model reference. Use invert_model() to '
                'produce resample-capable results.'
            )
            raise RuntimeError(msg)

        t_start = float(self.time[0])
        t_end = float(self.time[-1])
        new_time = np.arange(t_start, t_end + interval_s * 0.5, interval_s)

        new_hp = np.round(np.interp(new_time, self.time, self.heater_pct))
        new_hp = np.clip(new_hp, 0.0, 100.0)
        new_fan = np.round(np.interp(new_time, self.time, self.fan_pct))
        new_fan = np.clip(new_fan, 0.0, 100.0)
        new_drum = np.round(np.interp(new_time, self.time, self.drum_pct))
        new_drum = np.clip(new_drum, 0.0, 100.0)

        pred_bt = self._model.simulate(
            time=new_time,
            hp_schedule=new_hp,
            fan_schedule=new_fan,
            drum_schedule=new_drum,
            T0=self._T0,
            mass_kg=self._mass_kg,
        )

        if self._target_bt is not None:
            target_bt_resampled = np.interp(new_time, self.time, self._target_bt)
        else:
            target_bt_resampled = np.interp(
                new_time,
                self.time,
                self.predicted_bt + self.tracking_error,
            )

        err = target_bt_resampled - pred_bt
        max_err = float(np.max(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        yellowing_time = _crossing_time(new_time, pred_bt, _YELLOWING_BT_C)
        first_crack_time = _crossing_time(new_time, pred_bt, _FIRST_CRACK_BT_C)
        second_crack_time = _crossing_time(new_time, pred_bt, _SECOND_CRACK_BT_C)
        drop_time = float(new_time[-1])
        dtr_percent = _compute_dtr_percent(first_crack_time, drop_time)

        return InversionResult(
            time=new_time,
            heater_pct=new_hp,
            fan_pct=new_fan,
            drum_pct=new_drum,
            predicted_bt=pred_bt,
            tracking_error=err,
            max_tracking_error=max_err,
            rmse=rmse,
            objective_score=self.objective_score,
            exo_warning_time=self.exo_warning_time,
            yellowing_time=yellowing_time,
            first_crack_time=first_crack_time,
            second_crack_time=second_crack_time,
            drop_time=drop_time,
            dtr_percent=dtr_percent,
            _model=self._model,
            _mass_kg=self._mass_kg,
            _T0=self._T0,
            _target_bt=self._target_bt,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _moving_average(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Apply a centred moving-average filter."""
    window = max(1, window)
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(arr) <= window:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode='same')


def _compute_heater_schedule(
    model: KaleidoThermalModel,
    target_bt: NDArray[np.float64],
    dBTdt: NDArray[np.float64],
    fan_arr: NDArray[np.float64],
    drum_arr: NDArray[np.float64],
    mass_kg: float,
    hp_min: int,
    hp_max: int,
    smoothing_window: int,
) -> NDArray[np.float64]:
    p = model.params
    mass_scale = mass_kg / max(p.m_ref, _EPS)
    mA_eff = p.mA * mass_scale
    hp_raw = np.empty(len(target_bt), dtype=np.float64)

    for i, T in enumerate(target_bt):
        fan = fan_arr[i]
        drum = drum_arr[i]
        cp = max(p.cp0 + p.cp1 * T, _EPS)
        needed_heat = mA_eff * cp * dBTdt[i]

        sig_arg = (T - p.T_exo_onset) / max(p.T_exo_width, _EPS)
        Q_exo = p.q_exo * float(_safe_sigmoid(sig_arg))

        h_eff = max(
            (p.h0 + p.h1 * fan + p.h2 * drum) * regime_heat_multiplier(p, float(T)),
            _EPS,
        )
        k_hp = max(abs(p.k_hp), _EPS)

        # Algebraic solution for hp%:
        # hp = [(needed_heat - Q_exo) / h_eff - T_amb + k_fan * fan + T] / k_hp
        hp = ((needed_heat - Q_exo) / h_eff - p.T_amb + p.k_fan * fan + T) / k_hp
        hp_raw[i] = hp

    hp_smooth = _moving_average(hp_raw, smoothing_window)
    return np.clip(hp_smooth, hp_min, hp_max)


def _segment_edges(length: int, requested_segments: int) -> NDArray[np.int_]:
    segments = max(2, min(requested_segments, max(2, length // 8)))
    edges = np.linspace(0, length, segments + 1, dtype=int)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = length
    return edges


def _segment_levels(arr: NDArray[np.float64], edges: NDArray[np.int_]) -> NDArray[np.float64]:
    levels = np.zeros(len(edges) - 1, dtype=np.float64)
    for i in range(len(levels)):
        s, e = int(edges[i]), int(edges[i + 1])
        levels[i] = float(np.mean(arr[s:e])) if e > s else float(arr[min(s, len(arr) - 1)])
    return levels


def _expand_levels(
    levels: NDArray[np.float64],
    edges: NDArray[np.int_],
    length: int,
) -> NDArray[np.float64]:
    arr = np.zeros(length, dtype=np.float64)
    for i, level in enumerate(levels):
        s, e = int(edges[i]), int(edges[i + 1])
        arr[s:e] = level
    return arr


def _compute_objective(
    target_bt: NDArray[np.float64],
    predicted_bt: NDArray[np.float64],
    hp: NDArray[np.float64],
    fan: NDArray[np.float64],
    drum: NDArray[np.float64],
    smoothness_weight: float,
) -> float:
    err = target_bt - predicted_bt
    mse = float(np.mean(err ** 2))
    hp_smooth = float(np.mean(np.abs(np.diff(hp)))) if len(hp) > 1 else 0.0
    fan_smooth = float(np.mean(np.abs(np.diff(fan)))) if len(fan) > 1 else 0.0
    drum_smooth = float(np.mean(np.abs(np.diff(drum)))) if len(drum) > 1 else 0.0
    smooth_penalty = hp_smooth + 0.35 * fan_smooth + 0.25 * drum_smooth
    return mse + smoothness_weight * smooth_penalty


def _optimize_fan_and_drum(
    model: KaleidoThermalModel,
    target_time: NDArray[np.float64],
    target_bt: NDArray[np.float64],
    dBTdt: NDArray[np.float64],
    mass_kg: float,
    fan_init: NDArray[np.float64],
    drum_init: NDArray[np.float64],
    *,
    hp_min: int,
    hp_max: int,
    fan_min: int,
    fan_max: int,
    drum_min: int,
    drum_max: int,
    smoothing_window: int,
    optimizer_segments: int,
    optimizer_iterations: int,
    optimizer_step_pct: int,
    smoothness_weight: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Coordinate-descent optimizer over piecewise fan/drum schedules."""
    n = len(target_time)
    edges = _segment_edges(n, optimizer_segments)
    fan_levels = _segment_levels(fan_init, edges)
    drum_levels = _segment_levels(drum_init, edges)

    def evaluate(
        f_levels: NDArray[np.float64],
        d_levels: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        fan = np.clip(_expand_levels(f_levels, edges, n), fan_min, fan_max)
        drum = np.clip(_expand_levels(d_levels, edges, n), drum_min, drum_max)
        hp = _compute_heater_schedule(
            model=model,
            target_bt=target_bt,
            dBTdt=dBTdt,
            fan_arr=fan,
            drum_arr=drum,
            mass_kg=mass_kg,
            hp_min=hp_min,
            hp_max=hp_max,
            smoothing_window=smoothing_window,
        )
        pred = model.simulate(
            time=target_time,
            hp_schedule=hp,
            fan_schedule=fan,
            drum_schedule=drum,
            T0=float(target_bt[0]),
            mass_kg=mass_kg,
        )
        obj = _compute_objective(target_bt, pred, hp, fan, drum, smoothness_weight)
        return obj, hp, fan, drum, pred

    best_obj, best_hp, best_fan, best_drum, best_pred = evaluate(fan_levels, drum_levels)
    step = max(1, int(optimizer_step_pct))

    for _ in range(max(1, optimizer_iterations)):
        improved = False
        for seg in range(len(fan_levels)):
            for channel in ('fan', 'drum'):
                for delta in (-step, step):
                    if channel == 'fan':
                        trial_fan = fan_levels.copy()
                        trial_fan[seg] = np.clip(trial_fan[seg] + delta, fan_min, fan_max)
                        trial_drum = drum_levels
                    else:
                        trial_drum = drum_levels.copy()
                        trial_drum[seg] = np.clip(trial_drum[seg] + delta, drum_min, drum_max)
                        trial_fan = fan_levels

                    trial_obj, trial_hp, trial_fan_arr, trial_drum_arr, trial_pred = evaluate(
                        trial_fan,
                        trial_drum,
                    )
                    if trial_obj + 1e-9 < best_obj:
                        best_obj = trial_obj
                        best_hp = trial_hp
                        best_fan = trial_fan_arr
                        best_drum = trial_drum_arr
                        best_pred = trial_pred
                        fan_levels = trial_fan.copy()
                        drum_levels = trial_drum.copy()
                        improved = True
        if not improved:
            if step > 1:
                step = max(1, step // 2)
            else:
                break

    return best_hp, best_fan, best_drum, best_pred, float(best_obj)


# ---------------------------------------------------------------------------
# Model inversion
# ---------------------------------------------------------------------------


def invert_model(
    model: KaleidoThermalModel,
    target_time: NDArray[np.float64],
    target_bt: NDArray[np.float64],
    mass_kg: float,
    fan_schedule: NDArray[np.float64] | float = 30.0,
    drum_schedule: NDArray[np.float64] | float | None = None,
    hp_min: int = 0,
    hp_max: int = 100,
    fan_min: int = 0,
    fan_max: int = 100,
    drum_min: int = 0,
    drum_max: int = 100,
    smoothing_window: int = 5,
    optimize_actuators: bool = False,
    optimizer_iterations: int = 3,
    optimizer_segments: int = 8,
    optimizer_step_pct: int = 8,
    smoothness_weight: float = 0.02,
) -> InversionResult:
    """Invert the thermal model to find actuator schedules."""
    target_time = np.asarray(target_time, dtype=np.float64)
    target_bt = np.asarray(target_bt, dtype=np.float64)

    n = len(target_time)
    if len(target_bt) != n:
        msg = (
            f'target_time and target_bt must have the same length; '
            f'got {n} and {len(target_bt)}'
        )
        raise ValueError(msg)

    fan_arr = _as_schedule_array(fan_schedule, n, default=30.0)
    drum_arr = _as_schedule_array(drum_schedule, n, default=0.0)
    if len(fan_arr) != n:
        raise ValueError(
            f'fan_schedule length ({len(fan_arr)}) does not match target_time length ({n})'
        )
    if len(drum_arr) != n:
        raise ValueError(
            f'drum_schedule length ({len(drum_arr)}) does not match target_time length ({n})'
        )
    fan_arr = np.clip(fan_arr, fan_min, fan_max)
    drum_arr = np.clip(drum_arr, drum_min, drum_max)

    dBTdt = np.gradient(target_bt, target_time)
    dBTdt = _moving_average(dBTdt, smoothing_window)

    objective_score: float | None = None
    if optimize_actuators:
        hp, fan_arr, drum_arr, predicted_bt, objective_score = _optimize_fan_and_drum(
            model=model,
            target_time=target_time,
            target_bt=target_bt,
            dBTdt=dBTdt,
            mass_kg=mass_kg,
            fan_init=fan_arr,
            drum_init=drum_arr,
            hp_min=hp_min,
            hp_max=hp_max,
            fan_min=fan_min,
            fan_max=fan_max,
            drum_min=drum_min,
            drum_max=drum_max,
            smoothing_window=smoothing_window,
            optimizer_segments=optimizer_segments,
            optimizer_iterations=optimizer_iterations,
            optimizer_step_pct=optimizer_step_pct,
            smoothness_weight=smoothness_weight,
        )
    else:
        hp = _compute_heater_schedule(
            model=model,
            target_bt=target_bt,
            dBTdt=dBTdt,
            fan_arr=fan_arr,
            drum_arr=drum_arr,
            mass_kg=mass_kg,
            hp_min=hp_min,
            hp_max=hp_max,
            smoothing_window=smoothing_window,
        )
        predicted_bt = model.simulate(
            time=target_time,
            hp_schedule=hp,
            fan_schedule=fan_arr,
            drum_schedule=drum_arr,
            T0=float(target_bt[0]),
            mass_kg=mass_kg,
        )
        objective_score = _compute_objective(
            target_bt=target_bt,
            predicted_bt=predicted_bt,
            hp=hp,
            fan=fan_arr,
            drum=drum_arr,
            smoothness_weight=smoothness_weight,
        )

    err = target_bt - predicted_bt
    max_err = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    p = model.params
    exo_time = _crossing_time(target_time, predicted_bt, p.T_exo_onset)
    yellowing_time = _crossing_time(target_time, predicted_bt, _YELLOWING_BT_C)
    first_crack_time = _crossing_time(target_time, predicted_bt, _FIRST_CRACK_BT_C)
    second_crack_time = _crossing_time(target_time, predicted_bt, _SECOND_CRACK_BT_C)
    drop_time = float(target_time[-1])
    dtr_percent = _compute_dtr_percent(first_crack_time, drop_time)

    return InversionResult(
        time=target_time,
        heater_pct=hp,
        fan_pct=fan_arr,
        drum_pct=drum_arr,
        predicted_bt=predicted_bt,
        tracking_error=err,
        max_tracking_error=max_err,
        rmse=rmse,
        objective_score=objective_score,
        exo_warning_time=exo_time,
        yellowing_time=yellowing_time,
        first_crack_time=first_crack_time,
        second_crack_time=second_crack_time,
        drop_time=drop_time,
        dtr_percent=dtr_percent,
        _model=model,
        _mass_kg=mass_kg,
        _T0=float(target_bt[0]),
        _target_bt=target_bt,
    )
