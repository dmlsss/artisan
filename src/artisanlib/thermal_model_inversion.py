#
# ABOUT
# Invert the lumped-parameter thermal ODE model to compute heater%/fan%
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

from artisanlib.thermal_model import KaleidoThermalModel, _safe_sigmoid


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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InversionResult:
    """Result of inverting the thermal model for a target BT curve.

    Attributes:
        time: Time grid in seconds from CHARGE.
        heater_pct: Computed heater-% schedule (0-100).
        fan_pct: Fan-% schedule (given or co-optimised).
        predicted_bt: Forward-simulated BT from the computed controls.
        tracking_error: ``target_bt - predicted_bt`` at each time point.
        max_tracking_error: Maximum absolute tracking error (deg C).
        rmse: Root-mean-square tracking error (deg C).
        exo_warning_time: Time (s) when BT first crosses ``T_exo_onset``,
            or ``None`` if it never does.
    """

    time: NDArray[np.float64]
    heater_pct: NDArray[np.float64]
    fan_pct: NDArray[np.float64]
    predicted_bt: NDArray[np.float64]
    tracking_error: NDArray[np.float64]
    max_tracking_error: float
    rmse: float
    exo_warning_time: float | None
    yellowing_time: float | None
    first_crack_time: float | None
    second_crack_time: float | None
    drop_time: float
    dtr_percent: float | None

    # reference kept for resample forward-simulation
    _model: KaleidoThermalModel | None = None
    _mass_kg: float | None = None
    _T0: float | None = None
    _target_bt: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample_to_interval(self, interval_s: float) -> InversionResult:
        """Resample schedules to a fixed time interval.

        Creates a new uniform time grid spaced by *interval_s* seconds,
        interpolates the heater and fan schedules onto it, rounds the
        control values to integers (Kaleido only accepts int 0–100), and
        re-runs a forward simulation to recompute tracking error.

        Args:
            interval_s: Desired time spacing in seconds (e.g. 10.0).

        Returns:
            A new :class:`InversionResult` on the resampled grid.

        Raises:
            RuntimeError: If the result was created without the internal
                model reference needed for re-simulation.
        """
        if self._model is None or self._mass_kg is None or self._T0 is None:
            msg = (
                'Cannot resample: InversionResult was created without '
                'an internal model reference.  Use invert_model() to '
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

        # Forward-simulate with the resampled integer controls
        pred_bt = self._model.simulate(
            time=new_time,
            hp_schedule=new_hp,
            fan_schedule=new_fan,
            T0=self._T0,
            mass_kg=self._mass_kg,
        )

        # Interpolate original target BT onto new grid for error calc
        if self._target_bt is not None:
            target_bt_resampled = np.interp(new_time, self.time, self._target_bt)
        else:
            target_bt_resampled = np.interp(new_time, self.time,
                                            self.predicted_bt + self.tracking_error)

        err = target_bt_resampled - pred_bt
        max_err = float(np.max(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        yellowing_time = _crossing_time(new_time, pred_bt, _YELLOWING_BT_C)
        first_crack_time = _crossing_time(new_time, pred_bt, _FIRST_CRACK_BT_C)
        second_crack_time = _crossing_time(new_time, pred_bt, _SECOND_CRACK_BT_C)
        drop_time = float(new_time[-1])
        dtr_percent = _compute_dtr_percent(first_crack_time, drop_time)

        _log.info(
            'Resampled to %.1f s interval: %d points, '
            'max_err=%.2f C, RMSE=%.2f C',
            interval_s, len(new_time), max_err, rmse,
        )

        return InversionResult(
            time=new_time,
            heater_pct=new_hp,
            fan_pct=new_fan,
            predicted_bt=pred_bt,
            tracking_error=err,
            max_tracking_error=max_err,
            rmse=rmse,
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
# Moving-average smoothing helper
# ---------------------------------------------------------------------------

def _moving_average(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Apply a centred moving-average filter.

    Uses ``numpy.convolve`` with a uniform kernel.  The output has the
    same length as *arr* (``mode='same'``), so edge values are computed
    with a shorter effective window.

    Args:
        arr: 1-D input array.
        window: Kernel width (clamped to an odd number >= 1).

    Returns:
        Smoothed copy of *arr*.
    """
    window = max(1, window)
    if window % 2 == 0:
        window += 1  # ensure odd for centred smoothing
    if window <= 1 or len(arr) <= window:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode='same')


# ---------------------------------------------------------------------------
# Model inversion
# ---------------------------------------------------------------------------

def invert_model(
    model: KaleidoThermalModel,
    target_time: NDArray[np.float64],
    target_bt: NDArray[np.float64],
    mass_kg: float,
    fan_schedule: NDArray[np.float64] | float = 30.0,
    hp_min: int = 0,
    hp_max: int = 100,
    fan_min: int = 0,
    fan_max: int = 100,
    smoothing_window: int = 5,
) -> InversionResult:
    """Invert the thermal model to find a heater-% schedule.

    Given a target bean-temperature curve, algebraically solves for the
    heater-% at every time point that would reproduce the desired rate
    of temperature change under the lumped-parameter ODE model.

    The derivation starts from the energy balance::

        m_eff * cp(T) * dT/dt = h_eff * (T_env - T) + Q_exo

    Solving for ``hp%``::

        hp = [(needed_heat - Q_exo) / h_eff
              - T_amb + k_fan * fan + T] / k_hp

    where ``needed_heat = m_eff * cp(T) * dBT_target/dt`` and
    ``m_eff = mA * (mass_kg / m_ref)``.

    Args:
        model: Calibrated :class:`KaleidoThermalModel`.
        target_time: 1-D array of time points (seconds from CHARGE).
        target_bt: 1-D array of target bean temperatures (deg C).
        mass_kg: Batch mass in kg.
        fan_schedule: Either a scalar (constant fan-%) or a 1-D array
            of fan-% values aligned with *target_time*.
        hp_min: Minimum allowed heater-% (default 0).
        hp_max: Maximum allowed heater-% (default 100).
        fan_min: Minimum allowed fan-% (default 0).
        fan_max: Maximum allowed fan-% (default 100).
        smoothing_window: Width of the moving-average kernel applied to
            ``dBT/dt`` and to the computed heater-% schedule to reduce
            chatter (default 5).

    Returns:
        An :class:`InversionResult` with the computed schedules, the
        forward-simulated BT, and tracking-error metrics.
    """
    target_time = np.asarray(target_time, dtype=np.float64)
    target_bt = np.asarray(target_bt, dtype=np.float64)

    n = len(target_time)
    if len(target_bt) != n:
        msg = (
            f'target_time and target_bt must have the same length; '
            f'got {n} and {len(target_bt)}'
        )
        raise ValueError(msg)

    p = model.params

    # ---- Fan schedule ---------------------------------------------------
    if isinstance(fan_schedule, (int, float)):
        fan_arr = np.full(n, float(fan_schedule), dtype=np.float64)
    else:
        fan_arr = np.asarray(fan_schedule, dtype=np.float64)
        if len(fan_arr) != n:
            msg = (
                f'fan_schedule length ({len(fan_arr)}) does not match '
                f'target_time length ({n})'
            )
            raise ValueError(msg)
    fan_arr = np.clip(fan_arr, fan_min, fan_max)

    # ---- Compute dBT/dt from target curve -------------------------------
    dBTdt = np.gradient(target_bt, target_time)
    dBTdt = _moving_average(dBTdt, smoothing_window)

    # ---- Effective thermal mass -----------------------------------------
    mass_scale = mass_kg / max(p.m_ref, _EPS)
    mA_eff = p.mA * mass_scale

    # ---- Algebraic inversion at each time point -------------------------
    hp_raw = np.empty(n, dtype=np.float64)

    for i in range(n):
        T = target_bt[i]
        fan = fan_arr[i]

        # Temperature-dependent specific heat (clamped away from zero)
        cp = max(p.cp0 + p.cp1 * T, _EPS)

        # Needed energy input rate (W-equivalent) to achieve dBT/dt
        needed_heat = mA_eff * cp * dBTdt[i]

        # Exothermic heat release
        sig_arg = (T - p.T_exo_onset) / max(p.T_exo_width, _EPS)
        Q_exo = p.q_exo * float(_safe_sigmoid(sig_arg))

        # Effective heat-transfer coefficient (clamped away from zero)
        h_eff = max(p.h0 + p.h1 * fan, _EPS)

        # Effective heater-power gain (clamped away from zero)
        k_hp = max(abs(p.k_hp), _EPS)

        # Algebraic solution for hp%:
        #   hp = [(needed_heat - Q_exo) / h_eff
        #         - T_amb + k_fan * fan + T] / k_hp
        hp = ((needed_heat - Q_exo) / h_eff - p.T_amb + p.k_fan * fan + T) / k_hp
        hp_raw[i] = hp

    # ---- Smooth and clip the heater schedule ----------------------------
    hp_smooth = _moving_average(hp_raw, smoothing_window)
    hp_clipped = np.clip(hp_smooth, hp_min, hp_max)

    _log.info(
        'Inversion complete: hp range [%.1f, %.1f], mean %.1f',
        float(np.min(hp_clipped)),
        float(np.max(hp_clipped)),
        float(np.mean(hp_clipped)),
    )

    # ---- Forward-simulate to get predicted BT ---------------------------
    T0 = float(target_bt[0])
    predicted_bt = model.simulate(
        time=target_time,
        hp_schedule=hp_clipped,
        fan_schedule=fan_arr,
        T0=T0,
        mass_kg=mass_kg,
    )

    # ---- Tracking error -------------------------------------------------
    err = target_bt - predicted_bt
    max_err = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    _log.info('Tracking error: max=%.2f C, RMSE=%.2f C', max_err, rmse)

    # ---- Exothermic onset detection -------------------------------------
    exo_time = _crossing_time(target_time, predicted_bt, p.T_exo_onset)
    if exo_time is not None:
        _log.info('Exothermic onset at t=%.1f s (threshold=%.1f C)', exo_time, p.T_exo_onset)

    yellowing_time = _crossing_time(target_time, predicted_bt, _YELLOWING_BT_C)
    first_crack_time = _crossing_time(target_time, predicted_bt, _FIRST_CRACK_BT_C)
    second_crack_time = _crossing_time(target_time, predicted_bt, _SECOND_CRACK_BT_C)
    drop_time = float(target_time[-1])
    dtr_percent = _compute_dtr_percent(first_crack_time, drop_time)

    return InversionResult(
        time=target_time,
        heater_pct=hp_clipped,
        fan_pct=fan_arr,
        predicted_bt=predicted_bt,
        tracking_error=err,
        max_tracking_error=max_err,
        rmse=rmse,
        exo_warning_time=exo_time,
        yellowing_time=yellowing_time,
        first_crack_time=first_crack_time,
        second_crack_time=second_crack_time,
        drop_time=drop_time,
        dtr_percent=dtr_percent,
        _model=model,
        _mass_kg=mass_kg,
        _T0=T0,
        _target_bt=target_bt,
    )
