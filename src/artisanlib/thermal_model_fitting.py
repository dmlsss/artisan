#
# ABOUT
# Identifies thermal model parameters from calibration roast data using
# scipy global + local optimisation.  Standalone — no Artisan GUI dependency.

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
# Derek Mead, 2025

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import Final, TYPE_CHECKING

import numpy as np
from scipy.optimize import differential_evolution, minimize

from artisanlib.thermal_model import (
    KaleidoThermalModel,
    PARAM_NAMES,
    ThermalModelParams,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from artisanlib.thermal_profile_parser import CalibrationData


_log: Final[logging.Logger] = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Results from a thermal-model parameter fit."""

    params: ThermalModelParams
    rmse: float
    max_error: float
    r_squared: float
    per_roast_rmse: list[float]
    residuals: list[NDArray]
    predicted: list[NDArray]
    converged: bool
    message: str


# ---------------------------------------------------------------------------
# Default bounds
# ---------------------------------------------------------------------------

def default_bounds() -> dict[str, tuple[float, float]]:
    """Return physically-meaningful parameter bounds for the Kaleido M1 Lite.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from parameter name to ``(lower, upper)`` bound.
    """
    return {
        'h0':          (0.01, 5.0),
        'h1':          (0.0, 0.1),
        'h2':          (0.0, 0.1),
        'k_hp':        (1.0, 8.0),
        'k_fan':       (0.0, 3.0),
        'T_amb':       (15.0, 40.0),
        'mA':          (0.001, 0.1),
        'cp0':         (0.5, 3.0),
        'cp1':         (0.0, 0.01),
        'q_exo':       (0.0, 50.0),
        'T_exo_onset': (185.0, 215.0),
        'T_exo_width': (1.0, 20.0),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bounds_list(bounds: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Convert a bounds dict to a list ordered by :data:`PARAM_NAMES`."""
    return [bounds[name] for name in PARAM_NAMES]


def _simulate_roast(
    model: KaleidoThermalModel,
    calib: CalibrationData,
) -> NDArray:
    """Run a forward simulation for a single calibration roast."""
    return model.simulate(
        time=calib.time,
        hp_schedule=calib.heater_pct,
        fan_schedule=calib.fan_pct,
        drum_schedule=calib.drum_pct,
        T0=float(calib.bt[0]),
        mass_kg=calib.batch_mass_kg,
    )


def _compute_metrics(
    params: ThermalModelParams,
    calibration_data: list[CalibrationData],
) -> tuple[float, float, float, list[float], list[NDArray], list[NDArray]]:
    """Compute fit-quality metrics across all calibration roasts.

    Returns
    -------
    tuple
        ``(rmse, max_error, r_squared, per_roast_rmse, residuals, predicted)``
    """
    model = KaleidoThermalModel(params)

    all_residuals: list[NDArray] = []
    all_predicted: list[NDArray] = []
    per_roast_rmse: list[float] = []

    all_measured: list[NDArray] = []

    for calib in calibration_data:
        pred = _simulate_roast(model, calib)
        measured = np.asarray(calib.bt, dtype=np.float64)
        resid = pred - measured

        all_predicted.append(pred)
        all_residuals.append(resid)
        all_measured.append(measured)

        roast_rmse = float(np.sqrt(np.mean(resid ** 2)))
        per_roast_rmse.append(roast_rmse)

    # Global metrics across all roasts
    concat_resid = np.concatenate(all_residuals)
    concat_measured = np.concatenate(all_measured)

    rmse = float(np.sqrt(np.mean(concat_resid ** 2)))
    max_error = float(np.max(np.abs(concat_resid)))

    # R-squared (coefficient of determination)
    ss_res = float(np.sum(concat_resid ** 2))
    ss_tot = float(np.sum((concat_measured - np.mean(concat_measured)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return rmse, max_error, r_squared, per_roast_rmse, all_residuals, all_predicted


def _weighted_residual_objective(residuals_by_roast: list[NDArray]) -> float:
    """Return weighted mean-squared error across roasts.

    Each roast contributes by its fraction of total samples, so each sample
    is weighted once overall (equivalent to global MSE over concatenated data).
    """
    if not residuals_by_roast:
        raise ValueError('residuals_by_roast must not be empty')

    total_points = sum(len(resid) for resid in residuals_by_roast)
    if total_points <= 0:
        raise ValueError('residuals_by_roast contains no sample points')

    objective = 0.0
    for resid in residuals_by_roast:
        n_pts = len(resid)
        if n_pts <= 0:
            continue
        weight = n_pts / total_points
        objective += weight * float(np.mean(resid ** 2))
    return objective


# ---------------------------------------------------------------------------
# Main fitting routine
# ---------------------------------------------------------------------------

def fit_model(
    calibration_data: list[CalibrationData],
    initial_params: ThermalModelParams | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    progress_callback: Callable[[int, float], None] | None = None,
) -> FitResult:
    """Identify thermal model parameters from calibration roast data.

    Uses two-phase optimisation:
    1. **Global search** via ``scipy.optimize.differential_evolution``.
    2. **Local refinement** via ``scipy.optimize.minimize`` (L-BFGS-B).

    Parameters
    ----------
    calibration_data : list[CalibrationData]
        One or more parsed roast profiles.
    initial_params : ThermalModelParams | None
        Starting point for the fit.  When *None* the default
        :class:`ThermalModelParams` values are used.
    bounds : dict[str, tuple[float, float]] | None
        Per-parameter ``(lower, upper)`` bounds.  Defaults to
        :func:`default_bounds`.
    max_iter : int
        Maximum iterations for ``differential_evolution``.
    tol : float
        Convergence tolerance for the optimisers.
    progress_callback : Callable[[int, float], None] | None
        Called as ``progress_callback(iteration, best_cost)`` during the
        global search.  Useful for progress bars.

    Returns
    -------
    FitResult
        Identified parameters and fit-quality metrics.

    Raises
    ------
    ValueError
        If *calibration_data* is empty.
    """
    if not calibration_data:
        raise ValueError('calibration_data must not be empty')

    if bounds is None:
        bounds = default_bounds()

    if initial_params is None:
        initial_params = ThermalModelParams()

    # Determine m_ref from the first roast (or keep default)
    m_ref = initial_params.m_ref
    if calibration_data[0].batch_mass_kg > 0.0:
        m_ref = calibration_data[0].batch_mass_kg

    bounds_ordered = _bounds_list(bounds)

    total_points = sum(len(c.time) for c in calibration_data)
    if total_points <= 0:
        raise ValueError('calibration_data contains no sample points')

    _log.info(
        'Starting thermal model fit: %d roast(s), %d total points',
        len(calibration_data), total_points,
    )

    # ── Cost function ────────────────────────────────────────────────
    _PENALTY: Final[float] = 1e10

    def cost(x: NDArray) -> float:
        """Weighted MSE across calibration roasts."""
        try:
            params = ThermalModelParams.from_vector(x, m_ref=m_ref)
            model = KaleidoThermalModel(params)
            residuals: list[NDArray] = []
            for calib in calibration_data:
                pred = _simulate_roast(model, calib)
                measured = np.asarray(calib.bt, dtype=np.float64)
                resid = pred - measured
                residuals.append(resid)
            return _weighted_residual_objective(residuals)
        except Exception:  # noqa: BLE001
            _log.debug('Simulation failed during optimisation', exc_info=True)
            return _PENALTY

    # ── Differential-evolution callback ──────────────────────────────
    _iter_count = [0]

    def _de_callback(xk: NDArray, convergence: float = 0.0) -> None:  # noqa: ARG001
        _iter_count[0] += 1
        if progress_callback is not None:
            progress_callback(_iter_count[0], cost(xk))

    # ── Phase 1: global search ───────────────────────────────────────
    _log.info('Phase 1: differential evolution (max_iter=%d)', max_iter)
    de_result = differential_evolution(
        cost,
        bounds=bounds_ordered,
        x0=initial_params.to_vector(),
        maxiter=max_iter,
        tol=tol,
        seed=42,
        callback=_de_callback,
        polish=False,       # we do our own local refinement
        disp=False,
    )
    _log.info(
        'DE finished: success=%s, fun=%.6g, nit=%d — %s',
        de_result.success, de_result.fun, de_result.nit, de_result.message,
    )

    # ── Phase 2: local refinement ────────────────────────────────────
    _log.info('Phase 2: L-BFGS-B local refinement')
    local_result = minimize(
        cost,
        x0=de_result.x,
        method='L-BFGS-B',
        bounds=bounds_ordered,
        options={
            'maxiter': max_iter,
            'ftol': tol,
            'gtol': tol * 10,
        },
    )
    _log.info(
        'L-BFGS-B finished: success=%s, fun=%.6g, nit=%d — %s',
        local_result.success, local_result.fun, local_result.nit,
        local_result.message,
    )

    # Pick the better result
    if local_result.fun < de_result.fun:
        best_x = local_result.x
        converged = local_result.success
        message = f'L-BFGS-B: {local_result.message}'
    else:
        best_x = de_result.x
        converged = de_result.success
        message = f'DE: {de_result.message}'

    best_params = ThermalModelParams.from_vector(best_x, m_ref=m_ref)

    # ── Compute final metrics ────────────────────────────────────────
    rmse, max_error, r_squared, per_roast_rmse, residuals, predicted = (
        _compute_metrics(best_params, calibration_data)
    )

    _log.info(
        'Fit complete: RMSE=%.3f C, max_error=%.3f C, R^2=%.6f',
        rmse, max_error, r_squared,
    )

    return FitResult(
        params=best_params,
        rmse=rmse,
        max_error=max_error,
        r_squared=r_squared,
        per_roast_rmse=per_roast_rmse,
        residuals=residuals,
        predicted=predicted,
        converged=converged,
        message=message,
    )
