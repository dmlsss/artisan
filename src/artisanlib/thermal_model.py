#
# ABOUT
# Lumped-parameter thermal ODE model for the Kaleido M1 Lite coffee roaster.
# Provides forward simulation and residual computation for calibration.

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

import json
import logging
from dataclasses import dataclass, fields, asdict
from typing import Final, Protocol, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numpy.typing import NDArray  # pylint: disable=unused-import


_log: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_NAMES: Final[list[str]] = [
    'h0', 'h1', 'T_amb', 'k_hp', 'k_fan',
    'mA', 'cp0', 'cp1',
    'q_exo', 'T_exo_onset', 'T_exo_width',
]
"""Ordered names of the 11 fittable parameters (excludes m_ref)."""

_EXP_CLIP: Final[float] = 500.0
"""Max absolute exponent passed to numpy.exp to avoid overflow."""


# ---------------------------------------------------------------------------
# Calibration data protocol
# ---------------------------------------------------------------------------

class CalibrationData(Protocol):
    """Structural type for roast calibration data.

    Any object exposing these attributes is accepted (duck typing).
    """
    time: NDArray
    bt: NDArray
    heater_pct: NDArray
    fan_pct: NDArray
    batch_mass_kg: float


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_sigmoid(x: NDArray | float) -> NDArray | float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x)).

    Clips the exponent to [-_EXP_CLIP, _EXP_CLIP] to prevent overflow.
    """
    x_clipped = np.clip(x, -_EXP_CLIP, _EXP_CLIP)
    return 1.0 / (1.0 + np.exp(-x_clipped))


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThermalModelParams:
    """Lumped-parameter thermal model parameters for the Kaleido M1 Lite.

    11 fittable parameters plus the reference batch mass.
    """

    # Heat-transfer coefficients
    h0: float = 0.5
    h1: float = 0.01

    # Environment temperature model
    T_amb: float = 25.0
    k_hp: float = 3.5
    k_fan: float = 0.5

    # Thermal mass
    mA: float = 0.01

    # Specific heat model
    cp0: float = 1.2
    cp1: float = 0.001

    # Exothermic reaction
    q_exo: float = 5.0
    T_exo_onset: float = 200.0
    T_exo_width: float = 5.0

    # Reference batch mass (kg) — not fitted
    m_ref: float = 0.1

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_vector(self) -> NDArray:
        """Pack the 11 fittable parameters into a flat array.

        The order matches :data:`PARAM_NAMES`.
        """
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: NDArray, m_ref: float = 0.1) -> ThermalModelParams:
        """Unpack a flat array of 11 fittable parameters.

        Args:
            vec: Array of length 11, ordered per :data:`PARAM_NAMES`.
            m_ref: Reference batch mass in kg (not fitted).
        """
        if len(vec) != len(PARAM_NAMES):
            msg = f'Expected {len(PARAM_NAMES)} parameters, got {len(vec)}'
            raise ValueError(msg)
        kwargs = {name: float(vec[i]) for i, name in enumerate(PARAM_NAMES)}
        kwargs['m_ref'] = m_ref
        return cls(**kwargs)

    def save(self, filepath: str) -> None:
        """Save parameters to a JSON file."""
        data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        _log.info('Saved thermal model params to %s', filepath)

    @classmethod
    def load(cls, filepath: str) -> ThermalModelParams:
        """Load parameters from a JSON file."""
        with open(filepath, encoding='utf-8') as fh:
            data = json.load(fh)
        valid_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_names}
        params = cls(**filtered)
        _log.info('Loaded thermal model params from %s', filepath)
        return params

    def scale_for_mass(self, new_mass_kg: float) -> ThermalModelParams:
        """Return a copy with mA scaled for a different batch mass.

        ``mA_new = mA * (new_mass_kg / m_ref)``
        """
        scale = new_mass_kg / self.m_ref
        data = asdict(self)
        data['mA'] = self.mA * scale
        data['m_ref'] = new_mass_kg
        return ThermalModelParams(**data)


# ---------------------------------------------------------------------------
# Thermal model
# ---------------------------------------------------------------------------

class KaleidoThermalModel:
    """Lumped-parameter thermal ODE model for the Kaleido M1 Lite.

    Energy balance::

        m * cp(T) * dT/dt = h_eff(fan%) * (T_env(hp%, fan%) - T_bean)
                            + Q_exo(T)

    where:
        h_eff   = h0 + h1 * fan_pct
        T_env   = T_amb + k_hp * hp_pct - k_fan * fan_pct
        cp(T)   = cp0 + cp1 * T
        Q_exo   = q_exo * sigmoid((T - T_exo_onset) / T_exo_width)
    """

    __slots__ = ('params',)

    def __init__(self, params: ThermalModelParams) -> None:
        self.params: ThermalModelParams = params

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def ode_rhs(
        self,
        t: float,
        T: float,
        hp_func: callable,
        fan_func: callable,
        mass_kg: float,
    ) -> float:
        """Evaluate dT/dt at a single point.

        Args:
            t: Current time (seconds).
            T: Current bean temperature (deg C).
            hp_func: Callable ``hp_func(t) -> heater_%``.
            fan_func: Callable ``fan_func(t) -> fan_%``.
            mass_kg: Batch mass in kg (used for mA scaling).

        Returns:
            Rate of temperature change dT/dt (deg C / s).
        """
        p = self.params

        hp_pct: float = float(hp_func(t))
        fan_pct: float = float(fan_func(t))

        # Effective heat transfer coefficient
        h_eff: float = p.h0 + p.h1 * fan_pct

        # Effective environment temperature
        T_env: float = p.T_amb + p.k_hp * hp_pct - p.k_fan * fan_pct

        # Temperature-dependent specific heat
        cp: float = p.cp0 + p.cp1 * T

        # Exothermic heat release (first-crack region)
        sig_arg: float = (T - p.T_exo_onset) / p.T_exo_width
        Q_exo: float = p.q_exo * float(_safe_sigmoid(sig_arg))

        # Scale thermal mass for actual batch size
        mA_eff: float = p.mA * (mass_kg / p.m_ref)

        # Energy balance: m*cp*dT/dt = h_eff*(T_env - T) + Q_exo
        dTdt: float = (h_eff * (T_env - T) + Q_exo) / (mA_eff * cp)
        return dTdt

    # ------------------------------------------------------------------
    # Forward simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        time: NDArray,
        hp_schedule: NDArray,
        fan_schedule: NDArray,
        T0: float,
        mass_kg: float | None = None,
    ) -> NDArray:
        """Run a forward simulation of the thermal model.

        Args:
            time: 1-D array of time points (seconds).
            hp_schedule: Heater-% values at each time point.
            fan_schedule: Fan-% values at each time point.
            T0: Initial bean temperature (deg C).
            mass_kg: Batch mass in kg; defaults to ``params.m_ref``.

        Returns:
            1-D array of predicted bean temperatures at the given time points.
        """
        if mass_kg is None:
            mass_kg = self.params.m_ref

        time = np.asarray(time, dtype=np.float64)
        hp_schedule = np.asarray(hp_schedule, dtype=np.float64)
        fan_schedule = np.asarray(fan_schedule, dtype=np.float64)

        # Build interpolating control-input functions
        def hp_func(t: float) -> float:
            return np.interp(t, time, hp_schedule)

        def fan_func(t: float) -> float:
            return np.interp(t, time, fan_schedule)

        t_span = (float(time[0]), float(time[-1]))

        _log.debug(
            'Simulating thermal model: t=[%.1f, %.1f]s, T0=%.1f C, mass=%.4f kg',
            t_span[0], t_span[1], T0, mass_kg,
        )

        sol = solve_ivp(
            fun=lambda t, y: [self.ode_rhs(t, y[0], hp_func, fan_func, mass_kg)],
            t_span=t_span,
            y0=[T0],
            method='RK45',
            t_eval=time,
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            _log.warning('ODE integration failed: %s', sol.message)

        return sol.y[0]

    # ------------------------------------------------------------------
    # Residuals for calibration
    # ------------------------------------------------------------------

    def residuals(self, calib_data: CalibrationData) -> NDArray:
        """Compute (predicted - measured) bean temperature residuals.

        Args:
            calib_data: Object exposing ``time``, ``bt``, ``heater_pct``,
                ``fan_pct``, and ``batch_mass_kg`` attributes.

        Returns:
            1-D array of residuals (predicted BT - measured BT).
        """
        predicted = self.simulate(
            time=calib_data.time,
            hp_schedule=calib_data.heater_pct,
            fan_schedule=calib_data.fan_pct,
            T0=float(calib_data.bt[0]),
            mass_kg=calib_data.batch_mass_kg,
        )
        return predicted - np.asarray(calib_data.bt, dtype=np.float64)
