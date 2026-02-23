#
# ABOUT
# Safety validation for generated thermal schedules.

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

    from artisanlib.thermal_model import KaleidoThermalModel
    from artisanlib.thermal_model_inversion import InversionResult


@dataclass
class SafetyValidationResult:
    """Result of dry-run schedule safety validation."""

    is_safe: bool
    bt_peak_c: float
    et_peak_c: float
    ror_peak_c_per_min: float
    bt_limit_c: float | None = None
    et_limit_c: float | None = None
    max_ror_limit_c_per_min: float | None = None
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(
            f'Safety check: {"PASS" if self.is_safe else "FAIL"} '
            f'(BTmax={self.bt_peak_c:.1f}C, ETmax={self.et_peak_c:.1f}C, RoRmax={self.ror_peak_c_per_min:.1f}C/min)'
        )
        lines.extend(self.failures)
        lines.extend(self.warnings)
        return lines


def _estimate_environment_temperature(
    model: KaleidoThermalModel,
    heater_pct: NDArray[np.float64],
    fan_pct: NDArray[np.float64],
) -> NDArray[np.float64]:
    return model.environment_temperature(heater_pct, fan_pct)


def validate_schedule(
    model: KaleidoThermalModel,
    inversion: InversionResult,
    *,
    bt_limit_c: float | None = None,
    et_limit_c: float | None = None,
    max_ror_limit_c_per_min: float | None = None,
) -> SafetyValidationResult:
    """Validate schedule against BT/ET/RoR limits via dry-run outputs."""
    bt = np.asarray(inversion.predicted_bt, dtype=np.float64)
    et = _estimate_environment_temperature(
        model,
        np.asarray(inversion.heater_pct, dtype=np.float64),
        np.asarray(inversion.fan_pct, dtype=np.float64),
    )
    time = np.asarray(inversion.time, dtype=np.float64)
    if len(bt) != len(time):
        raise ValueError('predicted BT and time arrays must have same length')

    if len(bt) >= 2:
        ror = np.gradient(bt, time) * 60.0
        ror_peak = float(np.max(ror))
    else:
        ror_peak = 0.0

    bt_peak = float(np.max(bt)) if len(bt) else 0.0
    et_peak = float(np.max(et)) if len(et) else 0.0

    failures: list[str] = []
    warnings: list[str] = []

    if bt_limit_c is not None and bt_peak > bt_limit_c:
        failures.append(f'BT exceeds limit by {bt_peak - bt_limit_c:.1f}C')
    elif bt_limit_c is not None:
        warnings.append(f'BT margin: {bt_limit_c - bt_peak:.1f}C')

    if et_limit_c is not None and et_peak > et_limit_c:
        failures.append(f'ET exceeds limit by {et_peak - et_limit_c:.1f}C')
    elif et_limit_c is not None:
        warnings.append(f'ET margin: {et_limit_c - et_peak:.1f}C')

    if max_ror_limit_c_per_min is not None and ror_peak > max_ror_limit_c_per_min:
        failures.append(f'RoR exceeds limit by {ror_peak - max_ror_limit_c_per_min:.1f}C/min')
    elif max_ror_limit_c_per_min is not None:
        warnings.append(f'RoR margin: {max_ror_limit_c_per_min - ror_peak:.1f}C/min')

    return SafetyValidationResult(
        is_safe=len(failures) == 0,
        bt_peak_c=bt_peak,
        et_peak_c=et_peak,
        ror_peak_c_per_min=ror_peak,
        bt_limit_c=bt_limit_c,
        et_limit_c=et_limit_c,
        max_ror_limit_c_per_min=max_ror_limit_c_per_min,
        failures=failures,
        warnings=warnings,
    )
