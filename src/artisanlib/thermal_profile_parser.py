#
# ABOUT
# Parses Artisan .alog profile files and extracts synchronized timeseries
# data for thermal model fitting.  Standalone — no Artisan GUI dependency.

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

import ast
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt


_log: Final[logging.Logger] = logging.getLogger(__name__)

# Kaleido extra-device IDs
_KALEIDO_HP_FAN_DEVICE: Final[int] = 141   # extratemp1 = HP%, extratemp2 = Fan%
_KALEIDO_DRUM_DEVICE: Final[int] = 140     # extratemp1 = Drum%
_KALEIDO_SV_DEVICE: Final[int] = 139       # extratemp1 = SV (target temp)

# Weight-unit conversion factors → kilograms
_WEIGHT_TO_KG: Final[dict[str, float]] = {
    'Kg': 1.0,
    'kg': 1.0,
    'g':  0.001,
    'lb': 0.45359237,
    'oz': 0.028349523125,
}


@dataclass(slots=True)
class CalibrationData:
    """Synchronised timeseries extracted from a single roast profile."""

    time:       npt.NDArray[np.float64]  # seconds, rebased to 0 at CHARGE
    bt:         npt.NDArray[np.float64]  # bean temperature (°C or °F)
    et:         npt.NDArray[np.float64]  # environment temperature
    heater_pct: npt.NDArray[np.float64]  # heater power 0–100
    fan_pct:    npt.NDArray[np.float64]  # fan speed 0–100
    drum_pct:   npt.NDArray[np.float64]  # drum speed 0–100
    batch_mass_kg: float                 # green-bean charge weight in kg
    source_file:   str = field(default='')


@dataclass(slots=True)
class TargetCurveData:
    """Target BT curve extracted from a roast profile."""

    time: npt.NDArray[np.float64]       # seconds, rebased to 0 at CHARGE/start
    bt: npt.NDArray[np.float64]         # bean temperature (C/F as stored)
    batch_mass_kg: float                # green-bean charge weight in kg
    source_file: str = field(default='')


# ── helpers ──────────────────────────────────────────────────────────


def _load_profile(filepath: str) -> dict:
    """Read an .alog file and return the raw profile dict.

    Supports both legacy Python-literal and JSON profile formats.
    """
    fp = str(filepath)
    if not os.path.exists(fp):
        raise FileNotFoundError(f'Profile not found: {fp}')
    with open(fp, encoding='utf-8') as fh:
        data = fh.read()
    try:
        return ast.literal_eval(data)  # type: ignore[return-value]
    except (ValueError, SyntaxError):
        return json.loads(data)


def _find_device_index(extradevices: list[int], device_id: int) -> int | None:
    """Return the index of *device_id* inside *extradevices*, or ``None``."""
    try:
        return extradevices.index(device_id)
    except ValueError:
        return None


def _extract_channel(
    profile: dict,
    channel_key: str,
    device_idx: int,
    main_timex: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Extract a single extra-device channel and resample onto *main_timex*.

    Parameters
    ----------
    profile : dict
        The raw profile dictionary.
    channel_key : str
        ``'extratemp1'`` or ``'extratemp2'``.
    device_idx : int
        Index into the extra-device arrays.
    main_timex : ndarray
        The main time grid to interpolate onto.
    """
    raw_values = profile[channel_key][device_idx]
    raw_times = profile['extratimex'][device_idx]

    src_t = np.asarray(raw_times, dtype=np.float64)
    src_v = np.asarray(raw_values, dtype=np.float64)

    # Guard against length mismatches (corrupt files)
    n = min(len(src_t), len(src_v))
    src_t = src_t[:n]
    src_v = src_v[:n]

    if n == 0:
        return np.zeros_like(main_timex)

    return np.interp(main_timex, src_t, src_v)


def _resolve_charge_drop(timeindex: list[int], timex_len: int) -> tuple[int, int]:
    """Return validated (charge_idx, drop_idx) from a profile *timeindex*.

    Raises ``ValueError`` when CHARGE or DROP cannot be determined.
    """
    if len(timeindex) < 7:
        raise ValueError(
            f'timeindex has only {len(timeindex)} elements (need >= 7)')

    charge_idx: int = timeindex[0]
    drop_idx: int   = timeindex[6]

    # CHARGE: -1 or missing → treat index 0 as the start
    if charge_idx < 0:
        _log.info('CHARGE index is -1; defaulting to 0')
        charge_idx = 0

    if drop_idx <= 0:
        raise ValueError('DROP event not set in profile (timeindex[6] <= 0)')

    if charge_idx >= timex_len or drop_idx >= timex_len:
        raise ValueError(
            f'timeindex out of range: CHARGE={charge_idx}, '
            f'DROP={drop_idx}, timex length={timex_len}')

    if charge_idx >= drop_idx:
        raise ValueError(
            f'CHARGE ({charge_idx}) must precede DROP ({drop_idx})')

    return charge_idx, drop_idx


def _resolve_slice(
    profile: dict,
    timex_len: int,
    *,
    strict_timeindex: bool,
) -> tuple[slice, int]:
    """Resolve the CHARGE->DROP slice and CHARGE index.

    If *strict_timeindex* is ``False`` and the profile does not contain a
    valid CHARGE/DROP pair, the full timeline is used.
    """
    timeindex = profile.get('timeindex')
    if isinstance(timeindex, list):
        try:
            charge_idx, drop_idx = _resolve_charge_drop(timeindex, timex_len)
            return slice(charge_idx, drop_idx + 1), charge_idx
        except ValueError:
            if strict_timeindex:
                raise
            _log.info('Using full timeline: CHARGE/DROP window unavailable')

    if strict_timeindex:
        raise ValueError('Profile is missing valid timeindex data')

    return slice(0, timex_len), 0


def _batch_mass_kg(profile: dict) -> float:
    """Extract the green-bean charge weight in kilograms.

    Returns 0.0 when the weight field is absent or empty.
    """
    weight = profile.get('weight')
    if weight is None or len(weight) < 3:
        _log.warning('No weight data in profile')
        return 0.0

    try:
        value = float(weight[0])
    except (TypeError, ValueError):
        _log.warning('Unparsable weight value: %s', weight[0])
        return 0.0

    if value <= 0.0:
        return 0.0

    unit = str(weight[2]).strip()
    factor = _WEIGHT_TO_KG.get(unit)
    if factor is None:
        _log.warning('Unknown weight unit %r; assuming kg', unit)
        factor = 1.0

    return value * factor


# ── public API ───────────────────────────────────────────────────────


def parse_alog_profile(filepath: str) -> CalibrationData:
    """Parse a single ``.alog`` profile and return a :class:`CalibrationData`.

    Parameters
    ----------
    filepath : str
        Path to the ``.alog`` file.

    Returns
    -------
    CalibrationData
        Timeseries trimmed to the CHARGE → DROP window with time rebased to
        zero at CHARGE.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required data (extra devices, CHARGE/DROP events) is missing.
    """
    _log.info('Parsing profile: %s', filepath)
    profile = _load_profile(filepath)

    # ── main channels ────────────────────────────────────────────────
    timex_raw = profile.get('timex')
    temp1_raw = profile.get('temp1')  # ET
    temp2_raw = profile.get('temp2')  # BT

    if timex_raw is None or temp1_raw is None or temp2_raw is None:
        raise ValueError('Profile is missing timex, temp1, or temp2')

    main_timex = np.asarray(timex_raw, dtype=np.float64)

    # ── extra devices (Kaleido) ──────────────────────────────────────
    extradevices = profile.get('extradevices')
    if not extradevices:
        raise ValueError(
            'Profile contains no extra-device data (extradevices is empty)')

    idx_hp_fan = _find_device_index(extradevices, _KALEIDO_HP_FAN_DEVICE)
    idx_drum   = _find_device_index(extradevices, _KALEIDO_DRUM_DEVICE)

    if idx_hp_fan is None:
        raise ValueError(
            f'Kaleido HP/Fan device ({_KALEIDO_HP_FAN_DEVICE}) not found '
            f'in extradevices {extradevices}')

    heater = _extract_channel(profile, 'extratemp1', idx_hp_fan, main_timex)
    fan    = _extract_channel(profile, 'extratemp2', idx_hp_fan, main_timex)

    if idx_drum is not None:
        drum = _extract_channel(profile, 'extratemp1', idx_drum, main_timex)
    else:
        _log.warning(
            'Kaleido Drum device (%d) not found; drum_pct will be zeros',
            _KALEIDO_DRUM_DEVICE)
        drum = np.zeros_like(main_timex)

    et = np.asarray(temp1_raw, dtype=np.float64)
    bt = np.asarray(temp2_raw, dtype=np.float64)

    # Ensure lengths match the main time grid
    n = len(main_timex)
    et = et[:n]
    bt = bt[:n]

    # ── CHARGE / DROP window ─────────────────────────────────────────
    sl, charge_idx = _resolve_slice(profile, n, strict_timeindex=True)
    t  = main_timex[sl] - main_timex[charge_idx]  # rebase to 0

    return CalibrationData(
        time=t,
        bt=bt[sl],
        et=et[sl],
        heater_pct=heater[sl],
        fan_pct=fan[sl],
        drum_pct=drum[sl],
        batch_mass_kg=_batch_mass_kg(profile),
        source_file=os.path.abspath(filepath),
    )


def parse_target_profile(filepath: str) -> TargetCurveData:
    """Parse a profile into a target BT curve for schedule generation.

    Unlike :func:`parse_alog_profile`, this parser does not require Kaleido
    extra-device channels and accepts any profile containing `timex` + `temp2`.
    """
    _log.info('Parsing target curve profile: %s', filepath)
    profile = _load_profile(filepath)

    timex_raw = profile.get('timex')
    temp2_raw = profile.get('temp2')
    if timex_raw is None or temp2_raw is None:
        raise ValueError('Profile is missing timex or temp2')

    timex = np.asarray(timex_raw, dtype=np.float64)
    bt = np.asarray(temp2_raw, dtype=np.float64)
    n = min(len(timex), len(bt))
    if n < 2:
        raise ValueError('Profile must contain at least two BT samples')

    timex = timex[:n]
    bt = bt[:n]

    sl, start_idx = _resolve_slice(profile, n, strict_timeindex=False)
    t = timex[sl] - timex[start_idx]

    return TargetCurveData(
        time=t,
        bt=bt[sl],
        batch_mass_kg=_batch_mass_kg(profile),
        source_file=os.path.abspath(filepath),
    )


def parse_multiple_profiles(filepaths: list[str]) -> list[CalibrationData]:
    """Parse several ``.alog`` profiles, skipping any that fail.

    Parameters
    ----------
    filepaths : list[str]
        Paths to ``.alog`` files.

    Returns
    -------
    list[CalibrationData]
        Successfully parsed profiles.  Profiles that fail to parse are
        logged and omitted.
    """
    results: list[CalibrationData] = []
    for fp in filepaths:
        try:
            results.append(parse_alog_profile(fp))
        except Exception:  # pylint: disable=broad-except
            _log.exception('Failed to parse profile: %s', fp)
    return results
