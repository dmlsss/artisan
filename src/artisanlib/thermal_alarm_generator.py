#
# ABOUT
# Converts control schedules from thermal model inversion into
# Artisan-compatible alarm table files (.alrm JSON format).

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
from dataclasses import dataclass, field
from typing import Final, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray  # pylint: disable=unused-import


_log: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alarm action constants (Artisan slider mapping for Kaleido)
# ---------------------------------------------------------------------------

ACTION_POPUP: Final[int] = 0       # Pop-up message
ACTION_HEATER: Final[int] = 3      # Slider 1 → kaleido(HP,{})
ACTION_FAN: Final[int] = 4         # Slider 2 → kaleido(FC,{})
ACTION_DRUM: Final[int] = 5        # Slider 3 → kaleido(RC,{})

# Default alarm field values for time-only triggering
_FLAG_ENABLED: Final[int] = 1
_GUARD_NONE: Final[int] = -1
_TIME_ON_CHARGE: Final[int] = 0
_SOURCE_BT: Final[int] = 1
_COND_ABOVE: Final[int] = 1
_TEMP_ALWAYS: Final[float] = 500.0
_BEEP_OFF: Final[int] = 0

# Warning inserted before exothermic onset
_EXO_WARNING_LEAD_S: Final[int] = 30
_EXO_WARNING_MSG: Final[str] = 'Approaching First Crack - monitor closely'


# ---------------------------------------------------------------------------
# AlarmTableData
# ---------------------------------------------------------------------------

@dataclass
class AlarmTableData:
    """Parallel lists representing an Artisan alarm table.

    Each index *i* across all lists describes a single alarm row.
    """

    label: str = ''
    alarmflag: list[int] = field(default_factory=list)
    alarmguard: list[int] = field(default_factory=list)
    alarmnegguard: list[int] = field(default_factory=list)
    alarmtime: list[int] = field(default_factory=list)
    alarmoffset: list[int] = field(default_factory=list)
    alarmsource: list[int] = field(default_factory=list)
    alarmcond: list[int] = field(default_factory=list)
    alarmtemperature: list[float] = field(default_factory=list)
    alarmaction: list[int] = field(default_factory=list)
    alarmbeep: list[int] = field(default_factory=list)
    alarmstrings: list[str] = field(default_factory=list)

    # -- helpers ----------------------------------------------------------

    def alarm_count(self) -> int:
        """Return the total number of alarms."""
        return len(self.alarmflag)

    def to_alrm_dict(self) -> dict[str, list[int] | list[float] | list[str]]:
        """Convert to the dict format used by .alrm JSON files.

        Key names match what ``alarms.py`` ``importalarmsJSON()`` expects.
        """
        return {
            'alarmflags': list(self.alarmflag),
            'alarmguards': list(self.alarmguard),
            'alarmnegguards': list(self.alarmnegguard),
            'alarmtimes': list(self.alarmtime),
            'alarmoffsets': list(self.alarmoffset),
            'alarmconds': list(self.alarmcond),
            'alarmsources': list(self.alarmsource),
            'alarmtemperatures': list(self.alarmtemperature),
            'alarmactions': list(self.alarmaction),
            'alarmbeep': list(self.alarmbeep),
            'alarmstrings': list(self.alarmstrings),
        }

    def save_alrm(self, filepath: str) -> None:
        """Save alarm table as an Artisan .alrm JSON file.

        Format matches ``alarms.py`` ``exportalarmsJSON()``.
        """
        alrm = self.to_alrm_dict()
        with open(filepath, 'w', encoding='utf-8') as fh:
            json.dump(alrm, fh, indent=None, separators=(',', ':'),
                      ensure_ascii=False)
            fh.write('\n')
        _log.info('Saved %d alarms to %s', self.alarm_count(), filepath)


# ---------------------------------------------------------------------------
# Helper — append a single alarm row
# ---------------------------------------------------------------------------

def _append_alarm(
    data: AlarmTableData,
    offset: int,
    action: int,
    value: str,
) -> None:
    """Append one alarm entry with standard time-only defaults."""
    # Timed alarms with offset 0 can be skipped because they still pass
    # through the temperature condition branch in the alarm engine.
    safe_offset = max(1, int(offset))
    data.alarmflag.append(_FLAG_ENABLED)
    data.alarmguard.append(_GUARD_NONE)
    data.alarmnegguard.append(_GUARD_NONE)
    data.alarmtime.append(_TIME_ON_CHARGE)
    data.alarmoffset.append(safe_offset)
    data.alarmsource.append(_SOURCE_BT)
    data.alarmcond.append(_COND_ABOVE)
    data.alarmtemperature.append(_TEMP_ALWAYS)
    data.alarmaction.append(action)
    data.alarmbeep.append(_BEEP_OFF)
    data.alarmstrings.append(value)


# ---------------------------------------------------------------------------
# generate_alarm_table
# ---------------------------------------------------------------------------

def generate_alarm_table(
    time: NDArray[np.float64],
    heater_pct: NDArray[np.float64],
    fan_pct: NDArray[np.float64],
    exo_warning_time: float | None = None,
    label: str = 'Thermal Model Control',
) -> AlarmTableData:
    """Convert control schedules into an :class:`AlarmTableData`.

    Parameters
    ----------
    time:
        Seconds from CHARGE for each schedule sample.
    heater_pct:
        Heater power schedule (0-100 %).
    fan_pct:
        Fan speed schedule (0-100 %).
    exo_warning_time:
        Optional time in seconds (from CHARGE) at which a popup warning
        should appear.  The alarm is placed ``_EXO_WARNING_LEAD_S`` seconds
        *before* this time.
    label:
        Descriptive label stored in the returned data.

    Returns
    -------
    AlarmTableData
        Ready to be saved via :meth:`AlarmTableData.save_alrm`.
    """
    if len(time) != len(heater_pct) or len(time) != len(fan_pct):
        raise ValueError(
            f'Array lengths must match: time={len(time)}, '
            f'heater_pct={len(heater_pct)}, fan_pct={len(fan_pct)}'
        )

    data = AlarmTableData(label=label)

    # Round to integer percentages
    hp_int = np.rint(heater_pct).astype(int)
    fan_int = np.rint(fan_pct).astype(int)

    prev_hp: int | None = None
    prev_fan: int | None = None

    for i in range(len(time)):
        offset = int(time[i])
        hp = int(hp_int[i])
        fan = int(fan_int[i])

        # Heater alarm — deduplicate consecutive identical values
        if hp != prev_hp:
            _append_alarm(data, offset, ACTION_HEATER, str(hp))
            prev_hp = hp

        # Fan alarm — deduplicate consecutive identical values
        if fan != prev_fan:
            _append_alarm(data, offset, ACTION_FAN, str(fan))
            prev_fan = fan

    # Optional exothermic warning popup
    if exo_warning_time is not None:
        warning_offset = max(0, int(exo_warning_time) - _EXO_WARNING_LEAD_S)
        _append_alarm(data, warning_offset, ACTION_POPUP, _EXO_WARNING_MSG)

    # Sort all parallel lists by alarmoffset (stable — preserves insertion
    # order for identical offsets)
    if data.alarm_count() > 0:
        indices = sorted(range(data.alarm_count()),
                         key=lambda k: data.alarmoffset[k])
        data.alarmflag = [data.alarmflag[k] for k in indices]
        data.alarmguard = [data.alarmguard[k] for k in indices]
        data.alarmnegguard = [data.alarmnegguard[k] for k in indices]
        data.alarmtime = [data.alarmtime[k] for k in indices]
        data.alarmoffset = [data.alarmoffset[k] for k in indices]
        data.alarmsource = [data.alarmsource[k] for k in indices]
        data.alarmcond = [data.alarmcond[k] for k in indices]
        data.alarmtemperature = [data.alarmtemperature[k] for k in indices]
        data.alarmaction = [data.alarmaction[k] for k in indices]
        data.alarmbeep = [data.alarmbeep[k] for k in indices]
        data.alarmstrings = [data.alarmstrings[k] for k in indices]

    _log.info('Generated %d alarms (%s)', data.alarm_count(), label)
    return data


# ---------------------------------------------------------------------------
# generate_schedule_description
# ---------------------------------------------------------------------------

def generate_schedule_description(alarm_data: AlarmTableData) -> str:
    """Return a human-readable one-line summary of an alarm schedule.

    Example output::

        47 alarms over 12:30 (CHARGE+0s to CHARGE+750s), HP: 90->45%, Fan: 25->55%
    """
    n = alarm_data.alarm_count()
    if n == 0:
        return '0 alarms (empty schedule)'

    offsets = alarm_data.alarmoffset
    min_off = min(offsets)
    max_off = max(offsets)
    duration_m = max_off // 60
    duration_s = max_off % 60

    # Extract first / last heater and fan values
    hp_values = [
        alarm_data.alarmstrings[i]
        for i in range(n)
        if alarm_data.alarmaction[i] == ACTION_HEATER
    ]
    fan_values = [
        alarm_data.alarmstrings[i]
        for i in range(n)
        if alarm_data.alarmaction[i] == ACTION_FAN
    ]

    parts: list[str] = []
    parts.append(f'{n} alarms over {duration_m}:{duration_s:02d}')
    parts.append(f'(CHARGE+{min_off}s to CHARGE+{max_off}s)')

    if hp_values:
        parts.append(f'HP: {hp_values[0]}->{hp_values[-1]}%')
    if fan_values:
        parts.append(f'Fan: {fan_values[0]}->{fan_values[-1]}%')

    # Join first two with space, then comma-separate the rest
    desc = f'{parts[0]} {parts[1]}'
    if len(parts) > 2:
        desc += ', ' + ', '.join(parts[2:])
    return desc
