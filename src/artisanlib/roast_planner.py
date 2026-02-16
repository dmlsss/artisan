#
# ABOUT
# Feedforward roast planner utilities
#

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

import ast
import json
import logging
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from artisanlib.atypes import ProfileData
from artisanlib.util import events_external_to_internal_value, events_internal_to_external_value

_log: Final[logging.Logger] = logging.getLogger(__name__)

Channel = Literal['heat', 'fan', 'drum']
AlarmSetData = dict[str, list[int] | list[float] | list[str]]

_CHANNEL_ORDER: Final[dict[Channel, int]] = {'heat': 0, 'fan': 1, 'drum': 2}
_CHANNEL_CODES: Final[dict[Channel, str]] = {'heat': 'HP', 'fan': 'FC', 'drum': 'RC'}
_CHANNEL_EVENT_TYPE: Final[dict[Channel, int]] = {'heat': 0, 'fan': 1, 'drum': 2}


@dataclass(frozen=True)
class ControlPoint:
    time_s: float
    channel: Channel
    value_pct: int


@dataclass(frozen=True)
class PlanSummary:
    source_event_count: int
    output_event_count: int
    batch_scale: float
    time_scale: float
    heat_bias: int
    fan_bias: int
    drum_bias: int


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(float, value))
    except Exception:  # pylint: disable=broad-except
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(int, value))
    except Exception:  # pylint: disable=broad-except
        return default


def _clamp_int(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def _nearest_idx(times: list[float], target: float) -> int:
    if len(times) == 0:
        return 0
    pos = bisect_left(times, target)
    if pos <= 0:
        return 0
    if pos >= len(times):
        return len(times) - 1
    before = times[pos - 1]
    after = times[pos]
    if abs(target - before) <= abs(after - target):
        return pos - 1
    return pos


def load_profile_data(filename: str) -> ProfileData:
    text = Path(filename).read_text(encoding='utf-8').strip()
    if len(text) == 0:
        return cast(ProfileData, {})
    try:
        profile = ast.literal_eval(text)  # pylint: disable=eval-used
    except Exception:  # pylint: disable=broad-except
        profile = json.loads(text)
    if not isinstance(profile, dict):
        raise ValueError('Profile file does not contain an object')
    return cast(ProfileData, profile)


def save_profile_data(filename: str, profile: ProfileData) -> None:
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(profile, outfile, indent=2, ensure_ascii=False)
        outfile.write('\n')


def save_alarm_set(filename: str, alarms: AlarmSetData) -> None:
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(alarms, outfile, indent=None, separators=(',', ':'), ensure_ascii=False)
        outfile.write('\n')


def get_profile_batch_kg(profile: ProfileData) -> float | None:
    if 'weight' not in profile:
        return None
    weight = profile['weight']
    if len(weight) < 3:
        return None
    amount = _to_float(weight[0], -1.0)
    unit = str(weight[2]).strip().lower()
    if amount <= 0:
        return None
    if unit == 'kg':
        return amount
    if unit == 'g':
        return amount / 1000.0
    if unit in {'lb', 'lbs'}:
        return amount * 0.45359237
    return None


def _detect_channel(event_type: int, event_label: str) -> Channel | None:
    label = event_label.upper()

    if 'HP=' in label or 'HEAT' in label or 'BURNER' in label:
        return 'heat'
    if 'FC=' in label or 'SM=' in label or 'FAN' in label:
        return 'fan'
    if 'RC=' in label or 'RL=' in label or 'DRUM' in label:
        return 'drum'

    # legacy Kaleido imports map Fan->0, Drum->1, Burner->3
    if event_type == 3:
        return 'heat'
    if event_type == 0:
        return 'fan'
    if event_type in {1, 2}:
        return 'drum'

    return None


def extract_control_points(profile: ProfileData) -> list[ControlPoint]:
    timex = list(map(_to_float, profile.get('timex', [])))
    event_idx = profile.get('specialevents', [])
    event_types = profile.get('specialeventstype', [])
    event_values = profile.get('specialeventsvalue', [])
    event_strings = profile.get('specialeventsStrings', [])

    n_events = min(len(event_idx), len(event_types), len(event_values))
    if n_events == 0 or len(timex) == 0:
        return []

    points: list[ControlPoint] = []
    for i in range(n_events):
        idx = _to_int(event_idx[i], -1)
        if idx < 0 or idx >= len(timex):
            continue
        etype = _to_int(event_types[i], -1)
        label = str(event_strings[i]) if i < len(event_strings) else ''
        channel = _detect_channel(etype, label)
        if channel is None:
            continue

        internal_value = _to_float(event_values[i], 0.0)
        external_value = events_internal_to_external_value(internal_value)
        value_pct = _clamp_int(int(round(external_value)), 0, 100)
        points.append(ControlPoint(time_s=timex[idx], channel=channel, value_pct=value_pct))

    points.sort(key=lambda p: (p.time_s, _CHANNEL_ORDER[p.channel]))
    return points


def _compress_control_points(points: list[ControlPoint], min_event_gap_s: float) -> list[ControlPoint]:
    grouped: dict[Channel, list[ControlPoint]] = {'heat': [], 'fan': [], 'drum': []}

    for point in points:
        channel_points = grouped[point.channel]
        if len(channel_points) == 0:
            channel_points.append(point)
            continue

        last = channel_points[-1]
        if abs(point.time_s - last.time_s) < min_event_gap_s:
            channel_points[-1] = point
        elif point.value_pct != last.value_pct:
            channel_points.append(point)

    merged = grouped['heat'] + grouped['fan'] + grouped['drum']
    merged.sort(key=lambda p: (p.time_s, _CHANNEL_ORDER[p.channel]))
    return merged


def _scaled_timex(profile: ProfileData, time_scale: float, points: list[ControlPoint]) -> list[float]:
    source_timex = list(map(_to_float, profile.get('timex', [])))
    if len(source_timex) > 0:
        scaled = [t * time_scale for t in source_timex]
        for i in range(1, len(scaled)):
            if scaled[i] <= scaled[i - 1]:
                scaled[i] = scaled[i - 1] + 0.001
        return scaled

    max_time = 0.0
    for point in points:
        max_time = max(max_time, point.time_s)
    end_s = max(1, int(round(max_time)))
    return [float(t) for t in range(end_s + 1)]


def _scaled_control_points(
    points: list[ControlPoint],
    time_scale: float,
    heat_bias: int,
    fan_bias: int,
    drum_bias: int,
) -> list[ControlPoint]:
    scaled: list[ControlPoint] = []
    for point in points:
        bias = 0
        if point.channel == 'heat':
            bias = heat_bias
        elif point.channel == 'fan':
            bias = fan_bias
        elif point.channel == 'drum':
            bias = drum_bias
        scaled.append(
            ControlPoint(
                time_s=point.time_s * time_scale,
                channel=point.channel,
                value_pct=_clamp_int(point.value_pct + bias, 0, 100),
            )
        )
    scaled.sort(key=lambda p: (p.time_s, _CHANNEL_ORDER[p.channel]))
    return scaled


def build_planned_profile(
    source_profile: ProfileData,
    *,
    target_batch_kg: float | None = None,
    time_scale: float | None = None,
    min_event_gap_s: float = 2.0,
) -> tuple[ProfileData, PlanSummary]:
    source_points = extract_control_points(source_profile)
    if len(source_points) == 0:
        raise ValueError('No usable control events found in profile')

    source_batch_kg = get_profile_batch_kg(source_profile)
    if source_batch_kg is None or target_batch_kg is None or target_batch_kg <= 0:
        batch_scale = 1.0
    else:
        batch_scale = max(0.5, min(1.8, target_batch_kg / source_batch_kg))

    if time_scale is None or time_scale <= 0:
        computed_time_scale = max(0.7, min(1.4, batch_scale**0.35))
    else:
        computed_time_scale = max(0.5, min(2.0, time_scale))

    heat_bias = _clamp_int(int(round((batch_scale - 1.0) * 14.0)), -20, 20)
    fan_bias = _clamp_int(int(round((batch_scale - 1.0) * 8.0)), -15, 15)
    drum_bias = _clamp_int(int(round((batch_scale - 1.0) * 4.0)), -10, 10)

    scaled_points = _scaled_control_points(source_points, computed_time_scale, heat_bias, fan_bias, drum_bias)
    compressed_points = _compress_control_points(scaled_points, max(0.1, min_event_gap_s))

    planned_profile = cast(ProfileData, deepcopy(source_profile))
    timex = _scaled_timex(source_profile, computed_time_scale, compressed_points)

    event_map: dict[tuple[int, int], tuple[float, str]] = {}
    for point in compressed_points:
        idx = _nearest_idx(timex, point.time_s)
        etype = _CHANNEL_EVENT_TYPE[point.channel]
        event_map[(idx, etype)] = (
            events_external_to_internal_value(point.value_pct),
            f'{_CHANNEL_CODES[point.channel]}={point.value_pct}%',
        )

    specialevents: list[int] = []
    specialeventstype: list[int] = []
    specialeventsvalue: list[float] = []
    specialeventsStrings: list[str] = []
    for idx, etype in sorted(event_map.keys(), key=lambda p: (p[0], p[1])):
        value, description = event_map[(idx, etype)]
        specialevents.append(idx)
        specialeventstype.append(etype)
        specialeventsvalue.append(value)
        specialeventsStrings.append(description)

    planned_profile['timex'] = timex
    planned_profile['specialevents'] = specialevents
    planned_profile['specialeventstype'] = specialeventstype
    planned_profile['specialeventsvalue'] = specialeventsvalue
    planned_profile['specialeventsStrings'] = specialeventsStrings

    etypes = list(planned_profile.get('etypes', []))
    while len(etypes) < 5:
        etypes.append('--')
    etypes[0] = 'Heat'
    etypes[1] = 'Fan'
    etypes[2] = 'Drum'
    planned_profile['etypes'] = etypes

    title = str(planned_profile.get('title', '')).strip()
    if len(title) > 0:
        planned_profile['title'] = f'{title} [Planned]'
    else:
        planned_profile['title'] = 'Planned Roast'

    beans = str(planned_profile.get('beans', '')).strip()
    if len(beans) > 0:
        planned_profile['beans'] = f'{beans} [Feedforward]'

    return planned_profile, PlanSummary(
        source_event_count=len(source_points),
        output_event_count=len(specialevents),
        batch_scale=batch_scale,
        time_scale=computed_time_scale,
        heat_bias=heat_bias,
        fan_bias=fan_bias,
        drum_bias=drum_bias,
    )


def build_safety_alarm_set(
    profile: ProfileData,
    *,
    bt_ceiling: float | None = None,
    et_ceiling: float | None = None,
) -> AlarmSetData:
    mode = str(profile.get('mode', 'C')).upper()
    temp1 = list(map(_to_float, profile.get('temp1', [])))
    temp2 = list(map(_to_float, profile.get('temp2', [])))

    et_default = 260.0 if mode == 'C' else 500.0
    bt_default = 230.0 if mode == 'C' else 446.0
    margin = 6.0 if mode == 'C' else 10.0

    if len(temp1) > 0:
        et_default = max(temp1) + margin
    if len(temp2) > 0:
        bt_default = max(temp2) + margin

    et_limit = round(et_ceiling if et_ceiling is not None else et_default, 1)
    bt_limit = round(bt_ceiling if bt_ceiling is not None else bt_default, 1)

    alarms: AlarmSetData = {
        'alarmflags': [1, 1],
        'alarmguards': [-1, -1],
        'alarmnegguards': [-1, -1],
        'alarmtimes': [-1, -1],
        'alarmoffsets': [0, 0],
        'alarmconds': [1, 1],  # '>'
        'alarmsources': [0, 1],  # ET, BT
        'alarmtemperatures': [et_limit, bt_limit],
        'alarmactions': [0, 0],  # popup
        'alarmbeep': [1, 1],
        'alarmstrings': ['ET safety ceiling', 'BT safety ceiling'],
    }
    return alarms


def profile_path_stem(profile_path: str) -> str:
    path = Path(profile_path)
    return str(path.with_suffix(''))


def log_plan_summary(summary: PlanSummary) -> None:
    _log.info(
        'planner summary: source=%s output=%s batch_scale=%.3f time_scale=%.3f biases(h/f/d)=%s/%s/%s',
        summary.source_event_count,
        summary.output_event_count,
        summary.batch_scale,
        summary.time_scale,
        summary.heat_bias,
        summary.fan_bias,
        summary.drum_bias,
    )
