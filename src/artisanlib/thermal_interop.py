#
# ABOUT
# Interoperability adapters for thermal planning schedules.

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

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from artisanlib.thermal_alarm_generator import AlarmTableData, generate_alarm_table
from artisanlib.thermal_model_inversion import InversionResult

TriggerMode = Literal['time', 'bt']


@dataclass
class InteropSchedule:
    label: str
    trigger_mode: TriggerMode
    time_s: np.ndarray
    heater_pct: np.ndarray
    fan_pct: np.ndarray
    drum_pct: np.ndarray
    bt_threshold_c: np.ndarray | None = None


def schedule_from_inversion(
    inversion: InversionResult,
    *,
    trigger_mode: TriggerMode = 'time',
    label: str = 'Thermal Model Control',
) -> InteropSchedule:
    bt = inversion.predicted_bt if trigger_mode == 'bt' else None
    return InteropSchedule(
        label=label,
        trigger_mode=trigger_mode,
        time_s=np.asarray(inversion.time, dtype=np.float64),
        heater_pct=np.asarray(inversion.heater_pct, dtype=np.float64),
        fan_pct=np.asarray(inversion.fan_pct, dtype=np.float64),
        drum_pct=np.asarray(inversion.drum_pct, dtype=np.float64),
        bt_threshold_c=(None if bt is None else np.asarray(bt, dtype=np.float64)),
    )


def export_artisan_plan_json(path: str, schedule: InteropSchedule) -> None:
    payload: dict[str, object] = {
        'format': 'artisan-thermal-plan-v1',
        'label': schedule.label,
        'trigger_mode': schedule.trigger_mode,
        'time_s': schedule.time_s.tolist(),
        'heater_pct': np.rint(schedule.heater_pct).astype(int).tolist(),
        'fan_pct': np.rint(schedule.fan_pct).astype(int).tolist(),
        'drum_pct': np.rint(schedule.drum_pct).astype(int).tolist(),
    }
    if schedule.bt_threshold_c is not None:
        payload['bt_threshold_c'] = schedule.bt_threshold_c.tolist()
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, separators=(',', ':'), ensure_ascii=False)
        fh.write('\n')


def export_hibean_csv(path: str, schedule: InteropSchedule) -> None:
    """Export a HiBean-style replay CSV.

    The schema is a practical interchange format:
    ``time_s,bt_trigger_c,heater_pct,fan_pct,drum_pct``.
    """
    bt_values = schedule.bt_threshold_c
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['time_s', 'bt_trigger_c', 'heater_pct', 'fan_pct', 'drum_pct'])
        for i in range(len(schedule.time_s)):
            bt = '' if bt_values is None else f'{float(bt_values[i]):.3f}'
            writer.writerow(
                [
                    f'{float(schedule.time_s[i]):.3f}',
                    bt,
                    int(round(float(schedule.heater_pct[i]))),
                    int(round(float(schedule.fan_pct[i]))),
                    int(round(float(schedule.drum_pct[i]))),
                ]
            )


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # pylint: disable=broad-except
        return default


def import_interop_schedule(path: str, fmt: Literal['auto', 'json', 'csv'] = 'auto') -> InteropSchedule:
    suffix = Path(path).suffix.lower()
    if fmt == 'auto':
        fmt = 'csv' if suffix == '.csv' else 'json'
    if fmt == 'csv':
        return _import_csv_schedule(path)
    return _import_json_schedule(path)


def _import_csv_schedule(path: str) -> InteropSchedule:
    times: list[float] = []
    bt_vals: list[float] = []
    heater: list[float] = []
    fan: list[float] = []
    drum: list[float] = []
    has_bt = False
    with open(path, encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            times.append(_parse_float(row.get('time_s', 0.0)))
            heater.append(_parse_float(row.get('heater_pct', row.get('heat', 0.0))))
            fan.append(_parse_float(row.get('fan_pct', row.get('fan', 0.0))))
            drum.append(_parse_float(row.get('drum_pct', row.get('drum', 0.0))))
            raw_bt = row.get('bt_trigger_c', '')
            if raw_bt not in {'', None}:
                bt_vals.append(_parse_float(raw_bt))
                has_bt = True
            else:
                bt_vals.append(np.nan)
    bt_arr = np.asarray(bt_vals, dtype=np.float64) if has_bt else None
    if bt_arr is not None:
        bt_arr = np.where(np.isfinite(bt_arr), bt_arr, np.nan)
    return InteropSchedule(
        label=Path(path).stem,
        trigger_mode=('bt' if bt_arr is not None else 'time'),
        time_s=np.asarray(times, dtype=np.float64),
        heater_pct=np.asarray(heater, dtype=np.float64),
        fan_pct=np.asarray(fan, dtype=np.float64),
        drum_pct=np.asarray(drum, dtype=np.float64),
        bt_threshold_c=bt_arr,
    )


def _step_value(step: dict[str, object], keys: list[str], default: float | None = None) -> float | None:
    for key in keys:
        if key in step and step[key] is not None:
            return _parse_float(step[key])
    return default


def _import_json_schedule(path: str) -> InteropSchedule:
    with open(path, encoding='utf-8') as fh:
        payload = json.load(fh)

    label = str(payload.get('label', Path(path).stem))
    trigger_mode = str(payload.get('trigger_mode', 'time')).lower()
    if trigger_mode not in {'time', 'bt'}:
        trigger_mode = 'time'

    if 'time_s' in payload and 'heater_pct' in payload and 'fan_pct' in payload:
        time_arr = np.asarray(payload['time_s'], dtype=np.float64)
        hp_arr = np.asarray(payload['heater_pct'], dtype=np.float64)
        fan_arr = np.asarray(payload['fan_pct'], dtype=np.float64)
        drum_arr = np.asarray(payload.get('drum_pct', np.zeros_like(time_arr)), dtype=np.float64)
        bt_arr = payload.get('bt_threshold_c')
        bt = np.asarray(bt_arr, dtype=np.float64) if bt_arr is not None else None
        return InteropSchedule(
            label=label,
            trigger_mode=('bt' if bt is not None else trigger_mode),
            time_s=time_arr,
            heater_pct=hp_arr,
            fan_pct=fan_arr,
            drum_pct=drum_arr,
            bt_threshold_c=bt,
        )

    # HiBean-like flexible format with steps / automation arrays.
    steps = payload.get('steps') or payload.get('automation') or payload.get('rules') or []
    if not isinstance(steps, list):
        raise ValueError('Unsupported schedule JSON format')

    times: list[float] = []
    bt_vals: list[float] = []
    heater: list[float] = []
    fan: list[float] = []
    drum: list[float] = []
    has_bt = False
    for idx, step_raw in enumerate(steps):
        if not isinstance(step_raw, dict):
            continue
        step = step_raw
        controls = step.get('controls', {})
        if not isinstance(controls, dict):
            controls = {}
        time_val = _step_value(step, ['time_s', 'time', 'sec', 'seconds', 'offset'], float(idx))
        bt_val = _step_value(step, ['bt_c', 'bt', 'bean_temp', 'trigger_bt'])
        hp_val = _step_value(step, ['heater_pct', 'heater', 'heat', 'hp'])
        fan_val = _step_value(step, ['fan_pct', 'fan', 'fc'])
        drum_val = _step_value(step, ['drum_pct', 'drum', 'rc'])
        if hp_val is None:
            hp_val = _step_value(controls, ['heater_pct', 'heater', 'heat', 'hp'], 0.0)
        if fan_val is None:
            fan_val = _step_value(controls, ['fan_pct', 'fan', 'fc'], 0.0)
        if drum_val is None:
            drum_val = _step_value(controls, ['drum_pct', 'drum', 'rc'], 0.0)
        if bt_val is None and isinstance(step.get('trigger'), dict):
            bt_val = _step_value(step['trigger'], ['bt', 'bean_temp', 'temperature'])

        times.append(float(time_val or 0.0))
        heater.append(float(hp_val or 0.0))
        fan.append(float(fan_val or 0.0))
        drum.append(float(drum_val or 0.0))
        if bt_val is not None:
            bt_vals.append(float(bt_val))
            has_bt = True
        else:
            bt_vals.append(np.nan)

    bt = np.asarray(bt_vals, dtype=np.float64) if has_bt else None
    return InteropSchedule(
        label=label,
        trigger_mode=('bt' if has_bt else 'time'),
        time_s=np.asarray(times, dtype=np.float64),
        heater_pct=np.asarray(heater, dtype=np.float64),
        fan_pct=np.asarray(fan, dtype=np.float64),
        drum_pct=np.asarray(drum, dtype=np.float64),
        bt_threshold_c=bt,
    )


def interop_to_alarm_table(
    schedule: InteropSchedule,
    *,
    min_delta_pct: int = 1,
    bt_hysteresis_c: float = 1.0,
    bt_min_gap_c: float = 2.0,
) -> AlarmTableData:
    bt_profile = None
    if schedule.trigger_mode == 'bt':
        if schedule.bt_threshold_c is None:
            raise ValueError('BT-trigger schedule requires bt_threshold_c data')
        bt_profile = np.asarray(schedule.bt_threshold_c, dtype=np.float64)
        bt_profile = np.where(np.isfinite(bt_profile), bt_profile, np.nan)
        if np.any(~np.isfinite(bt_profile)):
            if np.any(np.isfinite(bt_profile)):
                fill = float(np.nanmean(bt_profile))
            else:
                fill = 0.0
            bt_profile = np.nan_to_num(bt_profile, nan=fill)

    return generate_alarm_table(
        time=np.asarray(schedule.time_s, dtype=np.float64),
        heater_pct=np.asarray(schedule.heater_pct, dtype=np.float64),
        fan_pct=np.asarray(schedule.fan_pct, dtype=np.float64),
        drum_pct=np.asarray(schedule.drum_pct, dtype=np.float64),
        trigger_mode=schedule.trigger_mode,
        bt_profile=bt_profile,
        min_delta_pct=min_delta_pct,
        bt_hysteresis_c=bt_hysteresis_c,
        bt_min_gap_c=bt_min_gap_c,
        label=schedule.label or 'Interop Schedule',
    )
