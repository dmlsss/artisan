import json
from pathlib import Path

import numpy as np

from artisanlib.thermal_alarm_generator import ACTION_FAN, ACTION_HEATER, generate_alarm_table
from artisanlib.thermal_profile_parser import parse_target_profile


def test_parse_target_profile_accepts_json_without_kaleido_extradevices(tmp_path: Path) -> None:
    profile = {
        'timex': [0.0, 30.0, 60.0, 90.0],
        'temp2': [25.0, 95.0, 150.0, 198.0],
        'weight': [120.0, 100.0, 'g'],
        'timeindex': [0, 0, 0, 0, 0, 0, 3, 0],
    }
    path = tmp_path / 'target_profile.alog'
    path.write_text(json.dumps(profile), encoding='utf-8')

    target = parse_target_profile(str(path))

    assert target.source_file.endswith('target_profile.alog')
    assert np.allclose(target.time, [0.0, 30.0, 60.0, 90.0])
    assert np.allclose(target.bt, profile['temp2'])
    assert target.batch_mass_kg == 0.12


def test_parse_target_profile_rebases_without_timeindex(tmp_path: Path) -> None:
    profile = {
        'timex': [10.0, 20.0, 35.0, 50.0],
        'temp2': [30.0, 60.0, 120.0, 180.0],
    }
    path = tmp_path / 'target_no_timeindex.alog'
    path.write_text(json.dumps(profile), encoding='utf-8')

    target = parse_target_profile(str(path))

    assert np.allclose(target.time, [0.0, 10.0, 25.0, 40.0])
    assert np.allclose(target.bt, profile['temp2'])


def test_generate_alarm_table_uses_positive_offsets_for_time_based_actions() -> None:
    alarms = generate_alarm_table(
        time=np.array([0.0, 10.0, 20.0], dtype=np.float64),
        heater_pct=np.array([75.0, 75.0, 65.0], dtype=np.float64),
        fan_pct=np.array([30.0, 30.0, 45.0], dtype=np.float64),
    )

    assert all(offset >= 1 for offset in alarms.alarmoffset)
    assert any(action == ACTION_HEATER for action in alarms.alarmaction)
    assert any(action == ACTION_FAN for action in alarms.alarmaction)
