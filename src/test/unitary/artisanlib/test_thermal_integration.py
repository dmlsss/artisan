import json
from pathlib import Path

import numpy as np

from artisanlib.thermal_alarm_generator import ACTION_FAN, ACTION_HEATER, generate_alarm_table
from artisanlib.thermal_control_dlg import _cap_calibration_file_selection
from artisanlib.thermal_profile_parser import parse_alog_profile, parse_target_profile


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


def test_parse_target_profile_converts_fahrenheit_to_celsius(tmp_path: Path) -> None:
    profile = {
        'mode': 'F',
        'timex': [0.0, 30.0, 60.0],
        'temp2': [32.0, 122.0, 212.0],
    }
    path = tmp_path / 'target_fahrenheit.alog'
    path.write_text(json.dumps(profile), encoding='utf-8')

    target = parse_target_profile(str(path))

    assert np.allclose(target.bt, [0.0, 50.0, 100.0])


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


def test_parse_alog_profile_converts_fahrenheit_to_celsius(tmp_path: Path) -> None:
    profile = {
        'mode': 'F',
        'timex': [0.0, 10.0, 20.0],
        'temp1': [122.0, 140.0, 158.0],  # ET: 50, 60, 70 C
        'temp2': [32.0, 122.0, 212.0],   # BT: 0, 50, 100 C
        'extradevices': [141],
        'extratimex': [[0.0, 10.0, 20.0]],
        'extratemp1': [[70.0, 60.0, 50.0]],
        'extratemp2': [[20.0, 30.0, 40.0]],
        'timeindex': [0, 0, 0, 0, 0, 0, 2, 0],
    }
    path = tmp_path / 'calibration_fahrenheit.alog'
    path.write_text(json.dumps(profile), encoding='utf-8')

    calib = parse_alog_profile(str(path))

    assert np.allclose(calib.et, [50.0, 60.0, 70.0])
    assert np.allclose(calib.bt, [0.0, 50.0, 100.0])


def test_generate_alarm_table_uses_positive_offsets_for_time_based_actions() -> None:
    alarms = generate_alarm_table(
        time=np.array([0.0, 10.0, 20.0], dtype=np.float64),
        heater_pct=np.array([75.0, 75.0, 65.0], dtype=np.float64),
        fan_pct=np.array([30.0, 30.0, 45.0], dtype=np.float64),
    )

    assert all(offset >= 1 for offset in alarms.alarmoffset)
    assert any(action == ACTION_HEATER for action in alarms.alarmaction)
    assert any(action == ACTION_FAN for action in alarms.alarmaction)


def test_cap_calibration_file_selection_respects_total_limit() -> None:
    selected, limited = _cap_calibration_file_selection(
        current_count=2,
        selected_files=['a.alog', 'b.alog'],
        max_profiles=3,
    )
    assert selected == ['a.alog']
    assert limited is True

    selected, limited = _cap_calibration_file_selection(
        current_count=3,
        selected_files=['a.alog'],
        max_profiles=3,
    )
    assert selected == []
    assert limited is True
