import json
from pathlib import Path

import numpy as np

from artisanlib.thermal_alarm_generator import (
    ACTION_DRUM,
    ACTION_FAN,
    ACTION_HEATER,
    ACTION_POPUP,
    generate_alarm_table,
)
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


def test_generate_alarm_table_bt_trigger_with_drum_and_deadband() -> None:
    alarms = generate_alarm_table(
        time=np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
        heater_pct=np.array([70.0, 71.0, 72.0, 78.0], dtype=np.float64),
        fan_pct=np.array([30.0, 30.0, 31.0, 35.0], dtype=np.float64),
        drum_pct=np.array([55.0, 55.0, 56.0, 60.0], dtype=np.float64),
        trigger_mode='bt',
        bt_profile=np.array([90.0, 110.0, 130.0, 150.0], dtype=np.float64),
        min_delta_pct=3,
    )

    # BT trigger mode should set alarmtime=-1 for control rows.
    control_rows = [
        i for i, action in enumerate(alarms.alarmaction)
        if action in {ACTION_HEATER, ACTION_FAN, ACTION_DRUM}
    ]
    assert control_rows
    assert all(alarms.alarmtime[i] == -1 for i in control_rows)
    assert all(alarms.alarmoffset[i] == 0 for i in control_rows)

    # Deadband should suppress tiny changes (71/72 and 31/56 do not emit).
    heater_values = [alarms.alarmstrings[i] for i in control_rows if alarms.alarmaction[i] == ACTION_HEATER]
    fan_values = [alarms.alarmstrings[i] for i in control_rows if alarms.alarmaction[i] == ACTION_FAN]
    drum_values = [alarms.alarmstrings[i] for i in control_rows if alarms.alarmaction[i] == ACTION_DRUM]
    assert heater_values == ['70', '78']
    assert fan_values == ['30', '35']
    assert drum_values == ['55', '60']


def test_generate_alarm_table_bt_trigger_hardening_reduces_dense_actions() -> None:
    alarms = generate_alarm_table(
        time=np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
        heater_pct=np.array([60.0, 65.0, 70.0, 75.0], dtype=np.float64),
        fan_pct=np.array([25.0, 30.0, 35.0, 40.0], dtype=np.float64),
        trigger_mode='bt',
        bt_profile=np.array([120.0, 121.0, 122.0, 123.0], dtype=np.float64),
        bt_hysteresis_c=2.0,
        bt_min_gap_c=3.0,
    )

    heater_rows = [i for i, action in enumerate(alarms.alarmaction) if action == ACTION_HEATER]
    # With a 3C minimum BT gap and only 1C per sample progression, we should
    # emit fewer than all four heater updates.
    assert len(heater_rows) < 4
    assert all(alarms.alarmtime[i] == -1 for i in heater_rows)


def test_generate_alarm_table_adds_milestones_and_safety_alarms() -> None:
    alarms = generate_alarm_table(
        time=np.array([0.0, 10.0], dtype=np.float64),
        heater_pct=np.array([75.0, 60.0], dtype=np.float64),
        fan_pct=np.array([30.0, 40.0], dtype=np.float64),
        milestone_offsets={'First Crack': 420.0, 'Drop': 510.0},
        bt_safety_ceiling=225.0,
        et_safety_ceiling=255.0,
    )

    popup_labels = [
        alarms.alarmstrings[i]
        for i, action in enumerate(alarms.alarmaction)
        if action == ACTION_POPUP
    ]
    assert 'First Crack target reached' in popup_labels
    assert 'Drop target reached' in popup_labels
    assert 'BT safety ceiling' in popup_labels
    assert 'ET safety ceiling' in popup_labels


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
