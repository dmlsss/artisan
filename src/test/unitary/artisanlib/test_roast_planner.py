from pathlib import Path

from artisanlib.roast_planner import (
    build_planned_profile,
    build_safety_alarm_set,
    load_profile_data,
    save_profile_data,
)
from artisanlib.util import events_external_to_internal_value


def make_profile() -> dict[str, object]:
    return {
        'mode': 'C',
        'title': 'Planner Source',
        'beans': 'Integration Test',
        'weight': [200.0, 170.0, 'g'],
        'timex': [0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
        'temp1': [165.0, 175.0, 190.0, 205.0, 220.0, 230.0],
        'temp2': [25.0, 70.0, 105.0, 145.0, 180.0, 205.0],
        'specialevents': [1, 2, 3, 4],
        # legacy Kaleido mapping Fan->0 Drum->1 Burner->3
        'specialeventstype': [3, 0, 1, 3],
        'specialeventsvalue': [
            events_external_to_internal_value(65),
            events_external_to_internal_value(40),
            events_external_to_internal_value(55),
            events_external_to_internal_value(50),
        ],
        'specialeventsStrings': ['HP=65%', 'SM=40%', 'RL=55%', 'HP=50%'],
        'etypes': ['Air', 'Drum', 'Damper', 'Burner', '--'],
    }


def test_load_profile_data_accepts_json(tmp_path: Path) -> None:
    p = make_profile()
    filename = tmp_path / 'planner_source.alog'
    save_profile_data(str(filename), p)  # writes JSON

    loaded = load_profile_data(str(filename))
    assert loaded.get('title') == p['title']
    assert loaded.get('specialevents') == p['specialevents']


def test_build_planned_profile_normalizes_kaleido_channels() -> None:
    planned, summary = build_planned_profile(make_profile(), target_batch_kg=0.30)

    assert summary.output_event_count > 0
    assert summary.batch_scale > 1.0
    assert planned['title'].endswith('[Planned]')
    assert planned['etypes'][0:3] == ['Heat', 'Fan', 'Drum']

    output_types = set(planned['specialeventstype'])
    assert output_types.issubset({0, 1, 2})
    assert len(planned['specialevents']) == len(planned['specialeventstype']) == len(planned['specialeventsStrings'])
    assert any(s.startswith('HP=') for s in planned['specialeventsStrings'])
    assert any(s.startswith('FC=') for s in planned['specialeventsStrings'])
    assert any(s.startswith('RC=') for s in planned['specialeventsStrings'])


def test_build_safety_alarm_set_defaults() -> None:
    alarms = build_safety_alarm_set(make_profile())

    assert alarms['alarmflags'] == [1, 1]
    assert alarms['alarmsources'] == [0, 1]  # ET, BT
    assert alarms['alarmconds'] == [1, 1]  # >
    assert alarms['alarmactions'] == [1, 1]  # popup
    assert alarms['alarmbeep'] == [1, 1]
    assert len(alarms['alarmtemperatures']) == 2
