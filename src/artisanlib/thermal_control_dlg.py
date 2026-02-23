#
# ABOUT
# PyQt6 dialog for thermal model calibration, schedule generation,
# and alarm export for the Kaleido M1 Lite roaster.

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

import logging
import os
import json
from dataclasses import dataclass

from PyQt6.QtCore import QThread, QSettings, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from artisanlib.dialogs import ArtisanDialog
from artisanlib.thermal_alarm_generator import (
    AlarmTableData,
    generate_alarm_table,
    generate_schedule_description,
)
from artisanlib.thermal_interop import (
    export_artisan_plan_json,
    export_hibean_csv,
    schedule_from_inversion,
)
from artisanlib.thermal_model import KaleidoThermalModel, ThermalModelParams
from artisanlib.thermal_model_fitting import FitResult, fit_model
from artisanlib.thermal_model_inversion import InversionResult, invert_model
from artisanlib.thermal_planner_quality import QualityReport, build_quality_report
from artisanlib.thermal_profile_parser import CalibrationData, parse_alog_profile, parse_target_profile
from artisanlib.thermal_schedule_validator import SafetyValidationResult, validate_schedule

import numpy as np

from typing import Final, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from artisanlib.main import ApplicationWindow  # pylint: disable=unused-import

_log: Final[logging.Logger] = logging.getLogger(__name__)
_MAX_CALIBRATION_PROFILES: Final[int] = 3
_BATCH_PRESET_MASSES: Final[tuple[int, ...]] = (50, 75, 100, 125, 150)
PlannerGoal = Literal['safe', 'balanced', 'precision']
_BT_TEMP_ALWAYS: Final[float] = 500.0
_FLAVOR_LEARNING_GROUP: Final[str] = 'ThermalModelControl'
_FLAVOR_LEARNING_KEY: Final[str] = 'flavorImpactLearningV1'
_FLAVOR_LEARNING_LIMIT: Final[int] = 512

ACTION_POPUP: Final[int] = 0
ACTION_HEATER: Final[int] = 3
ACTION_FAN: Final[int] = 4
ACTION_DRUM: Final[int] = 5


@dataclass(frozen=True)
class BatchPlannerPreset:
    mass_g: int
    fan_pct: int
    drum_pct: int
    interval_s: int
    min_delta_pct: int
    optimizer_passes: int
    bt_max_c: int
    et_max_c: int
    max_ror_c_per_min: float
    trigger_mode: Literal['time', 'bt'] = 'time'


@dataclass(frozen=True)
class TargetCurveStats:
    sample_count: int
    duration_s: float
    bt_min_c: float
    bt_max_c: float
    bt_start_c: float
    bt_end_c: float


_BASE_BATCH_PRESETS: Final[dict[int, BatchPlannerPreset]] = {
    50: BatchPlannerPreset(50, fan_pct=45, drum_pct=55, interval_s=5, min_delta_pct=3, optimizer_passes=2, bt_max_c=218, et_max_c=245, max_ror_c_per_min=24.0),
    75: BatchPlannerPreset(75, fan_pct=40, drum_pct=58, interval_s=5, min_delta_pct=3, optimizer_passes=2, bt_max_c=222, et_max_c=248, max_ror_c_per_min=26.0),
    100: BatchPlannerPreset(100, fan_pct=35, drum_pct=60, interval_s=10, min_delta_pct=2, optimizer_passes=3, bt_max_c=225, et_max_c=255, max_ror_c_per_min=30.0),
    125: BatchPlannerPreset(125, fan_pct=32, drum_pct=62, interval_s=10, min_delta_pct=2, optimizer_passes=3, bt_max_c=228, et_max_c=260, max_ror_c_per_min=32.0),
    150: BatchPlannerPreset(150, fan_pct=30, drum_pct=65, interval_s=10, min_delta_pct=2, optimizer_passes=4, bt_max_c=230, et_max_c=265, max_ror_c_per_min=34.0),
}


def _clamp_int(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def resolve_batch_planner_preset(mass_g: int, goal: PlannerGoal = 'balanced') -> BatchPlannerPreset:
    """Return a goal-adjusted preset snapped to the nearest supported batch size."""
    nearest_mass = min(_BATCH_PRESET_MASSES, key=lambda m: abs(m - int(mass_g)))
    base = _BASE_BATCH_PRESETS[nearest_mass]

    if goal == 'safe':
        return BatchPlannerPreset(
            mass_g=base.mass_g,
            fan_pct=_clamp_int(base.fan_pct + 4, 20, 90),
            drum_pct=base.drum_pct,
            interval_s=base.interval_s,
            min_delta_pct=_clamp_int(base.min_delta_pct + 1, 1, 20),
            optimizer_passes=_clamp_int(base.optimizer_passes - 1, 1, 8),
            bt_max_c=base.bt_max_c - 2,
            et_max_c=base.et_max_c - 4,
            max_ror_c_per_min=max(5.0, base.max_ror_c_per_min - 3.0),
            trigger_mode='time',
        )

    if goal == 'precision':
        return BatchPlannerPreset(
            mass_g=base.mass_g,
            fan_pct=_clamp_int(base.fan_pct - 3, 15, 90),
            drum_pct=base.drum_pct,
            interval_s=5 if base.interval_s > 5 else base.interval_s,
            min_delta_pct=_clamp_int(base.min_delta_pct - 1, 1, 20),
            optimizer_passes=_clamp_int(base.optimizer_passes + 1, 1, 8),
            bt_max_c=base.bt_max_c,
            et_max_c=base.et_max_c,
            max_ror_c_per_min=min(60.0, base.max_ror_c_per_min + 2.0),
            trigger_mode='time',
        )

    return base


def normalize_target_curve(
    target_time: np.ndarray,
    target_bt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, TargetCurveStats]:
    """Filter/normalize target arrays and return curve stats."""
    time_arr = np.asarray(target_time, dtype=np.float64).reshape(-1)
    bt_arr = np.asarray(target_bt, dtype=np.float64).reshape(-1)

    if time_arr.size != bt_arr.size:
        raise ValueError('Target time and BT arrays must have the same length')

    valid = np.isfinite(time_arr) & np.isfinite(bt_arr) & (bt_arr != -1.0)
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError('Target curve must include at least two valid points')

    time_valid = np.asarray(time_arr[valid], dtype=np.float64)
    bt_valid = np.asarray(bt_arr[valid], dtype=np.float64)

    order = np.argsort(time_valid, kind='stable')
    time_sorted = time_valid[order]
    bt_sorted = bt_valid[order]

    unique_mask = np.ones(time_sorted.shape[0], dtype=bool)
    unique_mask[1:] = np.diff(time_sorted) > 1e-9
    time_unique = time_sorted[unique_mask]
    bt_unique = bt_sorted[unique_mask]

    if time_unique.shape[0] < 2:
        raise ValueError('Target curve must include at least two distinct timestamps')

    duration_s = float(time_unique[-1] - time_unique[0])
    if duration_s <= 0.0:
        raise ValueError('Target curve duration must be positive')

    stats = TargetCurveStats(
        sample_count=int(time_unique.shape[0]),
        duration_s=duration_s,
        bt_min_c=float(np.min(bt_unique)),
        bt_max_c=float(np.max(bt_unique)),
        bt_start_c=float(bt_unique[0]),
        bt_end_c=float(bt_unique[-1]),
    )
    return time_unique, bt_unique, stats


def _cap_calibration_file_selection(
    current_count: int,
    selected_files: list[str],
    max_profiles: int = _MAX_CALIBRATION_PROFILES,
) -> tuple[list[str], bool]:
    """Cap selected file list so total loaded profiles never exceeds *max_profiles*."""
    if current_count >= max_profiles:
        return [], len(selected_files) > 0
    available = max_profiles - current_count
    capped = selected_files[:available]
    return capped, len(capped) < len(selected_files)


def _action_token(action: int) -> str:
    if action == ACTION_HEATER:
        return 'heater'
    if action == ACTION_FAN:
        return 'fan'
    if action == ACTION_DRUM:
        return 'drum'
    return 'popup'


def _sanitize_learning_token(text: str, limit: int = 32) -> str:
    token = ''.join(ch if ch.isalnum() else '_' for ch in str(text).strip().lower())
    token = token.strip('_')
    if token == '':
        token = 'generic'
    return token[:limit]


def _alarm_stage_bucket(
    alarm_data: AlarmTableData,
    row: int,
    max_control_offset: int,
) -> str:
    if int(alarm_data.alarmtime[row]) == -1:
        temp_c = float(alarm_data.alarmtemperature[row])
        if temp_c < 145.0:
            return 'drying'
        if temp_c < 196.0:
            return 'maillard'
        return 'development'

    offset = max(0, int(alarm_data.alarmoffset[row]))
    progress = offset / max(1, int(max_control_offset))
    if progress < 0.33:
        return 'drying'
    if progress < 0.72:
        return 'maillard'
    return 'development'


def _build_flavor_feature_rows(
    alarm_data: AlarmTableData,
) -> list[tuple[str, int, str, str, str]]:
    if alarm_data.alarm_count() == 0:
        return []

    control_actions = {ACTION_HEATER, ACTION_FAN, ACTION_DRUM}
    control_offsets = [
        int(alarm_data.alarmoffset[i])
        for i, action in enumerate(alarm_data.alarmaction)
        if action in control_actions
    ]
    if control_offsets:
        max_control_offset = max(1, int(max(control_offsets)))
    else:
        max_control_offset = max(1, int(max(alarm_data.alarmoffset)))

    rows: list[tuple[str, int, str, str, str]] = []
    last_value_by_action: dict[int, int] = {}
    for i, action in enumerate(alarm_data.alarmaction):
        trigger = 'bt' if int(alarm_data.alarmtime[i]) == -1 else 'time'
        stage = _alarm_stage_bucket(alarm_data, i, max_control_offset)
        popup_token = ''
        direction = 'popup'

        if action in control_actions:
            raw_value = str(alarm_data.alarmstrings[i]).strip()
            try:
                value = int(float(raw_value))
            except Exception:  # pylint: disable=broad-except
                value = last_value_by_action.get(action, 0)
            previous = last_value_by_action.get(action)
            if previous is None:
                direction = 'set'
            elif value > previous:
                direction = 'up'
            elif value < previous:
                direction = 'down'
            else:
                direction = 'hold'
            last_value_by_action[action] = value
            key = f'{_action_token(action)}:{direction}:{stage}:{trigger}'
        else:
            popup_token = _sanitize_learning_token(alarm_data.alarmstrings[i])
            key = f'{_action_token(action)}:{direction}:{stage}:{trigger}:{popup_token}'

        rows.append((key, int(action), direction, stage, popup_token))

    return rows


def _build_flavor_feature_keys(alarm_data: AlarmTableData) -> list[str]:
    return [row[0] for row in _build_flavor_feature_rows(alarm_data)]


def _heuristic_flavor_note(action: int, direction: str, stage: str, popup_token: str) -> str:
    if action == ACTION_POPUP:
        if 'first_crack' in popup_token:
            return 'First-crack marker: expect aroma shift and development-driven sweetness changes.'
        if 'safety' in popup_token:
            return 'Safety checkpoint: protects cup clarity and reduces defect risk.'
        if 'drop' in popup_token:
            return 'Drop milestone: sets final development balance and finish.'
        return 'Process milestone: use this point to correlate sensory changes after cupping.'

    if action == ACTION_HEATER:
        if direction == 'up':
            if stage == 'drying':
                return 'More early heat can increase body but may reduce floral clarity.'
            if stage == 'maillard':
                return 'More Maillard energy may boost caramel sweetness and body.'
            return 'More late heat can intensify roast tones with higher bitterness risk.'
        if direction == 'down':
            if stage == 'drying':
                return 'Gentler drying can improve cleanliness and reduce harshness risk.'
            if stage == 'maillard':
                return 'Lower mid-roast heat can preserve acidity and aromatic detail.'
            return 'Reduced development heat can preserve origin character and sweetness.'
        return 'Heat hold: stabilizes momentum and can improve repeatability of cup balance.'

    if action == ACTION_FAN:
        if direction == 'up':
            if stage == 'drying':
                return 'More airflow early can brighten the cup and reduce steam/chaff carryover.'
            if stage == 'maillard':
                return 'Higher airflow mid-roast can improve clarity and acidity definition.'
            return 'Higher airflow late can restrain RoR and reduce smoky/roasty carryover.'
        if direction == 'down':
            if stage == 'drying':
                return 'Lower airflow early can increase heat retention and body potential.'
            if stage == 'maillard':
                return 'Lower airflow mid-roast can push sweetness/body over acidity.'
            return 'Lower airflow late can deepen roast character and perceived weight.'
        return 'Fan hold: maintains convective balance and flavor consistency.'

    if action == ACTION_DRUM:
        if direction == 'up':
            return 'Higher drum speed can even bean exposure and smooth development.'
        if direction == 'down':
            return 'Lower drum speed can increase conductive influence and heavier cup weight.'
        return 'Drum hold: keeps mechanical energy stable for repeatable progression.'

    return 'Flavor impact estimate unavailable.'


def _load_flavor_learning_map() -> dict[str, str]:
    settings = QSettings()
    settings.beginGroup(_FLAVOR_LEARNING_GROUP)
    raw = settings.value(_FLAVOR_LEARNING_KEY, '')
    settings.endGroup()
    try:
        data = json.loads(str(raw)) if raw not in {None, ''} else {}
    except Exception:  # pylint: disable=broad-except
        data = {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in data.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            cleaned[k] = v
    return cleaned


def _save_flavor_learning_map(learning_map: dict[str, str]) -> None:
    cleaned_items = [
        (str(k).strip(), str(v).strip())
        for k, v in learning_map.items()
        if str(k).strip() and str(v).strip()
    ]
    if len(cleaned_items) > _FLAVOR_LEARNING_LIMIT:
        cleaned_items = cleaned_items[-_FLAVOR_LEARNING_LIMIT:]
    payload = dict(cleaned_items)
    settings = QSettings()
    settings.beginGroup(_FLAVOR_LEARNING_GROUP)
    settings.setValue(
        _FLAVOR_LEARNING_KEY,
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
    )
    settings.endGroup()


def _apply_flavor_learning(
    guessed_notes: list[str],
    feature_keys: list[str],
    learning_map: dict[str, str],
) -> list[str]:
    resolved: list[str] = []
    for idx, guess in enumerate(guessed_notes):
        key = feature_keys[idx] if idx < len(feature_keys) else ''
        learned = str(learning_map.get(key, '')).strip()
        resolved.append(learned if learned else guess)
    return resolved


def _guess_flavor_impact_notes(
    alarm_data: AlarmTableData,
    learning_map: dict[str, str] | None = None,
) -> list[str]:
    rows = _build_flavor_feature_rows(alarm_data)
    guessed = [
        _heuristic_flavor_note(action, direction, stage, popup_token)
        for _, action, direction, stage, popup_token in rows
    ]
    learned = _load_flavor_learning_map() if learning_map is None else dict(learning_map)
    return _apply_flavor_learning(guessed, [row[0] for row in rows], learned)


def _learn_flavor_impact_notes(
    alarm_data: AlarmTableData,
    notes: list[str],
    baseline_notes: list[str] | None = None,
    learning_map: dict[str, str] | None = None,
) -> dict[str, str]:
    learned = _load_flavor_learning_map() if learning_map is None else dict(learning_map)
    keys = _build_flavor_feature_keys(alarm_data)
    baseline = [str(x).strip() for x in (baseline_notes or [])]

    for idx, key in enumerate(keys):
        if idx >= len(notes):
            continue
        note = str(notes[idx]).strip()
        if note == '':
            continue
        if idx < len(baseline) and note == baseline[idx]:
            # unchanged auto-guess; only persist explicit user corrections
            continue
        learned[key] = note

    if learning_map is None:
        _save_flavor_learning_map(learned)
    return learned


# ---------------------------------------------------------------------------
# Fit worker thread
# ---------------------------------------------------------------------------

class FitWorker(QThread):
    """Run thermal model fitting in a background thread."""

    progress = pyqtSignal(int, float)
    finished = pyqtSignal(object)  # FitResult
    error = pyqtSignal(str)

    def __init__(self, calibration_data: list[CalibrationData], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.calibration_data = calibration_data

    def run(self) -> None:
        try:
            result = fit_model(
                self.calibration_data,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as e:  # pylint: disable=broad-except
            _log.exception('Fit failed')
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Alarm review dialog
# ---------------------------------------------------------------------------

class AlarmReviewDialog(QDialog):
    """Review and edit generated alarm rows before export/apply."""

    _ACTION_LABELS: Final[dict[int, str]] = {
        ACTION_POPUP: 'popup',
        ACTION_HEATER: 'heater',
        ACTION_FAN: 'fan',
        ACTION_DRUM: 'drum',
    }
    _ACTION_FROM_LABEL: Final[dict[str, int]] = {
        'popup': ACTION_POPUP,
        'heater': ACTION_HEATER,
        'fan': ACTION_FAN,
        'drum': ACTION_DRUM,
        'heat': ACTION_HEATER,
        'hp': ACTION_HEATER,
        'fc': ACTION_FAN,
        'rc': ACTION_DRUM,
    }
    _SOURCE_LABELS: Final[dict[int, str]] = {
        0: 'ET',
        1: 'BT',
    }
    _SOURCE_FROM_LABEL: Final[dict[str, int]] = {
        'et': 0,
        'bt': 1,
    }
    _COND_LABELS: Final[dict[int, str]] = {
        1: 'above',
        2: 'below',
    }
    _COND_FROM_LABEL: Final[dict[str, int]] = {
        'above': 1,
        'below': 2,
        '>': 1,
        '<': 2,
    }

    def __init__(
        self,
        parent: QWidget | None,
        alarm_data: AlarmTableData,
        flavor_notes: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._alarm_data = alarm_data
        self._flavor_notes = list(flavor_notes or [])
        if len(self._flavor_notes) < self._alarm_data.alarm_count():
            self._flavor_notes.extend([''] * (self._alarm_data.alarm_count() - len(self._flavor_notes)))
        elif len(self._flavor_notes) > self._alarm_data.alarm_count():
            self._flavor_notes = self._flavor_notes[:self._alarm_data.alarm_count()]

        self.updated_alarm_data: AlarmTableData | None = None
        self.updated_flavor_notes: list[str] = []

        self.setWindowTitle(QApplication.translate('Form', 'Review Alarm Table'))
        self.resize(1080, 520)

        layout = QVBoxLayout()

        info_label = QLabel(
            QApplication.translate(
                'Label',
                'Edit rows before finalizing. Flavor Impact notes are for collaboration and are not exported in .alrm.',
            )
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.table = QTableWidget(self._alarm_data.alarm_count(), 10)
        self.table.setHorizontalHeaderLabels(
            [
                QApplication.translate('Label', 'On'),
                QApplication.translate('Label', 'Trigger'),
                QApplication.translate('Label', 'Offset (s)'),
                QApplication.translate('Label', 'Source'),
                QApplication.translate('Label', 'Condition'),
                QApplication.translate('Label', 'BT/ET Temp'),
                QApplication.translate('Label', 'Action'),
                QApplication.translate('Label', 'Value'),
                QApplication.translate('Label', 'Beep'),
                QApplication.translate('Label', 'Flavor Impact'),
            ]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self._populate_rows()
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def _set_cell(self, row: int, column: int, value: str) -> None:
        self.table.setItem(row, column, QTableWidgetItem(value))

    def _populate_rows(self) -> None:
        for row in range(self._alarm_data.alarm_count()):
            enabled = '1' if int(self._alarm_data.alarmflag[row]) > 0 else '0'
            trigger = 'bt' if int(self._alarm_data.alarmtime[row]) == -1 else 'time'
            offset = str(int(self._alarm_data.alarmoffset[row]))
            source = self._SOURCE_LABELS.get(int(self._alarm_data.alarmsource[row]), 'BT')
            condition = self._COND_LABELS.get(int(self._alarm_data.alarmcond[row]), 'above')
            temp = f'{float(self._alarm_data.alarmtemperature[row]):.1f}'
            action = self._ACTION_LABELS.get(int(self._alarm_data.alarmaction[row]), 'popup')
            value = str(self._alarm_data.alarmstrings[row])
            beep = '1' if int(self._alarm_data.alarmbeep[row]) > 0 else '0'
            flavor_note = self._flavor_notes[row] if row < len(self._flavor_notes) else ''

            self._set_cell(row, 0, enabled)
            self._set_cell(row, 1, trigger)
            self._set_cell(row, 2, offset)
            self._set_cell(row, 3, source)
            self._set_cell(row, 4, condition)
            self._set_cell(row, 5, temp)
            self._set_cell(row, 6, action)
            self._set_cell(row, 7, value)
            self._set_cell(row, 8, beep)
            self._set_cell(row, 9, flavor_note)

    def _cell_text(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text().strip() if item is not None else ''

    @staticmethod
    def _parse_bool01(token: str, *, default: int = 0) -> int:
        value = token.strip().lower()
        if value in {'1', 'true', 'yes', 'y', 'on'}:
            return 1
        if value in {'0', 'false', 'no', 'n', 'off'}:
            return 0
        return int(default)

    @staticmethod
    def _parse_int(token: str, *, default: int = 0) -> int:
        value = token.strip()
        if value == '':
            return int(default)
        return int(float(value))

    @staticmethod
    def _parse_float(token: str, *, default: float = 0.0) -> float:
        value = token.strip()
        if value == '':
            return float(default)
        return float(value)

    @pyqtSlot()
    def _on_accept(self) -> None:
        try:
            self.updated_alarm_data, self.updated_flavor_notes = self._collect_data()
        except ValueError as e:
            QMessageBox.warning(
                self,
                QApplication.translate('Message', 'Invalid Alarm Table'),
                str(e),
            )
            return
        self.accept()

    def _collect_data(self) -> tuple[AlarmTableData, list[str]]:
        updated = AlarmTableData(label=self._alarm_data.label)
        notes: list[str] = []

        for row in range(self.table.rowCount()):
            trigger_token = self._cell_text(row, 1).lower()
            trigger_mode = 'bt' if trigger_token in {'bt', 'temp', 'temperature'} else 'time'

            source_token = self._cell_text(row, 3).lower()
            source = self._SOURCE_FROM_LABEL.get(source_token, 1)

            cond_token = self._cell_text(row, 4).lower()
            cond = self._COND_FROM_LABEL.get(cond_token, 1)

            action_token = self._cell_text(row, 6).lower()
            if action_token not in self._ACTION_FROM_LABEL:
                raise ValueError(
                    QApplication.translate(
                        'Message',
                        'Row {0}: action must be one of popup/heater/fan/drum',
                    ).format(row + 1)
                )
            action = self._ACTION_FROM_LABEL[action_token]

            value = self._cell_text(row, 7)
            if value == '':
                raise ValueError(
                    QApplication.translate('Message', 'Row {0}: value cannot be empty').format(row + 1)
                )
            if action in {ACTION_HEATER, ACTION_FAN, ACTION_DRUM}:
                numeric = self._parse_int(value, default=0)
                if numeric < 0 or numeric > 100:
                    raise ValueError(
                        QApplication.translate(
                            'Message',
                            'Row {0}: actuator value must be in 0..100',
                        ).format(row + 1)
                    )
                value = str(numeric)

            enabled = self._parse_bool01(self._cell_text(row, 0), default=1)
            beep = self._parse_bool01(self._cell_text(row, 8), default=0)

            guard = (
                int(self._alarm_data.alarmguard[row])
                if row < len(self._alarm_data.alarmguard)
                else -1
            )
            neg_guard = (
                int(self._alarm_data.alarmnegguard[row])
                if row < len(self._alarm_data.alarmnegguard)
                else -1
            )

            if trigger_mode == 'time':
                offset = max(1, self._parse_int(self._cell_text(row, 2), default=1))
                alarm_time = 0
                alarm_offset = offset
                alarm_source = 1
                alarm_cond = 1
                alarm_temp = _BT_TEMP_ALWAYS
            else:
                temperature = self._parse_float(
                    self._cell_text(row, 5),
                    default=float(self._alarm_data.alarmtemperature[row]),
                )
                alarm_time = -1
                alarm_offset = 0
                alarm_source = source
                alarm_cond = cond
                alarm_temp = float(temperature)

            updated.alarmflag.append(int(enabled))
            updated.alarmguard.append(int(guard))
            updated.alarmnegguard.append(int(neg_guard))
            updated.alarmtime.append(int(alarm_time))
            updated.alarmoffset.append(int(alarm_offset))
            updated.alarmsource.append(int(alarm_source))
            updated.alarmcond.append(int(alarm_cond))
            updated.alarmtemperature.append(float(alarm_temp))
            updated.alarmaction.append(int(action))
            updated.alarmbeep.append(int(beep))
            updated.alarmstrings.append(value)
            notes.append(self._cell_text(row, 9))

        if updated.alarm_count() > 0:
            indices = sorted(range(updated.alarm_count()), key=lambda k: updated.alarmoffset[k])
            updated.alarmflag = [updated.alarmflag[k] for k in indices]
            updated.alarmguard = [updated.alarmguard[k] for k in indices]
            updated.alarmnegguard = [updated.alarmnegguard[k] for k in indices]
            updated.alarmtime = [updated.alarmtime[k] for k in indices]
            updated.alarmoffset = [updated.alarmoffset[k] for k in indices]
            updated.alarmsource = [updated.alarmsource[k] for k in indices]
            updated.alarmcond = [updated.alarmcond[k] for k in indices]
            updated.alarmtemperature = [updated.alarmtemperature[k] for k in indices]
            updated.alarmaction = [updated.alarmaction[k] for k in indices]
            updated.alarmbeep = [updated.alarmbeep[k] for k in indices]
            updated.alarmstrings = [updated.alarmstrings[k] for k in indices]
            notes = [notes[k] for k in indices]

        return updated, notes


# ---------------------------------------------------------------------------
# ThermalControlDlg
# ---------------------------------------------------------------------------

class ThermalControlDlg(ArtisanDialog):
    """Dialog for thermal model calibration, schedule generation, and alarm export."""

    __slots__: list[str] = []

    def __init__(self, parent: QWidget | None, aw: ApplicationWindow) -> None:
        super().__init__(parent, aw)

        self.setWindowTitle(QApplication.translate('Form', 'Thermal Model Control'))
        self.setMinimumWidth(520)

        # ── state ──────────────────────────────────────────────────────
        self.calibration_data: list[CalibrationData] = []
        self.model: KaleidoThermalModel | None = None
        self.fit_result: FitResult | None = None
        self.inversion_result: InversionResult | None = None
        self.alarm_data: AlarmTableData | None = None
        self.safety_validation: SafetyValidationResult | None = None
        self.quality_report: QualityReport | None = None
        self.generated_trigger_mode: str = 'time'
        self.alarm_flavor_notes: list[str] = []
        self.alarm_flavor_guess_notes: list[str] = []
        self.alarm_review_required: bool = False
        self._fit_worker: FitWorker | None = None

        # ── tabs ───────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_calibrate_tab(), QApplication.translate('Tab', 'Calibrate'))
        self.tabs.addTab(self._build_generate_tab(), QApplication.translate('Tab', 'Generate Schedule'))
        self.tabs.addTab(self._build_export_tab(), QApplication.translate('Tab', 'Export'))

        # ── dialog buttons ─────────────────────────────────────────────
        # Replace the default Ok/Cancel with just Close
        self.dialogbuttons.removeButton(self.dialogbuttons.button(QDialogButtonBox.StandardButton.Ok))
        self.dialogbuttons.removeButton(self.dialogbuttons.button(QDialogButtonBox.StandardButton.Cancel))
        close_button: QPushButton | None = self.dialogbuttons.addButton(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            self.setButtonTranslations(close_button, 'Close', QApplication.translate('Button', 'Close'))
        self.dialogbuttons.rejected.connect(self.close)

        # ── main layout ───────────────────────────────────────────────
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.dialogbuttons)
        self.setLayout(layout)

        # initial button state
        self._update_button_states()

    # ===================================================================
    # Tab 1: Calibrate
    # ===================================================================

    def _build_calibrate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        # ── Load / Clear profiles ─────────────────────────────────────
        profile_group = QGroupBox(QApplication.translate('GroupBox', 'Calibration Profiles'))
        profile_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.load_profiles_button = QPushButton(QApplication.translate('Button', 'Load Profiles'))
        self.load_profiles_button.clicked.connect(self._on_load_profiles)
        btn_row.addWidget(self.load_profiles_button)

        self.clear_profiles_button = QPushButton(QApplication.translate('Button', 'Clear'))
        self.clear_profiles_button.clicked.connect(self._on_clear_profiles)
        btn_row.addWidget(self.clear_profiles_button)
        btn_row.addStretch()
        profile_layout.addLayout(btn_row)

        self.profile_list = QListWidget()
        profile_layout.addWidget(self.profile_list)

        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)

        # ── Fit model ─────────────────────────────────────────────────
        fit_group = QGroupBox(QApplication.translate('GroupBox', 'Model Fitting'))
        fit_layout = QVBoxLayout()

        fit_btn_row = QHBoxLayout()
        self.fit_button = QPushButton(QApplication.translate('Button', 'Fit Model'))
        self.fit_button.clicked.connect(self._on_fit_model)
        fit_btn_row.addWidget(self.fit_button)
        fit_btn_row.addStretch()
        fit_layout.addLayout(fit_btn_row)

        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 0)  # indeterminate until first progress
        self.fit_progress.setVisible(False)
        fit_layout.addWidget(self.fit_progress)

        self.fit_results_text = QTextEdit()
        self.fit_results_text.setReadOnly(True)
        self.fit_results_text.setMaximumHeight(160)
        fit_layout.addWidget(self.fit_results_text)

        fit_group.setLayout(fit_layout)
        layout.addWidget(fit_group)

        # ── Save / Load model ─────────────────────────────────────────
        model_btn_row = QHBoxLayout()
        self.save_model_button = QPushButton(QApplication.translate('Button', 'Save Model'))
        self.save_model_button.clicked.connect(self._on_save_model)
        model_btn_row.addWidget(self.save_model_button)

        self.load_model_button = QPushButton(QApplication.translate('Button', 'Load Model'))
        self.load_model_button.clicked.connect(self._on_load_model)
        model_btn_row.addWidget(self.load_model_button)
        model_btn_row.addStretch()
        layout.addLayout(model_btn_row)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    # ===================================================================
    # Tab 2: Generate Schedule
    # ===================================================================

    def _build_generate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        # ── Target curve ──────────────────────────────────────────────
        target_group = QGroupBox(QApplication.translate('GroupBox', 'Target Curve'))
        target_layout = QVBoxLayout()

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel(QApplication.translate('Label', 'Source:')))
        self.target_source_combo = QComboBox()
        self.target_source_combo.addItem(QApplication.translate('ComboBox', 'Background Profile'))
        self.target_source_combo.addItem(QApplication.translate('ComboBox', 'Load from file...'))
        self.target_source_combo.currentIndexChanged.connect(self._on_target_source_changed)
        source_row.addWidget(self.target_source_combo)
        self.inspect_target_button = QPushButton(QApplication.translate('Button', 'Inspect'))
        self.inspect_target_button.clicked.connect(self._on_inspect_target_curve)
        source_row.addWidget(self.inspect_target_button)
        source_row.addStretch()
        target_layout.addLayout(source_row)

        self.target_curve_summary_label = QLabel(
            QApplication.translate('Label', 'Target summary: choose source')
        )
        self.target_curve_summary_label.setWordWrap(True)
        target_layout.addWidget(self.target_curve_summary_label)

        self.target_curve_workflow_label = QLabel(
            QApplication.translate(
                'Label',
                'Workflow: inspect target -> choose batch preset + goal -> apply preset -> generate -> review alarms.',
            )
        )
        self.target_curve_workflow_label.setWordWrap(True)
        target_layout.addWidget(self.target_curve_workflow_label)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # ── Batch settings ────────────────────────────────────────────
        settings_group = QGroupBox(QApplication.translate('GroupBox', 'Schedule Settings'))
        settings_layout = QVBoxLayout()

        mass_row = QHBoxLayout()
        mass_row.addWidget(QLabel(QApplication.translate('Label', 'Batch mass (g):')))
        self.batch_mass_spin = QSpinBox()
        self.batch_mass_spin.setRange(50, 500)
        self.batch_mass_spin.setValue(100)
        self.batch_mass_spin.setSuffix(' g')
        self.batch_mass_spin.valueChanged.connect(self._on_batch_mass_changed)
        mass_row.addWidget(self.batch_mass_spin)
        mass_row.addStretch()
        settings_layout.addLayout(mass_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(QApplication.translate('Label', 'Batch preset:')))
        self.batch_preset_combo = QComboBox()
        for mass_g in _BATCH_PRESET_MASSES:
            self.batch_preset_combo.addItem(f'{mass_g} g', mass_g)
        self.batch_preset_combo.currentIndexChanged.connect(self._on_batch_preset_changed)
        preset_row.addWidget(self.batch_preset_combo)
        preset_row.addWidget(QLabel(QApplication.translate('Label', 'Goal:')))
        self.batch_goal_combo = QComboBox()
        self.batch_goal_combo.addItem(QApplication.translate('ComboBox', 'Safety-first'), 'safe')
        self.batch_goal_combo.addItem(QApplication.translate('ComboBox', 'Balanced'), 'balanced')
        self.batch_goal_combo.addItem(QApplication.translate('ComboBox', 'Precision tracking'), 'precision')
        self.batch_goal_combo.setCurrentIndex(1)
        self.batch_goal_combo.currentIndexChanged.connect(self._on_batch_goal_changed)
        preset_row.addWidget(self.batch_goal_combo)
        self.apply_preset_button = QPushButton(QApplication.translate('Button', 'Apply Preset'))
        self.apply_preset_button.clicked.connect(self._on_apply_batch_preset)
        preset_row.addWidget(self.apply_preset_button)
        preset_row.addStretch()
        settings_layout.addLayout(preset_row)

        self.batch_preset_summary_label = QLabel('')
        self.batch_preset_summary_label.setWordWrap(True)
        settings_layout.addWidget(self.batch_preset_summary_label)

        # Fan strategy
        fan_row = QHBoxLayout()
        fan_row.addWidget(QLabel(QApplication.translate('Label', 'Fan strategy:')))
        self.fan_strategy_combo = QComboBox()
        self.fan_strategy_combo.addItem(QApplication.translate('ComboBox', 'Constant'))
        self.fan_strategy_combo.addItem(QApplication.translate('ComboBox', 'Ramp'))
        self.fan_strategy_combo.currentIndexChanged.connect(self._on_fan_strategy_changed)
        fan_row.addWidget(self.fan_strategy_combo)
        fan_row.addStretch()
        settings_layout.addLayout(fan_row)

        # Fan constant value
        self.fan_constant_row = QHBoxLayout()
        self.fan_constant_row.addWidget(QLabel(QApplication.translate('Label', 'Fan %:')))
        self.fan_constant_spin = QSpinBox()
        self.fan_constant_spin.setRange(0, 100)
        self.fan_constant_spin.setValue(30)
        self.fan_constant_spin.setSuffix(' %')
        self.fan_constant_row.addWidget(self.fan_constant_spin)
        self.fan_constant_row.addStretch()

        self.fan_constant_widget = QWidget()
        self.fan_constant_widget.setLayout(self.fan_constant_row)
        settings_layout.addWidget(self.fan_constant_widget)

        # Fan ramp values
        self.fan_ramp_row = QHBoxLayout()
        self.fan_ramp_row.addWidget(QLabel(QApplication.translate('Label', 'Start %:')))
        self.fan_ramp_start_spin = QSpinBox()
        self.fan_ramp_start_spin.setRange(0, 100)
        self.fan_ramp_start_spin.setValue(20)
        self.fan_ramp_start_spin.setSuffix(' %')
        self.fan_ramp_row.addWidget(self.fan_ramp_start_spin)
        self.fan_ramp_row.addWidget(QLabel(QApplication.translate('Label', 'End %:')))
        self.fan_ramp_end_spin = QSpinBox()
        self.fan_ramp_end_spin.setRange(0, 100)
        self.fan_ramp_end_spin.setValue(60)
        self.fan_ramp_end_spin.setSuffix(' %')
        self.fan_ramp_row.addWidget(self.fan_ramp_end_spin)
        self.fan_ramp_row.addStretch()

        self.fan_ramp_widget = QWidget()
        self.fan_ramp_widget.setLayout(self.fan_ramp_row)
        self.fan_ramp_widget.setVisible(False)
        settings_layout.addWidget(self.fan_ramp_widget)

        # Drum strategy
        drum_row = QHBoxLayout()
        drum_row.addWidget(QLabel(QApplication.translate('Label', 'Drum strategy:')))
        self.drum_strategy_combo = QComboBox()
        self.drum_strategy_combo.addItem(QApplication.translate('ComboBox', 'Off'))
        self.drum_strategy_combo.addItem(QApplication.translate('ComboBox', 'Constant'))
        self.drum_strategy_combo.addItem(QApplication.translate('ComboBox', 'Ramp'))
        self.drum_strategy_combo.currentIndexChanged.connect(self._on_drum_strategy_changed)
        drum_row.addWidget(self.drum_strategy_combo)
        drum_row.addStretch()
        settings_layout.addLayout(drum_row)

        # Drum constant value
        self.drum_constant_row = QHBoxLayout()
        self.drum_constant_row.addWidget(QLabel(QApplication.translate('Label', 'Drum %:')))
        self.drum_constant_spin = QSpinBox()
        self.drum_constant_spin.setRange(0, 100)
        self.drum_constant_spin.setValue(60)
        self.drum_constant_spin.setSuffix(' %')
        self.drum_constant_row.addWidget(self.drum_constant_spin)
        self.drum_constant_row.addStretch()
        self.drum_constant_widget = QWidget()
        self.drum_constant_widget.setLayout(self.drum_constant_row)
        self.drum_constant_widget.setVisible(False)
        settings_layout.addWidget(self.drum_constant_widget)

        # Drum ramp values
        self.drum_ramp_row = QHBoxLayout()
        self.drum_ramp_row.addWidget(QLabel(QApplication.translate('Label', 'Start %:')))
        self.drum_ramp_start_spin = QSpinBox()
        self.drum_ramp_start_spin.setRange(0, 100)
        self.drum_ramp_start_spin.setValue(55)
        self.drum_ramp_start_spin.setSuffix(' %')
        self.drum_ramp_row.addWidget(self.drum_ramp_start_spin)
        self.drum_ramp_row.addWidget(QLabel(QApplication.translate('Label', 'End %:')))
        self.drum_ramp_end_spin = QSpinBox()
        self.drum_ramp_end_spin.setRange(0, 100)
        self.drum_ramp_end_spin.setValue(75)
        self.drum_ramp_end_spin.setSuffix(' %')
        self.drum_ramp_row.addWidget(self.drum_ramp_end_spin)
        self.drum_ramp_row.addStretch()
        self.drum_ramp_widget = QWidget()
        self.drum_ramp_widget.setLayout(self.drum_ramp_row)
        self.drum_ramp_widget.setVisible(False)
        settings_layout.addWidget(self.drum_ramp_widget)

        # Control interval
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel(QApplication.translate('Label', 'Control interval:')))
        self.interval_combo = QComboBox()
        self.interval_combo.addItem(QApplication.translate('ComboBox', '5 seconds'), 5)
        self.interval_combo.addItem(QApplication.translate('ComboBox', '10 seconds'), 10)
        self.interval_combo.addItem(QApplication.translate('ComboBox', '15 seconds'), 15)
        self.interval_combo.setCurrentIndex(1)  # default 10 seconds
        interval_row.addWidget(self.interval_combo)
        interval_row.addStretch()
        settings_layout.addLayout(interval_row)

        trigger_row = QHBoxLayout()
        trigger_row.addWidget(QLabel(QApplication.translate('Label', 'Trigger mode:')))
        self.trigger_mode_combo = QComboBox()
        self.trigger_mode_combo.addItem(QApplication.translate('ComboBox', 'Time from CHARGE'), 'time')
        self.trigger_mode_combo.addItem(QApplication.translate('ComboBox', 'BT temperature'), 'bt')
        self.trigger_mode_combo.currentIndexChanged.connect(self._on_trigger_mode_changed)
        trigger_row.addWidget(self.trigger_mode_combo)
        trigger_row.addStretch()
        settings_layout.addLayout(trigger_row)

        self.bt_trigger_hardening_row = QHBoxLayout()
        self.bt_trigger_hardening_row.addWidget(QLabel(QApplication.translate('Label', 'BT hysteresis (C):')))
        self.bt_hysteresis_spin = QDoubleSpinBox()
        self.bt_hysteresis_spin.setRange(0.0, 20.0)
        self.bt_hysteresis_spin.setDecimals(1)
        self.bt_hysteresis_spin.setSingleStep(0.5)
        self.bt_hysteresis_spin.setValue(1.0)
        self.bt_trigger_hardening_row.addWidget(self.bt_hysteresis_spin)
        self.bt_trigger_hardening_row.addWidget(QLabel(QApplication.translate('Label', 'BT min gap (C):')))
        self.bt_min_gap_spin = QDoubleSpinBox()
        self.bt_min_gap_spin.setRange(0.0, 30.0)
        self.bt_min_gap_spin.setDecimals(1)
        self.bt_min_gap_spin.setSingleStep(0.5)
        self.bt_min_gap_spin.setValue(2.0)
        self.bt_trigger_hardening_row.addWidget(self.bt_min_gap_spin)
        self.bt_trigger_hardening_row.addStretch()
        self.bt_trigger_hardening_widget = QWidget()
        self.bt_trigger_hardening_widget.setLayout(self.bt_trigger_hardening_row)
        self.bt_trigger_hardening_widget.setVisible(False)
        settings_layout.addWidget(self.bt_trigger_hardening_widget)

        deadband_row = QHBoxLayout()
        deadband_row.addWidget(QLabel(QApplication.translate('Label', 'Min control change:')))
        self.min_delta_spin = QSpinBox()
        self.min_delta_spin.setRange(1, 20)
        self.min_delta_spin.setValue(2)
        self.min_delta_spin.setSuffix(' %')
        deadband_row.addWidget(self.min_delta_spin)
        deadband_row.addStretch()
        settings_layout.addLayout(deadband_row)

        options_row = QHBoxLayout()
        self.milestone_checkbox = QCheckBox(QApplication.translate('CheckBox', 'Add milestone popups'))
        self.milestone_checkbox.setChecked(True)
        options_row.addWidget(self.milestone_checkbox)
        self.safety_checkbox = QCheckBox(QApplication.translate('CheckBox', 'Add safety ceilings'))
        self.safety_checkbox.setChecked(True)
        options_row.addWidget(self.safety_checkbox)
        self.validate_schedule_checkbox = QCheckBox(
            QApplication.translate('CheckBox', 'Validate schedule before export')
        )
        self.validate_schedule_checkbox.setChecked(True)
        options_row.addWidget(self.validate_schedule_checkbox)
        options_row.addStretch()
        settings_layout.addLayout(options_row)

        optimization_row = QHBoxLayout()
        self.optimize_actuators_checkbox = QCheckBox(
            QApplication.translate('CheckBox', 'Jointly optimize fan + drum')
        )
        self.optimize_actuators_checkbox.setChecked(True)
        optimization_row.addWidget(self.optimize_actuators_checkbox)
        optimization_row.addWidget(QLabel(QApplication.translate('Label', 'Passes:')))
        self.optimizer_passes_spin = QSpinBox()
        self.optimizer_passes_spin.setRange(1, 8)
        self.optimizer_passes_spin.setValue(3)
        optimization_row.addWidget(self.optimizer_passes_spin)
        optimization_row.addStretch()
        settings_layout.addLayout(optimization_row)

        safety_row = QHBoxLayout()
        safety_row.addWidget(QLabel(QApplication.translate('Label', 'BT max (C):')))
        self.bt_safety_spin = QSpinBox()
        self.bt_safety_spin.setRange(120, 260)
        self.bt_safety_spin.setValue(230)
        safety_row.addWidget(self.bt_safety_spin)
        safety_row.addWidget(QLabel(QApplication.translate('Label', 'ET max (C):')))
        self.et_safety_spin = QSpinBox()
        self.et_safety_spin.setRange(140, 320)
        self.et_safety_spin.setValue(260)
        safety_row.addWidget(self.et_safety_spin)
        safety_row.addWidget(QLabel(QApplication.translate('Label', 'RoR max (C/min):')))
        self.ror_safety_spin = QDoubleSpinBox()
        self.ror_safety_spin.setRange(5.0, 60.0)
        self.ror_safety_spin.setDecimals(1)
        self.ror_safety_spin.setSingleStep(1.0)
        self.ror_safety_spin.setValue(30.0)
        safety_row.addWidget(self.ror_safety_spin)
        safety_row.addStretch()
        settings_layout.addLayout(safety_row)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # ── Generate button ───────────────────────────────────────────
        gen_btn_row = QHBoxLayout()
        self.generate_button = QPushButton(QApplication.translate('Button', 'Generate'))
        self.generate_button.clicked.connect(self._on_generate)
        gen_btn_row.addWidget(self.generate_button)
        gen_btn_row.addStretch()
        layout.addLayout(gen_btn_row)

        # ── Results ───────────────────────────────────────────────────
        self.generate_results_text = QTextEdit()
        self.generate_results_text.setReadOnly(True)
        self.generate_results_text.setMaximumHeight(280)
        layout.addWidget(self.generate_results_text)

        # Ensure trigger-dependent controls reflect initial selection.
        self._on_trigger_mode_changed(self.trigger_mode_combo.currentIndex())
        self._sync_batch_preset_to_mass(self.batch_mass_spin.value())
        self._update_batch_preset_summary()
        self._refresh_target_curve_summary(show_errors=False)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    # ===================================================================
    # Tab 3: Export
    # ===================================================================

    def _build_export_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        # ── Export buttons ────────────────────────────────────────────
        export_group = QGroupBox(QApplication.translate('GroupBox', 'Export Alarm Schedule'))
        export_layout = QVBoxLayout()

        self.save_alrm_button = QPushButton(QApplication.translate('Button', 'Save as .alrm'))
        self.save_alrm_button.clicked.connect(self._on_save_alrm)
        export_layout.addWidget(self.save_alrm_button)

        self.apply_alarms_button = QPushButton(QApplication.translate('Button', 'Apply to Alarms'))
        self.apply_alarms_button.clicked.connect(self._on_apply_alarms)
        export_layout.addWidget(self.apply_alarms_button)

        self.save_interop_json_button = QPushButton(QApplication.translate('Button', 'Save Interop JSON'))
        self.save_interop_json_button.clicked.connect(self._on_save_interop_json)
        export_layout.addWidget(self.save_interop_json_button)

        self.save_hibean_csv_button = QPushButton(QApplication.translate('Button', 'Save HiBean CSV'))
        self.save_hibean_csv_button.clicked.connect(self._on_save_hibean_csv)
        export_layout.addWidget(self.save_hibean_csv_button)

        self.review_alarm_table_button = QPushButton(QApplication.translate('Button', 'Review/Edit Alarms'))
        self.review_alarm_table_button.clicked.connect(self._on_review_alarm_table)
        export_layout.addWidget(self.review_alarm_table_button)

        self.require_alarm_review_checkbox = QCheckBox(
            QApplication.translate('CheckBox', 'Require review before save/apply/store')
        )
        self.require_alarm_review_checkbox.setChecked(True)
        export_layout.addWidget(self.require_alarm_review_checkbox)

        # ── Store as alarm set ────────────────────────────────────────
        store_row = QHBoxLayout()
        self.store_alarm_set_button = QPushButton(QApplication.translate('Button', 'Store as Alarm Set'))
        self.store_alarm_set_button.clicked.connect(self._on_store_alarm_set)
        store_row.addWidget(self.store_alarm_set_button)

        store_row.addWidget(QLabel(QApplication.translate('Label', 'Slot:')))
        self.alarm_set_slot_spin = QSpinBox()
        self.alarm_set_slot_spin.setRange(0, 9)
        self.alarm_set_slot_spin.setValue(0)
        store_row.addWidget(self.alarm_set_slot_spin)
        store_row.addStretch()
        export_layout.addLayout(store_row)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    # ===================================================================
    # Button state management
    # ===================================================================

    def _update_button_states(self) -> None:
        """Enable / disable buttons based on current state."""
        has_profiles = len(self.calibration_data) > 0
        has_model = self.model is not None
        has_inversion = self.inversion_result is not None
        has_alarms = self.alarm_data is not None
        is_fitting = self._fit_worker is not None and self._fit_worker.isRunning()

        self.fit_button.setEnabled(has_profiles and not is_fitting)
        self.clear_profiles_button.setEnabled(has_profiles and not is_fitting)
        self.save_model_button.setEnabled(has_model)
        self.generate_button.setEnabled(has_model)
        self.save_alrm_button.setEnabled(has_alarms)
        self.apply_alarms_button.setEnabled(has_alarms)
        self.store_alarm_set_button.setEnabled(has_alarms)
        self.review_alarm_table_button.setEnabled(has_alarms)
        self.save_interop_json_button.setEnabled(has_inversion)
        self.save_hibean_csv_button.setEnabled(has_inversion)

    @staticmethod
    def _format_mmss(value_s: float | None) -> str:
        if value_s is None:
            return '--'
        total = max(0, int(round(value_s)))
        return f'{total // 60}:{total % 60:02d}'

    def _confirm_action_if_unsafe(self) -> bool:
        if not self.validate_schedule_checkbox.isChecked():
            return True
        if self.safety_validation is None:
            return True
        if self.safety_validation.is_safe:
            return True
        details = '\n'.join(self.safety_validation.failures) or QApplication.translate(
            'Message',
            'Schedule failed safety validation.',
        )
        reply = QMessageBox.warning(
            self,
            QApplication.translate('Message', 'Unsafe Schedule'),
            QApplication.translate(
                'Message',
                'Safety validation failed:\n{0}\n\nContinue anyway?',
            ).format(details),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # ===================================================================
    # Calibrate tab slots
    # ===================================================================

    @pyqtSlot()
    def _on_load_profiles(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            QApplication.translate('Dialog', 'Select Calibration Profiles'),
            '',
            QApplication.translate('Dialog', 'Artisan Profiles (*.alog);;All Files (*)'),
        )
        if not files:
            return

        files, limited = _cap_calibration_file_selection(len(self.calibration_data), files)
        if not files:
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Maximum of 3 calibration profiles already loaded')
            )
            return
        if limited:
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Limited to 3 calibration profiles'))

        for fp in files:
            try:
                calib = parse_alog_profile(fp)
                self.calibration_data.append(calib)
                duration_s = float(calib.time[-1] - calib.time[0])
                duration_m = int(duration_s) // 60
                duration_sec = int(duration_s) % 60
                mass_g = calib.batch_mass_kg * 1000.0
                basename = os.path.basename(fp)
                self.profile_list.addItem(
                    f'{basename}  ({duration_m}:{duration_sec:02d}, {mass_g:.0f} g)'
                )
            except Exception as e:  # pylint: disable=broad-except
                _log.exception('Failed to parse profile: %s', fp)
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Failed to parse {0}: {1}').format(
                        os.path.basename(fp), str(e)
                    )
                )

        self._update_button_states()
        self.aw.sendmessage(
            QApplication.translate('StatusBar', '{0} calibration profile(s) loaded').format(
                len(self.calibration_data)
            )
        )

    @pyqtSlot()
    def _on_clear_profiles(self) -> None:
        self.calibration_data.clear()
        self.profile_list.clear()
        self._update_button_states()
        self.aw.sendmessage(QApplication.translate('StatusBar', 'Calibration profiles cleared'))

    @pyqtSlot()
    def _on_fit_model(self) -> None:
        if not self.calibration_data:
            return

        self.fit_button.setEnabled(False)
        self.fit_progress.setVisible(True)
        self.fit_progress.setRange(0, 0)  # indeterminate
        self.fit_results_text.clear()
        self.aw.sendmessage(QApplication.translate('StatusBar', 'Fitting thermal model...'))

        self._fit_worker = FitWorker(self.calibration_data, parent=self)
        self._fit_worker.progress.connect(self._on_fit_progress)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.start()

    @pyqtSlot(int, float)
    def _on_fit_progress(self, iteration: int, _cost: float) -> None:
        # Switch to determinate mode after first callback
        if self.fit_progress.maximum() == 0:
            self.fit_progress.setRange(0, 100)
        # Map iteration to 0-100 range (DE typically runs ~100-1000 iterations)
        pct = min(iteration, 100)
        self.fit_progress.setValue(pct)

    @pyqtSlot(object)
    def _on_fit_finished(self, result: FitResult) -> None:
        self.fit_result = result
        self.model = KaleidoThermalModel(result.params)

        self.fit_progress.setRange(0, 100)
        self.fit_progress.setValue(100)

        # Build results text
        lines: list[str] = []
        lines.append(f'RMSE: {result.rmse:.3f} C')
        lines.append(f'R-squared: {result.r_squared:.6f}')
        lines.append(f'Max error: {result.max_error:.3f} C')
        lines.append(f'Converged: {result.converged}')
        lines.append(f'Message: {result.message}')
        lines.append('')
        for i, roast_rmse in enumerate(result.per_roast_rmse):
            lines.append(f'Roast {i + 1} RMSE: {roast_rmse:.3f} C')

        self.fit_results_text.setPlainText('\n'.join(lines))

        self._update_button_states()
        self.aw.sendmessage(
            QApplication.translate('StatusBar', 'Model fit complete: RMSE={0:.3f} C, R2={1:.6f}').format(
                result.rmse, result.r_squared
            )
        )

    @pyqtSlot(str)
    def _on_fit_error(self, error_msg: str) -> None:
        self.fit_progress.setVisible(False)
        self.fit_results_text.setPlainText(f'Fitting failed:\n{error_msg}')
        self._update_button_states()
        self.aw.sendmessage(
            QApplication.translate('StatusBar', 'Model fitting failed: {0}').format(error_msg)
        )

    @pyqtSlot()
    def _on_save_model(self) -> None:
        if self.model is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            QApplication.translate('Dialog', 'Save Thermal Model'),
            'thermal_model.json',
            QApplication.translate('Dialog', 'JSON Files (*.json);;All Files (*)'),
        )
        if filepath:
            try:
                self.model.params.save(filepath)
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Model saved to {0}').format(filepath)
                )
            except Exception as e:  # pylint: disable=broad-except
                _log.exception('Failed to save model')
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Failed to save model: {0}').format(str(e))
                )

    @pyqtSlot()
    def _on_load_model(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            QApplication.translate('Dialog', 'Load Thermal Model'),
            '',
            QApplication.translate('Dialog', 'JSON Files (*.json);;All Files (*)'),
        )
        if filepath:
            try:
                params = ThermalModelParams.load(filepath)
                self.model = KaleidoThermalModel(params)
                self.fit_results_text.setPlainText(
                    QApplication.translate('StatusBar', 'Model loaded from {0}').format(filepath)
                )
                self._update_button_states()
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Model loaded from {0}').format(filepath)
                )
            except Exception as e:  # pylint: disable=broad-except
                _log.exception('Failed to load model')
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Failed to load model: {0}').format(str(e))
                )

    # ===================================================================
    # Generate tab slots
    # ===================================================================

    @staticmethod
    def _goal_label(goal: PlannerGoal) -> str:
        if goal == 'safe':
            return QApplication.translate('Label', 'Safety-first')
        if goal == 'precision':
            return QApplication.translate('Label', 'Precision tracking')
        return QApplication.translate('Label', 'Balanced')

    def _current_planner_goal(self) -> PlannerGoal:
        goal_token = str(self.batch_goal_combo.currentData())
        if goal_token == 'safe':
            return 'safe'
        if goal_token == 'precision':
            return 'precision'
        return 'balanced'

    def _sync_batch_preset_to_mass(self, mass_g: int) -> None:
        nearest_mass = min(_BATCH_PRESET_MASSES, key=lambda m: abs(m - int(mass_g)))
        idx = self.batch_preset_combo.findData(nearest_mass)
        if idx >= 0 and idx != self.batch_preset_combo.currentIndex():
            self.batch_preset_combo.blockSignals(True)
            self.batch_preset_combo.setCurrentIndex(idx)
            self.batch_preset_combo.blockSignals(False)

    @pyqtSlot(int)
    def _on_batch_mass_changed(self, value: int) -> None:
        self._sync_batch_preset_to_mass(value)
        self._update_batch_preset_summary()

    @pyqtSlot(int)
    def _on_batch_preset_changed(self, _index: int) -> None:
        mass = int(self.batch_preset_combo.currentData() or self.batch_mass_spin.value())
        if mass != self.batch_mass_spin.value():
            self.batch_mass_spin.setValue(mass)
        else:
            self._update_batch_preset_summary()

    @pyqtSlot(int)
    def _on_batch_goal_changed(self, _index: int) -> None:
        self._update_batch_preset_summary()

    @pyqtSlot()
    def _on_apply_batch_preset(self) -> None:
        preset_mass = int(self.batch_preset_combo.currentData() or self.batch_mass_spin.value())
        goal = self._current_planner_goal()
        preset = resolve_batch_planner_preset(preset_mass, goal)

        self.batch_mass_spin.setValue(int(preset.mass_g))
        self.fan_strategy_combo.setCurrentIndex(0)
        self.fan_constant_spin.setValue(int(preset.fan_pct))
        self.drum_strategy_combo.setCurrentIndex(1)
        self.drum_constant_spin.setValue(int(preset.drum_pct))
        self.optimize_actuators_checkbox.setChecked(True)

        interval_idx = self.interval_combo.findData(int(preset.interval_s))
        if interval_idx >= 0:
            self.interval_combo.setCurrentIndex(interval_idx)
        trigger_idx = self.trigger_mode_combo.findData(str(preset.trigger_mode))
        if trigger_idx >= 0:
            self.trigger_mode_combo.setCurrentIndex(trigger_idx)

        self.min_delta_spin.setValue(int(preset.min_delta_pct))
        self.optimizer_passes_spin.setValue(int(preset.optimizer_passes))
        self.bt_safety_spin.setValue(int(preset.bt_max_c))
        self.et_safety_spin.setValue(int(preset.et_max_c))
        self.ror_safety_spin.setValue(float(preset.max_ror_c_per_min))
        self.safety_checkbox.setChecked(True)
        self.validate_schedule_checkbox.setChecked(True)
        self._update_batch_preset_summary()

        self.aw.sendmessage(
            QApplication.translate(
                'StatusBar',
                'Applied {0} preset ({1} g baseline)',
            ).format(self._goal_label(goal), preset.mass_g)
        )

    def _update_batch_preset_summary(self) -> None:
        goal = self._current_planner_goal()
        current_mass = int(self.batch_mass_spin.value())
        preset = resolve_batch_planner_preset(current_mass, goal)
        nearest_info = (
            QApplication.translate('Label', 'nearest preset {0} g')
            if current_mass != preset.mass_g
            else QApplication.translate('Label', 'exact preset')
        )
        summary = QApplication.translate(
            'Label',
            '{0}: fan {1}% / drum {2}% / interval {3}s / min change {4}% / passes {5} / safety BT-ET-RoR {6}C-{7}C-{8:.1f} ({9})',
        ).format(
            self._goal_label(goal),
            preset.fan_pct,
            preset.drum_pct,
            preset.interval_s,
            preset.min_delta_pct,
            preset.optimizer_passes,
            preset.bt_max_c,
            preset.et_max_c,
            preset.max_ror_c_per_min,
            nearest_info.format(preset.mass_g),
        )
        self.batch_preset_summary_label.setText(summary)

    def _load_selected_target_curve(
        self,
        *,
        show_errors: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, TargetCurveStats, int | None, str] | None:
        target_time: np.ndarray
        target_bt: np.ndarray
        source_label: str
        profile_mass_g: int | None = None

        if self.target_source_combo.currentIndex() == 0:
            timeB = self.aw.qmc.timeB
            temp2B = self.aw.qmc.temp2B
            if not timeB or not temp2B or len(timeB) < 2:
                if show_errors:
                    self.aw.sendmessage(
                        QApplication.translate('StatusBar', 'No background profile loaded')
                    )
                return None
            target_time = np.asarray(timeB, dtype=np.float64)
            target_bt = np.asarray(temp2B, dtype=np.float64)
            mode = str(getattr(self.aw.qmc, 'mode', 'C')).upper()
            if mode == 'F':
                valid = np.isfinite(target_bt) & (target_bt != -1.0)
                target_bt[valid] = (target_bt[valid] - 32.0) * (5.0 / 9.0)
            source_label = QApplication.translate('Label', 'Background profile')
        else:
            filepath = self.target_source_combo.itemData(1)
            if not filepath:
                if show_errors:
                    self.aw.sendmessage(
                        QApplication.translate('StatusBar', 'No target profile file selected')
                    )
                return None
            try:
                target = parse_target_profile(str(filepath))
            except Exception as e:  # pylint: disable=broad-except
                _log.exception('Failed to parse target profile')
                if show_errors:
                    self.aw.sendmessage(
                        QApplication.translate(
                            'StatusBar',
                            'Failed to parse target profile: {0}',
                        ).format(str(e))
                    )
                return None
            target_time = np.asarray(target.time, dtype=np.float64)
            target_bt = np.asarray(target.bt, dtype=np.float64)
            if target.batch_mass_kg > 0.0:
                profile_mass_g = int(round(target.batch_mass_kg * 1000.0))
            source_label = QApplication.translate('Label', 'File: {0}').format(
                os.path.basename(str(filepath))
            )

        try:
            clean_time, clean_bt, stats = normalize_target_curve(target_time, target_bt)
            return clean_time, clean_bt, stats, profile_mass_g, source_label
        except Exception as e:  # pylint: disable=broad-except
            if show_errors:
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Target curve is not usable: {0}').format(str(e))
                )
            return None

    def _refresh_target_curve_summary(self, *, show_errors: bool = False) -> bool:
        loaded = self._load_selected_target_curve(show_errors=show_errors)
        if loaded is None:
            self.target_curve_summary_label.setText(
                QApplication.translate('Label', 'Target summary: not available')
            )
            return False

        _, _, stats, profile_mass_g, source_label = loaded
        mass_info = (
            QApplication.translate('Label', ' | profile mass {0} g').format(profile_mass_g)
            if profile_mass_g is not None
            else ''
        )
        self.target_curve_summary_label.setText(
            QApplication.translate(
                'Label',
                'Target summary: {0} | {1} points | {2} | BT {3:.1f}-{4:.1f} C | start/end {5:.1f}/{6:.1f} C{7}',
            ).format(
                source_label,
                stats.sample_count,
                self._format_mmss(stats.duration_s),
                stats.bt_min_c,
                stats.bt_max_c,
                stats.bt_start_c,
                stats.bt_end_c,
                mass_info,
            )
        )
        return True

    @pyqtSlot()
    def _on_inspect_target_curve(self) -> None:
        if self._refresh_target_curve_summary(show_errors=True):
            self.aw.sendmessage(self.target_curve_summary_label.text())

    @pyqtSlot(int)
    def _on_target_source_changed(self, index: int) -> None:
        if index == 1:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                QApplication.translate('Dialog', 'Select Target Profile'),
                '',
                QApplication.translate('Dialog', 'Artisan Profiles (*.alog);;All Files (*)'),
            )
            if not filepath:
                self.target_source_combo.blockSignals(True)
                self.target_source_combo.setCurrentIndex(0)
                self.target_source_combo.blockSignals(False)
                self._refresh_target_curve_summary(show_errors=False)
                return
            self.target_source_combo.setItemData(1, filepath)
            self.target_source_combo.setItemText(
                1,
                QApplication.translate('ComboBox', 'File: {0}').format(os.path.basename(filepath)),
            )

        loaded = self._load_selected_target_curve(show_errors=False)
        if loaded is not None:
            _, _, _, profile_mass_g, _ = loaded
            if index == 1 and profile_mass_g is not None:
                clamped_mass = max(
                    self.batch_mass_spin.minimum(),
                    min(int(profile_mass_g), self.batch_mass_spin.maximum()),
                )
                self.batch_mass_spin.setValue(clamped_mass)
        self._refresh_target_curve_summary(show_errors=False)

    @pyqtSlot(int)
    def _on_fan_strategy_changed(self, index: int) -> None:
        self.fan_constant_widget.setVisible(index == 0)
        self.fan_ramp_widget.setVisible(index == 1)

    @pyqtSlot(int)
    def _on_drum_strategy_changed(self, index: int) -> None:
        # 0=Off, 1=Constant, 2=Ramp
        self.drum_constant_widget.setVisible(index == 1)
        self.drum_ramp_widget.setVisible(index == 2)

    @pyqtSlot(int)
    def _on_trigger_mode_changed(self, index: int) -> None:
        # 1 = BT-trigger mode
        self.bt_trigger_hardening_widget.setVisible(index == 1)

    def _review_alarm_table(self) -> bool:
        if self.alarm_data is None:
            return False
        dialog = AlarmReviewDialog(self, self.alarm_data, self.alarm_flavor_notes)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return False
        if dialog.updated_alarm_data is not None:
            self.alarm_data = dialog.updated_alarm_data
        updated_notes = list(dialog.updated_flavor_notes)
        if self.alarm_data is not None:
            _learn_flavor_impact_notes(
                self.alarm_data,
                updated_notes,
                baseline_notes=self.alarm_flavor_guess_notes,
            )
        self.alarm_flavor_notes = updated_notes
        self.alarm_flavor_guess_notes = list(updated_notes)
        self.alarm_review_required = False
        self._update_button_states()
        self.aw.sendmessage(
            QApplication.translate(
                'StatusBar',
                'Alarm review applied ({0} rows)',
            ).format(self.alarm_data.alarm_count())
        )
        return True

    def _ensure_alarm_review_if_required(self) -> bool:
        if self.alarm_data is None:
            return False
        if not self.require_alarm_review_checkbox.isChecked():
            return True
        if not self.alarm_review_required:
            return True
        if self._review_alarm_table():
            return True
        self.aw.sendmessage(
            QApplication.translate(
                'StatusBar',
                'Action cancelled: complete alarm review first',
            )
        )
        return False

    @pyqtSlot()
    def _on_review_alarm_table(self) -> None:
        if self.alarm_data is None:
            return
        self._review_alarm_table()

    @pyqtSlot()
    def _on_generate(self) -> None:
        if self.model is None:
            return
        self.safety_validation = None
        self.quality_report = None

        loaded_target = self._load_selected_target_curve(show_errors=True)
        if loaded_target is None:
            return
        target_time, target_bt, target_stats, _, target_source_label = loaded_target

        # ── Build fan schedule ────────────────────────────────────────
        mass_kg = self.batch_mass_spin.value() / 1000.0

        if self.fan_strategy_combo.currentIndex() == 0:
            # Constant
            fan_schedule: np.ndarray | float = float(self.fan_constant_spin.value())
        else:
            # Ramp
            start_pct = float(self.fan_ramp_start_spin.value())
            end_pct = float(self.fan_ramp_end_spin.value())
            fan_schedule = np.linspace(start_pct, end_pct, len(target_time))

        # ── Build drum schedule ───────────────────────────────────────
        drum_schedule: np.ndarray | float | None
        drum_mode = self.drum_strategy_combo.currentIndex()
        if drum_mode == 0:
            drum_schedule = None
        elif drum_mode == 1:
            drum_schedule = float(self.drum_constant_spin.value())
        else:
            drum_schedule = np.linspace(
                float(self.drum_ramp_start_spin.value()),
                float(self.drum_ramp_end_spin.value()),
                len(target_time),
            )

        trigger_mode = str(self.trigger_mode_combo.currentData())
        min_delta_pct = int(self.min_delta_spin.value())
        bt_hysteresis_c = float(self.bt_hysteresis_spin.value())
        bt_min_gap_c = float(self.bt_min_gap_spin.value())
        add_milestones = self.milestone_checkbox.isChecked()
        add_safety = self.safety_checkbox.isChecked()
        validate_schedule_p = self.validate_schedule_checkbox.isChecked()
        optimize_actuators = self.optimize_actuators_checkbox.isChecked()
        optimizer_passes = int(self.optimizer_passes_spin.value())
        ror_limit = float(self.ror_safety_spin.value())

        # ── Run inversion ─────────────────────────────────────────────
        try:
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Generating schedule...'))
            inv_result = invert_model(
                model=self.model,
                target_time=target_time,
                target_bt=target_bt,
                mass_kg=mass_kg,
                fan_schedule=fan_schedule,
                drum_schedule=drum_schedule,
                optimize_actuators=optimize_actuators,
                optimizer_iterations=optimizer_passes,
            )

            # Resample to chosen interval
            interval_s = float(self.interval_combo.currentData())
            resampled = inv_result.resample_to_interval(interval_s)
            self.inversion_result = resampled
            self.generated_trigger_mode = ('bt' if trigger_mode == 'bt' else 'time')

            milestone_offsets: dict[str, float] | None = None
            if add_milestones:
                milestone_offsets = {}
                if resampled.yellowing_time is not None:
                    milestone_offsets['Yellowing'] = resampled.yellowing_time
                if resampled.first_crack_time is not None:
                    milestone_offsets['First Crack'] = resampled.first_crack_time
                milestone_offsets['Drop'] = resampled.drop_time

            # Generate alarm table
            self.alarm_data = generate_alarm_table(
                time=resampled.time,
                heater_pct=resampled.heater_pct,
                fan_pct=resampled.fan_pct,
                exo_warning_time=resampled.exo_warning_time,
                drum_pct=resampled.drum_pct,
                min_delta_pct=min_delta_pct,
                trigger_mode=('bt' if trigger_mode == 'bt' else 'time'),
                bt_profile=(resampled.predicted_bt if trigger_mode == 'bt' else None),
                bt_hysteresis_c=bt_hysteresis_c,
                bt_min_gap_c=bt_min_gap_c,
                milestone_offsets=milestone_offsets,
                bt_safety_ceiling=(float(self.bt_safety_spin.value()) if add_safety else None),
                et_safety_ceiling=(float(self.et_safety_spin.value()) if add_safety else None),
            )
            self.alarm_flavor_guess_notes = _guess_flavor_impact_notes(self.alarm_data)
            self.alarm_flavor_notes = list(self.alarm_flavor_guess_notes)
            self.alarm_review_required = True

            self.safety_validation = None
            if validate_schedule_p:
                self.safety_validation = validate_schedule(
                    self.model,
                    resampled,
                    bt_limit_c=(float(self.bt_safety_spin.value()) if add_safety else None),
                    et_limit_c=(float(self.et_safety_spin.value()) if add_safety else None),
                    max_ror_limit_c_per_min=ror_limit,
                )

            self.quality_report = build_quality_report(
                target_time=np.asarray(target_time, dtype=np.float64),
                target_bt=np.asarray(target_bt, dtype=np.float64),
                inversion=resampled,
                control_change_count=self.alarm_data.control_change_count(),
                safety=self.safety_validation,
            )

            # Show results
            desc = generate_schedule_description(self.alarm_data)
            lines: list[str] = []
            goal = self._current_planner_goal()
            preset = resolve_batch_planner_preset(self.batch_mass_spin.value(), goal)
            lines.append(
                f'Target: {target_source_label} | {target_stats.sample_count} pts | '
                f'{self._format_mmss(target_stats.duration_s)} | BT {target_stats.bt_min_c:.1f}-{target_stats.bt_max_c:.1f} C'
            )
            lines.append(
                f'Planner goal: {self._goal_label(goal)} (nearest baseline {preset.mass_g} g)'
            )
            lines.append(f'Alarm count: {self.alarm_data.alarm_count()}')
            lines.append(f'Max tracking error: {resampled.max_tracking_error:.2f} C')
            lines.append(f'RMSE: {resampled.rmse:.2f} C')
            if resampled.objective_score is not None:
                lines.append(f'Objective score: {resampled.objective_score:.3f}')
            lines.append(
                f'Trigger mode: {"BT temperature" if trigger_mode == "bt" else "Time from CHARGE"}'
            )
            lines.append(f'Min control change: {min_delta_pct}%')
            if trigger_mode == 'bt':
                lines.append(f'BT hysteresis/min gap: {bt_hysteresis_c:.1f}C / {bt_min_gap_c:.1f}C')
            lines.append(f'Joint fan+drum optimization: {"on" if optimize_actuators else "off"}')
            lines.append(f'Yellowing estimate: {self._format_mmss(resampled.yellowing_time)}')
            lines.append(f'First crack estimate: {self._format_mmss(resampled.first_crack_time)}')
            lines.append(f'Drop estimate: {self._format_mmss(resampled.drop_time)}')
            if resampled.dtr_percent is not None:
                lines.append(f'DTR estimate: {resampled.dtr_percent:.1f}%')
            if self.safety_validation is not None:
                lines.extend(self.safety_validation.summary_lines())
            if self.quality_report is not None:
                lines.extend(self.quality_report.summary_lines())
            lines.append(f'Schedule: {desc}')
            if self.require_alarm_review_checkbox.isChecked():
                lines.append('Review step: required before save/apply/store')
            self.generate_results_text.setPlainText('\n'.join(lines))

            self._update_button_states()
            safety_state = (
                'safe'
                if self.safety_validation is None or self.safety_validation.is_safe
                else 'unsafe'
            )
            self.aw.sendmessage(
                QApplication.translate(
                    'StatusBar',
                    'Schedule generated: {0} alarms, RMSE={1:.2f} C ({2}, {3})',
                ).format(
                    self.alarm_data.alarm_count(),
                    resampled.rmse,
                    'BT trigger' if trigger_mode == 'bt' else 'time trigger',
                    safety_state,
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.exception('Schedule generation failed')
            self.inversion_result = None
            self.alarm_data = None
            self.alarm_flavor_notes = []
            self.alarm_flavor_guess_notes = []
            self.alarm_review_required = False
            self.safety_validation = None
            self.quality_report = None
            self._update_button_states()
            self.generate_results_text.setPlainText(f'Generation failed:\n{e}')
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Schedule generation failed: {0}').format(str(e))
            )

    # ===================================================================
    # Export tab slots
    # ===================================================================

    @pyqtSlot()
    def _on_save_alrm(self) -> None:
        if self.alarm_data is None:
            return
        if not self._ensure_alarm_review_if_required():
            return
        if not self._confirm_action_if_unsafe():
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Export cancelled: schedule marked unsafe'))
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            QApplication.translate('Dialog', 'Save Alarm Schedule'),
            'thermal_schedule.alrm',
            QApplication.translate('Dialog', 'Alarm Files (*.alrm);;All Files (*)'),
        )
        if filepath:
            try:
                self.alarm_data.save_alrm(filepath)
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Alarm schedule saved to {0}').format(filepath)
                )
            except Exception as e:  # pylint: disable=broad-except
                _log.exception('Failed to save alarm file')
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Failed to save alarm file: {0}').format(str(e))
                )

    @pyqtSlot()
    def _on_save_interop_json(self) -> None:
        if self.inversion_result is None:
            return
        if not self._confirm_action_if_unsafe():
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Export cancelled: schedule marked unsafe'))
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            QApplication.translate('Dialog', 'Save Interop Schedule JSON'),
            'thermal_schedule_interop.json',
            QApplication.translate('Dialog', 'JSON Files (*.json);;All Files (*)'),
        )
        if not filepath:
            return
        try:
            schedule = schedule_from_inversion(
                self.inversion_result,
                trigger_mode=('bt' if self.generated_trigger_mode == 'bt' else 'time'),
                label='Thermal Model Control',
            )
            export_artisan_plan_json(filepath, schedule)
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Interop JSON saved to {0}').format(filepath)
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.exception('Failed to save interop JSON')
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Failed to save interop JSON: {0}').format(str(e))
            )

    @pyqtSlot()
    def _on_save_hibean_csv(self) -> None:
        if self.inversion_result is None:
            return
        if not self._confirm_action_if_unsafe():
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Export cancelled: schedule marked unsafe'))
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            QApplication.translate('Dialog', 'Save HiBean Replay CSV'),
            'thermal_schedule_hibean.csv',
            QApplication.translate('Dialog', 'CSV Files (*.csv);;All Files (*)'),
        )
        if not filepath:
            return
        try:
            schedule = schedule_from_inversion(
                self.inversion_result,
                trigger_mode=('bt' if self.generated_trigger_mode == 'bt' else 'time'),
                label='Thermal Model Control',
            )
            export_hibean_csv(filepath, schedule)
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'HiBean-style CSV saved to {0}').format(filepath)
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.exception('Failed to save HiBean CSV')
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Failed to save HiBean CSV: {0}').format(str(e))
            )

    @pyqtSlot()
    def _on_apply_alarms(self) -> None:
        if self.alarm_data is None:
            return
        if not self._ensure_alarm_review_if_required():
            return
        if not self._confirm_action_if_unsafe():
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Apply cancelled: schedule marked unsafe'))
            return
        ad = self.alarm_data
        qmc = self.aw.qmc
        qmc.alarmflag = list(ad.alarmflag)
        qmc.alarmguard = list(ad.alarmguard)
        qmc.alarmnegguard = list(ad.alarmnegguard)
        qmc.alarmtime = list(ad.alarmtime)
        qmc.alarmoffset = list(ad.alarmoffset)
        qmc.alarmsource = list(ad.alarmsource)
        qmc.alarmcond = list(ad.alarmcond)
        qmc.alarmtemperature = list(ad.alarmtemperature)
        qmc.alarmaction = list(ad.alarmaction)
        qmc.alarmbeep = list(ad.alarmbeep)
        qmc.alarmstrings = list(ad.alarmstrings)
        qmc.alarmstate = [-1] * len(ad.alarmflag)
        self.aw.sendmessage(
            QApplication.translate('StatusBar', 'Applied {0} alarms to alarm table').format(
                len(ad.alarmflag)
            )
        )

    @pyqtSlot()
    def _on_store_alarm_set(self) -> None:
        if self.alarm_data is None:
            return
        if not self._ensure_alarm_review_if_required():
            return
        if not self._confirm_action_if_unsafe():
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Store cancelled: schedule marked unsafe')
            )
            return
        slot = self.alarm_set_slot_spin.value()
        ad = self.alarm_data
        qmc = self.aw.qmc

        from artisanlib.canvas import tgraphcanvas
        alarm_set = tgraphcanvas.makeAlarmSet(
            ad.label or f'Thermal Schedule (slot {slot})',
            list(ad.alarmflag),
            list(ad.alarmguard),
            list(ad.alarmnegguard),
            list(ad.alarmtime),
            list(ad.alarmoffset),
            list(ad.alarmsource),
            list(ad.alarmcond),
            list(ad.alarmtemperature),
            list(ad.alarmaction),
            list(ad.alarmbeep),
            list(ad.alarmstrings),
        )
        qmc.setAlarmSet(slot, alarm_set)
        self.aw.sendmessage(
            QApplication.translate('StatusBar', 'Stored {0} alarms in alarm set slot {1}').format(
                len(ad.alarmflag), slot
            )
        )
