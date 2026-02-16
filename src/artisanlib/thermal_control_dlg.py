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

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
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
from artisanlib.thermal_model import KaleidoThermalModel, ThermalModelParams
from artisanlib.thermal_model_fitting import FitResult, fit_model
from artisanlib.thermal_model_inversion import InversionResult, invert_model
from artisanlib.thermal_profile_parser import CalibrationData, parse_alog_profile, parse_target_profile

import numpy as np

from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    from artisanlib.main import ApplicationWindow  # pylint: disable=unused-import

_log: Final[logging.Logger] = logging.getLogger(__name__)


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
        source_row.addStretch()
        target_layout.addLayout(source_row)

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
        mass_row.addWidget(self.batch_mass_spin)
        mass_row.addStretch()
        settings_layout.addLayout(mass_row)

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
        self.generate_results_text.setMaximumHeight(120)
        layout.addWidget(self.generate_results_text)

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
        has_alarms = self.alarm_data is not None
        is_fitting = self._fit_worker is not None and self._fit_worker.isRunning()

        self.fit_button.setEnabled(has_profiles and not is_fitting)
        self.clear_profiles_button.setEnabled(has_profiles and not is_fitting)
        self.save_model_button.setEnabled(has_model)
        self.generate_button.setEnabled(has_model)
        self.save_alrm_button.setEnabled(has_alarms)
        self.apply_alarms_button.setEnabled(has_alarms)
        self.store_alarm_set_button.setEnabled(has_alarms)

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

        # Limit to 3 profiles
        if len(files) > 3:
            files = files[:3]
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

    @pyqtSlot(int)
    def _on_target_source_changed(self, index: int) -> None:
        # index 1 = "Load from file..."
        if index == 1:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                QApplication.translate('Dialog', 'Select Target Profile'),
                '',
                QApplication.translate('Dialog', 'Artisan Profiles (*.alog);;All Files (*)'),
            )
            if not filepath:
                # Reset to background profile if cancelled
                self.target_source_combo.blockSignals(True)
                self.target_source_combo.setCurrentIndex(0)
                self.target_source_combo.blockSignals(False)
                return
            # Store the loaded file path as item data
            self.target_source_combo.setItemData(1, filepath)
            basename = os.path.basename(filepath)
            self.target_source_combo.setItemText(
                1, QApplication.translate('ComboBox', 'File: {0}').format(basename)
            )
            try:
                target = parse_target_profile(str(filepath))
                if target.batch_mass_kg > 0.0:
                    mass_g = int(round(target.batch_mass_kg * 1000.0))
                    clamped_mass = max(
                        self.batch_mass_spin.minimum(),
                        min(mass_g, self.batch_mass_spin.maximum()),
                    )
                    self.batch_mass_spin.setValue(clamped_mass)
            except Exception:  # pylint: disable=broad-except
                # Profile parsing is re-validated on Generate.
                pass

    @pyqtSlot(int)
    def _on_fan_strategy_changed(self, index: int) -> None:
        self.fan_constant_widget.setVisible(index == 0)
        self.fan_ramp_widget.setVisible(index == 1)

    @pyqtSlot()
    def _on_generate(self) -> None:
        if self.model is None:
            return

        # ── Get target BT curve ──────────────────────────────────────
        target_time: np.ndarray | None = None
        target_bt: np.ndarray | None = None

        if self.target_source_combo.currentIndex() == 0:
            # Background profile
            timeB = self.aw.qmc.timeB
            temp2B = self.aw.qmc.temp2B
            if not timeB or not temp2B or len(timeB) < 2:
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'No background profile loaded')
                )
                return
            target_time = np.array(timeB, dtype=np.float64)
            target_bt = np.array(temp2B, dtype=np.float64)
        else:
            # Loaded file
            filepath = self.target_source_combo.itemData(1)
            if not filepath:
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'No target profile file selected')
                )
                return
            try:
                target = parse_target_profile(str(filepath))
                target_time = target.time
                target_bt = target.bt
            except Exception as e:  # pylint: disable=broad-except
                _log.exception('Failed to parse target profile')
                self.aw.sendmessage(
                    QApplication.translate('StatusBar', 'Failed to parse target profile: {0}').format(str(e))
                )
                return

        if target_time is None or target_bt is None:
            return

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

        # ── Run inversion ─────────────────────────────────────────────
        try:
            self.aw.sendmessage(QApplication.translate('StatusBar', 'Generating schedule...'))
            inv_result = invert_model(
                model=self.model,
                target_time=target_time,
                target_bt=target_bt,
                mass_kg=mass_kg,
                fan_schedule=fan_schedule,
            )

            # Resample to chosen interval
            interval_s = float(self.interval_combo.currentData())
            resampled = inv_result.resample_to_interval(interval_s)
            self.inversion_result = resampled

            # Generate alarm table
            self.alarm_data = generate_alarm_table(
                time=resampled.time,
                heater_pct=resampled.heater_pct,
                fan_pct=resampled.fan_pct,
                exo_warning_time=resampled.exo_warning_time,
            )

            # Show results
            desc = generate_schedule_description(self.alarm_data)
            lines: list[str] = []
            lines.append(f'Alarm count: {self.alarm_data.alarm_count()}')
            lines.append(f'Max tracking error: {resampled.max_tracking_error:.2f} C')
            lines.append(f'RMSE: {resampled.rmse:.2f} C')
            lines.append(f'Schedule: {desc}')
            self.generate_results_text.setPlainText('\n'.join(lines))

            self._update_button_states()
            self.aw.sendmessage(
                QApplication.translate('StatusBar', 'Schedule generated: {0} alarms, RMSE={1:.2f} C').format(
                    self.alarm_data.alarm_count(), resampled.rmse
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.exception('Schedule generation failed')
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
    def _on_apply_alarms(self) -> None:
        if self.alarm_data is None:
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
