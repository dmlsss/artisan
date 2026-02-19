#
# ABOUT
# Standalone CLI tool for the Kaleido thermal model system.
# Provides subcommands for fitting, profile generation, and forward
# simulation — no Artisan GUI dependency required.

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

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Final

import numpy as np

from artisanlib.thermal_model import (
    KaleidoThermalModel,
    PARAM_NAMES,
    ThermalModelParams,
)
from artisanlib.thermal_model_fitting import (
    FitResult,
    default_bounds,
    fit_model,
)
from artisanlib.thermal_model_inversion import invert_model
from artisanlib.thermal_profile_parser import parse_multiple_profiles, parse_target_profile
from artisanlib.thermal_alarm_generator import (
    generate_alarm_table,
    generate_schedule_description,
)
from artisanlib.thermal_interop import (
    export_artisan_plan_json,
    export_hibean_csv,
    import_interop_schedule,
    interop_to_alarm_table,
    schedule_from_inversion,
)
from artisanlib.thermal_planner_quality import build_quality_report
from artisanlib.thermal_schedule_validator import validate_schedule


_log: Final[logging.Logger] = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _print_params_table(params: ThermalModelParams) -> None:
    """Print a formatted table of parameter values with their bounds."""
    bounds = default_bounds()
    header = f'{"Parameter":<16} {"Value":>12}    {"Bounds"}'
    print(header)
    print('-' * len(header))
    for name in PARAM_NAMES:
        val = getattr(params, name)
        lo, hi = bounds[name]
        print(f'{name:<16} {val:>12.6f}    [{lo:.2f}, {hi:.2f}]')
    # m_ref is not fitted but still useful to display
    print(f'{"m_ref":<16} {params.m_ref:>12.6f}    (fixed)')


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f'{m}:{s:02d}'


# ---------------------------------------------------------------------------
# Subcommand: fit
# ---------------------------------------------------------------------------

def _cmd_fit(args: argparse.Namespace) -> int:
    """Fit thermal model parameters from calibration roast profiles."""
    print(f'Parsing {len(args.profiles)} profile(s)...')
    profiles = parse_multiple_profiles(args.profiles)

    if not profiles:
        print('Error: No profiles could be parsed.', file=sys.stderr)
        return 1

    print(f'Successfully parsed {len(profiles)} profile(s):')
    for i, p in enumerate(profiles):
        duration = float(p.time[-1] - p.time[0])
        print(f'  [{i+1}] {os.path.basename(p.source_file)}  '
              f'({len(p.time)} pts, {_format_time(duration)}, '
              f'{p.batch_mass_kg:.3f} kg)')

    # Progress callback: print every 50 iterations
    def progress_callback(iteration: int, cost: float) -> None:
        if iteration % 50 == 0 or iteration == 1:
            print(f'  iteration {iteration:>5d}  cost = {cost:.6f}')

    print(f'\nFitting model (max_iter={args.max_iter})...')
    result: FitResult = fit_model(
        calibration_data=profiles,
        max_iter=args.max_iter,
        progress_callback=progress_callback,
    )

    # Print results summary
    print('\n' + '=' * 60)
    print('FIT RESULTS')
    print('=' * 60)
    print(f'Converged:   {result.converged}')
    print(f'Message:     {result.message}')
    print(f'RMSE:        {result.rmse:.4f} C')
    print(f'R-squared:   {result.r_squared:.6f}')
    print(f'Max error:   {result.max_error:.4f} C')
    print()

    # Per-roast RMSE
    print('Per-roast RMSE:')
    for i, (rmse, prof) in enumerate(zip(result.per_roast_rmse, profiles, strict=False)):
        print(f'  [{i+1}] {os.path.basename(prof.source_file)}: '
              f'{rmse:.4f} C')
    print()

    # Parameter table
    print('Fitted parameters:')
    _print_params_table(result.params)
    print()

    # Save model
    output_path = args.output
    result.params.save(output_path)
    print(f'Model saved to: {output_path}')

    # Optional plot
    if args.plot:
        _plot_fit(result, profiles)

    return 0


def _plot_fit(result: FitResult, profiles: list) -> None:
    """Show matplotlib figure with measured vs predicted BT for each roast."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Warning: matplotlib not available, skipping plot.',
              file=sys.stderr)
        return

    try:
        n_roasts = len(profiles)
        fig, axes = plt.subplots(n_roasts, 1, figsize=(10, 4 * n_roasts),
                                 squeeze=False)
        fig.suptitle('Thermal Model Fit: Measured vs Predicted BT',
                     fontsize=14)

        for i, (prof, pred) in enumerate(zip(profiles, result.predicted, strict=False)):
            ax = axes[i, 0]
            time_min = prof.time / 60.0
            ax.plot(time_min, prof.bt, 'b-', label='Measured BT', linewidth=1.5)
            ax.plot(time_min, pred, 'r--', label='Predicted BT', linewidth=1.5)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Temperature (C)')
            roast_name = os.path.basename(prof.source_file)
            ax.set_title(f'{roast_name}  (RMSE={result.per_roast_rmse[i]:.2f} C)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f'Warning: Could not display plot: {exc}', file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------

def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate alarm schedule from target profile using model inversion."""
    if args.mass <= 0:
        print('Error: --mass must be > 0 grams', file=sys.stderr)
        return 1
    if args.interval <= 0:
        print('Error: --interval must be > 0 seconds', file=sys.stderr)
        return 1
    if args.fan < 0 or args.fan > 100:
        print('Error: --fan must be in [0,100]', file=sys.stderr)
        return 1
    if args.min_delta < 1:
        print('Error: --min-delta must be >= 1', file=sys.stderr)
        return 1
    if args.bt_hysteresis < 0 or args.bt_min_gap < 0:
        print('Error: --bt-hysteresis and --bt-min-gap must be >= 0', file=sys.stderr)
        return 1
    if args.drum > 100 or args.drum < -1:
        print('Error: --drum must be in [0,100] or -1 to disable', file=sys.stderr)
        return 1
    if args.optimizer_iterations < 1 or args.optimizer_segments < 2 or args.optimizer_step < 1:
        print('Error: optimizer settings must be positive (segments >= 2)', file=sys.stderr)
        return 1

    # Load model
    print(f'Loading model from: {args.model}')
    params = ThermalModelParams.load(args.model)
    model = KaleidoThermalModel(params)

    # Parse target profile
    print(f'Parsing target profile: {args.target}')
    try:
        target = parse_target_profile(args.target)
    except Exception as exc:  # pylint: disable=broad-except
        print(f'Error: Could not parse target profile: {exc}', file=sys.stderr)
        return 1

    mass_kg = args.mass / 1000.0  # CLI takes grams, model uses kg
    print(f'Batch mass: {args.mass} g ({mass_kg:.4f} kg)')
    print(f'Fan setting: {args.fan}%')
    if args.drum >= 0:
        print(f'Drum setting: {args.drum}%')
    else:
        print('Drum setting: off')
    print(f'Actuator optimization: {"on" if args.optimize_actuators else "off"}')
    print(f'Trigger mode: {args.trigger_mode}')
    print(f'Min control change: {args.min_delta}%')
    print(f'Resample interval: {args.interval} s')

    # Invert model
    print('\nInverting model...')
    inv_result = invert_model(
        model=model,
        target_time=target.time,
        target_bt=target.bt,
        mass_kg=mass_kg,
        fan_schedule=float(args.fan),
        drum_schedule=(None if args.drum < 0 else float(args.drum)),
        optimize_actuators=bool(args.optimize_actuators),
        optimizer_iterations=int(args.optimizer_iterations),
        optimizer_segments=int(args.optimizer_segments),
        optimizer_step_pct=int(args.optimizer_step),
    )

    print(f'Inversion tracking error: RMSE={inv_result.rmse:.2f} C, '
          f'max={inv_result.max_tracking_error:.2f} C')
    if inv_result.objective_score is not None:
        print(f'Optimization objective: {inv_result.objective_score:.3f}')

    if inv_result.exo_warning_time is not None:
        print(f'Exothermic onset: {_format_time(inv_result.exo_warning_time)}')
    if inv_result.first_crack_time is not None:
        print(f'Estimated first crack: {_format_time(inv_result.first_crack_time)}')
    print(f'Estimated drop: {_format_time(inv_result.drop_time)}')
    if inv_result.dtr_percent is not None:
        print(f'Estimated DTR: {inv_result.dtr_percent:.1f}%')

    # Resample to desired interval
    print(f'\nResampling to {args.interval}s interval...')
    resampled = inv_result.resample_to_interval(float(args.interval))
    print(f'Resampled: {len(resampled.time)} points, '
          f'RMSE={resampled.rmse:.2f} C, '
          f'max error={resampled.max_tracking_error:.2f} C')

    # Generate alarms
    milestone_offsets: dict[str, float] | None = None
    if not args.no_milestones:
        milestone_offsets = {}
        if resampled.yellowing_time is not None:
            milestone_offsets['Yellowing'] = resampled.yellowing_time
        if resampled.first_crack_time is not None:
            milestone_offsets['First Crack'] = resampled.first_crack_time
        milestone_offsets['Drop'] = resampled.drop_time

    alarms = generate_alarm_table(
        time=resampled.time,
        heater_pct=resampled.heater_pct,
        fan_pct=resampled.fan_pct,
        exo_warning_time=inv_result.exo_warning_time,
        drum_pct=resampled.drum_pct,
        min_delta_pct=int(args.min_delta),
        trigger_mode=args.trigger_mode,
        bt_profile=(resampled.predicted_bt if args.trigger_mode == 'bt' else None),
        bt_hysteresis_c=float(args.bt_hysteresis),
        bt_min_gap_c=float(args.bt_min_gap),
        milestone_offsets=milestone_offsets,
        bt_safety_ceiling=args.bt_max,
        et_safety_ceiling=args.et_max,
    )

    safety = validate_schedule(
        model,
        resampled,
        bt_limit_c=args.bt_max,
        et_limit_c=args.et_max,
        max_ror_limit_c_per_min=args.max_ror,
    )
    print('')
    for line in safety.summary_lines():
        print(line)

    quality = build_quality_report(
        target_time=target.time,
        target_bt=target.bt,
        inversion=resampled,
        control_change_count=alarms.control_change_count(),
        safety=safety,
    )
    print('')
    for line in quality.summary_lines():
        print(line)

    # Print schedule description
    desc = generate_schedule_description(alarms)
    print(f'\nSchedule: {desc}')

    # Save .alrm file
    output_path = args.output
    alarms.save_alrm(output_path)
    print(f'Alarm file saved to: {output_path}')

    interop_schedule = schedule_from_inversion(
        resampled,
        trigger_mode=args.trigger_mode,
        label=alarms.label or 'Thermal Model Control',
    )
    if args.interop_json:
        export_artisan_plan_json(args.interop_json, interop_schedule)
        print(f'Interop JSON saved to: {args.interop_json}')
    if args.hibean_csv:
        export_hibean_csv(args.hibean_csv, interop_schedule)
        print(f'HiBean-style CSV saved to: {args.hibean_csv}')

    # Optional plot
    if args.plot:
        _plot_generate(target, resampled)

    return 0


def _plot_generate(target, inv_result) -> None:
    """Show matplotlib figure with target BT, predicted BT, heater%, fan%."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Warning: matplotlib not available, skipping plot.',
              file=sys.stderr)
        return

    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.suptitle('Thermal Model Inversion: Target vs Predicted', fontsize=14)

        target_time_min = target.time / 60.0
        inv_time_min = inv_result.time / 60.0

        # Temperature axis (left)
        ax1.plot(target_time_min, target.bt, 'b-',
                 label='Target BT', linewidth=2)
        ax1.plot(inv_time_min, inv_result.predicted_bt, 'r--',
                 label='Predicted BT', linewidth=1.5)
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Temperature (C)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Control axis (right)
        ax2 = ax1.twinx()
        ax2.plot(inv_time_min, inv_result.heater_pct, 'orange',
                 label='Heater %', linewidth=1, alpha=0.8)
        ax2.plot(inv_time_min, inv_result.fan_pct, 'green',
                 label='Fan %', linewidth=1, alpha=0.8)
        ax2.plot(inv_time_min, inv_result.drum_pct, 'purple',
                 label='Drum %', linewidth=1, alpha=0.8)
        ax2.set_ylabel('Control (%)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, 110)

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f'Warning: Could not display plot: {exc}', file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: simulate
# ---------------------------------------------------------------------------

def _cmd_simulate(args: argparse.Namespace) -> int:
    """Forward-simulate the thermal model with fixed controls."""
    # Load model
    print(f'Loading model from: {args.model}')
    params = ThermalModelParams.load(args.model)
    model = KaleidoThermalModel(params)

    mass_kg = args.mass / 1000.0  # CLI takes grams, model uses kg
    duration = args.duration
    T0 = args.T0

    print(f'Heater: {args.hp}%')
    print(f'Fan:    {args.fan}%')
    print(f'Mass:   {args.mass} g ({mass_kg:.4f} kg)')
    print(f'T0:     {T0} C')
    print(f'Duration: {duration} s ({_format_time(duration)})')

    # Create constant control arrays at 1-second resolution
    time = np.arange(0, duration + 1, 1.0)
    hp_schedule = np.full_like(time, float(args.hp))
    fan_schedule = np.full_like(time, float(args.fan))

    # Simulate
    print('\nSimulating...')
    predicted = model.simulate(
        time=time,
        hp_schedule=hp_schedule,
        fan_schedule=fan_schedule,
        drum_schedule=np.zeros_like(time, dtype=np.float64),
        T0=T0,
        mass_kg=mass_kg,
    )

    # Print results
    final_bt = predicted[-1]
    print(f'\nFinal BT: {final_bt:.1f} C')

    # Time to reach 200 C
    above_200 = np.where(predicted >= 200.0)[0]
    if len(above_200) > 0:
        t_200 = time[above_200[0]]
        print(f'Time to 200 C: {_format_time(t_200)} ({t_200:.0f} s)')
    else:
        print('Time to 200 C: not reached')

    # Time to reach 210 C
    above_210 = np.where(predicted >= 210.0)[0]
    if len(above_210) > 0:
        t_210 = time[above_210[0]]
        print(f'Time to 210 C: {_format_time(t_210)} ({t_210:.0f} s)')
    else:
        print('Time to 210 C: not reached')

    # Optional plot
    if args.plot:
        _plot_simulate(time, predicted)

    return 0


def _plot_simulate(time, predicted) -> None:
    """Show matplotlib figure with BT curve and RoR."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Warning: matplotlib not available, skipping plot.',
              file=sys.stderr)
        return

    try:
        time_min = time / 60.0

        # Compute RoR (rate of rise in C/min)
        ror = np.gradient(predicted, time) * 60.0  # convert C/s to C/min

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle('Thermal Model Simulation', fontsize=14)

        # BT axis (left)
        ax1.plot(time_min, predicted, 'b-', label='Bean Temp (BT)',
                 linewidth=2)
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Temperature (C)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # RoR axis (right)
        ax2 = ax1.twinx()
        ax2.plot(time_min, ror, 'r-', label='RoR (C/min)',
                 linewidth=1, alpha=0.7)
        ax2.set_ylabel('RoR (C/min)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f'Warning: Could not display plot: {exc}', file=sys.stderr)


def _cmd_interop_convert(args: argparse.Namespace) -> int:
    """Convert an interop schedule into an Artisan alarm file."""
    schedule = import_interop_schedule(args.input, fmt=args.format)
    alarms = interop_to_alarm_table(
        schedule,
        min_delta_pct=args.min_delta,
        bt_hysteresis_c=args.bt_hysteresis,
        bt_min_gap_c=args.bt_min_gap,
    )
    alarms.save_alrm(args.output)
    print(f'Loaded schedule: {schedule.label}')
    print(f'Trigger mode: {schedule.trigger_mode}')
    print(f'Generated alarms: {alarms.alarm_count()}')
    print(f'Saved: {args.output}')
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='thermal_model_cli',
        description='Kaleido thermal model CLI — fit, generate, simulate, and interop conversion.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose (DEBUG) logging.',
    )

    sub = parser.add_subparsers(dest='command', required=True,
                                help='Available subcommands')

    # ── fit ────────────────────────────────────────────────────────────
    p_fit = sub.add_parser(
        'fit',
        help='Fit model parameters from calibration roast profiles.',
    )
    p_fit.add_argument(
        'profiles', nargs='+',
        help='One or more .alog profile files.',
    )
    p_fit.add_argument(
        '-o', '--output', default='kaleido_model.json',
        help='Output path for the model JSON (default: kaleido_model.json).',
    )
    p_fit.add_argument(
        '--max-iter', type=int, default=1000,
        help='Maximum iterations for differential evolution (default: 1000).',
    )
    p_fit.add_argument(
        '--plot', action='store_true',
        help='Show matplotlib plot of measured vs predicted BT.',
    )

    # ── generate ──────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        'generate',
        help='Generate alarm schedule from a target profile.',
    )
    p_gen.add_argument(
        'model',
        help='Path to the model JSON file.',
    )
    p_gen.add_argument(
        'target',
        help='Path to the target .alog profile.',
    )
    p_gen.add_argument(
        '--mass', type=float, required=True,
        help='Batch mass in grams.',
    )
    p_gen.add_argument(
        '--fan', type=float, default=30.0,
        help='Constant fan percentage (default: 30).',
    )
    p_gen.add_argument(
        '--drum', type=float, default=-1.0,
        help='Constant drum percentage (0-100). Use -1 to disable (default: -1).',
    )
    p_gen.add_argument(
        '--trigger-mode', choices=['time', 'bt'], default='time',
        help='Control trigger mode: time or BT threshold (default: time).',
    )
    p_gen.add_argument(
        '--optimize-actuators', action='store_true',
        help='Co-optimise fan and drum schedules during inversion.',
    )
    p_gen.add_argument(
        '--optimizer-iterations', type=int, default=3,
        help='Coordinate-descent iterations for actuator optimisation (default: 3).',
    )
    p_gen.add_argument(
        '--optimizer-segments', type=int, default=8,
        help='Piecewise schedule segments for actuator optimisation (default: 8).',
    )
    p_gen.add_argument(
        '--optimizer-step', type=int, default=8,
        help='Initial actuator step size in %% for optimisation (default: 8).',
    )
    p_gen.add_argument(
        '--min-delta', type=int, default=2,
        help='Minimum control change (%%) required to emit a new alarm (default: 2).',
    )
    p_gen.add_argument(
        '--bt-hysteresis', type=float, default=1.0,
        help='BT trigger hysteresis in C (default: 1.0).',
    )
    p_gen.add_argument(
        '--bt-min-gap', type=float, default=2.0,
        help='Minimum BT trigger spacing in C (default: 2.0).',
    )
    p_gen.add_argument(
        '--no-milestones', action='store_true',
        help='Disable milestone popup alarms (yellowing/first crack/drop).',
    )
    p_gen.add_argument(
        '--bt-max', type=float, default=None,
        help='Optional BT safety ceiling popup alarm in C.',
    )
    p_gen.add_argument(
        '--et-max', type=float, default=None,
        help='Optional ET safety ceiling popup alarm in C.',
    )
    p_gen.add_argument(
        '--max-ror', type=float, default=None,
        help='Optional RoR safety ceiling for dry-run validation (C/min).',
    )
    p_gen.add_argument(
        '--interval', type=float, default=10.0,
        help='Resample interval in seconds (default: 10).',
    )
    p_gen.add_argument(
        '-o', '--output', default='schedule.alrm',
        help='Output path for the .alrm file (default: schedule.alrm).',
    )
    p_gen.add_argument(
        '--plot', action='store_true',
        help='Show matplotlib plot of target/predicted BT and controls.',
    )
    p_gen.add_argument(
        '--interop-json', default=None,
        help='Optional path to export artisan-thermal-plan JSON.',
    )
    p_gen.add_argument(
        '--hibean-csv', default=None,
        help='Optional path to export a HiBean-style replay CSV.',
    )

    # ── simulate ──────────────────────────────────────────────────────
    p_sim = sub.add_parser(
        'simulate',
        help='Forward-simulate the model with fixed controls.',
    )
    p_sim.add_argument(
        'model',
        help='Path to the model JSON file.',
    )
    p_sim.add_argument(
        '--hp', type=float, default=90.0,
        help='Constant heater percentage (default: 90).',
    )
    p_sim.add_argument(
        '--fan', type=float, default=25.0,
        help='Constant fan percentage (default: 25).',
    )
    p_sim.add_argument(
        '--mass', type=float, default=100.0,
        help='Batch mass in grams (default: 100).',
    )
    p_sim.add_argument(
        '--duration', type=float, default=720.0,
        help='Simulation duration in seconds (default: 720).',
    )
    p_sim.add_argument(
        '--T0', type=float, default=200.0,
        help='Initial bean temperature in C (default: 200).',
    )
    p_sim.add_argument(
        '--plot', action='store_true',
        help='Show matplotlib plot of BT and RoR curves.',
    )

    # ── interop-convert ───────────────────────────────────────────────
    p_io = sub.add_parser(
        'interop-convert',
        help='Convert interop schedule JSON/CSV into Artisan .alrm format.',
    )
    p_io.add_argument(
        'input',
        help='Input schedule file (.json or .csv).',
    )
    p_io.add_argument(
        'output',
        help='Output .alrm file path.',
    )
    p_io.add_argument(
        '--format', choices=['auto', 'json', 'csv'], default='auto',
        help='Input format (default: auto).',
    )
    p_io.add_argument(
        '--min-delta', type=int, default=2,
        help='Minimum control change (%%) required to emit a new alarm (default: 2).',
    )
    p_io.add_argument(
        '--bt-hysteresis', type=float, default=1.0,
        help='BT trigger hysteresis in C (default: 1.0).',
    )
    p_io.add_argument(
        '--bt-min-gap', type=float, default=2.0,
        help='Minimum BT trigger spacing in C (default: 2.0).',
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    'fit':      _cmd_fit,
    'generate': _cmd_generate,
    'simulate': _cmd_simulate,
    'interop-convert': _cmd_interop_convert,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(name)s: %(message)s',
    )

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f'Unexpected error: {exc}', file=sys.stderr)
        _log.debug('Traceback:', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
