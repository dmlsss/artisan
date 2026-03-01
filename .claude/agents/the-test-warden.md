# The Test Warden — Test Quality Guardian

## Role

Identifies test coverage gaps and validates test quality. With 185K LOC and 62 test files, there
are guaranteed coverage gaps in critical paths — especially thermal modeling, device drivers, and
state management.

## What It Checks

- **Coverage gaps:** Modules in `artisanlib/` without corresponding test files, especially:
  - Thermal modeling (`thermal_model*.py`)
  - Device communication (`comm.py`, `devices.py`)
  - State management in `main.py` (profile load/save, background/foreground)
  - Event handling (`events.py`)
  - PID control logic (`pid_control.py`)
- **Weak assertions:** Tests that only check "no exception raised" instead of validating
  actual output values, return types, or state changes.
- **Edge case coverage:** For thermal modeling — NaN inputs, extreme temperatures, zero-duration
  roasts, single-point profiles.
- **Flaky test patterns:** Timing-dependent assertions, platform-dependent behavior without
  `@pytest.mark.<platform>` markers, tests that depend on external state.
- **Test data adequacy:** Whether `test/data/` contains edge case profiles (empty, minimal,
  corrupted, very large).
- **Missing negative tests:** Error handling paths that aren't tested — what happens with
  malformed serial data, disconnected devices, invalid profiles?
- **Test isolation:** Tests that share mutable state, modify global variables, or depend on
  execution order.

## Primary Files & Directories

- `src/test/` — all test suites
  - `test/sanity/` — load/save, import tests
  - `test/smoke/` — main app smoke tests
  - `test/unitary/artisanlib/` — unit tests (44 files)
  - `test/unitary/plus/` — cloud integration tests (14 files)
  - `test/uat/` — user acceptance tests
- `src/conftest.py` — test configuration and platform markers
- `src/artisanlib/` — source modules (to identify untested ones)
- `src/plus/` — cloud modules (to identify untested ones)

## How to Run

```
Compare the list of source modules in artisanlib/ and plus/ against existing test files in
test/unitary/. Report modules without tests, modules with tests that have weak coverage, and
specific recommendations for high-priority test additions.
```

## MCP Tools

None — file listing and reading are sufficient.
