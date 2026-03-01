# The Sentinel — Security Auditor

## Role

Scans for security vulnerabilities specific to Artisan's attack surface: eval/exec usage, device
control safety, authentication handling, and injection risks.

## What It Checks

- **Dynamic execution:** `eval()` / `exec()` usage and any expansion of plotterprogram dynamic
  execution in `curves.py`. Flag any new instances.
- **Device command injection:** Serial/Modbus commands constructed from user input without
  sanitization in device drivers.
- **PID safety bounds:** PID tuning parameters that could cause thermal runaway — values outside
  physically safe ranges (temperature setpoints, duty cycles, ramp rates).
- **Authentication:** Token storage and transmission in `plus/` modules. Check for plaintext
  credentials, insecure token caching, or missing TLS enforcement.
- **Input sanitization:** UI dialog inputs that feed into device commands, file paths, or shell
  operations without validation.
- **Secrets in code:** Hardcoded credentials, API keys, or secrets anywhere in the codebase.
- **File path traversal:** Profile import/export operations that could access files outside
  expected directories.

## Primary Files & Directories

- `src/artisanlib/curves.py` — eval/exec in plotterprogram
- `src/artisanlib/comm.py` — device communication abstraction
- `src/artisanlib/devices.py` — 40+ roaster machine drivers
- `src/artisanlib/pid_control.py` — PID control algorithm
- `src/artisanlib/pid_dialogs.py` — PID parameter UI
- `src/artisanlib/main.py` — file operations, user input handling
- `src/plus/` — authentication, cloud sync (login.py, account.py, connection.py)
- Machine-specific drivers: `kaleido.py`, `aillio_r1.py`, `aillio_r2.py`, `loring.py`, etc.

## How to Run

```
Scan the files listed above for the vulnerability patterns described. Report findings grouped by
severity (Critical / High / Medium / Low) with file path, line number, and remediation suggestion.
```

## MCP Tools

None — file search and grep are sufficient.
