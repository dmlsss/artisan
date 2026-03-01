# Infrastructure Summary

## What Was Set Up

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **CLAUDE.md** | `/CLAUDE.md` | Project-specific commands, architecture overview, code conventions, critical rules (~580 tokens) |
| **.claudeignore** | `/.claudeignore` | Excludes ~770MB of build artifacts, generated code, translations, fonts/icons/themes |
| **The Sentinel** | `/.claude/agents/the-sentinel.md` | Security auditor: eval/exec, device command injection, PID safety, auth tokens |
| **The Roast Scientist** | `/.claude/agents/the-roast-scientist.md` | Domain correctness: thermal model ODE stability, unit conversions, event ordering, calibration bounds |
| **The Test Warden** | `/.claude/agents/the-test-warden.md` | Test quality: coverage gaps, weak assertions, missing edge cases, flaky patterns |
| **Task tracking** | `/tasks/todo.md` | Active work items |
| **Lessons log** | `/tasks/lessons.md` | Accumulated gotchas and debugging insights |
| **Project profile** | `/tasks/project-profile.md` | Full project analysis (tech stack, architecture, complexity) |

## What Was Deliberately Skipped

- **MCP Server** — Grep/Glob are sufficient for this codebase; no repeated reference data lookups
- **Custom Skills** — Pre-commit hooks + individual CLI commands cover existing workflows
- **Style/Lint Agents** — Already covered by mypy strict + pylint + ruff + codespell + pre-commit

## Agents at a Glance

| Agent | One-liner |
|-------|-----------|
| **The Sentinel** | Finds security vulnerabilities: eval/exec, device injection, PID bounds, auth leaks |
| **The Roast Scientist** | Validates thermal modeling math, numerical stability, unit conversions, physics constraints |
| **The Test Warden** | Identifies test coverage gaps and weak assertions across the 185K LOC codebase |

## Key Commands

```bash
# Run the app
cd src && python artisan.py

# Run tests
cd src && pytest

# Run a single test module
cd src && pytest test/unitary/artisanlib/test_<module>.py -v

# Type check
cd src && mypy artisanlib plus

# Lint
cd src && ruff check .

# Pre-commit (all hooks)
pre-commit run --all-files
```

## Verification Results

| Check | Result |
|-------|--------|
| CLAUDE.md under 3,000 tokens | ~580 tokens |
| Agent path references valid | All 3 agents verified (after fix) |
| .claudeignore covers key exclusions | Build, generated, translations, resources, IDE |
| tasks/lessons.md exists and formatted | Yes |
| Smoke test (Test Warden) | 47 of 84 modules (56%) have zero tests; top gaps identified |

## Smoke Test Highlights

The Test Warden agent found:
- **47 of 84 artisanlib modules have zero unit tests** (56% untested)
- Highest-risk untested: `roast_properties.py` (5.9K LOC), `devices.py` (4.9K LOC), `events.py` (3.9K LOC)
- Even tested modules are thin: `canvas.py` (20K LOC) has only 95 tests
- Recommended starting point: pure static methods in `events.py` (zero Qt dependencies, high impact)

## How to Add New Agents

1. Create a new `.md` file in `/.claude/agents/`
2. Follow the existing format: Role, What It Checks, Primary Files, How to Run, MCP Tools
3. Verify all file paths exist before committing
4. Commit with message: `chore: add <agent-name> agent definition`

## How to Update Lessons

When you discover a gotcha, debugging insight, or important decision, add it to `tasks/lessons.md`
under the appropriate section. Keep entries concise and actionable.
