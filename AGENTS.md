# Repository Guidelines

## Project Structure & Module Organization
- `agnitra/` — core SDK packages (`core/`, `telemetry/`, `demo/`).
- `cli/` — CLI entry points (`main.py`, `optimize.py`).
- `tests/` — PyTest suite mirroring package structure (`test_*.py`).
- `docs/` — documentation (e.g., `prd.md`).
- `pyproject.toml` — build metadata (setuptools), CLI scripts.
- `README.md`, `notes.yaml` — usage and project notes.

## Build, Test, and Development Commands
- Install (editable) with optional extras: `pip install -e .[openai,rl]`
- Run tests quietly: `pytest -q`
- CLI help: `agnitra --help`  • Optimize example: `agnitra-optimize --model demo-model`
- Profile a TorchScript model: `python -m cli.main profile path/to/model.pt --input-shape 1,3,224,224 --output telemetry.json`
- Build a wheel (optional): `python -m build` (requires `build` package).

## Coding Style & Naming Conventions
- Python 3.8+ • 4‑space indentation • PEP 8 compliant.
- Use type hints and concise, NumPy‑style docstrings where helpful.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Keep files focused and imports grouped: stdlib, third‑party, local.

## Testing Guidelines
- Framework: PyTest. Place tests under `tests/` as `test_*.py`.
- Mirror module paths (e.g., `agnitra/telemetry/...` → `tests/test_telemetry_*.py`).
- Prefer fast, isolated tests; use `tmp_path`, `monkeypatch`, `capsys` as needed (see existing tests).
- Ensure CLI paths return explicit exit codes and emit clear messages.

## Commit & Pull Request Guidelines
- Commits: imperative mood (“Add profiler”), short subject (≤72 chars), body explains Why/What/How; reference issues (`Closes #123`).
- Before opening a PR: run `pytest`, verify CLI examples in README, and update docs (`README.md`, `docs/`) for user‑visible changes.
- PRs should include: clear description, reproduction or usage example, tests for new behavior, and screenshots/output snippets when changing CLI UX.

## Security & Configuration Tips
- Do not commit secrets. Use environment variables (e.g., `OPENAI_API_KEY`).
- GPU metrics require optional deps (`pynvml`); code degrades gracefully when unavailable.
- Large models are optional; prefer lightweight fixtures for tests.

## Agent Runtime Expectations
- Do not abort tasks prematurely; each assigned task should receive at least 10 minutes of active effort before yielding or escalating.
- Do not freeze on long-running scripts; if no useful results appear, stop, report the outcome, and move to the next task.
