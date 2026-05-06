# Contributing to Agnitra

Thanks for considering a contribution. This document covers the dev
setup, the test pattern, and what makes a good PR.

## Dev setup

```bash
git clone https://github.com/agnitraai/agnitraai
cd agnitraai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,rl,nvml,quantize]"
pip install pytest pytest-cov ruff mypy
```

The `[quantize]` extra pulls in `torchao` for the INT8/INT4/FP8 paths.

## Running tests

```bash
pytest tests/                              # full suite (94 tests, runs on CPU)
pytest tests/test_optimizers_detection.py  # one file
pytest -k "quantize"                       # by keyword
```

Tests are designed to run **without a GPU**. We monkeypatch the
underlying optimizer everywhere — the unit tests verify routing,
dispatch, validation, and integration shape, not real model speedup.
Real-speedup verification lives in `benchmarks/llama3_h100/`.

If you add a new architecture-specific optimization, add:

1. A unit test that monkeypatches the optimizer call site and verifies
   your code path is reached.
2. A line in `benchmarks/llama3_h100/runners/agnitra_runner.py` that
   exercises the new path.
3. An entry in `agnitra/optimizers/registry.py` if it's a new
   architecture.

## What makes a good PR

- **Honest scope.** "Bug fix" PRs should fix one bug. "Feature" PRs
  should add one feature. Mixed PRs are hard to review and harder to
  revert.
- **Tests for new behavior.** Use the monkeypatched-optimizer pattern
  in existing tests as your template.
- **README + CHANGELOG entry.** If your change is user-visible, add a
  bullet under the appropriate `## [Unreleased]` section in
  `CHANGELOG.md`. README updates only when the change affects the
  install / quickstart / supported architectures table.
- **Honest commit messages.** What you changed, *why*, and what you
  considered and rejected. The commit log is the project's memory.

## What gets rejected

- **Silent-no-op fallbacks.** Agnitra explicitly returns a passthrough
  result with `notes["passthrough"] = True` when an architecture is
  unsupported. Don't add fallbacks that pretend the optimizer ran.
- **Inflated benchmark claims.** All speedup numbers in this repo are
  reproducible from `benchmarks/`. If you add a benchmark, the run
  must be reproducible from a single command.
- **Quantization without validation.** New quantization modes must
  ship with output-drift validation enabled by default.

## Reporting issues

Use the [issue templates](.github/ISSUE_TEMPLATE/). The most useful
bug reports include:

- The exact `pip install` command you used
- Output of `agnitra doctor`
- A minimal reproduction (5-10 lines of Python)
- The exact error message and traceback

## Adversarial benchmarks

If you find Agnitra is handicapping a competitor (vLLM, TensorRT-LLM,
etc.) in our published benchmark, open an issue with the specific
configuration that would make the comparison fair. We treat this as
**adversarial review**, not criticism — the benchmark is only useful
if reviewers trust it.

## License

By contributing, you agree your code will be licensed under
[Apache 2.0](LICENSE).
