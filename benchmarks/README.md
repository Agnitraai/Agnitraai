# Agnitra Benchmarks

Public, reproducible benchmarks. Each subdirectory pins one workload on
one GPU SKU and ships a one-command repro plus a committed results file.

| Suite | Workload | Hardware | Status |
|---|---|---|---|
| [`llama3_h100/`](llama3_h100/) | Llama-3-8B-Instruct, 512→128 tokens | H100 80GB SXM5 | scaffolded — numbers pending first run |

## Why these directories exist

Inference optimization claims ("2x faster") are unfalsifiable without a
fixed workload, fixed hardware, fixed software versions, and a one-command
repro. This directory exists so a skeptic with a GPU can reproduce — or
refute — every number Agnitra publishes within ±5%.

## Adding a new suite

1. Copy `llama3_h100/` to a new directory named `<workload>_<gpu>/`.
2. Edit `workload.py` — change the model and prompt, never change the
   measurement plumbing in `common.py` / `_hf_base.py`.
3. Update `Dockerfile` if the new workload needs different deps.
4. Run `./run.sh` on the target hardware, commit the regenerated
   `RESULTS.md` and the raw JSON outputs.
5. Add a row to the table above.

## CI gating

The benchmark CI workflow at `.github/workflows/benchmark.yml` re-runs
`llama3_h100/run.sh` on a self-hosted H100 runner for every release tag,
fails the build if Agnitra's throughput regresses >5% vs. the previous
tagged release, and uploads the new `RESULTS.md` as a release asset.
