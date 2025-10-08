# Agnitra Product Design Handoff (v0.1)

## 1. Purpose & Product Snapshot
- **Mission**: Eliminate manual GPU kernel tuning by pairing telemetry-led diagnostics with LLM- and RL-generated optimizations delivered as instant runtime patches.
- **Differentiators**: Runtime agent with live profiling, deterministic JSON artifacts, pay-per-optimization metering, and drop-in CLI/Python SDK touchpoints.
- **MVP surfaces**: Interactive CLI (`agnitra profile`, `agnitra benchmark`), Python SDK helper (`agnitra.optimize()`), structured telemetry and billing artifacts, and the milestone demo script.


## 2. Experience Overview
- **CLI & SDK entry**: Maintain a concise, metric-forward tone. `agnitra profile` prints telemetry paths plus next-step hints; `agnitra benchmark` surfaces speedup and latency deltas. Both commands degrade gracefully when dependencies are missing. The Python SDK mirrors CLI semantics—`optimize()` returns baseline versus optimized snapshots and emits a billing-friendly usage event.
- **Runtime optimization report**: Present stacked metric cards for latency (ms), tokens/s, GPU utilization, GPU hours saved, and cost impact. A secondary panel lists agent notes (RL enabled flag, LLM source model, applied presets). Provide download/copy affordances for underlying JSON artifacts.
- **Telemetry & billing artifacts**: Telemetry JSON captures op-wise timings, memory, and device metadata with bottleneck annotations referencing IR nodes. Usage events detail GPU hours before/after, savings, charge breakdown (usage plus success fee), timestamps, and metadata. Kernel/patch artifacts include Triton kernels, FX node patch descriptors, and runtime logs with deterministic naming for CI/CD diffing.
- **Dashboard preview (post-MVP)**: Dark navy canvas with electric teal accents framing hero KPIs, run timelines, and layer-level drilldowns. Maintain data parity with CLI outputs to simplify developer verification.

## 3. Core Flows & States
1. **CLI first run**: Install → `agnitra profile` → telemetry JSON written → call-to-action to benchmark. Empty and error states cover missing Torch installs, absent model paths, or sandbox hints.
2. **Optimization and patch loop**: Telemetry capture → LLM suggestion (JSON) → optional RL refinement → kernel generation → runtime patch → benchmark diff. Surface explicit checkpoints so users can track progress.
3. **Telemetry export & billing**: Persist snapshots alongside usage events; SDK emits structured events ready for control-plane ingestion.
4. **Errors & fallbacks**: LLM timeout triggers a heuristic preset banner, regressions prompt rollback guidance, and absent GPU metrics soft-fail with degraded copy while preserving transparency.

## 4. Data & Schema Contracts
- **Telemetry JSON**:
  ```json
  {
    "bottleneck": {"op": "aten::matmul", "latency_ms": 12.4, "shape": [1, 4096, 4096]},
    "events": [],
    "gpu": {"gpu_utilisation": 0.71},
    "behavior": {"gpu_util_mean": 0.68}
  }
  ```
- **LLM optimization suggestion fields**: `block_size:int`, `tile_shape:[int,int]`, `unroll_factor:int`, `target_latency_ms:float`, `expected_latency_ms:float`, `rationale:str`, `source:str`, `raw_text:str?`.
- **Usage event schema**:
  ```json
  {
    "project_id": "demo",
    "model_name": "TinyLlama",
    "tokens_processed": 262144,
    "baseline_latency_ms": 83.2,
    "optimized_latency_ms": 62.9,
    "baseline_tokens_per_sec": 3149.0,
    "optimized_tokens_per_sec": 4168.0,
    "gpu_hours_before": 0.0231,
    "gpu_hours_after": 0.0175,
    "gpu_hours_saved": 0.0056,
    "performance_uplift_pct": 32.5,
    "cost_before": 0.0578,
    "cost_after": 0.0437,
    "cost_savings": 0.0141,
    "usage_charge": 0.0437,
    "success_fee": 0.0028,
    "total_billable": 0.0465,
    "currency": "USD",
    "metadata": {"stage_notes": "baseline_vs_optimized"}
  }
  ```
- **Kernel/patch artifacts**: Triton kernel source, FX node patch descriptors, and runtime patch logs; enforce deterministic naming for automated diffing.

## 5. Developer Handoff Checklist
1. Annotated Figma frames (hero, CLI snapshot render, report cards, error banners) with spacing, typography, and motion notes.
2. Copy deck covering CLI strings, tooltips, CTA labels, and fallback messaging.
3. Data fixtures: telemetry JSON, usage event JSON, LLM suggestion JSON, and kernel patch logs for design QA.
4. Interaction spec outlining flow diagrams, success/failure state matrices, and repeatable triggers for heuristic versus RL paths.
5. Accessibility requirements covering color contrast ratios, keyboard focus order (future dashboard), and CLI output guidance for screen readers.
6. QA plan mapping acceptance tests (unit, CLI integration, regression thresholds) plus logging expectations and latency guardrails.
7. Release checklist detailing dependency toggles (env vars), telemetry retention policy, and security notes (no secrets in logs).

## 6. Success Metrics & Instrumentation
- **Primary performance**: ≥15% latency reduction, ≥20% tokens/sec gain, ≥99.9% output equivalence.
- **Business health**: GPU hours saved, total billable, uplift per project.
- **Engagement**: CLI completion rate from `profile` to `benchmark`, SDK optimize call success rate, fallback frequency.
- **Instrumentation hooks**: Structured events (`usage.attach`, `optimization.completed`, `optimization.fallback`), CLI exit codes, optional NVML metrics flag.

## 7. Dependencies & Open Questions
- Confirm brand assets (logo, final palette) and partner logos for the proof band.
- Validate GPU telemetry availability on non-NVIDIA hardware and define degraded-mode copy.
- Align on control-plane ingestion endpoint contract and authentication strategy.
- Decide on storage and retention for telemetry artifacts (local versus remote).
- Determine roadmap communication style on the landing page (timeline versus cards) before visual design lock.
- Clarify timeline for the sandbox try-in-browser experiment to inform CTA hierarchy.

## 8. Immediate Next Steps
1. Review this handoff with design and product to lock visual references and resolve open questions.
2. Package required fixtures (sample telemetry, usage event, kernel log) into the developer handoff bundle.
