# Agnitra AI – UI/UX Handoff Brief (v0.1)

## 1. Product Snapshot
- **Mission**: Deliver runtime AI model optimization that boosts throughput, lowers latency, and improves GPU utilization without code rewrites.
- **Core differentiators**: live telemetry feedback loop, LLM+RL co-pilot for kernels, cross-vendor GPU abstraction, drop-in CLI/SDK.
- **Primary launch assets**: marketing landing page, CLI quickstart surface, optimization report view, kernel suggestion inspection.

## 2. Target Personas
| Persona | Needs | Desired Outcomes |
|---------|-------|------------------|
| **ML Engineer** | Faster inference without custom kernels | Single command integration, clear before/after metrics |
| **Compiler / Performance Engineer** | Fine control & insight into kernels | Inspectable suggestions, telemetry drill-down, reproducible patches |
| **Infra / DevOps Lead** | Higher GPU utilization & observability | Fleet-wide telemetry, alerts, automated rollbacks |
| **Chip Startup BD / Solutions** | Showcase silicon performance uplift | Branded benchmark stories, partner-tailored pitch CTA |

## 3. Experience Principles
- **Evidence-first**: every claim backed by quantified metrics (tokens/sec, latency, memory).
- **Trust through transparency**: surface telemetry, kernel diffs, model fallback logic.
- **Low friction**: highlight 1-line integration, minimize jargon, progressive disclosure for advanced mechanics.
- **Future-ready**: clearly flag roadmap modules (dashboard, API) while keeping CTA focused on today’s offering.

## 4. Landing Page Information Architecture
1. **Hero**
   - Headline: “AI-native runtime optimization for every GPU.”
   - Sub-head: emphasize telemetry-driven, LLM+RL agent.
   - Primary CTA: `Book a demo` (enterprise focus); Secondary CTA: `Get CLI` (developer download).
   - Visual: animated graph showing latency drop or dual-before/after KPI cards.
2. **Proof Band**
   - Slider with real benchmark stats (e.g., +20% tokens/sec, -15% latency).
   - Logos for partner chips/clouds (placeholder until confirmed).
3. **How It Works (3-step)**
   - Step 1: Profile & collect telemetry (CLI).
   - Step 2: LLM/RL propose tuned kernels.
   - Step 3: Runtime patches & benchmark validation.
4. **Deep Dive Modules**
   - **Telemetry Collector**: live capture, JSON artifact.
   - **Optimizer Engine**: Codex/Responses API integration; fallback heuristics.
   - **Kernel Generator & Runtime Patcher**: automated Triton/CUDA code injection.
5. **Product Tour / UI Preview**
   - Tabs or carousel: CLI snapshot, optimization report JSON → human-readable card, planned dashboard mock.
6. **Why Agnitra vs Traditional Stacks**
   - Comparison table mirroring PRD unique value matrix.
7. **Use Cases**
   - Cards for “LLM inference teams,” “GPU fleet ops,” “Chip partners.”
8. **Roadmap Strip**
   - Timeline referencing PRD roadmap (v0.2 dashboard, v0.3 MLIR/TVM).
9. **Social Proof & Security**
   - Quotes (placeholder), compliance/SLA statements, environment variable best practices.
10. **Final CTA**
    - Dual path: Enterprise contact form + “Install CLI (pip command)” code snippet.

## 5. Key Flows & States

### 5.1 CLI First-Run Flow
1. User installs via `pip install agnitra`.
2. Runs `agnitra profile ./model.pt --input-shape ...`.
3. CLI outputs telemetry JSON path; success notification plus next-step suggestion (`agnitra benchmark`).
4. Edge states:
   - Missing Torch: friendly error copy from `cli/main.py`.
   - Model not found: highlight path check, link to docs.

### 5.2 Optimization Loop Visualization
- Show sequence: Telemetry → LLM suggestion (JSON) → RL refinement (if enabled) → Kernel patch → Benchmark report.
- Provide toggle for “LLM-only” vs “LLM + RL” to mirror CLI flags.

### 5.3 Telemetry & Report Consumption
- Card layout summarizing:
  - Bottleneck layer (op, shape, latency).
  - Suggested params (block size, tile shape, unroll factor).
  - Expected latency vs baseline.
- Include “Download JSON” action and tooltip explaining deterministic schema.

### 5.4 Error & Recovery UX
- **LLM timeout / malformed output**: show fallback heuristic results with banner “Optimized via heuristic fallback (LLM unavailable).”
- **Benchmark regression**: instruct rollback; link to docs.

## 6. Content & Voice Guidelines
- Tone: confident, technical, optimistic; avoid hype.
- Messaging pillars:
  1. “Agentic optimization that learns from your workloads.”
  2. “Telemetry-driven confidence—no black box.”
  3. “Hardware-agnostic acceleration.”
- Microcopy style: short, direct sentences; highlight metrics numerically.
- Glossary tooltip for terms like “Triton,” “torch.fx” to keep landing page accessible.

## 7. Visual & Interaction Direction
- **Palette**: dark navy base (precision), electric teal accents (performance), warm gradient highlight (agent intelligence).
- **Typography**: geometric sans for headlines (innovation), mono or tech serif for code snippets.
- **Iconography**: thin-line data visual icons; pair with holographic gradient backgrounds for hero.
- **Illustration style**: abstract GPU grids, waveform/telemetry patterns.
- **Motion**: subtle pulse on metric improvements; stepper animation for pipeline.

## 8. Assets & Data Sources
- Telemetry JSON sample: `demo_artifacts/llm_optimizer_report.json`
- CLI states reference: `cli/main.py`, `docs/non_interactive_codex_usage.txt`
- Optimization logic: `agnitra/core/optimizer/llm_optimizer.py`, `agnitra/core/optimizer/rl_optimizer.py`
- Roadmap & business context: `docs/prd.md`, `docs/advanced.md`
- Benchmark results mock: `profile_result_tinyllama.json`

## 9. Open Questions for Design Sync
1. What brand guidelines or logo treatments already exist?
2. Confirm hero visual assets (motion design vs static illustration).
3. Are partner logos or testimonial copy available at launch?
4. Determine placement for optional “Try CLI in-browser” if sandbox emerges.
5. Need to visualize roadmap phases? (e.g., timeline vs stacked cards)

## 10. Next Steps
- Align on landing page wireframe (hero → proof → how-it-works → comparison → CTA).
- Define telemetry report component layout for future dashboard.
- Prepare responsive breakpoints (desktop-first, workable tablet/mobile stack).
- Schedule copy review with product marketing once wireframes are ready.

---
_For internal use: tuned to Agnitra v0.1 (MVP) scope and current CLI capabilities. Update metric claims when latest benchmark data is available._
