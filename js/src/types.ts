// Public types mirror the Python agnitra-api JSON shape.
// Keep in sync with agnitra/api/app.py — if a field name changes there,
// change it here in the same commit.

export interface OptimizeRequest {
  /** Identifier used for billing / metering and for logs. */
  project_id: string;
  /** Human-readable model name (logging only; not used as an identifier). */
  model_name?: string;
  /** Hardware target hint, e.g. "H100", "A100", "CPU". */
  target?: string;
  /** Serialized FX graph or model URI — the server-side accepts whichever. */
  model_graph?: unknown;
  /** Sample telemetry payload for the optimizer to calibrate against. */
  telemetry?: unknown;
  /** Optional metadata attached to the resulting usage event. */
  metadata?: Record<string, unknown>;
  /** Customer identifier (defaults to project_id when omitted). */
  customer_id?: string;
  /** URL the server POSTs to when the job completes. */
  webhook_url?: string;
  /** Number of input tokens this run will process (for metering). */
  tokens_processed?: number;
}

export type JobState = "pending" | "running" | "completed" | "failed";

export interface JobStatus {
  job_id: string;
  status: JobState;
  result?: OptimizeResult;
  error?: string;
}

export interface OptimizeResult {
  /** Latency / throughput / memory snapshot before and after. */
  baseline?: PerfSnapshot;
  optimized?: PerfSnapshot;
  /** Detected architecture (e.g. "llama"); null when unsupported. */
  detected_architecture?: string | null;
  /** Per-call notes — passthrough flag, validation drift, cache hit, etc. */
  notes?: Record<string, unknown>;
}

export interface PerfSnapshot {
  latency_ms: number;
  tokens_per_sec: number;
  tokens_processed: number;
  gpu_utilization?: number | null;
  telemetry?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface ClientOptions {
  /** Base URL of the agnitra-api server. */
  baseUrl: string;
  /** API key sent as `X-Agnitra-Api-Key` header. */
  apiKey?: string;
  /** Timeout for individual HTTP requests, in milliseconds. Default 30000. */
  timeoutMs?: number;
  /** Optional fetch implementation (for Node <18 or testing). */
  fetch?: typeof fetch;
}
