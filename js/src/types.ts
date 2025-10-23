export interface AgnitraClientOptions {
  /**
   * Base URL for the Agnitra API (e.g. https://api.agnitra.ai).
   * Defaults to the AGNITRA_API_BASE_URL environment variable or http://127.0.0.1:8080.
   */
  baseUrl?: string;
  /**
   * API key used for authentication. Falls back to AGNITRA_API_KEY when omitted.
   */
  apiKey?: string;
  /**
   * Default project identifier applied to optimize calls when not provided explicitly.
   */
  projectId?: string;
  /**
   * Additional headers merged into every request.
   */
  defaultHeaders?: HeadersInit;
  /**
   * Custom fetch implementation (useful for tests or non-Node environments).
   */
  fetch?: typeof fetch;
  /**
   * Custom User-Agent header. Defaults to agnitra-js/<package-version>.
   */
  userAgent?: string;
  /**
   * Default request timeout in milliseconds (applies to all requests unless overridden).
   */
  timeoutMs?: number;
}

export interface RequestOptions {
  signal?: AbortSignal | null;
  headers?: HeadersInit;
  timeoutMs?: number;
}

export interface OptimizeRequest {
  /**
   * Target accelerator label (e.g. "A100", "H100").
   */
  target: string;
  /**
   * FX graph payload describing the profiled model.
   */
  modelGraph: unknown;
  /**
   * Profiler telemetry payload associated with the graph.
   */
  telemetry: unknown;
  projectId?: string;
  modelName?: string;
  tokensProcessed?: number;
  metadata?: Record<string, unknown>;
  customerId?: string;
  webhookUrl?: string;
  /**
   * When true the API enqueues the optimization job and returns immediately.
   */
  async?: boolean;
  /**
   * Alias for async (kept for ergonomics when mapping to queue=true flags).
   */
  queue?: boolean;
  /**
   * Explicit mode override ("async" requests queue, "sync" runs inline).
   */
  mode?: "sync" | "async";
}

export interface OptimizeResponse {
  target: string;
  telemetry_summary: Record<string, unknown>;
  bottleneck: {
    name?: string;
    op?: string;
    baseline_latency_ms: number;
    expected_latency_ms: number;
    expected_speedup_pct: number;
  };
  ir_graph: {
    nodes: unknown[];
    metadata: Record<string, unknown>;
  };
  kernel: Record<string, unknown>;
  patch_instructions: unknown[];
  usage?: UsageEvent;
  billing?: Record<string, unknown>;
}

export interface OptimizeQueueResponse {
  status: string;
  job_id: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  result?: OptimizeResponse;
  error?: unknown;
}

export interface UsageSnapshotInput {
  latencyMs: number;
  tokensPerSec: number;
  tokensProcessed?: number;
  gpuUtilization?: number;
  telemetry?: Record<string, unknown>;
}

export interface UsageRequest {
  projectId: string;
  modelName?: string;
  baseline: UsageSnapshotInput;
  optimized: UsageSnapshotInput;
  tokensProcessed?: number;
  ratePerGpuHour?: number;
  successMarginPct?: number;
  currency?: string;
  providers?: string[];
  metadata?: Record<string, unknown>;
  meterName?: string;
  quantityField?: string;
  /**
    * Fully precomputed usage event; when provided the snapshot fields are ignored.
    */
  usageEvent?: UsageEvent;
}

export interface UsageEvent {
  project_id: string;
  model_name: string;
  tokens_processed: number;
  baseline_latency_ms: number;
  optimized_latency_ms: number;
  baseline_tokens_per_sec: number;
  optimized_tokens_per_sec: number;
  gpu_util_before: number | null;
  gpu_util_after: number | null;
  gpu_hours_before: number;
  gpu_hours_after: number;
  gpu_hours_saved: number;
  performance_uplift_pct: number;
  cost_before: number;
  cost_after: number;
  cost_savings: number;
  usage_charge: number;
  success_fee: number;
  total_billable: number;
  currency: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface DispatchResult {
  provider: string;
  status: string;
  detail: string;
  payload: Record<string, unknown>;
}

export interface UsageResponse {
  status: string;
  usage_event: UsageEvent;
  dispatch: DispatchResult[];
}
