import packageJson from "../package.json";

import {
  AgnitraClientOptions,
  JobStatusResponse,
  OptimizeQueueResponse,
  OptimizeRequest,
  OptimizeResponse,
  RequestOptions,
  UsageRequest,
  UsageResponse
} from "./types";
import { AgnitraError, AgnitraHttpError, AgnitraTimeoutError } from "./errors";

type FetchLike = typeof fetch;

const DEFAULT_TIMEOUT_MS = 30_000;

function getEnv(): Record<string, string> | undefined {
  if (typeof process !== "undefined" && process.env) {
    return process.env as Record<string, string>;
  }
  return undefined;
}

function getDefaultFetch(): FetchLike {
  if (typeof fetch === "function") {
    return fetch;
  }
  throw new AgnitraError(
    "Fetch API is not available in this environment. Provide a custom fetch implementation via AgnitraClientOptions.fetch."
  );
}

function ensureNonEmpty(value: string | undefined | null, label: string): string {
  if (!value || !value.trim()) {
    throw new AgnitraError(`${label} must be a non-empty string.`);
  }
  return value.trim();
}

function mergeHeaders(base: Headers, extra?: HeadersInit): Headers {
  if (!extra) {
    return base;
  }
  const additional = new Headers(extra);
  additional.forEach((value, key) => {
    base.set(key, value);
  });
  return base;
}

function toNumber(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export class AgnitraClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly projectId?: string;
  private readonly fetchImpl: FetchLike;
  private readonly defaultHeaders: HeadersInit;
  private readonly userAgent?: string;
  private readonly defaultTimeoutMs: number | undefined;

  constructor(options: AgnitraClientOptions = {}) {
    const env = getEnv();
    const defaultBase =
      env?.AGNITRA_API_BASE_URL ??
      env?.AGNITRA_BASE_URL ??
      env?.AGNITRA_API_URL ??
      "http://127.0.0.1:8080";
    const configuredBase = options.baseUrl ?? defaultBase;
    this.baseUrl = this.normaliseBaseUrl(configuredBase);
    this.apiKey = options.apiKey ?? env?.AGNITRA_API_KEY ?? undefined;
    this.projectId = options.projectId ?? env?.AGNITRA_PROJECT_ID ?? undefined;
    this.fetchImpl = options.fetch ?? getDefaultFetch();
    this.userAgent = options.userAgent ?? `agnitra-js/${packageJson.version}`;
    this.defaultTimeoutMs =
      typeof options.timeoutMs === "number"
        ? options.timeoutMs
        : env?.AGNITRA_HTTP_TIMEOUT_MS
        ? toNumber(env.AGNITRA_HTTP_TIMEOUT_MS, DEFAULT_TIMEOUT_MS)
        : DEFAULT_TIMEOUT_MS;
    this.defaultHeaders = {
      Accept: "application/json",
      ...options.defaultHeaders
    };
  }

  /**
   * Run the optimization pipeline synchronously and return the enriched response.
   */
  async optimize(request: OptimizeRequest, options: RequestOptions = {}): Promise<OptimizeResponse> {
    const payload = this.buildOptimizePayload(request, false);
    const headers: HeadersInit = {
      "Content-Type": "application/json"
    };
    if (request.customerId) {
      headers["x-customer-id"] = request.customerId;
    }
    if (request.webhookUrl) {
      headers["x-webhook-url"] = request.webhookUrl;
    }

    return this.request<OptimizeResponse>("/optimize", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: mergeHeaders(new Headers(headers), options.headers),
      timeoutMs: options.timeoutMs,
      signal: options.signal ?? null
    });
  }

  /**
   * Enqueue an optimization job and receive the job identifier.
   */
  async optimizeAsync(request: OptimizeRequest, options: RequestOptions = {}): Promise<OptimizeQueueResponse> {
    const payload = this.buildOptimizePayload(request, true);
    const headers: HeadersInit = {
      "Content-Type": "application/json"
    };
    if (request.customerId) {
      headers["x-customer-id"] = request.customerId;
    }
    if (request.webhookUrl) {
      headers["x-webhook-url"] = request.webhookUrl;
    }

    return this.request<OptimizeQueueResponse>("/optimize", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: mergeHeaders(new Headers(headers), options.headers),
      timeoutMs: options.timeoutMs,
      signal: options.signal ?? null
    });
  }

  /**
   * Retrieve the status (and result) of a previously queued optimization job.
   */
  async getJob(jobId: string, options: RequestOptions = {}): Promise<JobStatusResponse> {
    const identifier = ensureNonEmpty(jobId, "jobId");
    const path = `/jobs/${encodeURIComponent(identifier)}`;
    return this.request<JobStatusResponse>(path, {
      method: "GET",
      headers: options.headers,
      timeoutMs: options.timeoutMs,
      signal: options.signal ?? null
    });
  }

  /**
   * Record a usage event for marketplace dispatching.
   */
  async recordUsage(request: UsageRequest, options: RequestOptions = {}): Promise<UsageResponse> {
    const payload = this.buildUsagePayload(request);
    return this.request<UsageResponse>("/usage", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: mergeHeaders(
        new Headers({
          "Content-Type": "application/json"
        }),
        options.headers
      ),
      timeoutMs: options.timeoutMs,
      signal: options.signal ?? null
    });
  }

  private normaliseBaseUrl(url: string): string {
    const trimmed = url.trim();
    if (!trimmed) {
      return "http://127.0.0.1:8080";
    }
    return trimmed.replace(/\/+$/, "");
  }

  private buildOptimizePayload(request: OptimizeRequest, forceAsync: boolean): Record<string, unknown> {
    if (!request) {
      throw new AgnitraError("Optimize request payload is required.");
    }
    const target = ensureNonEmpty(request.target, "target");
    if (typeof request.modelGraph === "undefined" || request.modelGraph === null) {
      throw new AgnitraError("optimize request requires a modelGraph payload.");
    }
    if (typeof request.telemetry === "undefined" || request.telemetry === null) {
      throw new AgnitraError("optimize request requires a telemetry payload.");
    }

    const payload: Record<string, unknown> = {
      target,
      model_graph: request.modelGraph,
      telemetry: request.telemetry,
      project_id: request.projectId ?? this.projectId ?? "default"
    };

    if (request.modelName) {
      payload.model_name = request.modelName;
    }
    if (typeof request.tokensProcessed === "number") {
      payload.tokens_processed = request.tokensProcessed;
    }
    if (request.metadata) {
      payload.metadata = request.metadata;
    }
    if (request.customerId) {
      payload.customer_id = request.customerId;
    }
    if (request.webhookUrl) {
      payload.webhook_url = request.webhookUrl;
    }

    const shouldQueue =
      forceAsync ||
      request.async === true ||
      request.queue === true ||
      request.mode?.toLowerCase() === "async";
    if (shouldQueue) {
      payload.async = true;
    }

    return payload;
  }

  private buildUsagePayload(request: UsageRequest): Record<string, unknown> {
    if (!request) {
      throw new AgnitraError("Usage request payload is required.");
    }
    const projectId = ensureNonEmpty(request.projectId, "projectId");

    const payload: Record<string, unknown> = {
      project_id: projectId
    };

    if (request.meterName) {
      payload.meter_name = request.meterName;
    }
    if (request.quantityField) {
      payload.quantity_field = request.quantityField;
    }
    if (request.providers?.length) {
      payload.providers = request.providers;
    }
    if (request.metadata) {
      payload.metadata = request.metadata;
    }
    if (typeof request.tokensProcessed === "number") {
      payload.tokens_processed = request.tokensProcessed;
    }
    if (typeof request.ratePerGpuHour === "number") {
      payload.rate_per_gpu_hour = request.ratePerGpuHour;
    }
    if (typeof request.successMarginPct === "number") {
      payload.success_margin_pct = request.successMarginPct;
    }
    if (request.currency) {
      payload.currency = request.currency;
    }
    if (request.modelName) {
      payload.model_name = request.modelName;
    }

    if (request.usageEvent) {
      payload.usage_event = request.usageEvent;
      return payload;
    }

    if (!request.baseline || !request.optimized) {
      throw new AgnitraError("baseline and optimized snapshots are required when usageEvent is not provided.");
    }

    payload.baseline = this.snapshotToPayload(request.baseline, "baseline");
    payload.optimized = this.snapshotToPayload(request.optimized, "optimized");

    return payload;
  }

  private snapshotToPayload(snapshot: UsageRequest["baseline"], label: string): Record<string, unknown> {
    if (typeof snapshot.latencyMs !== "number") {
      throw new AgnitraError(`${label} snapshot must include latencyMs.`);
    }
    if (typeof snapshot.tokensPerSec !== "number") {
      throw new AgnitraError(`${label} snapshot must include tokensPerSec.`);
    }
    const payload: Record<string, unknown> = {
      latency_ms: snapshot.latencyMs,
      tokens_per_sec: snapshot.tokensPerSec
    };
    if (typeof snapshot.tokensProcessed === "number") {
      payload.tokens_processed = snapshot.tokensProcessed;
    }
    if (typeof snapshot.gpuUtilization === "number") {
      payload.gpu_utilization = snapshot.gpuUtilization;
    }
    if (snapshot.telemetry) {
      payload.telemetry = snapshot.telemetry;
    }
    return payload;
  }

  private async request<T>(path: string, init: RequestInit & RequestOptions): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers = mergeHeaders(new Headers(this.defaultHeaders), init.headers);
    if (this.apiKey && !headers.has("x-api-key")) {
      headers.set("x-api-key", this.apiKey);
    }
    if (this.userAgent && typeof window === "undefined" && !headers.has("user-agent")) {
      headers.set("user-agent", this.userAgent);
    }

    const effectiveTimeout = init.timeoutMs ?? this.defaultTimeoutMs;
    const upstreamSignal = init.signal ?? null;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let upstreamListener: (() => void) | undefined;

    if (typeof effectiveTimeout === "number" && effectiveTimeout > 0) {
      controller = new AbortController();
      timeoutId = setTimeout(() => controller?.abort(), effectiveTimeout);
      if (upstreamSignal) {
        if (upstreamSignal.aborted) {
          controller.abort();
        } else {
          upstreamListener = () => controller?.abort();
          upstreamSignal.addEventListener("abort", upstreamListener, { once: true });
        }
      }
    }

    const signal = controller?.signal ?? upstreamSignal ?? undefined;

    try {
      const response = await this.fetchImpl(url, {
        ...init,
        headers,
        signal
      });
      const text = await response.text();
      let data: unknown = undefined;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
      }
      if (!response.ok) {
        throw new AgnitraHttpError(
          `Request to ${url} failed with status ${response.status}`,
          response.status,
          data,
          response.headers
        );
      }
      return data as T;
    } catch (error: unknown) {
      if (controller?.signal.aborted && (effectiveTimeout ?? 0) > 0) {
        throw new AgnitraTimeoutError(`Request to ${url} timed out after ${effectiveTimeout}ms`, effectiveTimeout!);
      }
      if (error instanceof AgnitraError) {
        throw error;
      }
      const message = error instanceof Error ? error.message : String(error);
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new AgnitraError(`Request to ${url} was aborted: ${message}`);
      }
      throw new AgnitraError(`Request to ${url} failed: ${message}`);
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (upstreamListener && init.signal) {
        init.signal.removeEventListener("abort", upstreamListener);
      }
    }
  }
}

export function createAgnitraClient(options: AgnitraClientOptions = {}): AgnitraClient {
  return new AgnitraClient(options);
}
