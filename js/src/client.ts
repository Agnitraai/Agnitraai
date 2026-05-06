// Minimal HTTP client for the agnitra-api server.
//
// What this is: a typed wrapper over fetch() that calls /optimize,
// /jobs/{id}, /health, and the /ws/jobs/{id} WebSocket route. Browser
// and Node.js compatible.
//
// What this is NOT: a port of the Python optimizer. The actual model
// optimization happens on the server (which is Python). This client
// only schedules and fetches results.

import {
  AgnitraError,
  AgnitraHttpError,
  AgnitraTimeoutError,
} from "./errors.js";
import type {
  ClientOptions,
  JobStatus,
  OptimizeRequest,
  OptimizeResult,
} from "./types.js";

export class AgnitraClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions) {
    if (!options.baseUrl) {
      throw new AgnitraError("baseUrl is required");
    }
    // Strip trailing slash so callers can pass either form.
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30000;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    if (!this.fetchImpl) {
      throw new AgnitraError(
        "No fetch implementation found. Pass `fetch` in options or run on Node 18+."
      );
    }
  }

  /** GET /health — quick liveness probe. */
  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>("GET", "/health");
  }

  /** POST /optimize — submit a single optimization job. */
  async optimize(request: OptimizeRequest): Promise<JobStatus> {
    return this.request<JobStatus>("POST", "/optimize", request);
  }

  /** GET /jobs/{id} — poll for job status. */
  async getJob(jobId: string): Promise<JobStatus> {
    return this.request<JobStatus>("GET", `/jobs/${encodeURIComponent(jobId)}`);
  }

  /**
   * Wait for a job to reach a terminal state (completed / failed) by
   * polling /jobs/{id}. Returns the final OptimizeResult.
   *
   * Prefer `subscribeToJob` (WebSocket) for low-latency updates. This
   * method exists for environments without WebSocket support.
   */
  async waitForJob(
    jobId: string,
    options: { intervalMs?: number; timeoutMs?: number } = {}
  ): Promise<OptimizeResult> {
    const interval = options.intervalMs ?? 1000;
    const deadline = Date.now() + (options.timeoutMs ?? 600000);
    while (Date.now() < deadline) {
      const status = await this.getJob(jobId);
      if (status.status === "completed" && status.result) {
        return status.result;
      }
      if (status.status === "failed") {
        throw new AgnitraError(`Job ${jobId} failed: ${status.error ?? "unknown"}`);
      }
      await sleep(interval);
    }
    throw new AgnitraTimeoutError(options.timeoutMs ?? 600000);
  }

  /**
   * Subscribe to job updates via WebSocket on /ws/jobs/{id}.
   * The async iterator yields JobStatus updates until the job terminates,
   * then closes the socket.
   *
   * Browser-compatible (uses native WebSocket) and Node-compatible
   * (Node 22+ has native WebSocket; older Node needs `ws` polyfill).
   */
  async *subscribeToJob(jobId: string): AsyncIterable<JobStatus> {
    const wsUrl =
      this.baseUrl.replace(/^http/, "ws") +
      `/ws/jobs/${encodeURIComponent(jobId)}`;
    const WSImpl: typeof WebSocket | undefined = (globalThis as { WebSocket?: typeof WebSocket }).WebSocket;
    if (!WSImpl) {
      throw new AgnitraError(
        "WebSocket not available. Use waitForJob() or polyfill with `ws`."
      );
    }
    const socket = new WSImpl(wsUrl);
    const queue: JobStatus[] = [];
    let resolver: ((value: void) => void) | null = null;
    let closed = false;
    let errored: Error | null = null;

    socket.addEventListener("message", (ev: MessageEvent) => {
      try {
        const parsed = JSON.parse(typeof ev.data === "string" ? ev.data : "") as JobStatus;
        queue.push(parsed);
        resolver?.();
      } catch (err) {
        errored = err instanceof Error ? err : new Error(String(err));
        resolver?.();
      }
    });
    socket.addEventListener("close", () => {
      closed = true;
      resolver?.();
    });
    socket.addEventListener("error", (ev: Event) => {
      errored = new Error(`WebSocket error: ${(ev as ErrorEvent).message ?? "unknown"}`);
      resolver?.();
    });

    try {
      while (!closed) {
        if (queue.length === 0) {
          await new Promise<void>((resolve) => {
            resolver = resolve;
          });
          resolver = null;
        }
        if (errored) throw errored;
        while (queue.length > 0) {
          const status = queue.shift()!;
          yield status;
          if (status.status === "completed" || status.status === "failed") {
            return;
          }
        }
      }
    } finally {
      try {
        socket.close();
      } catch {
        // Ignore close failures during cleanup.
      }
    }
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    if (this.apiKey) {
      headers["X-Agnitra-Api-Key"] = this.apiKey;
    }
    let response: Response;
    try {
      response = await this.fetchImpl(this.baseUrl + path, {
        method,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new AgnitraTimeoutError(this.timeoutMs);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new AgnitraHttpError(response.status, response.statusText, text);
    }
    return (await response.json()) as T;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
