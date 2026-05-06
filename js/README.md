# `agnitra` (npm)

Minimal TypeScript HTTP client for the [Agnitra](https://github.com/agnitraai/agnitraai) inference-optimization API. Browser- and Node.js-compatible.

```bash
npm install agnitra
```

## What this is

A typed wrapper over `fetch` and `WebSocket` for calling the `agnitra-api` server (`POST /optimize`, `GET /jobs/{id}`, `WebSocket /ws/jobs/{id}`, `GET /health`).

## What this is NOT

A port of the Agnitra optimizer. The actual model optimization runs in the Python SDK on the server. This client only schedules and fetches results.

If you want to optimize PyTorch models, install the Python package: `pip install agnitra`.

## Usage

```typescript
import { AgnitraClient } from "agnitra";

const client = new AgnitraClient({
  baseUrl: "https://your-agnitra-server.example.com",
  apiKey: process.env.AGNITRA_API_KEY,
});

// Submit a job and wait for completion via polling.
const job = await client.optimize({
  project_id: "my-project",
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct",
  target: "H100",
});

const result = await client.waitForJob(job.job_id);
console.log("speedup:", result.baseline?.latency_ms, "->", result.optimized?.latency_ms);
```

### Streaming updates via WebSocket

```typescript
for await (const update of client.subscribeToJob(jobId)) {
  console.log(update.status, update.result?.optimized);
  if (update.status === "completed" || update.status === "failed") break;
}
```

## Versioning

This package's version tracks the Python `agnitra` package. `agnitra@0.2.0` (npm) is compatible with `agnitra==0.2.0` (PyPI) and the `agnitra-api` server in that release.

## License

[Apache 2.0](LICENSE).
