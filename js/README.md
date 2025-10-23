# Agnitra JavaScript SDK

`agnitra` is the JavaScript/TypeScript SDK for Agnitra's optimization API and usage-based metering service. It mirrors the Python SDK so browser and Node.js workloads can submit optimization jobs, poll asynchronous queues, and forward pay-per-optimization telemetry to cloud marketplaces.

## Installation

```bash
npm install agnitra
# or
yarn add agnitra
```

Requires Node.js 18+ (or any runtime that provides the Fetch API).

## Quick Start

```ts
import { AgnitraClient } from "agnitra";

const client = new AgnitraClient({
  apiKey: process.env.AGNITRA_API_KEY,
  baseUrl: process.env.AGNITRA_API_BASE_URL ?? "http://127.0.0.1:8080"
});

const result = await client.optimize({
  target: "A100",
  modelGraph: graphPayload,   // JSON payload or object describing the FX graph
  telemetry: telemetryPayload // JSON payload or object with profiler telemetry
});

console.log(result.bottleneck.expected_speedup_pct);
```

Submit a background optimization job and poll for completion:

```ts
const queued = await client.optimizeAsync({
  target: "A100",
  modelGraph,
  telemetry,
  webhookUrl: "https://example.com/webhooks/agnitra"
});

const status = await client.getJob(queued.job_id);
if (status.status === "completed" && status.result) {
  console.log(status.result.kernel);
}
```

Report usage to cloud marketplace adapters:

```ts
await client.recordUsage({
  projectId: "demo-project",
  modelName: "tinyllama",
  baseline: { latencyMs: 120, tokensPerSec: 90 },
  optimized: { latencyMs: 78, tokensPerSec: 138 },
  providers: ["aws", "gcp"]
});
```

See the generated TypeScript definitions in `dist/index.d.ts` for the full surface area.

## Publishing

To ship a new version of the SDK:

```bash
cd js
npm version <major|minor|patch>
npm publish
```

The `prepare` script builds the TypeScript sources before publishing. Ensure you have staged the compiled `dist/` artifacts (generated automatically by `npm publish`) when testing locally via `npm pack`.
