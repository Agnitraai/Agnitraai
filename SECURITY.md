# Security policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security
vulnerabilities. Instead:

- Email **security@agnitra.ai** with the details.
- We aim to acknowledge within 72 hours and provide a timeline for
  remediation within 7 days.

If you don't get a response in a week, escalate by mentioning
`@drvt69talati` on a private GitHub Security Advisory at
https://github.com/Agnitraai/Agnitraai/security/advisories.

## What to include

- Affected version(s) (`pip show agnitra`).
- Reproduction steps. The shorter, the better.
- Impact assessment. RCE? Data exfiltration? Denial of service?
- Suggested fix or mitigation, if you have one.

## Scope

In scope:

- The Python SDK (`agnitra`)
- The npm HTTP client (`agnitra` on npm)
- The `agnitra-api` HTTP server
- The CLI (`agnitra`, `agnitra-api`, `agnitra-optimize`)
- The benchmark harness in `benchmarks/`

Out of scope:

- Vulnerabilities in upstream dependencies (report those upstream;
  notify us if a transitive issue specifically needs to be vendored).
- Issues in pre-built TensorRT-LLM or NIM containers — report those
  to NVIDIA directly.

## What Agnitra collects

By default, Agnitra is a local SDK and collects nothing. With explicit
configuration:

- `OPENAI_API_KEY` set + `enable_rl=True`: graph IR (no weights, no
  inputs, no outputs) is sent to OpenAI for kernel suggestions.
- `AGNITRA_NOTIFY_WEBHOOK_URL` set: optimization summaries are POSTed
  to the configured Slack / Discord / Telegram / generic webhook.
- `AGNITRA_LICENSE_PATH` set: the license file is read at optimize
  time. Seat IDs are derived from the workload fingerprint.

The default specialist optimization path (`use_specialist=True`,
default since 0.2.0) makes **no network calls**.

## Token handling

Agnitra never logs API keys or auth tokens. Environment-variable
based configuration is the recommended pattern for all secrets.
Never paste production tokens into issue reports — redact them first.
