// Discrete error classes so callers can `instanceof`-check rather
// than parsing message strings.

export class AgnitraError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgnitraError";
  }
}

export class AgnitraHttpError extends AgnitraError {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly body: string
  ) {
    super(`HTTP ${status} ${statusText}: ${body.slice(0, 200)}`);
    this.name = "AgnitraHttpError";
  }
}

export class AgnitraTimeoutError extends AgnitraError {
  constructor(public readonly timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`);
    this.name = "AgnitraTimeoutError";
  }
}
