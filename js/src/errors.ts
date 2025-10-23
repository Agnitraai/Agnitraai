export class AgnitraError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgnitraError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class AgnitraHttpError extends AgnitraError {
  readonly status: number;
  readonly data?: unknown;
  readonly headers: Headers;

  constructor(message: string, status: number, data: unknown, headers: Headers) {
    super(message);
    this.name = "AgnitraHttpError";
    this.status = status;
    this.data = data;
    this.headers = headers;
  }
}

export class AgnitraTimeoutError extends AgnitraError {
  readonly timeoutMs: number;

  constructor(message: string, timeoutMs: number) {
    super(message);
    this.name = "AgnitraTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}
