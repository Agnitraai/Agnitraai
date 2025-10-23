import packageJson from "../package.json";

export { AgnitraClient, createAgnitraClient } from "./client";
export { AgnitraError, AgnitraHttpError, AgnitraTimeoutError } from "./errors";
export type {
  AgnitraClientOptions,
  DispatchResult,
  JobStatusResponse,
  OptimizeQueueResponse,
  OptimizeRequest,
  OptimizeResponse,
  RequestOptions,
  UsageEvent,
  UsageRequest,
  UsageResponse,
  UsageSnapshotInput
} from "./types";

export const VERSION = packageJson.version;
