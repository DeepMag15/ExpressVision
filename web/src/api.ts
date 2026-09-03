/* Typed client for the console API.
 *
 * The types mirror src/expressvision/api/queries.py. Where a number can be
 * `null` that is load-bearing rather than sloppy: a null rate means the store
 * cannot say what the rate is, and the views must render that as "coverage
 * unknown" rather than as zero. Typing those fields as `number | null` makes
 * TypeScript refuse to let a view forget the distinction.
 */

export type Verdict = "confirmed" | "rejected" | "reclassified";

export const LABELS = [
  "rodent",
  "bird",
  "reptile",
  "carnivore",
  "insect",
  "human",
  "other",
] as const;

export type Label = (typeof LABELS)[number];

/** Keyboard shortcuts for reclassification, matching the offline review sheet. */
export const LABEL_KEYS: Record<string, Label> = {
  "1": "rodent",
  "2": "bird",
  "3": "reptile",
  "4": "carnivore",
  "5": "insect",
  "6": "human",
  "7": "other",
};

export interface Coverage {
  runs: number;
  runs_missing_uptime: number;
  observed_seconds: number;
  observed_hours: number;
  /** False when any run lacks uptime; every derived rate is then null. */
  complete: boolean;
}

export interface Health {
  ok: boolean;
  store: string;
  store_exists: boolean;
  demo: boolean;
}

export interface Overview {
  demo: boolean;
  cameras: number;
  counts: {
    runs: number;
    events: number;
    unverified: number;
    confirmed: number;
    rejected: number;
    reclassified: number;
    judged: number;
    discarded_tracks: number;
  };
  coverage: Coverage;
  events_per_camera_hour: number | null;
  review_progress: number;
  first_event_at: string | null;
  last_event_at: string | null;
}

export interface EventRow {
  id: string;
  run_id: string;
  camera_id: string;
  site_id: string;
  track_id: number;
  label: string;
  started_at: string;
  duration_s: number;
  n_detections: number;
  displacement_px: number;
  straightness: number;
  heading_deg: number | null;
  origin_x: number | null;
  origin_y: number | null;
  terminus_x: number | null;
  terminus_y: number | null;
  clip_sha256: string | null;
  model_version: string;
  verdict: Verdict | null;
  corrected_label: string | null;
  verified_by: string | null;
  verified_at: string | null;
  has_keyframe: boolean;
  has_clip: boolean;
}

export interface EventPage {
  total: number;
  limit: number;
  offset: number;
  events: EventRow[];
}

export interface CameraRow {
  camera_id: string;
  events: number;
  pending: number;
  confirmed: number;
  rejected: number;
  last_event_at: string | null;
  last_run_at: string | null;
  coverage: Coverage;
  events_per_camera_hour: number | null;
  gate_pass_rate: number | null;
  tiles_per_gated_frame: number | null;
  tamper_frames: number;
  global_change_frames: number;
}

export interface FunnelStage {
  key: string;
  label: string;
  count: number;
  share_of_previous: number | null;
}

export interface Funnel {
  stages: FunnelStage[];
  gate_pass_rate: number | null;
  tiles_per_gated_frame: number | null;
  inference_saving: number | null;
  global_change_frames: number;
  tamper_frames: number;
}

export interface FeatureSummary {
  n: number;
  min: number;
  p50: number;
  max: number;
  mean: number;
}

export interface RejectionReason {
  reason: string;
  count: number;
  features: Record<string, FeatureSummary | null>;
}

export interface Rejections {
  total: number;
  reasons: RejectionReason[];
}

export interface RunRow {
  id: string;
  started_at: string;
  finished_at: string | null;
  camera_id: string;
  source: string;
  fps: number | null;
  observed_seconds: number | null;
  frames_processed: number | null;
  frames_gated: number | null;
  tiles: number | null;
  events: number | null;
  tamper_frames: number | null;
  rejected: Record<string, number>;
}

export interface OriginPoint {
  x: number;
  y: number;
  tx: number | null;
  ty: number | null;
  label: string;
  verdict: Verdict | null;
}

export interface Analytics {
  coverage: Coverage;
  hour_of_day: { timezone: string; bins: number[]; normalised: boolean };
  headings: number[];
  origins: OriginPoint[];
  labels: Record<string, number>;
  durations: FeatureSummary | null;
  n: number;
  extent: { width: number; height: number };
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    // A dead backend is the common case during development, and "Failed to
    // fetch" tells an operator nothing useful.
    throw new ApiError(
      "Cannot reach the API. Is `exv serve` running?",
      0,
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* a non-JSON error body is not worth failing over */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

export const api = {
  health: () => request<Health>("/api/health"),
  overview: () => request<Overview>("/api/overview"),

  events: (params: {
    camera_id?: string;
    verdict?: string;
    label?: string;
    limit?: number;
    offset?: number;
    order?: "asc" | "desc";
  }) => request<EventPage>(`/api/events${query(params)}`),

  setVerdict: (id: string, verdict: Verdict, corrected_label?: Label) =>
    request<{ ok: boolean }>(`/api/events/${id}/verdict`, {
      method: "POST",
      body: JSON.stringify({ verdict, corrected_label: corrected_label ?? null }),
    }),

  cameras: () => request<CameraRow[]>("/api/cameras"),
  runs: () => request<RunRow[]>("/api/runs"),
  funnel: (run_id?: string) => request<Funnel>(`/api/funnel${query({ run_id })}`),
  rejections: () => request<Rejections>("/api/rejections"),
  analytics: (camera_id?: string) => request<Analytics>(`/api/analytics${query({ camera_id })}`),

  keyframeUrl: (id: string) => `/api/events/${id}/keyframe`,
  clipUrl: (id: string) => `/api/events/${id}/clip`,
};
