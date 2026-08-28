import { getBackendBaseURL } from "../config";

import { fetchWithTimeout } from "./fetch";
import { fetch } from "./fetcher";

export interface FeedbackData {
  feedback_id: string;
  /** Fields returned by the run-scoped feedback endpoint. */
  run_id?: string;
  thread_id?: string;
  rating: number;
  comment: string | null;
  created_at?: string;
  updated_at?: string | null;
}

export async function upsertFeedback(
  threadId: string,
  runId: string,
  rating: number,
  comment?: string,
): Promise<FeedbackData> {
  const res = await fetch(
    `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/feedback`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating, comment: comment ?? null }),
    },
  );
  if (!res.ok) {
    throw new Error(`Failed to submit feedback: ${res.status}`);
  }
  return res.json();
}

export async function deleteFeedback(
  threadId: string,
  runId: string,
): Promise<void> {
  const res = await fetch(
    `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/feedback`,
    { method: "DELETE" },
  );
  if (!res.ok && res.status !== 404) {
    throw new Error(`Failed to delete feedback: ${res.status}`);
  }
}

/**
 * Compatibility API retained for the original Anaxa message controls.
 *
 * DeerFlow's newer UI uses `upsertFeedback`/`deleteFeedback`; older callers
 * use the more explicit run-scoped names and expect the timeout wrapper. Keep
 * both surfaces pointed at the same Gateway endpoint while leaving the newer
 * CSRF-aware mutation path untouched.
 */
export async function getRunFeedback(
  threadId: string,
  runId: string,
): Promise<FeedbackData | null> {
  const response = await fetchWithTimeout(feedbackURL(threadId, runId));
  if (!response.ok) {
    throw new Error(await feedbackError(response, "Failed to load feedback."));
  }
  return response.json() as Promise<FeedbackData | null>;
}

export async function putRunFeedback(
  threadId: string,
  runId: string,
  rating: 1 | -1,
): Promise<FeedbackData> {
  const response = await fetchWithTimeout(feedbackURL(threadId, runId), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rating }),
  });
  if (!response.ok) {
    throw new Error(await feedbackError(response, "Failed to save feedback."));
  }
  return response.json() as Promise<FeedbackData>;
}

export async function deleteRunFeedback(
  threadId: string,
  runId: string,
): Promise<void> {
  const response = await fetchWithTimeout(feedbackURL(threadId, runId), {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 404) {
    throw new Error(await feedbackError(response, "Failed to delete feedback."));
  }
}

function feedbackURL(threadId: string, runId: string) {
  return `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/feedback`;
}

async function feedbackError(response: Response, fallback: string) {
  const payload = (await response.json().catch(() => ({}))) as {
    detail?: string;
  };
  return payload.detail ?? fallback;
}
