import { getBackendBaseURL } from "../config";

import type { Model } from "./types";

export async function loadModels() {
  const res = await fetch(`${getBackendBaseURL()}/api/models`);
  const payload = (await res.json().catch(() => null)) as unknown;
  if (!res.ok) {
    const detail =
      typeof payload === "object" && payload !== null
        ? Reflect.get(payload, "detail")
        : undefined;
    throw new Error(
      typeof detail === "string" && detail.trim()
        ? detail
        : `Model request failed (${res.status}).`,
    );
  }

  const models =
    typeof payload === "object" && payload !== null
      ? Reflect.get(payload, "models")
      : undefined;
  if (!Array.isArray(models)) {
    throw new Error("Model response was invalid.");
  }
  return models as Model[];
}
