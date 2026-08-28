import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import type {
  AdvanceResearchQuestRequest,
  CreateResearchQuestRequest,
  GateStatus,
  ResearchAdvanceResult,
  ResearchGate,
  ResearchQuest,
  ResearchQuestSnapshot,
  ResearchStage,
} from "./types";

async function parseError(res: Response, fallback: string): Promise<Error> {
  const data = (await res.json().catch(() => ({}))) as { detail?: unknown };
  return new Error(typeof data.detail === "string" ? data.detail : fallback);
}

export async function listResearchQuests(
  threadId?: string,
): Promise<ResearchQuest[]> {
  const params = threadId ? `?thread_id=${encodeURIComponent(threadId)}` : "";
  const res = await fetchWithAuth(
    `${getBackendBaseURL()}/api/research/quests${params}`,
  );
  if (!res.ok) {
    throw await parseError(
      res,
      `Failed to load research quests: ${res.statusText}`,
    );
  }
  const data = (await res.json()) as { data: ResearchQuest[] };
  return data.data;
}

export async function createResearchQuest(
  request: CreateResearchQuestRequest,
): Promise<ResearchQuest> {
  const res = await fetchWithAuth(
    `${getBackendBaseURL()}/api/research/quests`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    },
  );
  if (!res.ok) {
    throw await parseError(
      res,
      `Failed to create research quest: ${res.statusText}`,
    );
  }
  const data = (await res.json()) as { quest: ResearchQuest };
  return data.quest;
}

export async function getResearchQuest(
  questId: string,
): Promise<ResearchQuestSnapshot> {
  const res = await fetchWithAuth(
    `${getBackendBaseURL()}/api/research/quests/${encodeURIComponent(questId)}`,
  );
  if (!res.ok) {
    throw await parseError(
      res,
      `Failed to load research quest: ${res.statusText}`,
    );
  }
  return (await res.json()) as ResearchQuestSnapshot;
}

export async function advanceResearchQuest(
  questId: string,
  request: AdvanceResearchQuestRequest = {},
): Promise<ResearchAdvanceResult> {
  const res = await fetchWithAuth(
    `${getBackendBaseURL()}/api/research/quests/${encodeURIComponent(questId)}/advance`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    },
  );
  if (!res.ok) {
    throw await parseError(
      res,
      `Failed to advance research quest: ${res.statusText}`,
    );
  }
  return (await res.json()) as ResearchAdvanceResult;
}

export async function decideResearchGate({
  questId,
  stage,
  gateType,
  status,
  reason,
}: {
  questId: string;
  stage: ResearchStage;
  gateType: string;
  status: GateStatus;
  reason?: string;
}): Promise<ResearchGate> {
  const res = await fetchWithAuth(
    `${getBackendBaseURL()}/api/research/quests/${encodeURIComponent(questId)}/gate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stage,
        gate_type: gateType,
        status,
        reason,
      }),
    },
  );
  if (!res.ok) {
    throw await parseError(
      res,
      `Failed to update research gate: ${res.statusText}`,
    );
  }
  const data = (await res.json()) as { gate: ResearchGate };
  return data.gate;
}
