import { getBackendBaseURL } from "../config";
import type { AgentThread } from "../threads";

export interface ThreadArtifactRecord {
  filepath: string;
  filename?: string;
  size?: number;
  modified_at?: string;
}

export interface ArtifactInventoryEntry {
  filepath: string;
  size?: number;
  modifiedAt?: string;
  source: "thread" | "outputs";
}

function decodePathSegment(segment: string) {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

function encodeArtifactPath(filepath: string) {
  return filepath
    .split("/")
    .map((segment) => encodeURIComponent(decodePathSegment(segment)))
    .join("/");
}

function splitPathSuffix(src: string) {
  const [path = ""] = src.split(/[?#]/, 1);
  return { path, suffix: src.slice(path.length) };
}

export function urlOfArtifact({
  filepath,
  threadId,
  download = false,
  isMock = false,
}: {
  filepath: string;
  threadId: string;
  download?: boolean;
  isMock?: boolean;
}) {
  const encodedThreadId = encodeURIComponent(threadId);
  const encodedFilepath = encodeArtifactPath(filepath);
  if (isMock) {
    return `${getBackendBaseURL()}/mock/api/threads/${encodedThreadId}/artifacts${encodedFilepath}${download ? "?download=true" : ""}`;
  }
  return `${getBackendBaseURL()}/api/threads/${encodedThreadId}/artifacts${encodedFilepath}${download ? "?download=true" : ""}`;
}

export function extractArtifactsFromThread(thread: AgentThread) {
  return thread.values.artifacts ?? [];
}

export function resolveArtifactURL(absolutePath: string, threadId: string) {
  const { path, suffix } = splitPathSuffix(absolutePath);
  return `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/artifacts${encodeArtifactPath(path)}${suffix}`;
}

function parseArtifactTime(value?: string) {
  if (!value) {
    return Number.NEGATIVE_INFINITY;
  }
  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp;
}

export function mergeArtifactEntries(
  threadArtifacts: string[],
  discoveredArtifacts: ThreadArtifactRecord[],
): ArtifactInventoryEntry[] {
  const seedOrder = new Map(
    threadArtifacts.map((filepath, index) => [filepath, index]),
  );
  const merged = new Map<string, ArtifactInventoryEntry>();

  for (const filepath of threadArtifacts) {
    merged.set(filepath, { filepath, source: "thread" });
  }

  for (const artifact of discoveredArtifacts) {
    const previous = merged.get(artifact.filepath);
    merged.set(artifact.filepath, {
      filepath: artifact.filepath,
      size: artifact.size,
      modifiedAt: artifact.modified_at,
      source: previous?.source ?? "outputs",
    });
  }

  return [...merged.values()].sort((left, right) => {
    const timeDiff = parseArtifactTime(right.modifiedAt) - parseArtifactTime(left.modifiedAt);
    if (timeDiff !== 0) {
      return timeDiff;
    }

    const leftSeed = seedOrder.get(left.filepath) ?? Number.MAX_SAFE_INTEGER;
    const rightSeed = seedOrder.get(right.filepath) ?? Number.MAX_SAFE_INTEGER;
    if (leftSeed !== rightSeed) {
      return leftSeed - rightSeed;
    }

    return left.filepath.localeCompare(right.filepath);
  });
}

export function buildArtifactVersionKey(
  artifact?: Pick<ArtifactInventoryEntry, "modifiedAt" | "size">,
) {
  if (!artifact?.modifiedAt && artifact?.size === undefined) {
    return undefined;
  }
  return `${artifact?.modifiedAt ?? ""}:${artifact?.size ?? ""}`;
}

export function getLatestArtifactFilepath(
  artifacts: ArtifactInventoryEntry[],
): string | null {
  return artifacts[0]?.filepath ?? null;
}
