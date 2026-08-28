import { getBackendBaseURL } from "../config";
import { isStaticWebsiteOnly } from "../static-mode";
import type { AgentThreadState } from "../threads";

const EMPTY_ARTIFACT_PATHS: readonly string[] = [];

/** Metadata returned by the Gateway's artifact inventory endpoint. */
export interface ThreadArtifactRecord {
  filepath: string;
  filename?: string;
  size?: number;
  modified_at?: string;
}

/** One artifact as projected into the thread-scoped file list. */
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

function splitPathSuffix(src: string) {
  const [path = ""] = src.split(/[?#]/, 1);
  return {
    path,
    suffix: src.slice(path.length),
  };
}

function encodeArtifactPath(filepath: string) {
  return filepath
    .split("/")
    .map((segment) => encodeURIComponent(decodePathSegment(segment)))
    .join("/");
}

export function buildWriteFileArtifactURL({
  filepath,
  messageId,
  toolCallId,
}: {
  filepath: string;
  messageId?: string;
  toolCallId?: string;
}) {
  const url = new URL("write-file:/");
  url.pathname = filepath.replaceAll("%", "%25");
  if (messageId) {
    url.searchParams.set("message_id", messageId);
  }
  if (toolCallId) {
    url.searchParams.set("tool_call_id", toolCallId);
  }
  return url.toString();
}

function decodeRelativeArtifactPath(filepath: string) {
  return filepath.split("/").map(decodePathSegment).join("/");
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
  if (isStaticWebsiteOnly()) {
    return staticDemoArtifactURL({ filepath, threadId, download });
  }
  const encodedThreadId = encodeURIComponent(threadId);
  const encodedFilepath = encodeArtifactPath(filepath);
  if (isMock) {
    return `${getBackendBaseURL()}/mock/api/threads/${encodedThreadId}/artifacts${encodedFilepath}${download ? "?download=true" : ""}`;
  }
  return `${getBackendBaseURL()}/api/threads/${encodedThreadId}/artifacts${encodedFilepath}${download ? "?download=true" : ""}`;
}

export function extractArtifactsFromThread(thread: {
  values: Pick<AgentThreadState, "artifacts">;
}) {
  return thread.values.artifacts ?? EMPTY_ARTIFACT_PATHS;
}

export function resolveArtifactURL(absolutePath: string, threadId: string) {
  if (isStaticWebsiteOnly()) {
    return staticDemoArtifactURL({ filepath: absolutePath, threadId });
  }
  return `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/artifacts${encodeArtifactPath(absolutePath)}`;
}

export function resolveMarkdownArtifactURL(src: string, threadId: string) {
  const { path, suffix } = splitPathSuffix(src);
  return `${resolveArtifactURL(path, threadId)}${suffix}`;
}

export function resolveMessageImageURL(
  src: string,
  threadId: string,
  artifactPaths: readonly string[],
) {
  if (src.startsWith("/mnt/")) {
    return resolveMarkdownArtifactURL(src, threadId);
  }

  const { path: relativePath, suffix } = splitPathSuffix(src);
  const normalizedPath = relativePath.replace(/^(?:\.\/)+/, "");
  const decodedNormalizedPath = decodeRelativeArtifactPath(normalizedPath);
  if (
    !normalizedPath ||
    normalizedPath.startsWith("/") ||
    /^[a-z][a-z\d+.-]*:/i.test(normalizedPath) ||
    normalizedPath.startsWith("//") ||
    normalizedPath.split("/").includes("..")
  ) {
    return src;
  }

  const matches = artifactPaths.filter((path) =>
    path.endsWith(`/${decodedNormalizedPath}`),
  );
  if (matches.length !== 1) {
    return src;
  }

  return `${resolveArtifactURL(matches[0]!, threadId)}${suffix}`;
}

function staticDemoArtifactURL({
  filepath,
  threadId,
  download = false,
}: {
  filepath: string;
  threadId: string;
  download?: boolean;
}) {
  const demoPath = encodeArtifactPath(filepath.replace(/^\/mnt\//, "/"));
  return `${getBackendBaseURL()}/demo/threads/${encodeURIComponent(threadId)}${demoPath}${download ? "?download=true" : ""}`;
}

function parseArtifactTime(value?: string) {
  if (!value) {
    return Number.NEGATIVE_INFINITY;
  }
  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp;
}

/**
 * Merge artifact paths recorded in thread state with the output-directory
 * inventory. Thread-only paths are retained, while discovered metadata wins
 * for duplicate paths; recency determines the display order.
 */
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
    const timeDiff =
      parseArtifactTime(right.modifiedAt) - parseArtifactTime(left.modifiedAt);
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
