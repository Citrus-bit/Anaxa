import { describe, expect, it } from "vitest";

import {
  buildArtifactVersionKey,
  getLatestArtifactFilepath,
  mergeArtifactEntries,
} from "./utils";

describe("mergeArtifactEntries", () => {
  it("prioritizes discovered outputs by recency and keeps thread-only files", () => {
    const merged = mergeArtifactEntries(
      [
        "/mnt/user-data/outputs/report.md",
        "/mnt/user-data/outputs/metrics.json",
        "/mnt/user-data/uploads/dataset.csv",
      ],
      [
        {
          filepath: "/mnt/user-data/outputs/metrics.json",
          modified_at: "2026-05-07T08:00:00Z",
          size: 24,
        },
        {
          filepath: "/mnt/user-data/outputs/figures/roc.png",
          modified_at: "2026-05-07T09:00:00Z",
          size: 42,
        },
        {
          filepath: "/mnt/user-data/outputs/report.md",
          modified_at: "2026-05-07T07:00:00Z",
          size: 12,
        },
      ],
    );

    expect(merged.map((item) => item.filepath)).toEqual([
      "/mnt/user-data/outputs/figures/roc.png",
      "/mnt/user-data/outputs/metrics.json",
      "/mnt/user-data/outputs/report.md",
      "/mnt/user-data/uploads/dataset.csv",
    ]);
    expect(
      merged.find((item) => item.filepath.endsWith("metrics.json"))?.size,
    ).toBe(24);
    expect(getLatestArtifactFilepath(merged)).toBe(
      "/mnt/user-data/outputs/figures/roc.png",
    );
  });
});

describe("buildArtifactVersionKey", () => {
  it("returns a stable key from artifact metadata", () => {
    expect(
      buildArtifactVersionKey({
        modifiedAt: "2026-05-07T09:00:00Z",
        size: 42,
      }),
    ).toBe("2026-05-07T09:00:00Z:42");
  });

  it("returns undefined when no version metadata exists", () => {
    expect(buildArtifactVersionKey(undefined)).toBeUndefined();
  });
});

describe("artifact URL helpers", () => {
  it("encodes thread IDs and reserved characters in path segments", async () => {
    const { resolveArtifactURL, urlOfArtifact } = await import("./utils");

    expect(
      urlOfArtifact({
        filepath: "/mnt/user-data/outputs/a#b?.txt",
        threadId: "thread #1",
        download: true,
      }),
    ).toContain(
      "/api/threads/thread%20%231/artifacts/mnt/user-data/outputs/a%23b%3F.txt?download=true",
    );
    expect(
      resolveArtifactURL(
        "/mnt/user-data/outputs/%E4%B8%AD%20%E6%96%87%23%3F.png?v=2#detail",
        "thread #1",
      ),
    ).toContain(
      "/api/threads/thread%20%231/artifacts/mnt/user-data/outputs/%E4%B8%AD%20%E6%96%87%23%3F.png?v=2#detail",
    );
  });
});
