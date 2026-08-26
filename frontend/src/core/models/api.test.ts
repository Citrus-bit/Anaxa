import { afterEach, describe, expect, it, vi } from "vitest";

import { loadModels } from "./api";

describe("loadModels", () => {
  afterEach(() => vi.restoreAllMocks());

  it("returns models from a successful response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ models: [{ id: "m1" }] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(loadModels()).resolves.toEqual([{ id: "m1" }]);
  });

  it("preserves a backend error detail for non-2xx responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Provider is unavailable" }), {
          status: 503,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(loadModels()).rejects.toThrow("Provider is unavailable");
  });

  it("rejects malformed successful payloads", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ models: null }), { status: 200 }),
      ),
    );

    await expect(loadModels()).rejects.toThrow("Model response was invalid.");
  });
});
