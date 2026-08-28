import {
  afterEach,
  beforeEach,
  describe,
  expect,
  rs,
  test,
} from "@rstest/core";
import { NextRequest } from "next/server";

import { config, middleware } from "@/middleware";

const ENV_KEYS = [
  "BETTER_AUTH_SECRET",
  "MEDRIX_FLOW_UI_PASSWORD",
  "MEDRIX_FLOW_ENV",
  "NODE_ENV",
] as const;
function setEnv(key: (typeof ENV_KEYS)[number], value: string | undefined) {
  const env = process.env as Record<string, string | undefined>;
  if (value === undefined) {
    delete env[key];
  } else {
    env[key] = value;
  }
}

let savedEnv: Record<(typeof ENV_KEYS)[number], string | undefined>;

beforeEach(() => {
  savedEnv = Object.fromEntries(
    ENV_KEYS.map((key) => [key, process.env[key]]),
  ) as Record<(typeof ENV_KEYS)[number], string | undefined>;
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    setEnv(key, savedEnv[key]);
  }
});

describe("local UI-password middleware", () => {
  async function loginToken(password: string): Promise<string> {
    // local-auth reads the validated env object at module load. Reload it after
    // changing BETTER_AUTH_SECRET so this test exercises the same fallback as
    // the /api/session/login route on each case.
    rs.resetModules();
    const { createLocalAuthToken } = await import("@/server/local-auth");
    return createLocalAuthToken(password);
  }

  test("redirects a locked setup request through canonical login", async () => {
    setEnv("MEDRIX_FLOW_UI_PASSWORD", "outer-pass");
    setEnv("MEDRIX_FLOW_ENV", "development");

    const response = await middleware(
      new NextRequest("http://localhost:3000/setup"),
    );

    expect(config.matcher).toContain("/setup");
    expect(response.status).toBe(307);
    const location = response.headers.get("location");
    expect(location).toBe("http://localhost:3000/login?next=%2Fsetup");
  });

  test("keeps setup open when no UI-password is configured", async () => {
    setEnv("MEDRIX_FLOW_UI_PASSWORD", undefined);
    setEnv("MEDRIX_FLOW_ENV", "development");

    const response = await middleware(
      new NextRequest("http://localhost:3000/setup"),
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("location")).toBeNull();
  });

  test("accepts the login token when Better Auth secret is an empty string", async () => {
    const password = "outer-pass";
    setEnv("BETTER_AUTH_SECRET", "");
    setEnv("MEDRIX_FLOW_UI_PASSWORD", password);
    setEnv("MEDRIX_FLOW_ENV", "development");
    setEnv("NODE_ENV", "test");

    const response = await middleware(
      new NextRequest("http://localhost:3000/setup", {
        headers: {
          cookie: `medrix_flow_session=${await loginToken(password)}`,
        },
      }),
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("location")).toBeNull();
  });

  test("accepts the login token when Better Auth secret is configured", async () => {
    const password = "outer-pass";
    setEnv("BETTER_AUTH_SECRET", "shared-auth-secret");
    setEnv("MEDRIX_FLOW_UI_PASSWORD", password);
    setEnv("MEDRIX_FLOW_ENV", "development");
    setEnv("NODE_ENV", "test");

    const response = await middleware(
      new NextRequest("http://localhost:3000/setup", {
        headers: {
          cookie: `medrix_flow_session=${await loginToken(password)}`,
        },
      }),
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("location")).toBeNull();
  });
});
