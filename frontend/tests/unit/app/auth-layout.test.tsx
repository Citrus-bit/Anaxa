import { beforeEach, describe, expect, test, rs } from "@rstest/core";
import { renderToStaticMarkup } from "react-dom/server";

const authState = rs.hoisted(() => ({
  gatewayResult: {
    tag: "authenticated" as const,
    user: {
      id: "test-user",
      email: "test@example.com",
      system_role: "admin" as const,
      needs_setup: false,
      oauth_provider: null,
    },
  },
  getServerSideUser: rs.fn(),
  localAuthEnabled: true,
  localAuthValid: false,
  redirect: rs.fn(),
}));

rs.mock("next/headers", () => ({
  cookies: rs.fn(async () => ({
    get: rs.fn(() => ({ value: "local-session" })),
  })),
}));

rs.mock("next/navigation", () => ({
  redirect: authState.redirect,
}));

rs.mock("@/components/workspace/gateway-offline-fallback", () => ({
  GatewayOfflineFallback: ({ children }: { children: React.ReactNode }) =>
    children,
}));

rs.mock("@/core/auth/AuthProvider", () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => children,
}));

rs.mock("@/core/auth/server", () => ({
  getServerSideUser: authState.getServerSideUser,
}));

rs.mock("@/core/auth/types", () => ({
  assertNever: () => {
    throw new Error("Unexpected auth result");
  },
}));

rs.mock("@/core/i18n/context", () => ({
  I18nProvider: ({ children }: { children: React.ReactNode }) => children,
}));

rs.mock("@/core/i18n/server", () => ({
  detectLocaleServer: rs.fn(async () => "en-US"),
}));

rs.mock("@/server/local-auth", () => ({
  isLocalAuthEnabled: () => authState.localAuthEnabled,
  isValidLocalAuthToken: () => authState.localAuthValid,
  LOCAL_AUTH_COOKIE_NAME: "medrix_flow_session",
}));

import AuthLayout from "@/app/(auth)/layout";

describe("AuthLayout local UI-password gate", () => {
  beforeEach(() => {
    authState.localAuthEnabled = true;
    authState.localAuthValid = false;
    authState.getServerSideUser.mockReset();
    authState.getServerSideUser.mockImplementation(async () =>
      Promise.resolve(authState.gatewayResult),
    );
    authState.redirect.mockReset();
  });

  test("does not probe or redirect Gateway auth before local unlock", async () => {
    const result = await AuthLayout({
      children: <p>Local unlock</p>,
    });

    expect(renderToStaticMarkup(result)).toContain("Local unlock");
    expect(authState.getServerSideUser).not.toHaveBeenCalled();
    expect(authState.redirect).not.toHaveBeenCalled();
  });

  test("resumes Gateway auth after local unlock", async () => {
    authState.localAuthValid = true;
    authState.getServerSideUser.mockResolvedValue({ tag: "unauthenticated" });

    const result = await AuthLayout({
      children: <p>Account login</p>,
    });

    expect(renderToStaticMarkup(result)).toContain("Account login");
    expect(authState.getServerSideUser).toHaveBeenCalledOnce();
  });
});
