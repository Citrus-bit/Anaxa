import { createHash, timingSafeEqual } from "crypto";

import { env } from "@/env";

export const LOCAL_AUTH_COOKIE_NAME = "medrix_flow_session";
export const LOCAL_AUTH_LOGIN_PATH = "/login";
export const LOCAL_AUTH_DEFAULT_REDIRECT = "/workspace";
export const ADMIN_TOKEN_HEADER_NAME = "x-medrix-admin-token";
export const PROXY_AUTH_HEADER_NAME = "x-medrix-proxy-authorized";
const LOCAL_AUTH_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;
const LOCAL_AUTH_CONFIGURATION_ERROR =
  "MEDRIX_FLOW_UI_PASSWORD must be configured before serving protected routes in production.";

function getConfiguredPassword(): string {
  const password = env.MEDRIX_FLOW_UI_PASSWORD?.trim();
  return password?.length ? password : "";
}

function getSessionSigningMaterial(password: string): string {
  const configuredSecret = env.BETTER_AUTH_SECRET?.trim();
  const signingSecret = configuredSecret?.length ? configuredSecret : password;
  return `${signingSecret}:${password}`;
}

export function isLocalAuthEnabled(): boolean {
  return getConfiguredPassword().length > 0;
}

export function isProductionLikeEnvironment(): boolean {
  const runtime = env.MEDRIX_FLOW_ENV?.trim().toLowerCase();
  if (runtime === "prod" || runtime === "production") {
    return true;
  }
  return env.NODE_ENV === "production";
}

export function isLocalAuthMisconfigured(): boolean {
  return isProductionLikeEnvironment() && !isLocalAuthEnabled();
}

export function getLocalAuthConfigurationError(): string {
  return LOCAL_AUTH_CONFIGURATION_ERROR;
}

export function normalizeNextPath(nextPath: string | null | undefined): string {
  if (!nextPath?.startsWith("/") || nextPath.startsWith("//")) {
    return LOCAL_AUTH_DEFAULT_REDIRECT;
  }
  return nextPath;
}

export function createLocalAuthToken(password: string): string {
  return createHash("sha256").update(getSessionSigningMaterial(password)).digest("hex");
}

export function getExpectedLocalAuthToken(): string | null {
  const password = getConfiguredPassword();
  if (!password) {
    return null;
  }
  return createLocalAuthToken(password);
}

export function isValidLocalAuthToken(value: string | null | undefined): boolean {
  const expected = getExpectedLocalAuthToken();
  if (!expected) {
    return true;
  }
  if (typeof value !== "string" || value.length !== expected.length) {
    return false;
  }
  return timingSafeEqual(Buffer.from(value), Buffer.from(expected));
}

export function isValidLocalAuthPassword(password: string): boolean {
  const configuredPassword = getConfiguredPassword();
  if (!configuredPassword) {
    return true;
  }
  const provided = password.trim();
  if (provided.length !== configuredPassword.length) {
    return false;
  }
  return timingSafeEqual(Buffer.from(provided), Buffer.from(configuredPassword));
}

function getProxyAuthSigningSecret(): string {
  // Only BETTER_AUTH_SECRET — never fall back to the user password, otherwise the
  // proxy token becomes directly derivable from the UI password and anyone who
  // knows it can forge nginx-bypass tokens against the gateway.
  const signingSecret = env.BETTER_AUTH_SECRET?.trim();
  return signingSecret?.length ? signingSecret : "";
}

export function getProxyAuthorizationToken(): string | null {
  const signingSecret = getProxyAuthSigningSecret();
  if (!signingSecret) {
    return null;
  }
  return createHash("sha256")
    .update(`${signingSecret}:medrix-flow-proxy-auth`)
    .digest("hex");
}

export function isValidAdminAccessToken(value: string | null | undefined): boolean {
  const expected = env.MEDRIX_GATEWAY_ADMIN_TOKEN?.trim();
  if (!expected) {
    return false;
  }
  if (typeof value !== "string" || value.length !== expected.length) {
    return false;
  }
  return timingSafeEqual(Buffer.from(value), Buffer.from(expected));
}

export function getLocalAuthCookieSettings() {
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure: env.NODE_ENV === "production",
    path: "/",
    maxAge: LOCAL_AUTH_COOKIE_MAX_AGE_SECONDS,
  };
}
