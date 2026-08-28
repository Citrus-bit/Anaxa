import { NextResponse, type NextRequest } from "next/server";

const LOCAL_AUTH_COOKIE_NAME = "medrix_flow_session";
const LOCAL_AUTH_DEFAULT_REDIRECT = "/workspace";
const LOCAL_AUTH_CONFIGURATION_ERROR =
  "MEDRIX_FLOW_UI_PASSWORD must be configured before serving protected routes in production.";

function normalizeNextPath(nextPath: string): string {
  if (!nextPath.startsWith("/") || nextPath.startsWith("//")) {
    return LOCAL_AUTH_DEFAULT_REDIRECT;
  }
  return nextPath;
}

function isProductionLikeEnvironment(): boolean {
  const runtime = process.env.MEDRIX_FLOW_ENV?.trim().toLowerCase();
  if (runtime === "prod" || runtime === "production") {
    return true;
  }
  return process.env.NODE_ENV === "production";
}

async function buildExpectedToken(): Promise<string | null> {
  // Match `emptyStringAsUndefined` in src/env.js and the server-side login
  // helper: an empty or whitespace-only secret falls back to the password.
  const configuredPassword = process.env.MEDRIX_FLOW_UI_PASSWORD?.trim();
  const password = configuredPassword?.length ? configuredPassword : "";
  if (!password) {
    return null;
  }

  const configuredSecret = process.env.BETTER_AUTH_SECRET?.trim();
  const signingSecret = configuredSecret?.length ? configuredSecret : password;
  const data = new TextEncoder().encode(`${signingSecret}:${password}`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function middleware(request: NextRequest) {
  const expectedToken = await buildExpectedToken();
  if (!expectedToken) {
    if (isProductionLikeEnvironment()) {
      return new NextResponse(LOCAL_AUTH_CONFIGURATION_ERROR, { status: 503 });
    }
    return NextResponse.next();
  }

  const { pathname, search } = request.nextUrl;
  const sessionToken = request.cookies.get(LOCAL_AUTH_COOKIE_NAME)?.value;
  const authenticated = sessionToken === expectedToken;

  if (pathname === "/login") {
    // `/login` is the canonical DeerFlow account/SSO page.  When the optional
    // Anaxa UI-password gate is enabled, the page itself renders the unlock
    // form until this cookie is present.  Do not redirect a locally-unlocked
    // browser to `/workspace`: it may still need to complete the Gateway
    // account login, and doing so would create a two-layer auth redirect loop.
    return NextResponse.next();
  }

  if (!authenticated) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("next", normalizeNextPath(`${pathname}${search}`));
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/login", "/setup", "/workspace/:path*"],
};
