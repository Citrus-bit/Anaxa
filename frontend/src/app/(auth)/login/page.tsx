import { cookies } from "next/headers";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  getLocalAuthConfigurationError,
  isLocalAuthEnabled,
  isLocalAuthMisconfigured,
  isValidLocalAuthToken,
  LOCAL_AUTH_COOKIE_NAME,
  LOCAL_AUTH_DEFAULT_REDIRECT,
  normalizeNextPath,
} from "@/server/local-auth";

import LoginForm from "./login-form";

export const dynamic = "force-dynamic";

type LoginSearchParams = Promise<
  Record<string, string | string[] | undefined>
>;

function firstParam(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function isSetupPath(path: string): boolean {
  try {
    return new URL(path, "http://local.invalid").pathname === "/setup";
  } catch {
    return false;
  }
}

/**
 * Keep the original account-login target after the outer UI-password gate.
 * The local session endpoint redirects to this path after setting its cookie.
 * Unwrapping a nested `/login?next=...` also keeps failed unlock attempts from
 * growing the query string indefinitely.
 */
function resolveAccountLoginPath(nextValue: string | undefined): string {
  const candidate = normalizeNextPath(nextValue);
  // Setup is a Gateway bootstrap flow, not an account-protected workspace
  // route. Once the outer UI-password is accepted, return there directly so
  // setup-status/initialize can run before the first account exists.
  if (isSetupPath(candidate)) {
    return candidate;
  }
  if (!candidate.startsWith("/login?")) {
    return `/login?next=${encodeURIComponent(candidate)}`;
  }

  try {
    const nested = new URL(candidate, "http://local.invalid").searchParams.get(
      "next",
    );
    const target = normalizeNextPath(nested);
    if (isSetupPath(target)) {
      return target;
    }
    return `/login?next=${encodeURIComponent(target)}`;
  } catch {
    return `/login?next=${encodeURIComponent(LOCAL_AUTH_DEFAULT_REDIRECT)}`;
  }
}

function LocalAuthConfigurationError() {
  return (
    <main className="from-background via-muted/30 to-background flex min-h-screen items-center justify-center bg-linear-to-br px-6 py-12">
      <Card className="w-full max-w-md border shadow-lg">
        <CardHeader className="space-y-1">
          <CardTitle>Anaxa</CardTitle>
          <CardDescription>
            Protected mode is enabled, but no UI password is configured.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm leading-6">
            {getLocalAuthConfigurationError()}
          </p>
        </CardContent>
      </Card>
    </main>
  );
}

function LocalUnlockForm({
  nextPath,
  invalidPassword,
}: {
  nextPath: string;
  invalidPassword: boolean;
}) {
  return (
    <main className="from-background via-muted/30 to-background flex min-h-screen items-center justify-center bg-linear-to-br px-6 py-12">
      <Card className="w-full max-w-md border shadow-lg">
        <CardHeader className="space-y-1">
          <CardTitle>Anaxa</CardTitle>
          <CardDescription>
            Enter the workspace password to access the UI and API.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form action="/api/session/login" method="post" className="space-y-4">
            <input type="hidden" name="next" value={nextPath} />
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="password">
                Password
              </label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                autoFocus
              />
            </div>
            {invalidPassword && (
              <p className="text-destructive text-sm">Invalid password.</p>
            )}
            <Button type="submit" className="w-full">
              Unlock Workspace
            </Button>
          </form>
        </CardContent>
      </Card>
    </main>
  );
}

export default async function LoginPage({
  searchParams,
}: {
  searchParams: LoginSearchParams;
}) {
  if (isLocalAuthMisconfigured()) {
    return <LocalAuthConfigurationError />;
  }

  if (isLocalAuthEnabled()) {
    const cookieStore = await cookies();
    const localSession = cookieStore.get(LOCAL_AUTH_COOKIE_NAME)?.value;
    if (!isValidLocalAuthToken(localSession)) {
      const params = await searchParams;
      const nextPath = resolveAccountLoginPath(firstParam(params.next));
      return (
        <LocalUnlockForm
          nextPath={nextPath}
          invalidPassword={firstParam(params.error) === "invalid_password"}
        />
      );
    }
  }

  return <LoginForm />;
}
