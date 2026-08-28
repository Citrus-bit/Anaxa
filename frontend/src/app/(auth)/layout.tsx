import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { type ReactNode } from "react";

import { GatewayOfflineFallback } from "@/components/workspace/gateway-offline-fallback";
import { AuthProvider } from "@/core/auth/AuthProvider";
import { getServerSideUser } from "@/core/auth/server";
import { assertNever } from "@/core/auth/types";
import { I18nProvider } from "@/core/i18n/context";
import { detectLocaleServer } from "@/core/i18n/server";
import {
  isLocalAuthEnabled,
  isValidLocalAuthToken,
  LOCAL_AUTH_COOKIE_NAME,
} from "@/server/local-auth";

export const dynamic = "force-dynamic";

async function hasLocalAuthSession(): Promise<boolean> {
  if (!isLocalAuthEnabled()) {
    return true;
  }

  const cookieStore = await cookies();
  return isValidLocalAuthToken(cookieStore.get(LOCAL_AUTH_COOKIE_NAME)?.value);
}

export default async function AuthLayout({
  children,
}: {
  children: ReactNode;
}) {
  const locale = await detectLocaleServer();
  // Anaxa's optional UI-password is an outer deployment gate.  Evaluate it
  // before the Gateway probe so an account session cannot redirect an
  // as-yet-unlocked browser to `/workspace`, where middleware would send it
  // back to `/login` forever.  It also keeps the unlock form usable while the
  // Gateway is temporarily unavailable.
  const localAuthSession = await hasLocalAuthSession();
  const result = localAuthSession
    ? await getServerSideUser()
    : ({ tag: "unauthenticated" } as const);

  let content: ReactNode;

  switch (result.tag) {
    case "authenticated":
      redirect("/workspace");
    case "needs_setup":
      // Allow access to setup page
      content = (
        <AuthProvider initialUser={result.user}>{children}</AuthProvider>
      );
      break;
    case "system_setup_required":
    case "unauthenticated":
      content = <AuthProvider initialUser={null}>{children}</AuthProvider>;
      break;
    case "gateway_unavailable":
      // Auth pages have no banner of their own, so render one here. The
      // fallback's AuthProvider replaces the bare-HTML branch that
      // previously locked users out without any logout/retry capability.
      content = (
        <GatewayOfflineFallback renderBanner>
          <div className="flex h-screen flex-col items-center justify-center gap-4">
            <p className="text-muted-foreground">
              Service temporarily unavailable.
            </p>
          </div>
        </GatewayOfflineFallback>
      );
      break;
    case "config_error":
      throw new Error(result.message);
    default:
      assertNever(result);
  }

  return <I18nProvider initialLocale={locale}>{content}</I18nProvider>;
}
