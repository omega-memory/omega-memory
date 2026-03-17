/**
 * Generic OAuth 2.0/OIDC provider — supports Google and GitHub.
 * Web Crypto API only (CF Workers compatible).
 */
import { eq, and } from "drizzle-orm";
import { users, oauthAccounts, oauthStates } from "../../db/schema";
import { AuthEngine } from "../index";
import { decodeJwt, verifyJwtRsa, fetchJwks, findJwkByKid, base64urlEncode } from "../jwt";
import type { AuthProvider, AuthResult, LoginMethod } from "../provider";
import type { Database } from "../../db";

// ===================================================================
// Provider configs
// ===================================================================

interface OAuthConfig {
  authorizationUrl: string;
  tokenUrl: string;
  userinfoUrl: string;
  jwksUri?: string;
  emailsUrl?: string;
  scopes: string[];
  isOidc: boolean;
}

const OAUTH_CONFIGS: Record<string, OAuthConfig> = {
  google: {
    authorizationUrl: "https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl: "https://oauth2.googleapis.com/token",
    userinfoUrl: "https://openidconnect.googleapis.com/v1/userinfo",
    jwksUri: "https://www.googleapis.com/oauth2/v3/certs",
    scopes: ["openid", "email", "profile"],
    isOidc: true,
  },
  github: {
    authorizationUrl: "https://github.com/login/oauth/authorize",
    tokenUrl: "https://github.com/login/oauth/access_token",
    userinfoUrl: "https://api.github.com/user",
    emailsUrl: "https://api.github.com/user/emails",
    scopes: ["user:email"],
    isOidc: false,
  },
};

// ===================================================================
// Helpers (Web Crypto only)
// ===================================================================

function generateRandomString(length: number): string {
  const bytes = crypto.getRandomValues(new Uint8Array(length));
  return base64urlEncode(bytes);
}

async function sha256(plain: string): Promise<Uint8Array> {
  const enc = new TextEncoder();
  const digest = await crypto.subtle.digest("SHA-256", enc.encode(plain));
  return new Uint8Array(digest);
}

async function generatePkce(): Promise<{ codeVerifier: string; codeChallenge: string }> {
  const codeVerifier = generateRandomString(32);
  const hash = await sha256(codeVerifier);
  const codeChallenge = base64urlEncode(hash);
  return { codeVerifier, codeChallenge };
}

// ===================================================================
// Environment helpers
// ===================================================================

interface OAuthEnv {
  OAUTH_PROVIDERS?: string;
  OAUTH_GOOGLE_CLIENT_ID?: string;
  OAUTH_GOOGLE_CLIENT_SECRET?: string;
  OAUTH_GITHUB_CLIENT_ID?: string;
  OAUTH_GITHUB_CLIENT_SECRET?: string;
  AUTH_SECRET?: string;
}

function getEnv(locals: Record<string, unknown>): OAuthEnv {
  const runtime = locals.runtime as { env?: OAuthEnv } | undefined;
  return {
    OAUTH_PROVIDERS: runtime?.env?.OAUTH_PROVIDERS ?? process.env.OAUTH_PROVIDERS,
    OAUTH_GOOGLE_CLIENT_ID: runtime?.env?.OAUTH_GOOGLE_CLIENT_ID ?? process.env.OAUTH_GOOGLE_CLIENT_ID,
    OAUTH_GOOGLE_CLIENT_SECRET: runtime?.env?.OAUTH_GOOGLE_CLIENT_SECRET ?? process.env.OAUTH_GOOGLE_CLIENT_SECRET,
    OAUTH_GITHUB_CLIENT_ID: runtime?.env?.OAUTH_GITHUB_CLIENT_ID ?? process.env.OAUTH_GITHUB_CLIENT_ID,
    OAUTH_GITHUB_CLIENT_SECRET: runtime?.env?.OAUTH_GITHUB_CLIENT_SECRET ?? process.env.OAUTH_GITHUB_CLIENT_SECRET,
    AUTH_SECRET: runtime?.env?.AUTH_SECRET ?? process.env.AUTH_SECRET,
  };
}

function getClientCredentials(
  env: OAuthEnv,
  provider: string
): { clientId: string; clientSecret: string } | null {
  const key = provider.toUpperCase();
  const clientId = env[`OAUTH_${key}_CLIENT_ID` as keyof OAuthEnv];
  const clientSecret = env[`OAUTH_${key}_CLIENT_SECRET` as keyof OAuthEnv];
  if (!clientId || !clientSecret) return null;
  return { clientId, clientSecret };
}

function getEnabledProviders(env: OAuthEnv): string[] {
  const raw = env.OAUTH_PROVIDERS ?? "";
  return raw
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s.length > 0 && OAUTH_CONFIGS[s]);
}

// ===================================================================
// OAuth Provider class
// ===================================================================

export class OAuthProvider implements AuthProvider {
  readonly name = "oauth";

  getLoginMethods(): LoginMethod[] {
    // Return available providers based on env; called from provider-config
    // Since we don't have env here, this returns all known configs.
    // The actual filtering happens in getEnabledLoginMethods().
    return Object.keys(OAUTH_CONFIGS).map((provider) => ({
      type: "oauth" as const,
      provider,
      label: provider.charAt(0).toUpperCase() + provider.slice(1),
    }));
  }

  getEnabledLoginMethods(env: OAuthEnv): LoginMethod[] {
    const enabled = getEnabledProviders(env);
    return enabled.map((provider) => ({
      type: "oauth" as const,
      provider,
      label: provider.charAt(0).toUpperCase() + provider.slice(1),
    }));
  }

  async authenticate(
    _request: Request,
    _db: Database,
    _secret: string
  ): Promise<AuthResult | null> {
    // OAuth doesn't use direct authentication — it uses authorize + callback flow
    return null;
  }

  /**
   * Start the OAuth flow: generate state, PKCE, store in DB, return authorization URL.
   */
  async startAuthorization(
    provider: string,
    redirectUri: string,
    db: Database,
  ): Promise<string> {
    const config = OAUTH_CONFIGS[provider];
    if (!config) throw new Error(`Unknown OAuth provider: ${provider}`);

    const state = generateRandomString(32);
    const { codeVerifier, codeChallenge } = await generatePkce();
    const now = new Date();
    const expiresAt = new Date(now.getTime() + 5 * 60 * 1000); // 5 min TTL

    // Store state in DB
    await db.insert(oauthStates).values({
      id: crypto.randomUUID(),
      state,
      codeVerifier,
      redirectUri,
      provider,
      createdAt: now.toISOString(),
      expiresAt: expiresAt.toISOString(),
    });

    // Build authorization URL
    const params = new URLSearchParams({
      response_type: "code",
      client_id: "PLACEHOLDER", // Will be replaced by caller
      redirect_uri: redirectUri,
      scope: config.scopes.join(" "),
      state,
      code_challenge: codeChallenge,
      code_challenge_method: "S256",
    });

    // Google needs access_type=offline for refresh tokens
    if (provider === "google") {
      params.set("access_type", "offline");
      params.set("prompt", "consent");
    }

    return `${config.authorizationUrl}?${params.toString()}`;
  }

  /**
   * Full authorize flow with env-based client ID.
   */
  async authorize(
    provider: string,
    redirectUri: string,
    db: Database,
    env: OAuthEnv,
  ): Promise<string> {
    const config = OAUTH_CONFIGS[provider];
    if (!config) throw new Error(`Unknown OAuth provider: ${provider}`);

    const creds = getClientCredentials(env, provider);
    if (!creds) throw new Error(`Missing OAuth credentials for ${provider}`);

    const state = generateRandomString(32);
    const { codeVerifier, codeChallenge } = await generatePkce();
    const now = new Date();
    const expiresAt = new Date(now.getTime() + 5 * 60 * 1000);

    await db.insert(oauthStates).values({
      id: crypto.randomUUID(),
      state,
      codeVerifier,
      redirectUri,
      provider,
      createdAt: now.toISOString(),
      expiresAt: expiresAt.toISOString(),
    });

    const params = new URLSearchParams({
      response_type: "code",
      client_id: creds.clientId,
      redirect_uri: redirectUri,
      scope: config.scopes.join(" "),
      state,
      code_challenge: codeChallenge,
      code_challenge_method: "S256",
    });

    if (provider === "google") {
      params.set("access_type", "offline");
      params.set("prompt", "consent");
    }

    return `${config.authorizationUrl}?${params.toString()}`;
  }

  /**
   * Handle the OAuth callback: validate state, exchange code, upsert user, create session.
   */
  async handleCallback(
    request: Request,
    db: Database,
    secret: string,
  ): Promise<AuthResult | null> {
    const url = new URL(request.url);
    const code = url.searchParams.get("code");
    const stateParam = url.searchParams.get("state");
    const error = url.searchParams.get("error");

    if (error || !code || !stateParam) return null;

    // Look up and validate state
    const [storedState] = await db
      .select()
      .from(oauthStates)
      .where(eq(oauthStates.state, stateParam))
      .limit(1);

    if (!storedState) return null;

    // Check expiry
    if (new Date(storedState.expiresAt) < new Date()) {
      // Clean up expired state
      await db.delete(oauthStates).where(eq(oauthStates.id, storedState.id));
      return null;
    }

    // Delete the state (single use)
    await db.delete(oauthStates).where(eq(oauthStates.id, storedState.id));

    const provider = storedState.provider;
    const config = OAUTH_CONFIGS[provider];
    if (!config) return null;

    // Get client credentials from env embedded in request
    // The caller must pass env info; we extract from the URL's origin
    const env = this.extractEnvFromRequest(request);
    const creds = getClientCredentials(env, provider);
    if (!creds) return null;

    // Exchange code for tokens
    const tokenData = await this.exchangeCode(
      config,
      code,
      storedState.redirectUri!,
      storedState.codeVerifier!,
      creds.clientId,
      creds.clientSecret
    );

    if (!tokenData) return null;

    // Get user info
    let email: string | null = null;
    let displayName: string | null = null;
    let providerAccountId: string | null = null;

    if (config.isOidc && tokenData.id_token) {
      // Google OIDC: verify id_token
      const userInfo = await this.verifyOidcToken(config, tokenData.id_token);
      if (!userInfo) return null;
      email = userInfo.email as string;
      displayName = (userInfo.name as string) ?? null;
      providerAccountId = userInfo.sub as string;
    } else {
      // GitHub: fetch user info from API
      const userInfo = await this.fetchUserInfo(
        config,
        tokenData.access_token
      );
      if (!userInfo) return null;
      email = userInfo.email;
      displayName = userInfo.name;
      providerAccountId = userInfo.id;
    }

    if (!email || !providerAccountId) return null;

    // Upsert user and link OAuth account
    const user = await this.upsertUser(
      db,
      email,
      displayName,
      provider,
      providerAccountId,
      tokenData.access_token,
      tokenData.refresh_token ?? null,
      tokenData.expires_in
        ? new Date(Date.now() + tokenData.expires_in * 1000).toISOString()
        : null
    );

    // Create session
    const auth = new AuthEngine(db, secret);
    const deviceInfo = request.headers.get("user-agent") ?? undefined;
    const sessionToken = await auth.createSession(user.id, deviceInfo);

    return {
      user: {
        id: user.id,
        email: user.email,
        displayName: user.displayName,
        role: user.role,
      },
      sessionToken,
    };
  }

  // Env extraction helper — used by handleCallback when called via the API route
  private envCache: OAuthEnv | null = null;

  setEnv(env: OAuthEnv) {
    this.envCache = env;
  }

  private extractEnvFromRequest(_request: Request): OAuthEnv {
    // Return cached env (set by API route before calling handleCallback)
    return this.envCache ?? {};
  }

  /**
   * Exchange authorization code for tokens.
   */
  private async exchangeCode(
    config: OAuthConfig,
    code: string,
    redirectUri: string,
    codeVerifier: string,
    clientId: string,
    clientSecret: string
  ): Promise<TokenResponse | null> {
    const body = new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      client_id: clientId,
      client_secret: clientSecret,
      code_verifier: codeVerifier,
    });

    const headers: Record<string, string> = {
      "Content-Type": "application/x-www-form-urlencoded",
    };

    // GitHub needs Accept: application/json
    if (config.tokenUrl.includes("github.com")) {
      headers["Accept"] = "application/json";
    }

    try {
      const res = await fetch(config.tokenUrl, {
        method: "POST",
        headers,
        body: body.toString(),
      });

      if (!res.ok) return null;
      return (await res.json()) as TokenResponse;
    } catch {
      return null;
    }
  }

  /**
   * Verify OIDC id_token against the provider's JWKS.
   */
  private async verifyOidcToken(
    config: OAuthConfig,
    idToken: string
  ): Promise<Record<string, unknown> | null> {
    if (!config.jwksUri) return null;

    try {
      const decoded = decodeJwt(idToken);
      const kid = decoded.header.kid as string;
      if (!kid) return null;

      const keys = await fetchJwks(config.jwksUri);
      const jwk = findJwkByKid(keys, kid);
      if (!jwk) return null;

      return await verifyJwtRsa(idToken, jwk);
    } catch {
      return null;
    }
  }

  /**
   * Fetch user info from the provider's API (for non-OIDC providers like GitHub).
   */
  private async fetchUserInfo(
    config: OAuthConfig,
    accessToken: string
  ): Promise<{ email: string; name: string | null; id: string } | null> {
    try {
      const res = await fetch(config.userinfoUrl, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          Accept: "application/json",
          "User-Agent": "omega-memory",
        },
      });

      if (!res.ok) return null;
      const data = (await res.json()) as Record<string, unknown>;

      let email = data.email as string | null;
      const name = (data.name as string) ?? (data.login as string) ?? null;
      const id = String(data.id ?? data.sub ?? "");

      // GitHub may not return email in /user — fetch from /user/emails
      if (!email && config.emailsUrl) {
        email = await this.fetchGitHubPrimaryEmail(config.emailsUrl, accessToken);
      }

      if (!email || !id) return null;
      return { email, name, id };
    } catch {
      return null;
    }
  }

  private async fetchGitHubPrimaryEmail(
    emailsUrl: string,
    accessToken: string
  ): Promise<string | null> {
    try {
      const res = await fetch(emailsUrl, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          Accept: "application/json",
          "User-Agent": "omega-memory",
        },
      });

      if (!res.ok) return null;

      const emails = (await res.json()) as Array<{
        email: string;
        primary: boolean;
        verified: boolean;
      }>;

      // Prefer primary + verified
      const primary = emails.find((e) => e.primary && e.verified);
      if (primary) return primary.email;

      // Fall back to any verified
      const verified = emails.find((e) => e.verified);
      return verified?.email ?? null;
    } catch {
      return null;
    }
  }

  /**
   * Upsert user in users table and link via oauthAccounts.
   */
  private async upsertUser(
    db: Database,
    email: string,
    displayName: string | null,
    provider: string,
    providerAccountId: string,
    accessToken: string | null,
    refreshToken: string | null,
    expiresAt: string | null
  ) {
    const now = new Date().toISOString();

    // Check if OAuth account already exists
    const [existingOauth] = await db
      .select()
      .from(oauthAccounts)
      .where(
        and(
          eq(oauthAccounts.provider, provider),
          eq(oauthAccounts.providerAccountId, providerAccountId)
        )
      )
      .limit(1);

    if (existingOauth) {
      // Update tokens
      await db
        .update(oauthAccounts)
        .set({
          accessToken,
          refreshToken: refreshToken ?? existingOauth.refreshToken,
          expiresAt,
        })
        .where(eq(oauthAccounts.id, existingOauth.id));

      // Fetch the linked user
      const [user] = await db
        .select()
        .from(users)
        .where(eq(users.id, existingOauth.userId))
        .limit(1);

      return user;
    }

    // Check if user with this email exists
    let [user] = await db
      .select()
      .from(users)
      .where(eq(users.email, email))
      .limit(1);

    if (!user) {
      // Create new user (no password — OAuth-only)
      const userId = crypto.randomUUID();
      await db.insert(users).values({
        id: userId,
        email,
        passwordHash: null,
        displayName,
        role: "user",
        createdAt: now,
        updatedAt: now,
        isActive: 1,
      });

      [user] = await db
        .select()
        .from(users)
        .where(eq(users.id, userId))
        .limit(1);
    }

    // Link OAuth account
    await db.insert(oauthAccounts).values({
      id: crypto.randomUUID(),
      userId: user.id,
      provider,
      providerAccountId,
      accessToken,
      refreshToken,
      expiresAt,
      createdAt: now,
    });

    return user;
  }
}

interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in?: number;
  refresh_token?: string;
  id_token?: string;
  scope?: string;
}

export { OAUTH_CONFIGS, getEnv, getEnabledProviders, getClientCredentials };
export type { OAuthEnv };
