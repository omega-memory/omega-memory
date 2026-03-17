/**
 * WebAuthn/Passkey auth provider.
 * Uses @simplewebauthn/server for registration and authentication.
 */
import {
  generateRegistrationOptions,
  verifyRegistrationResponse,
  generateAuthenticationOptions,
  verifyAuthenticationResponse,
} from "@simplewebauthn/server";
import type {
  RegistrationResponseJSON,
  AuthenticationResponseJSON,
  AuthenticatorTransportFuture,
} from "@simplewebauthn/server";
import { eq } from "drizzle-orm";
import type { AuthProvider, AuthResult, LoginMethod } from "../provider";
import type { Database } from "../../db";
import { AuthEngine } from "../index";
import { users, webauthnCredentials, oauthStates } from "../../db/schema";

// ===================================================================
// Config from env
// ===================================================================

function getConfig() {
  const rpId = process.env.WEBAUTHN_RP_ID ?? "localhost";
  const rpName = process.env.WEBAUTHN_RP_NAME ?? "OMEGA";
  const origin = process.env.WEBAUTHN_ORIGIN ?? "http://localhost:4321";
  return { rpId, rpName, origin };
}

// ===================================================================
// Challenge storage helpers (uses oauthStates table with provider="webauthn")
// ===================================================================

async function storeChallenge(db: Database, challenge: string, userId?: string): Promise<string> {
  const id = crypto.randomUUID();
  const now = new Date();
  const expiresAt = new Date(now.getTime() + 5 * 60 * 1000); // 5 min TTL
  await db.insert(oauthStates).values({
    id,
    state: challenge,
    codeVerifier: userId ?? null,
    provider: "webauthn",
    createdAt: now.toISOString(),
    expiresAt: expiresAt.toISOString(),
  });
  return id;
}

async function getAndDeleteChallenge(db: Database, challenge: string): Promise<{ challenge: string; userId: string | null } | null> {
  const [row] = await db
    .select()
    .from(oauthStates)
    .where(eq(oauthStates.state, challenge))
    .limit(1);

  if (!row || row.provider !== "webauthn") return null;
  if (new Date(row.expiresAt) < new Date()) {
    await db.delete(oauthStates).where(eq(oauthStates.id, row.id));
    return null;
  }

  // Clean up
  await db.delete(oauthStates).where(eq(oauthStates.id, row.id));
  return { challenge: row.state, userId: row.codeVerifier ?? null };
}

// ===================================================================
// Registration
// ===================================================================

export async function generateWebAuthnRegistrationOptions(
  db: Database,
  userId: string,
  userEmail: string,
  userName?: string,
) {
  const config = getConfig();

  // Get existing credentials for this user
  const existing = await db
    .select()
    .from(webauthnCredentials)
    .where(eq(webauthnCredentials.userId, userId));

  const excludeCredentials = existing.map((cred: any) => ({
    id: cred.credentialId,
    type: "public-key" as const,
    transports: cred.transports ? JSON.parse(cred.transports) as AuthenticatorTransportFuture[] : undefined,
  }));

  const options = await generateRegistrationOptions({
    rpName: config.rpName,
    rpID: config.rpId,
    userName: userEmail,
    userDisplayName: userName ?? userEmail,
    attestationType: "none",
    excludeCredentials,
    authenticatorSelection: {
      residentKey: "preferred",
      userVerification: "preferred",
    },
  });

  // Store challenge
  await storeChallenge(db, options.challenge, userId);

  return options;
}

export async function verifyWebAuthnRegistration(
  db: Database,
  userId: string,
  response: RegistrationResponseJSON,
  expectedChallenge: string,
) {
  const config = getConfig();

  const challengeData = await getAndDeleteChallenge(db, expectedChallenge);
  if (!challengeData) {
    throw new Error("Challenge not found or expired");
  }

  const verification = await verifyRegistrationResponse({
    response,
    expectedChallenge: expectedChallenge,
    expectedOrigin: config.origin,
    expectedRPID: config.rpId,
  });

  if (!verification.verified || !verification.registrationInfo) {
    throw new Error("Registration verification failed");
  }

  const { credential, credentialDeviceType, credentialBackedUp } = verification.registrationInfo;

  // Store credential
  const id = crypto.randomUUID();
  const now = new Date().toISOString();
  await db.insert(webauthnCredentials).values({
    id,
    userId,
    credentialId: credential.id,
    publicKey: Buffer.from(credential.publicKey).toString("base64url"),
    counter: credential.counter,
    deviceType: credentialDeviceType,
    backedUp: credentialBackedUp ? 1 : 0,
    transports: credential.transports ? JSON.stringify(credential.transports) : null,
    createdAt: now,
    lastUsedAt: now,
  });

  return { verified: true, credentialId: id };
}

// ===================================================================
// Authentication
// ===================================================================

export async function generateWebAuthnAuthenticationOptions(db: Database) {
  const config = getConfig();

  // Get all credentials to allow any registered passkey
  const allCreds = await db.select().from(webauthnCredentials);
  const allowCredentials = allCreds.map((cred: any) => ({
    id: cred.credentialId,
    type: "public-key" as const,
    transports: cred.transports ? JSON.parse(cred.transports) as AuthenticatorTransportFuture[] : undefined,
  }));

  const options = await generateAuthenticationOptions({
    rpID: config.rpId,
    userVerification: "preferred",
    // Empty allowCredentials lets the browser show all available passkeys (discoverable credentials)
    allowCredentials: allowCredentials.length > 0 ? allowCredentials : undefined,
  });

  await storeChallenge(db, options.challenge);

  return options;
}

export async function verifyWebAuthnAuthentication(
  db: Database,
  secret: string,
  response: AuthenticationResponseJSON,
  expectedChallenge: string,
  request?: Request,
): Promise<AuthResult | null> {
  const config = getConfig();

  const challengeData = await getAndDeleteChallenge(db, expectedChallenge);
  if (!challengeData) return null;

  // Find credential by ID
  const [cred] = await db
    .select()
    .from(webauthnCredentials)
    .where(eq(webauthnCredentials.credentialId, response.id))
    .limit(1);

  if (!cred) return null;

  // Decode the stored public key
  const publicKeyBytes = new Uint8Array(
    Buffer.from(cred.publicKey, "base64url")
  );

  const verification = await verifyAuthenticationResponse({
    response,
    expectedChallenge: expectedChallenge,
    expectedOrigin: config.origin,
    expectedRPID: config.rpId,
    credential: {
      id: cred.credentialId,
      publicKey: publicKeyBytes,
      counter: cred.counter,
      transports: cred.transports ? JSON.parse(cred.transports) : undefined,
    },
  });

  if (!verification.verified) return null;

  // Update counter and lastUsedAt
  await db
    .update(webauthnCredentials)
    .set({
      counter: verification.authenticationInfo.newCounter,
      lastUsedAt: new Date().toISOString(),
    })
    .where(eq(webauthnCredentials.id, cred.id));

  // Get the user
  const [user] = await db
    .select()
    .from(users)
    .where(eq(users.id, cred.userId))
    .limit(1);

  if (!user) return null;

  // Create session
  const auth = new AuthEngine(db, secret);
  const deviceInfo = request?.headers.get("user-agent") ?? undefined;
  const sessionToken = await auth.createSession(user.id, deviceInfo);

  return {
    user: {
      id: user.id,
      email: user.email,
      displayName: user.displayName ?? null,
      role: user.role ?? null,
    },
    sessionToken,
  };
}

// ===================================================================
// Provider class (implements AuthProvider interface)
// ===================================================================

export class WebAuthnProvider implements AuthProvider {
  readonly name = "webauthn";

  async authenticate(request: Request, db: Database, secret: string): Promise<AuthResult | null> {
    // WebAuthn authenticate expects JSON body with credentialResponse + challenge
    const body = await request.json();
    const { credentialResponse, challenge } = body;
    if (!credentialResponse || !challenge) return null;

    return verifyWebAuthnAuthentication(db, secret, credentialResponse, challenge, request);
  }

  getLoginMethods(): LoginMethod[] {
    return [{ type: "webauthn", label: "Sign in with Passkey" }];
  }
}
