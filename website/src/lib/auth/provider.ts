import type { Database } from "../db";

export interface AuthUser {
  id: string;
  email: string;
  displayName: string | null;
  role: string | null;
}

export interface AuthResult {
  user: AuthUser;
  sessionToken: string;
}

export type LoginMethod =
  | { type: "password" }
  | { type: "oauth"; provider: string; label: string; icon?: string }
  | { type: "webauthn"; label: string }
  | { type: "supabase"; providers: string[] };

export interface AuthProvider {
  readonly name: string;
  authenticate(request: Request, db: Database, secret: string): Promise<AuthResult | null>;
  getAuthorizationUrl?(state: string, redirectUri: string): Promise<string>;
  handleCallback?(request: Request, db: Database, secret: string): Promise<AuthResult | null>;
  getLoginMethods(): LoginMethod[];
}
