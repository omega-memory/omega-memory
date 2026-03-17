import type { AuthProvider, AuthResult, LoginMethod } from "./provider";
import type { Database } from "../db";
import { InternalAuthProvider } from "./providers/internal";
import { WebAuthnProvider } from "./providers/webauthn";

const providers: Record<string, () => AuthProvider> = {
  internal: () => new InternalAuthProvider(),
  webauthn: () => new WebAuthnProvider(),
  // oauth, supabase will be added by other agents
};

export function getAuthProvider(providerName?: string): AuthProvider {
  const name = providerName ||
    (typeof process !== "undefined" ? process.env?.AUTH_PROVIDER : undefined) ||
    "internal";

  // Support composite: "internal,webauthn"
  const names = name.split(",").map(n => n.trim());

  if (names.length === 1) {
    const factory = providers[names[0]];
    if (!factory) throw new Error(`Unknown auth provider: ${names[0]}`);
    return factory();
  }

  // Composite provider
  return new CompositeAuthProvider(names.map(n => {
    const factory = providers[n];
    if (!factory) throw new Error(`Unknown auth provider: ${n}`);
    return factory();
  }));
}

// Register a provider (used by other provider modules)
export function registerAuthProvider(name: string, factory: () => AuthProvider) {
  providers[name] = factory;
}

/**
 * Composite provider that merges multiple auth providers.
 * Dispatches authenticate() to each provider in order until one succeeds.
 * Merges getLoginMethods() from all providers.
 */
class CompositeAuthProvider implements AuthProvider {
  readonly name: string;

  constructor(private readonly delegates: AuthProvider[]) {
    this.name = delegates.map(d => d.name).join("+");
  }

  async authenticate(request: Request, db: Database, secret: string): Promise<AuthResult | null> {
    // We need to clone the request for each provider since body can only be read once
    for (let i = 0; i < this.delegates.length; i++) {
      const provider = this.delegates[i];
      const req = i < this.delegates.length - 1 ? request.clone() : request;
      try {
        const result = await provider.authenticate(req, db, secret);
        if (result) return result;
      } catch {
        // Provider couldn't handle this request format, try next
        continue;
      }
    }
    return null;
  }

  async getAuthorizationUrl(state: string, redirectUri: string): Promise<string> {
    for (const provider of this.delegates) {
      if (provider.getAuthorizationUrl) {
        return provider.getAuthorizationUrl(state, redirectUri);
      }
    }
    throw new Error("No provider supports getAuthorizationUrl");
  }

  async handleCallback(request: Request, db: Database, secret: string): Promise<AuthResult | null> {
    for (const provider of this.delegates) {
      if (provider.handleCallback) {
        const result = await provider.handleCallback(request.clone(), db, secret);
        if (result) return result;
      }
    }
    return null;
  }

  getLoginMethods(): LoginMethod[] {
    const methods: LoginMethod[] = [];
    for (const provider of this.delegates) {
      methods.push(...provider.getLoginMethods());
    }
    return methods;
  }
}
