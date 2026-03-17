import type { AuthProvider, AuthResult, LoginMethod } from "../provider";
import type { Database } from "../../db";
import { AuthEngine } from "../index";

export class InternalAuthProvider implements AuthProvider {
  readonly name = "internal";

  async authenticate(request: Request, db: Database, secret: string): Promise<AuthResult | null> {
    const auth = new AuthEngine(db, secret);
    // Parse email+password from request body (support both JSON and FormData)
    let email: string;
    let password: string;
    const contentType = request.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const body = await request.json();
      email = body.email;
      password = body.password;
    } else {
      const form = await request.formData();
      email = form.get("email") as string;
      password = form.get("password") as string;
    }
    if (!email || !password) return null;

    const user = await auth.authenticateUser(email, password);
    if (!user) return null;

    const deviceInfo = request.headers.get("user-agent") ?? undefined;
    const sessionToken = await auth.createSession(user.id, deviceInfo);
    return {
      user: { id: user.id, email: user.email, displayName: user.displayName ?? null, role: user.role ?? null },
      sessionToken,
    };
  }

  getLoginMethods(): LoginMethod[] {
    return [{ type: "password" }];
  }
}
