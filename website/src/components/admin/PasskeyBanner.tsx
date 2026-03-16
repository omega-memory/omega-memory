import { useState, useEffect } from "react";

export default function PasskeyBanner() {
  const [showBanner, setShowBanner] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");

  // Check whether user already has a passkey registered
  useEffect(() => {
    (async () => {
      try {
        if (localStorage.getItem("omega-passkey-registered")) return;
        const res = await fetch("/api/auth/passkey-status");
        if (res.ok) {
          const { hasPasskey } = await res.json();
          if (hasPasskey) {
            localStorage.setItem("omega-passkey-registered", "1");
          } else {
            setShowBanner(true);
          }
        }
      } catch { /* ignore */ }
    })();
  }, []);

  async function handleSetup() {
    setLoading(true);
    setError("");
    try {
      const optRes = await fetch("/api/auth/webauthn/register-options", { method: "POST" });
      if (!optRes.ok) {
        setError("Failed to get registration options");
        return;
      }
      const options = await optRes.json();

      const { startRegistration } = await import("@simplewebauthn/browser");
      const regResp = await startRegistration({ optionsJSON: options });
      const verRes = await fetch("/api/auth/webauthn/register-verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(regResp),
      });
      if (verRes.ok) {
        localStorage.setItem("omega-passkey-registered", "1");
        setShowBanner(false);
        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      } else {
        const err = await verRes.json().catch(() => ({}));
        setError(err.error || "Passkey verification failed");
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name === "NotAllowedError") return;
      setError(e instanceof Error ? e.message : "Passkey setup failed");
    } finally { setLoading(false); }
  }

  return (
    <>
      {showBanner && (
        <div className="mx-4 mt-3 p-3 rounded-xl flex items-center justify-between gap-3 bg-surface-elevated border border-edge card-enter">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-gold/[0.06] border border-gold/[0.1] shrink-0">
              <svg className="w-4 h-4 text-gold-dim" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.864 4.243A7.5 7.5 0 0119.5 10.5c0 2.92-.556 5.709-1.568 8.268M5.742 6.364A7.465 7.465 0 004.5 10.5a48.667 48.667 0 00-1.109 8.18M12 10.5a3 3 0 11-6 0 3 3 0 016 0zm-3 3a7.5 7.5 0 017.5 7.5H1.5A7.5 7.5 0 019 13.5z" />
              </svg>
            </div>
            <span className="text-[13px] text-ink-secondary truncate">
              Secure your account with Touch ID
            </span>
          </div>
          <button
            onClick={handleSetup}
            disabled={loading}
            className="text-[12px] font-semibold px-3.5 py-1.5 rounded-lg shrink-0 bg-gold text-canvas hover:bg-gold-dim transition-colors disabled:opacity-50"
          >
            {loading ? "..." : "Set up"}
          </button>
        </div>
      )}
      {error && (
        <div className="mx-4 mt-2 p-2.5 rounded-xl text-[12px] text-red-400 bg-red-500/[0.06] border border-red-500/[0.1] card-enter">
          {error}
        </div>
      )}
      {success && (
        <div className="mx-4 mt-3 p-3 rounded-xl text-center text-[13px] text-type-lesson bg-type-lesson/[0.06] border border-type-lesson/[0.1] card-enter">
          Passkey registered - you can now sign in with Touch ID.
        </div>
      )}
    </>
  );
}
