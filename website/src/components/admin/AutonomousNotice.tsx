import { useState, useEffect, useCallback } from "react";

interface AutonomousModeState {
  enabled: boolean;
  tweets_today: number;
  replies_today: number;
}

export default function AutonomousNotice() {
  const [state, setState] = useState<AutonomousModeState | null>(null);
  const [toggling, setToggling] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/autonomous-mode");
      if (res.ok) setState(await res.json());
    } catch {}
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  const handleToggle = async () => {
    if (!state || toggling) return;
    setToggling(true);
    try {
      const res = await fetch("/api/admin/autonomous-mode", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !state.enabled }),
      });
      if (res.ok) {
        const data = await res.json();
        setState((prev) => prev ? { ...prev, enabled: data.enabled } : prev);
      }
    } catch {} finally {
      setToggling(false);
    }
  };

  if (!state) {
    return (
      <div className="px-5 pt-8 pb-4">
        <div className="h-48 rounded-xl skeleton" />
      </div>
    );
  }

  return (
    <div className="px-5 pt-8 pb-4">
      <div className={`rounded-xl border p-6 ${
        state.enabled
          ? "border-emerald-500/20 bg-emerald-500/[0.04]"
          : "border-amber-500/20 bg-amber-500/[0.04]"
      }`}>
        {/* Icon + headline */}
        <div className="flex items-start gap-4">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
            state.enabled
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-amber-500/10 text-amber-400"
          }`}>
            {state.enabled ? (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 4.5 8.25v9a2.25 2.25 0 0 0 2.25 2.25Z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" />
              </svg>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <h3 className="text-[15px] font-semibold text-ink">
              {state.enabled ? "AI Autonomous Mode" : "Autonomous Mode Paused"}
            </h3>
            <p className="mt-1 text-[13px] text-ink-secondary leading-relaxed">
              {state.enabled
                ? "Content for @omega_memory is generated and posted automatically. A 30-minute veto window applies before each post."
                : "The scan-and-reply cron will not generate new content for @omega_memory."}
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="mt-5 flex gap-6">
          <div>
            <div className="text-[22px] font-bold text-ink tabular-nums">{state.tweets_today}</div>
            <div className="text-[11px] font-medium uppercase tracking-wider text-ink-faint mt-0.5">Tweets today</div>
          </div>
          <div>
            <div className="text-[22px] font-bold text-ink tabular-nums">{state.replies_today}</div>
            <div className="text-[11px] font-medium uppercase tracking-wider text-ink-faint mt-0.5">Replies today</div>
          </div>
        </div>

        {/* Toggle button */}
        <div className="mt-5">
          <button
            onClick={handleToggle}
            disabled={toggling}
            className={`px-4 py-2 text-[13px] font-semibold rounded-lg border transition-colors disabled:opacity-50 ${
              state.enabled
                ? "border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/15"
                : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15"
            }`}
          >
            {toggling
              ? "Updating..."
              : state.enabled
                ? "Disable Autonomous Mode"
                : "Enable Autonomous Mode"}
          </button>
        </div>
      </div>
    </div>
  );
}
