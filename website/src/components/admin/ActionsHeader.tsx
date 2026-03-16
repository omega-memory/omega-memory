import { useState } from "react";
import type { XAccount } from "./lib/types";

const ACCOUNT_OPTIONS: { value: XAccount; label: string }[] = [
  { value: "jasonsosa", label: "@jasonsosa" },
  { value: "omega_memory", label: "@omega_memory" },
];

interface ActionsHeaderProps {
  account: XAccount;
  onAccountChange: (account: XAccount) => void;
  onGenerate?: () => void;
  generating?: boolean;
  genProgress?: string;
}

export default function ActionsHeader({
  account,
  onAccountChange,
  onGenerate,
  generating,
  genProgress,
}: ActionsHeaderProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const targetLabel = `@${account}`;
  const isAutonomous = account === "omega_memory";

  return (
    <div className="flex items-center justify-between px-6 pt-5 pb-3">
      {/* Account selector with title */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <div
            className="flex items-center gap-1.5 cursor-help"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
          >
            <span className="text-[14px] font-medium uppercase tracking-wider text-ink-faint">Account</span>
            <svg className="w-3 h-3 text-ink-faint/60" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z" />
            </svg>
          </div>
          {/* Tooltip */}
          {showTooltip && (
            <div className="absolute left-0 top-full mt-1.5 z-50 px-3 py-2 rounded-lg bg-surface-elevated border border-edge-strong shadow-lg text-[14px] text-ink-secondary leading-relaxed w-[240px] pointer-events-none">
              Filter tweets, suggestions, and events by X account. Does not affect automations or schedules.
            </div>
          )}
        </div>

        <div className="relative">
          <select
            value={account}
            onChange={(e) => onAccountChange(e.target.value as XAccount)}
            className="appearance-none pl-7 pr-7 py-2 text-[15px] font-medium rounded-lg bg-surface border border-edge text-ink cursor-pointer hover:bg-surface-hover transition-colors"
          >
            {ACCOUNT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          {/* X icon overlay */}
          <svg className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-tertiary pointer-events-none" viewBox="0 0 24 24" fill="currentColor">
            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
          </svg>
          {/* Chevron overlay */}
          <svg className="absolute right-2 top-1/2 -translate-y-1/2 w-2.5 h-2.5 text-ink-faint pointer-events-none" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
          </svg>
        </div>

        {/* Mode badge */}
        {account === "omega_memory" && (
          <span className="flex items-center gap-1.5 text-[14px] font-medium text-emerald-400 bg-emerald-500/[0.08] border border-emerald-500/[0.12] px-2.5 py-1.5 rounded-md">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            Autonomous
          </span>
        )}
        {account === "jasonsosa" && (
          <span className="flex items-center gap-1.5 text-[14px] font-medium text-ink-tertiary bg-surface border border-edge px-2.5 py-1.5 rounded-md">
            <span className="w-2 h-2 rounded-full bg-ink-tertiary" />
            Manual
          </span>
        )}
      </div>

      {/* Generate button */}
      <div className="flex items-center gap-3">
        <span className="text-[14px] text-ink-faint hidden sm:block">
          Target: <span className="font-medium text-ink-secondary">{targetLabel}</span>
        </span>
        {onGenerate && (
          <div className="relative">
            <button
              onClick={isAutonomous ? undefined : onGenerate}
              disabled={generating || isAutonomous}
              title={isAutonomous ? "Content is auto-generated for @omega_memory. Switch to @jasonsosa to generate manually." : undefined}
              className={`flex items-center gap-2 px-5 py-2.5 text-[15px] font-semibold rounded-lg border min-h-[44px] transition-colors ${
                isAutonomous
                  ? "bg-surface text-ink-faint border-edge cursor-not-allowed"
                  : "bg-gold/10 text-gold border-gold/20 hover:bg-gold/15 disabled:opacity-50"
              }`}
            >
              {generating ? (
                <>
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
                  </svg>
                  <span className="max-w-[160px] truncate">{genProgress || "Generating..."}</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 0 0-2.455 2.456Z" />
                  </svg>
                  {isAutonomous ? "Auto-generated" : "Generate"}
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
