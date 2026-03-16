import { useState } from "react";

const MAX_CHARS = 280;

interface Props {
  onClose: () => void;
  onCreated: () => void;
}

export default function ThreadComposer({ onClose, onCreated }: Props) {
  const [parts, setParts] = useState<string[]>(["", ""]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scheduleDate, setScheduleDate] = useState("");
  const [scheduleTime, setScheduleTime] = useState("09:00");
  const [xAccount, setXAccount] = useState<"jasonsosa" | "omega_memory">("jasonsosa");

  function updatePart(index: number, value: string) {
    setParts((prev) => prev.map((p, i) => (i === index ? value : p)));
  }

  function addPart() {
    if (parts.length >= 10) return;
    setParts((prev) => [...prev, ""]);
  }

  function removePart(index: number) {
    if (parts.length <= 2) return;
    setParts((prev) => prev.filter((_, i) => i !== index));
  }

  function movePart(index: number, direction: -1 | 1) {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= parts.length) return;
    setParts((prev) => {
      const updated = [...prev];
      [updated[index], updated[newIndex]] = [updated[newIndex], updated[index]];
      return updated;
    });
  }

  const isValid = parts.every((p) => p.trim().length > 0) && parts.every((p) => p.length <= MAX_CHARS);
  const totalChars = parts.reduce((sum, p) => sum + p.length, 0);

  async function handleSave() {
    if (!isValid) return;
    setSaving(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        thread_parts: parts.map((p) => p.trim()),
        x_account: xAccount,
      };
      if (scheduleDate) {
        body.scheduled_for = new Date(`${scheduleDate}T${scheduleTime}:00`).toISOString();
      }

      const res = await fetch("/api/tweets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Failed to create thread (${res.status})`);
      }

      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create thread");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mx-5 mb-4 bg-surface border border-gold/20 rounded-xl overflow-hidden card-enter">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-edge">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-gold" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 0 1 0 3.75H5.625a1.875 1.875 0 0 1 0-3.75Z" />
          </svg>
          <span className="text-[15px] font-semibold text-ink">New Thread</span>
          <span className="text-[12px] text-ink-faint">{parts.length} parts</span>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-surface-hover text-ink-faint hover:text-ink transition-colors touch-manipulation"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Account selector */}
      <div className="px-4 pt-3 flex gap-1 p-1.5 bg-canvas/50 rounded-lg mx-4 mt-3">
        {(["jasonsosa", "omega_memory"] as const).map((acct) => (
          <button
            key={acct}
            onClick={() => setXAccount(acct)}
            className={`flex-1 py-2 text-[13px] font-medium rounded-md transition-all ${
              xAccount === acct
                ? "bg-surface-elevated text-gold shadow-sm"
                : "text-ink-tertiary hover:text-ink-secondary"
            }`}
          >
            @{acct === "omega_memory" ? "omega_memory" : "jasonsosa"}
          </button>
        ))}
      </div>

      {/* Thread parts */}
      <div className="p-4 space-y-3">
        {parts.map((part, idx) => {
          const charCount = part.length;
          const isOverLimit = charCount > MAX_CHARS;
          const charColor = isOverLimit ? "text-type-error" : charCount > MAX_CHARS * 0.9 ? "text-gold" : "text-ink-faint";

          return (
            <div key={idx} className="relative group/part">
              <div className="flex items-start gap-2">
                {/* Thread connector */}
                <div className="flex flex-col items-center pt-3 shrink-0">
                  <div className="w-6 h-6 rounded-full bg-gold/10 border border-gold/25 flex items-center justify-center">
                    <span className="text-[11px] font-bold text-gold">{idx + 1}</span>
                  </div>
                  {idx < parts.length - 1 && (
                    <div className="w-[2px] h-full min-h-[20px] bg-gold/15 mt-1" />
                  )}
                </div>

                {/* Text area */}
                <div className="flex-1 min-w-0">
                  <textarea
                    value={part}
                    onChange={(e) => updatePart(idx, e.target.value)}
                    placeholder={idx === 0 ? "Hook: grab attention with a bold claim or question..." : idx === parts.length - 1 ? "Closer: CTA or key takeaway..." : "Continue the thread..."}
                    rows={3}
                    className={`w-full bg-surface-elevated border rounded-lg px-3 py-2.5 text-[14px] text-ink placeholder:text-ink-faint resize-none focus:outline-none focus:ring-1 transition-all ${
                      isOverLimit
                        ? "border-type-error/40 focus:border-type-error/60 focus:ring-type-error/20"
                        : "border-edge focus:border-gold/40 focus:ring-gold/20"
                    }`}
                  />
                  <div className="flex items-center justify-between mt-1">
                    <span className={`text-[11px] ${charColor}`}>
                      {charCount}/{MAX_CHARS}
                    </span>
                    <div className="flex gap-1 opacity-0 group-hover/part:opacity-100 transition-opacity">
                      {idx > 0 && (
                        <button
                          onClick={() => movePart(idx, -1)}
                          className="p-1 rounded text-ink-faint hover:text-ink-secondary touch-manipulation"
                          title="Move up"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 15.75 7.5-7.5 7.5 7.5" />
                          </svg>
                        </button>
                      )}
                      {idx < parts.length - 1 && (
                        <button
                          onClick={() => movePart(idx, 1)}
                          className="p-1 rounded text-ink-faint hover:text-ink-secondary touch-manipulation"
                          title="Move down"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
                          </svg>
                        </button>
                      )}
                      {parts.length > 2 && (
                        <button
                          onClick={() => removePart(idx)}
                          className="p-1 rounded text-ink-faint hover:text-type-error touch-manipulation"
                          title="Remove"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}

        {/* Add part button */}
        {parts.length < 10 && (
          <button
            onClick={addPart}
            className="w-full py-2.5 rounded-lg border border-dashed border-edge hover:border-gold/30 text-ink-faint hover:text-gold text-[13px] transition-all flex items-center justify-center gap-1.5 touch-manipulation"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            Add part
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-3 p-3 bg-type-error/[0.12] border border-type-error/25 rounded-lg text-[13px] text-type-error">
          {error}
        </div>
      )}

      {/* Footer actions */}
      <div className="px-4 pb-4 space-y-3">
        {/* Schedule row */}
        <div className="flex items-center gap-2">
          <input
            type="date"
            value={scheduleDate}
            onChange={(e) => setScheduleDate(e.target.value)}
            min={new Date().toISOString().slice(0, 10)}
            className="flex-1 bg-surface-elevated border border-edge rounded-lg px-3 py-2 text-[13px] text-ink focus:outline-none focus:border-gold/40"
            placeholder="Schedule date (optional)"
          />
          <input
            type="time"
            value={scheduleTime}
            onChange={(e) => setScheduleTime(e.target.value)}
            className="w-28 bg-surface-elevated border border-edge rounded-lg px-3 py-2 text-[13px] text-ink focus:outline-none focus:border-gold/40"
          />
        </div>

        {/* Save */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleSave}
            disabled={saving || !isValid}
            className="flex-1 py-3 rounded-lg bg-gold text-canvas text-[14px] font-semibold transition-all disabled:opacity-50 touch-manipulation hover:bg-gold-dim flex items-center justify-center gap-2"
          >
            {saving ? (
              <>
                <span className="w-4 h-4 border-2 border-canvas/30 border-t-canvas rounded-full animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 0 1 0 3.75H5.625a1.875 1.875 0 0 1 0-3.75Z" />
                </svg>
                Create thread ({parts.length} parts)
              </>
            )}
          </button>
          <button
            onClick={onClose}
            className="px-5 py-3 rounded-lg border border-edge text-[14px] text-ink-secondary font-medium hover:bg-surface-hover transition-colors touch-manipulation"
          >
            Cancel
          </button>
        </div>

        <div className="text-[11px] text-ink-faint text-center">
          {totalChars} total characters across {parts.length} tweets
          {scheduleDate ? ` · Scheduled for ${scheduleDate} ${scheduleTime}` : " · Auto-scheduled to next slot"}
        </div>
      </div>
    </div>
  );
}
