import { useCallback, useEffect, useState } from "react";

type TargetTier = 1 | 2 | 3;

interface EngagementTarget {
  id: string;
  handle: string;
  display_name: string | null;
  tier: TargetTier;
  notes: string | null;
  last_checked: string | null;
  active: boolean;
  x_account: string | null; // null = both accounts
  replies_sent_jasonsosa: number;
  replies_sent_omega_memory: number;
  last_replied_at_jasonsosa: string | null;
  last_replied_at_omega_memory: string | null;
  created_at: string;
}

const TIER_CONFIG: Record<TargetTier, { label: string; color: string; bg: string; border: string; desc: string }> = {
  1: { label: "Tier 1", color: "text-type-error", bg: "bg-type-error/10", border: "border-type-error/25", desc: "Reply daily" },
  2: { label: "Tier 2", color: "text-gold", bg: "bg-gold/10", border: "border-gold/25", desc: "Reply when relevant" },
  3: { label: "Tier 3", color: "text-ink-secondary", bg: "bg-ink-faint/10", border: "border-edge", desc: "Monitor" },
};

export default function TargetsManager() {
  const [targets, setTargets] = useState<EngagementTarget[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [seeding, setSeeding] = useState(false);

  // Add form state
  const [newHandle, setNewHandle] = useState("");
  const [newDisplayName, setNewDisplayName] = useState("");
  const [newTier, setNewTier] = useState<TargetTier>(2);
  const [newNotes, setNewNotes] = useState("");
  const [newAccount, setNewAccount] = useState<string | null>(null); // null = both
  const [saving, setSaving] = useState(false);

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch("/api/targets");
      if (!res.ok) throw new Error("Failed to load targets");
      const data = await res.json();
      setTargets(data.targets ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function showToast(msg: string) {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  }

  async function handleAdd() {
    if (!newHandle.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/targets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "upsert",
          handle: newHandle.trim().replace(/^@/, ""),
          display_name: newDisplayName.trim() || null,
          tier: newTier,
          notes: newNotes.trim() || null,
          x_account: newAccount,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Failed to add target");
      }
      const { target } = await res.json();
      setTargets((prev) => {
        const exists = prev.findIndex((t) => t.handle === target.handle);
        if (exists >= 0) {
          const updated = [...prev];
          updated[exists] = target;
          return updated;
        }
        return [...prev, target];
      });
      setNewHandle("");
      setNewDisplayName("");
      setNewTier(2);
      setNewNotes("");
      setNewAccount(null);
      setAdding(false);
      showToast(`@${target.handle} added`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Add failed");
    } finally {
      setSaving(false);
    }
  }

  async function handleChangeTier(target: EngagementTarget, tier: TargetTier) {
    try {
      const res = await fetch("/api/targets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "upsert",
          handle: target.handle,
          display_name: target.display_name,
          tier,
          notes: target.notes,
        }),
      });
      if (!res.ok) throw new Error("Failed to update tier");
      const { target: updated } = await res.json();
      setTargets((prev) => prev.map((t) => (t.id === target.id ? updated : t)));
      setEditingId(null);
      showToast(`@${target.handle} moved to Tier ${tier}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Update failed");
    }
  }

  async function handleDeactivate(target: EngagementTarget) {
    try {
      const res = await fetch("/api/targets", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handle: target.handle }),
      });
      if (!res.ok) throw new Error("Failed to remove target");
      setTargets((prev) => prev.filter((t) => t.id !== target.id));
      showToast(`@${target.handle} removed`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Remove failed");
    }
  }

  async function handleSeedDefaults() {
    setSeeding(true);
    setError(null);
    try {
      const res = await fetch("/api/targets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "seed" }),
      });
      if (!res.ok) throw new Error("Failed to seed defaults");
      const { seeded } = await res.json();
      if (seeded.targets === 0 && seeded.keywords === 0) {
        showToast("Defaults already loaded");
      } else {
        showToast(`Seeded ${seeded.targets} targets, ${seeded.keywords} keywords`);
        await load();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Seed failed");
    } finally {
      setSeeding(false);
    }
  }

  function formatTimeAgo(dateStr: string | null) {
    if (!dateStr) return "Never";
    const d = new Date(dateStr);
    const mins = Math.floor((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  const grouped = {
    1: targets.filter((t) => t.tier === 1),
    2: targets.filter((t) => t.tier === 2),
    3: targets.filter((t) => t.tier === 3),
  };

  if (loading) {
    return (
      <div className="space-y-4 px-5 pt-2 animate-pulse">
        {[0, 1, 2].map((i) => (
          <div key={i} className="admin-card p-4">
            <div className="h-5 w-24 rounded-md skeleton mb-3" />
            <div className="space-y-2">
              <div className="h-10 rounded-lg skeleton" />
              <div className="h-10 rounded-lg skeleton" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="pb-4">
      {/* Header with actions */}
      <div className="flex items-center justify-between mx-5 mb-4">
        <div>
          <h3 className="admin-section-label text-gold/80">Reply Targets</h3>
          <p className="text-[13px] text-ink-faint mt-0.5 font-mono tabular-nums">{targets.length} accounts tracked</p>
        </div>
        <div className="flex gap-2">
          {targets.length === 0 && (
            <button
              onClick={handleSeedDefaults}
              disabled={seeding}
              className="px-3 py-1.5 rounded-lg border border-edge hover:bg-surface-hover text-[13px] text-ink-secondary font-medium transition-colors touch-manipulation disabled:opacity-50"
            >
              {seeding ? "..." : "Load defaults"}
            </button>
          )}
          <button
            onClick={() => setAdding(!adding)}
            className="px-3 py-1.5 rounded-lg bg-gold/[0.12] border border-gold/20 text-[13px] text-gold font-semibold hover:bg-gold/[0.18] transition-colors touch-manipulation"
          >
            {adding ? "Cancel" : "+ Add"}
          </button>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="mx-5 mb-4 p-3 bg-type-lesson/[0.12] border border-type-lesson/25 rounded-xl text-[14px] font-medium text-type-lesson card-enter flex items-center gap-2">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
          {toast}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mx-5 mb-4 p-3 bg-type-error/[0.12] border border-type-error/25 rounded-xl text-[14px] text-type-error flex items-center gap-2">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
          </svg>
          {error}
        </div>
      )}

      {/* Add form */}
      {adding && (
        <div className="mx-5 mb-4 p-4 bg-surface border border-gold/20 rounded-xl space-y-3 card-enter">
          <div className="grid grid-cols-2 gap-3">
            <input
              type="text"
              value={newHandle}
              onChange={(e) => setNewHandle(e.target.value)}
              placeholder="@handle"
              className="bg-surface-elevated border border-edge rounded-lg px-3 py-2.5 text-[14px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40 focus:ring-1 focus:ring-gold/20 transition-all"
            />
            <input
              type="text"
              value={newDisplayName}
              onChange={(e) => setNewDisplayName(e.target.value)}
              placeholder="Display name (optional)"
              className="bg-surface-elevated border border-edge rounded-lg px-3 py-2.5 text-[14px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40 focus:ring-1 focus:ring-gold/20 transition-all"
            />
          </div>
          <div className="flex gap-2">
            {([1, 2, 3] as TargetTier[]).map((t) => {
              const cfg = TIER_CONFIG[t];
              return (
                <button
                  key={t}
                  onClick={() => setNewTier(t)}
                  className={`flex-1 py-2 rounded-lg text-[13px] font-medium border transition-all touch-manipulation ${
                    newTier === t
                      ? `${cfg.bg} ${cfg.color} ${cfg.border}`
                      : "border-edge text-ink-faint hover:text-ink-secondary"
                  }`}
                >
                  {cfg.label}
                </button>
              );
            })}
          </div>
          <div className="flex gap-2">
            {([
              { value: null, label: "Both accounts" },
              { value: "jasonsosa", label: "@jasonsosa" },
              { value: "omega_memory", label: "@omega_memory" },
            ] as const).map((opt) => (
              <button
                key={String(opt.value)}
                onClick={() => setNewAccount(opt.value as string | null)}
                className={`flex-1 py-2 rounded-lg text-[13px] font-medium border transition-all touch-manipulation ${
                  newAccount === opt.value
                    ? "bg-ink-faint/15 text-ink border-edge"
                    : "border-edge text-ink-faint hover:text-ink-secondary"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <input
            type="text"
            value={newNotes}
            onChange={(e) => setNewNotes(e.target.value)}
            placeholder="Notes (e.g. 'reply to AI memory topics')"
            className="w-full bg-surface-elevated border border-edge rounded-lg px-3 py-2.5 text-[14px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40 focus:ring-1 focus:ring-gold/20 transition-all"
          />
          <button
            onClick={handleAdd}
            disabled={saving || !newHandle.trim()}
            className="w-full py-2.5 rounded-lg bg-gold text-canvas text-[14px] font-semibold transition-all disabled:opacity-50 touch-manipulation hover:bg-gold-dim"
          >
            {saving ? "Adding..." : "Add target"}
          </button>
        </div>
      )}

      {/* Tier groups */}
      {([1, 2, 3] as TargetTier[]).map((tier) => {
        const cfg = TIER_CONFIG[tier];
        const group = grouped[tier];
        if (group.length === 0) return null;

        return (
          <div key={tier} className="mx-5 mb-4">
            <div className="flex items-center gap-2 mb-2.5">
              <span className={`text-[12px] font-semibold px-2 py-0.5 rounded-md ${cfg.bg} ${cfg.color} uppercase tracking-wider`}>
                {cfg.label}
              </span>
              <span className="text-[12px] text-ink-faint">{cfg.desc}</span>
              <span className="text-[12px] text-ink-faint ml-auto">{group.length}</span>
            </div>
            <div className="space-y-1.5 admin-stagger">
              {group.map((target) => (
                <div
                  key={target.id}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-surface border border-edge hover:bg-surface-hover group/target transition-colors"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[14px] font-medium text-ink">@{target.handle}</span>
                      {target.display_name && (
                        <span className="text-[13px] text-ink-faint truncate">{target.display_name}</span>
                      )}
                      <span className="text-[11px] text-ink-faint font-mono">
                        {target.x_account === "omega_memory" ? "@omega" : target.x_account === "jasonsosa" ? "@jason" : "both"}
                      </span>
                    </div>
                    {target.notes && (
                      <p className="text-[12px] text-ink-faint mt-0.5 truncate">{target.notes}</p>
                    )}
                  </div>
                  {/* Per-account reply counts */}
                  <div className="flex gap-2 shrink-0">
                    {(target.x_account === null || target.x_account === "jasonsosa") && (
                      <div className="text-center" title={target.last_replied_at_jasonsosa ? `Last: ${formatTimeAgo(target.last_replied_at_jasonsosa)}` : "No replies yet from @jasonsosa"}>
                        <div className="text-[13px] font-semibold font-mono tabular-nums text-ink-secondary">{target.replies_sent_jasonsosa ?? 0}</div>
                        <div className="text-[9px] font-mono text-ink-faint uppercase">J</div>
                      </div>
                    )}
                    {(target.x_account === null || target.x_account === "omega_memory") && (
                      <div className="text-center" title={target.last_replied_at_omega_memory ? `Last: ${formatTimeAgo(target.last_replied_at_omega_memory)}` : "No replies yet from @omega_memory"}>
                        <div className="text-[13px] font-semibold font-mono tabular-nums text-gold">{target.replies_sent_omega_memory ?? 0}</div>
                        <div className="text-[9px] font-mono text-ink-faint uppercase">O</div>
                      </div>
                    )}
                  </div>
                  <span className="text-[11px] text-ink-faint shrink-0 font-mono">
                    {formatTimeAgo(target.last_checked)}
                  </span>

                  {/* Tier change dropdown */}
                  {editingId === target.id ? (
                    <div className="flex gap-1 shrink-0">
                      {([1, 2, 3] as TargetTier[]).map((t) => {
                        const tcfg = TIER_CONFIG[t];
                        return (
                          <button
                            key={t}
                            onClick={() => handleChangeTier(target, t)}
                            className={`px-2 py-1 rounded text-[11px] font-medium border transition-all touch-manipulation ${
                              target.tier === t
                                ? `${tcfg.bg} ${tcfg.color} ${tcfg.border}`
                                : "border-edge text-ink-faint hover:text-ink-secondary"
                            }`}
                          >
                            T{t}
                          </button>
                        );
                      })}
                      <button
                        onClick={() => setEditingId(null)}
                        className="px-1.5 py-1 text-[11px] text-ink-faint hover:text-ink-secondary"
                      >
                        Done
                      </button>
                    </div>
                  ) : (
                    <div className="flex gap-1 shrink-0 opacity-0 group-hover/target:opacity-100 transition-opacity">
                      <button
                        onClick={() => {
                          setEditingId(target.id);
                        }}
                        className="px-2 py-1 rounded text-[11px] text-ink-faint hover:text-ink-secondary border border-transparent hover:border-edge transition-all touch-manipulation"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeactivate(target)}
                        className="px-2 py-1 rounded text-[11px] text-ink-faint hover:text-type-error border border-transparent hover:border-type-error/25 transition-all touch-manipulation"
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* Empty state */}
      {targets.length === 0 && !adding && (
        <div className="text-center px-8 py-16">
          <div className="w-14 h-14 mx-auto mb-5 rounded-full bg-gold/10 flex items-center justify-center">
            <svg className="w-7 h-7 text-gold" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 0 0 3.741-.479 3 3 0 0 0-4.682-2.72m.94 3.198.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0 1 12 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 0 1 6 18.719m12 0a5.971 5.971 0 0 0-.941-3.197m0 0A5.995 5.995 0 0 0 12 12.75a5.995 5.995 0 0 0-5.058 2.772m0 0a3 3 0 0 0-4.681 2.72 8.986 8.986 0 0 0 3.74.477m.94-3.197a5.971 5.971 0 0 0-.94 3.197M15 6.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm6 3a2.25 2.25 0 1 1-4.5 0 2.25 2.25 0 0 1 4.5 0Zm-13.5 0a2.25 2.25 0 1 1-4.5 0 2.25 2.25 0 0 1 4.5 0Z" />
            </svg>
          </div>
          <p className="text-ink-secondary text-[17px] font-medium mb-1.5">No reply targets</p>
          <p className="text-ink-tertiary text-[15px] mb-4">
            Add accounts to track for the "reply guy" strategy. Reply to their tweets within 15 minutes for maximum algorithmic boost.
          </p>
          <button
            onClick={handleSeedDefaults}
            disabled={seeding}
            className="px-5 py-2.5 rounded-lg bg-gold/[0.12] border border-gold/20 text-[14px] text-gold font-semibold hover:bg-gold/[0.18] transition-colors touch-manipulation disabled:opacity-50"
          >
            {seeding ? "Loading..." : "Load recommended targets"}
          </button>
        </div>
      )}
    </div>
  );
}
