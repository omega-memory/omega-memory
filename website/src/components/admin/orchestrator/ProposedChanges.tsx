/** Renders a before/after diff of proposed changes inline on proposal cards. */

interface PoolEntry {
  type: string;
  weight: number;
}

interface ProposedChangesProps {
  metadata: Record<string, unknown>;
}

export default function ProposedChanges({ metadata }: ProposedChangesProps) {
  const eventParams = metadata.event_params as
    | { payload?: { changes?: Record<string, unknown> } }
    | undefined;
  const changes = eventParams?.payload?.changes;
  const currentValues = metadata.current_values as Record<string, unknown> | undefined;

  if (!changes) return null;

  const sections: React.ReactNode[] = [];

  // ── Content type pool diff ──────────────────────────────────
  if (changes.content_type_pool) {
    const newPool = changes.content_type_pool as PoolEntry[];
    const oldPool = currentValues?.content_type_pool as PoolEntry[] | undefined;

    sections.push(
      <div key="pool">
        <p className="text-[11px] font-semibold text-ink-faint uppercase tracking-[0.08em] mb-1.5">
          Content weights
        </p>
        <table className="w-full text-[13px] tabular-nums">
          <tbody>
            {newPool.map((entry) => {
              const old = oldPool?.find((p) => p.type === entry.type);
              const delta = old ? entry.weight - old.weight : 0;
              const colorClass =
                delta > 0
                  ? "text-type-task"
                  : delta < 0
                    ? "text-type-reminder"
                    : "text-ink-faint";

              return (
                <tr key={entry.type} className="border-b border-edge/50 last:border-0">
                  <td className="py-1 pr-3 text-ink-secondary">{entry.type}</td>
                  {old ? (
                    <>
                      <td className="py-1 pr-1.5 text-right text-ink-faint w-10">{old.weight}</td>
                      <td className="py-1 px-1 text-ink-faint w-5 text-center">&rarr;</td>
                      <td className={`py-1 pl-1.5 text-right font-medium w-10 ${colorClass}`}>
                        {entry.weight}
                      </td>
                    </>
                  ) : (
                    <td className="py-1 text-right font-medium text-ink-secondary" colSpan={3}>
                      {entry.weight}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>,
    );
  }

  // ── Slot hours diff ─────────────────────────────────────────
  if (changes.slot_hours) {
    const newHours = changes.slot_hours as number[];
    const oldHours = currentValues?.slot_hours as number[] | undefined;
    const oldSet = new Set(oldHours ?? []);
    const newSet = new Set(newHours);

    // Collect all hours for display, sorted
    const allHours = Array.from(new Set([...(oldHours ?? []), ...newHours])).sort((a, b) => a - b);

    sections.push(
      <div key="slots">
        <p className="text-[11px] font-semibold text-ink-faint uppercase tracking-[0.08em] mb-1.5">
          Posting slots
        </p>
        <div className="flex flex-wrap gap-1.5">
          {allHours.map((h) => {
            const inOld = oldSet.has(h);
            const inNew = newSet.has(h);

            let cls = "px-2 py-0.5 rounded text-[12px] font-medium tabular-nums ";
            if (inOld && inNew) {
              cls += "bg-surface text-ink-secondary"; // unchanged
            } else if (inNew && !inOld) {
              cls += "bg-type-task/15 text-type-task"; // added
            } else {
              cls += "bg-type-reminder/10 text-type-reminder line-through"; // removed
            }

            return (
              <span key={h} className={cls}>
                {String(h).padStart(2, "0")}:00
              </span>
            );
          })}
        </div>
      </div>,
    );
  }

  // ── Fallback for unknown change keys ────────────────────────
  const knownKeys = new Set(["content_type_pool", "slot_hours"]);
  const unknownKeys = Object.keys(changes).filter((k) => !knownKeys.has(k));
  if (unknownKeys.length > 0) {
    const unknownChanges: Record<string, unknown> = {};
    for (const k of unknownKeys) unknownChanges[k] = changes[k];

    sections.push(
      <div key="unknown">
        <p className="text-[11px] font-semibold text-ink-faint uppercase tracking-[0.08em] mb-1.5">
          Other changes
        </p>
        <pre className="text-[12px] text-ink-secondary bg-surface rounded p-2 overflow-x-auto">
          {JSON.stringify(unknownChanges, null, 2)}
        </pre>
      </div>,
    );
  }

  if (sections.length === 0) return null;

  return (
    <div className="mt-3 border border-edge/60 rounded-lg p-3 space-y-3">
      <p className="text-[11px] font-semibold text-ink-faint uppercase tracking-[0.08em]">
        Proposed changes
      </p>
      {sections}
    </div>
  );
}
