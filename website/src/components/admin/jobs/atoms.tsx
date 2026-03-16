import { type JobStatus, STATUS_CONFIG, ownerBadge } from "./jobUtils";

// ─── StatusPill ───────────────────────────────────────

export function StatusPill({ status }: { status: JobStatus }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span
      className={`inline-flex items-center gap-2 text-[14px] font-medium px-2.5 py-1 rounded-full ${cfg.bgCls} ${cfg.textCls}`}
    >
      <span className={`w-2 h-2 rounded-full ${cfg.dotCls}`} />
      {cfg.label}
    </span>
  );
}

// ─── ScheduleChip ──────────────────────────────────────

export function ScheduleChip({ text }: { text: string }) {
  return (
    <span className="text-[15px] font-mono text-ink-secondary tracking-wide">
      {text}
    </span>
  );
}

// ─── ToggleSwitch ──────────────────────────────────────

export { default as ToggleSwitch } from "../shared/ToggleSwitch";

// ─── OwnerBadge ────────────────────────────────────────

export function OwnerBadge({ label }: { label: string }) {
  const { text, cls } = ownerBadge(label);
  return (
    <span className={`text-[14px] font-semibold px-2.5 py-0.5 rounded-full ${cls}`}>
      {text}
    </span>
  );
}
