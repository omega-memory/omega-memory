import { SUBSYSTEM_COLORS } from "./helpers";

export default function SubsystemPill({ subsystem }: { subsystem: string }) {
  const style =
    SUBSYSTEM_COLORS[subsystem] ??
    "bg-ink-faint/10 text-ink-tertiary border-ink-faint/20";
  return (
    <span
      className={`px-2.5 py-1 rounded-md border text-[14px] font-medium ${style}`}
    >
      {subsystem}
    </span>
  );
}
