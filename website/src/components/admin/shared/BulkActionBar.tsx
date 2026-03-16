interface BulkActionBarProps {
  count: number;
  onClear: () => void;
  children: React.ReactNode;
}

export default function BulkActionBar({ count, onClear, children }: BulkActionBarProps) {
  if (count === 0) return null;

  return (
    <div className="sticky bottom-0 z-10 mx-4 mb-4 card-enter">
      <div className="flex items-center justify-between gap-3 px-4 py-3 rounded-xl bg-surface border border-gold/20 shadow-[0_-4px_24px_rgba(0,0,0,0.3)]">
        <div className="flex items-center gap-3">
          <span className="text-[14px] font-semibold text-ink tabular-nums">
            {count} selected
          </span>
          <button
            onClick={onClear}
            className="text-[13px] text-ink-tertiary hover:text-ink-secondary transition-colors"
          >
            Clear
          </button>
        </div>
        <div className="flex items-center gap-2">
          {children}
        </div>
      </div>
    </div>
  );
}
