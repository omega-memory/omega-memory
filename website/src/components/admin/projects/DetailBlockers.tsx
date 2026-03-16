interface DetailBlockersProps {
  blockedItems: string[];
}

export default function DetailBlockers({ blockedItems }: DetailBlockersProps) {
  if (blockedItems.length === 0) return null;

  return (
    <div className="rounded-lg border border-red-400/20 bg-red-400/[0.04] p-3 space-y-2">
      <h3 className="text-xs font-semibold text-red-400/80 uppercase tracking-wider">
        Blocked ({blockedItems.length})
      </h3>
      <div className="space-y-1.5">
        {blockedItems.map((item, i) => (
          <div key={i} className="flex items-start gap-2 text-sm">
            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-red-400 flex-shrink-0" />
            <span className="text-white/60">{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
