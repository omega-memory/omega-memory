interface DetailNextStepsProps {
  nextSteps: string[];
}

export default function DetailNextSteps({ nextSteps }: DetailNextStepsProps) {
  if (nextSteps.length === 0) return null;

  return (
    <div className="rounded-lg border border-amber-400/20 bg-amber-400/[0.04] p-3 space-y-2">
      <h3 className="text-xs font-semibold text-amber-400/80 uppercase tracking-wider">
        Next Steps ({nextSteps.length})
      </h3>
      <div className="space-y-1.5">
        {nextSteps.map((step, i) => (
          <div key={i} className="flex items-start gap-2 text-[13px]">
            <span className="text-amber-400/60 flex-shrink-0 font-medium w-4 text-right">
              {i + 1}.
            </span>
            <span className="text-white/60">{step}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
