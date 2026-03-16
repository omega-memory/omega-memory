import Tooltip from "./Tooltip";

export default function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip content={<span className="text-ink-secondary">{text}</span>} side="bottom" delayMs={300}>
      <svg
        width={14}
        height={14}
        viewBox="0 0 16 16"
        fill="none"
        className="text-ink-tertiary hover:text-ink-secondary transition-colors cursor-help shrink-0"
      >
        <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.2" />
        <path d="M8 7v4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="8" cy="5" r="0.75" fill="currentColor" />
      </svg>
    </Tooltip>
  );
}
