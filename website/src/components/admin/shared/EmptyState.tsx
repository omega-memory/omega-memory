interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export default function EmptyState({ icon, title, description, action, className = "" }: EmptyStateProps) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 px-6 text-center ${className}`}>
      {icon && (
        <div className="w-12 h-12 rounded-xl bg-surface-elevated border border-edge flex items-center justify-center mb-4 text-ink-faint">
          {icon}
        </div>
      )}
      <h3 className="text-[15px] font-medium text-ink-secondary mb-1">{title}</h3>
      {description && (
        <p className="text-[14px] text-ink-faint max-w-[320px] leading-relaxed">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
