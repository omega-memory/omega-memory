interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
  className?: string;
}

export default function ErrorBanner({ message, onDismiss, className = "" }: ErrorBannerProps) {
  return (
    <div
      className={`p-3.5 bg-type-error/[0.12] border border-type-error/25 rounded-xl text-[15px] text-type-error flex items-center gap-2.5 ${className}`}
      role="alert"
    >
      <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
      </svg>
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="shrink-0 text-type-error/60 hover:text-type-error transition-colors"
          aria-label="Dismiss error"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
}
