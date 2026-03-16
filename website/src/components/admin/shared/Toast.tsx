interface ToastProps {
  message: string;
  variant?: "success" | "error" | "info";
}

const VARIANT_STYLES = {
  success: "bg-type-lesson/[0.12] border-type-lesson/25 text-type-lesson",
  error: "bg-type-error/[0.12] border-type-error/25 text-type-error",
  info: "bg-gold/[0.12] border-gold/25 text-gold",
};

const VARIANT_ICONS = {
  success: (
    <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
    </svg>
  ),
  error: (
    <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
    </svg>
  ),
  info: (
    <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
    </svg>
  ),
};

export default function Toast({ message, variant = "success" }: ToastProps) {
  return (
    <div
      className={`p-3.5 border rounded-xl text-[15px] font-medium card-enter flex items-center gap-2.5 ${VARIANT_STYLES[variant]}`}
      role="status"
    >
      {VARIANT_ICONS[variant]}
      {message}
    </div>
  );
}
