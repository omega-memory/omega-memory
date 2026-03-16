export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

export default function ToggleSwitch({ checked, onChange, disabled }: ToggleSwitchProps) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={(e) => {
        e.stopPropagation();
        onChange(!checked);
      }}
      className={`
        relative w-10 h-6 rounded-full transition-colors duration-200
        touch-manipulation shrink-0
        ${checked ? "bg-emerald-500/40" : "bg-white/[0.08]"}
        ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer hover:brightness-110"}
      `}
    >
      <span
        className={`
          absolute top-0.5 left-0.5 w-5 h-5 rounded-full
          transition-transform duration-200 shadow-sm
          ${checked ? "translate-x-4 bg-emerald-500" : "translate-x-0 bg-ink-tertiary"}
        `}
      />
    </button>
  );
}
