interface OmegaMarkProps {
  size?: number;
  className?: string;
}

export default function OmegaMark({
  size = 28,
  className = "",
}: OmegaMarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="OmegaMax"
    >
      {/* Zero capsule */}
      <rect
        x="9"
        y="5"
        width="14"
        height="22"
        rx="7"
        stroke="currentColor"
        strokeWidth="2.4"
        fill="none"
      />
      {/* Center dot */}
      <circle cx="16" cy="16" r="2.5" fill="currentColor" />
    </svg>
  );
}
