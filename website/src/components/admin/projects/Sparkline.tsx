interface SparklineProps {
  data: number[];
  color?: string;
  width?: number;
  height?: number;
}

export default function Sparkline({
  data,
  color = "rgba(212, 168, 67, 0.6)",
  width = 120,
  height = 28,
}: SparklineProps) {
  if (!data.length || data.every((v) => v === 0)) {
    return (
      <svg width={width} height={height} className="opacity-20">
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="2 4"
          className="text-white/20"
        />
      </svg>
    );
  }

  const max = Math.max(...data, 1);
  const padding = 2;
  const usableH = height - padding * 2;
  const step = (width - padding * 2) / Math.max(data.length - 1, 1);

  const points = data
    .map((v, i) => {
      const x = padding + i * step;
      const y = padding + usableH - (v / max) * usableH;
      return `${x},${y}`;
    })
    .join(" ");

  // Area fill path: polyline + close along bottom
  const firstX = padding;
  const lastX = padding + (data.length - 1) * step;
  const areaPath = `M${firstX},${padding + usableH} L${points
    .split(" ")
    .map((p) => `L${p}`)
    .join(" ")} L${lastX},${padding + usableH} Z`;

  return (
    <svg width={width} height={height}>
      <path d={areaPath} fill={color} opacity={0.15} />
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
