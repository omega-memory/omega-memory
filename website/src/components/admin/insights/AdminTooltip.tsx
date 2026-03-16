import React from "react";

interface PayloadEntry {
  value?: number;
  name?: string;
  color?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload?: Record<string, any>;
}

interface AdminTooltipProps {
  active?: boolean;
  payload?: PayloadEntry[];
  label?: string | number;
  formatLabel?: (label: string) => string;
  formatValue?: (value: number) => string;
  colorKey?: string;
}

/**
 * Shared Recharts tooltip styled in Midnight Gold.
 *
 * Usage:
 *   <Tooltip content={<AdminTooltip />} />
 *   <Tooltip content={<AdminTooltip formatLabel={dateFormatter} />} />
 */
export default function AdminTooltip({
  active,
  payload,
  label,
  formatLabel,
  formatValue,
  colorKey,
}: AdminTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const displayLabel = formatLabel ? formatLabel(String(label)) : String(label);

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-edge-strong)",
        borderRadius: 8,
        padding: "8px 12px",
        boxShadow: "0 4px 16px rgba(0,0,0,0.4), 0 0 1px rgba(212,168,67,0.15)",
      }}
    >
      {displayLabel && (
        <div
          style={{
            color: "var(--color-ink-secondary)",
            fontSize: 13,
            marginBottom: payload.length > 1 ? 6 : 4,
            fontFamily: "var(--font-mono, monospace)",
          }}
        >
          {displayLabel}
        </div>
      )}
      {payload.map((entry, i) => {
        const color =
          colorKey && entry.payload?.[colorKey]
            ? entry.payload[colorKey]
            : entry.color || "var(--color-gold)";
        const value = formatValue
          ? formatValue(entry.value as number)
          : String(entry.value);

        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              marginTop: i > 0 ? 3 : 0,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                backgroundColor: color,
                flexShrink: 0,
              }}
            />
            {entry.name && payload.length > 1 && (
              <span style={{ color: "var(--color-ink-tertiary)", fontSize: 13 }}>
                {entry.name}
              </span>
            )}
            <span
              style={{
                color: "var(--color-gold)",
                fontSize: 14,
                fontWeight: 600,
                fontVariantNumeric: "tabular-nums",
                marginLeft: "auto",
              }}
            >
              {value}
            </span>
          </div>
        );
      })}
    </div>
  );
}
