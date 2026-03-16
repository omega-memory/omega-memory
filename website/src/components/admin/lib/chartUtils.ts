/**
 * Shared sparkline path utilities for admin charts.
 */

/** Generate SVG path strings from date-keyed data. */
export function makePathFromDates(
  dates: string[],
  data: Record<string, number>,
  width: number,
  height: number,
): { line: string; area: string } {
  const values = dates.map((d) => data[d] || 0);
  return makePathFromValues(values, width, height);
}

/** Generate SVG path strings from a raw number array. */
export function makePathFromValues(
  values: number[],
  width: number,
  height: number,
): { line: string; area: string } {
  const max = Math.max(1, ...values);
  const count = values.length;
  if (count < 2) return { line: `M0,${height}`, area: `M0,${height} L${width},${height} Z` };
  const points = values.map((v, i) => ({
    x: (i / (count - 1)) * width,
    y: height - (v / max) * (height - 4),
  }));
  const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const area = `${line} L${width},${height} L0,${height} Z`;
  return { line, area };
}
