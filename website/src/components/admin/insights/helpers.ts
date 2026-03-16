export function formatContentType(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function distColor(type: string, TYPE_REGISTRY: Record<string, { color?: string }>): string {
  if (type === "decision") return "var(--color-gold)";
  return TYPE_REGISTRY[type]?.color ?? "var(--color-ink-faint)";
}
