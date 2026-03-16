// Shared entity styling constants and utilities

export const TYPE_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  company:            { bg: "bg-blue-500/15",    text: "text-blue-400",    dot: "bg-blue-400" },
  llc:                { bg: "bg-blue-500/15",    text: "text-blue-400",    dot: "bg-blue-400" },
  s_corp:             { bg: "bg-blue-500/15",    text: "text-blue-400",    dot: "bg-blue-400" },
  c_corp:             { bg: "bg-blue-500/15",    text: "text-blue-400",    dot: "bg-blue-400" },
  foundation:         { bg: "bg-purple-500/15",  text: "text-purple-400",  dot: "bg-purple-400" },
  startup:            { bg: "bg-amber-500/15",   text: "text-amber-400",   dot: "bg-amber-400" },
  trust:              { bg: "bg-slate-500/15",   text: "text-slate-400",   dot: "bg-slate-400" },
  partnership:        { bg: "bg-emerald-500/15", text: "text-emerald-400", dot: "bg-emerald-400" },
  sole_proprietorship:{ bg: "bg-cyan-500/15",    text: "text-cyan-400",    dot: "bg-cyan-400" },
  nonprofit:          { bg: "bg-pink-500/15",    text: "text-pink-400",    dot: "bg-pink-400" },
  person:             { bg: "bg-emerald-500/15", text: "text-emerald-400", dot: "bg-emerald-400" },
  project:            { bg: "bg-gold/15",        text: "text-gold",        dot: "bg-gold" },
  tool:               { bg: "bg-orange-500/15",  text: "text-orange-400",  dot: "bg-orange-400" },
  concept:            { bg: "bg-cyan-500/15",    text: "text-cyan-400",    dot: "bg-cyan-400" },
  technology:         { bg: "bg-sky-500/15",     text: "text-sky-400",     dot: "bg-sky-400" },
  service:            { bg: "bg-pink-500/15",    text: "text-pink-400",    dot: "bg-pink-400" },
};

export const DEFAULT_TYPE_STYLE = { bg: "bg-ink-faint/20", text: "text-ink-secondary", dot: "bg-ink-secondary" };

export const STATUS_STYLES: Record<string, string> = {
  active: "text-emerald-400",
  acquired: "text-blue-400",
  dissolved: "text-red-400",
  dormant: "text-ink-tertiary",
  pending: "text-amber-400",
};

export function formatRelType(type: string): string {
  return type.replace(/_/g, " ");
}

// Hex colors for 3D graph rendering (mirrors TYPE_STYLES)
export const ENTITY_TYPE_COLORS: Record<string, string> = {
  company: "#6b9fff",
  llc: "#5b8fee",
  s_corp: "#4b7fdd",
  c_corp: "#3b6fcc",
  foundation: "#b088e8",
  startup: "#e8a040",
  trust: "#7878a0",
  partnership: "#5ec9a0",
  sole_proprietorship: "#40c8c8",
  nonprofit: "#d488c8",
  person: "#5ec9a0",
  project: "#d4a837",
  tool: "#e88040",
  concept: "#40c8c8",
  technology: "#60a8d8",
  service: "#d070a0",
  other: "#505068",
};

export const DEFAULT_NODE_COLOR = "#505068";
