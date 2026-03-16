import { PLUGIN_COLORS } from "./skillsManifest";
import type { SkillNode, SkillsGraphData } from "./useSkillsGraph";

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function resolveId(ref: unknown): string {
  return typeof ref === "string" ? ref : (ref as any).id;
}

export function SkillDetailPanel({
  node,
  data,
  onClose,
}: {
  node: SkillNode;
  data: SkillsGraphData;
  onClose: () => void;
}) {
  const pluginColor = PLUGIN_COLORS[node.plugin] || "#505068";
  const nodeMap = new Map(data.nodes.map((n) => [n.id, n]));

  // Invocation links FROM this node (this node invokes others)
  const invokes = data.links
    .filter((l) => l.type === "invocation" && resolveId(l.source) === node.id)
    .map((l) => nodeMap.get(resolveId(l.target)))
    .filter(Boolean) as SkillNode[];

  // Invocation links TO this node (others invoke this node)
  const invokedBy = data.links
    .filter((l) => l.type === "invocation" && resolveId(l.target) === node.id)
    .map((l) => nodeMap.get(resolveId(l.source)))
    .filter(Boolean) as SkillNode[];

  // Co-occurrence links
  const coOccurrence = data.links
    .filter((l) => {
      if (l.type !== "co-occurrence") return false;
      const src = resolveId(l.source);
      const tgt = resolveId(l.target);
      return src === node.id || tgt === node.id;
    })
    .map((l) => {
      const src = resolveId(l.source);
      const otherId = src === node.id ? resolveId(l.target) : src;
      return nodeMap.get(otherId);
    })
    .filter(Boolean) as SkillNode[];

  return (
    <>
      {/* Backdrop */}
      <div className="absolute inset-0 z-10" onClick={onClose} />

      {/* Panel */}
      <div className="absolute right-0 top-0 bottom-0 z-20 w-[340px] border-l border-edge bg-surface/80 backdrop-blur-xl animate-slide-in-right overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Header */}
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <span
                className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider shrink-0"
                style={{
                  backgroundColor: `${pluginColor}20`,
                  color: pluginColor,
                }}
              >
                {node.plugin}
              </span>
              <span className="text-[14px] font-semibold text-ink truncate">
                {node.name}
              </span>
            </div>
            <button
              onClick={onClose}
              className="text-ink-tertiary hover:text-ink text-[13px] p-1 -m-1 shrink-0"
            >
              Esc
            </button>
          </div>

          {/* Type badge + usage count */}
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider ${
                node.type === "rigid"
                  ? "bg-type-error/10 text-type-error"
                  : "bg-type-lesson/10 text-type-lesson"
              }`}
            >
              {node.type}
            </span>
            {node.usageCount > 0 && (
              <span className="text-[12px] text-ink-tertiary">
                {node.usageCount} uses
              </span>
            )}
          </div>

          {/* Description */}
          <div className="text-[13px] text-ink leading-relaxed">
            {node.description}
          </div>

          {/* Last used */}
          {node.lastUsed && (
            <div className="text-[12px] text-ink-tertiary">
              Last used {formatDate(node.lastUsed)}
            </div>
          )}

          {/* Invokes section */}
          {invokes.length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Invokes
              </div>
              <div className="space-y-1.5">
                {invokes.map((s) => (
                  <div key={s.id} className="flex items-center gap-2 text-[12px]">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: PLUGIN_COLORS[s.plugin] || "#505068" }}
                    />
                    <span className="text-ink-secondary">{s.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Invoked by section */}
          {invokedBy.length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Invoked by
              </div>
              <div className="space-y-1.5">
                {invokedBy.map((s) => (
                  <div key={s.id} className="flex items-center gap-2 text-[12px]">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: PLUGIN_COLORS[s.plugin] || "#505068" }}
                    />
                    <span className="text-ink-secondary">{s.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Often used with section */}
          {coOccurrence.length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Often used with
              </div>
              <div className="space-y-1.5">
                {coOccurrence.map((s) => (
                  <div key={s.id} className="flex items-center gap-2 text-[12px]">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: PLUGIN_COLORS[s.plugin] || "#505068" }}
                    />
                    <span className="text-ink-secondary">{s.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Skill ID */}
          <div className="pt-2 border-t border-edge-subtle">
            <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
              Skill ID
            </div>
            <div className="text-[12px] text-ink-secondary font-mono">
              {node.id}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
