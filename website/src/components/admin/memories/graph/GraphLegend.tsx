import { useState } from "react";

// Grouped by engine weight tier for compact legend display
const EVENT_TYPE_GROUPS: { heading: string; items: { color: string; label: string }[] }[] = [
  {
    heading: "Core Knowledge",
    items: [
      { color: "#ff9f43", label: "Constraint" },
      { color: "#ffc048", label: "Checkpoint" },
      { color: "#e8a040", label: "Reminder" },
      { color: "#6b9fff", label: "Decision" },
      { color: "#5ec9a0", label: "Lesson" },
      { color: "#f06060", label: "Error Pattern" },
      { color: "#b088e8", label: "Preference" },
    ],
  },
  {
    heading: "Reflections & Research",
    items: [
      { color: "#40c8c8", label: "Task Completion" },
      { color: "#78dbb8", label: "Reflexion" },
      { color: "#a088e8", label: "Advisor Insight" },
      { color: "#d4a843", label: "System Insight" },
      { color: "#60b8e8", label: "Research" },
    ],
  },
  {
    heading: "Coordination",
    items: [
      { color: "#7878a0", label: "Session" },
      { color: "#887070", label: "Merge/File/Branch" },
      { color: "#585878", label: "Coordination" },
    ],
  },
  {
    heading: "Say/Do Tracking",
    items: [
      { color: "#e8c060", label: "Statement" },
      { color: "#c8d860", label: "Resolution" },
      { color: "#f07878", label: "Contradiction" },
    ],
  },
];

const LINK_LEGEND: { color: string; label: string; synth?: boolean }[] = [
  { color: "rgba(240, 96, 96, 0.6)", label: "Contradicts" },
  { color: "rgba(240, 96, 96, 0.3)", label: "Supersedes" },
  { color: "rgba(176, 136, 232, 0.5)", label: "Evolution" },
  { color: "rgba(94, 201, 160, 0.35)", label: "Contains fact" },
  { color: "rgba(212, 168, 67, 0.5)", label: "Same entity" },
  { color: "rgba(140, 120, 180, 0.15)", label: "Temporal cluster" },
  { color: "rgba(160, 160, 180, 0.15)", label: "Related" },
  { color: "rgba(212, 168, 67, 0.15)", label: "Session (inferred)", synth: true },
  { color: "rgba(107, 159, 255, 0.10)", label: "Tag (inferred)", synth: true },
];

export function GraphLegend() {
  const [open, setOpen] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("omega-graph-legend") !== "0";
    }
    return true;
  });

  const toggle = () => {
    const next = !open;
    setOpen(next);
    localStorage.setItem("omega-graph-legend", next ? "1" : "0");
  };

  return (
    <div className="absolute bottom-4 left-4 z-10 max-w-[220px] max-h-[calc(100vh-120px)] overflow-y-auto">
      <div className="bg-surface/80 backdrop-blur-xl border border-edge rounded-lg overflow-hidden">
        {/* Toggle header */}
        <button
          onClick={toggle}
          className="flex items-center justify-between w-full px-3 py-2 text-[11px] font-semibold text-ink-secondary uppercase tracking-wider hover:text-ink transition-colors"
        >
          Legend
          <svg
            className={`w-3.5 h-3.5 text-ink-tertiary transition-transform ${open ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Collapsible content */}
        <div
          style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
          className="grid transition-[grid-template-rows] duration-200"
        >
          <div className="overflow-hidden">
            <div className="px-3 pb-3 space-y-3">
              {/* Node colors = event_type (grouped by tier) */}
              {EVENT_TYPE_GROUPS.map((group) => (
                <div key={group.heading}>
                  <div className="text-[10px] text-ink-faint uppercase tracking-wider mb-1.5">
                    {group.heading}
                  </div>
                  <div className="space-y-1">
                    {group.items.map(({ color, label }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-[11px] text-ink-secondary">{label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {/* Size = priority */}
              <div>
                <div className="text-[10px] text-ink-faint uppercase tracking-wider mb-1.5">
                  Size = Priority
                </div>
                <div className="flex items-center gap-1.5">
                  {[1, 3, 5].map((p) => (
                    <div key={p} className="flex items-center gap-1">
                      <span
                        className="rounded-full bg-ink-tertiary"
                        style={{
                          width: `${4 + p * 2}px`,
                          height: `${4 + p * 2}px`,
                        }}
                      />
                      <span className="text-[10px] text-ink-faint">{p}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Special markers */}
              <div>
                <div className="text-[10px] text-ink-faint uppercase tracking-wider mb-1.5">
                  Markers
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: "radial-gradient(circle, rgba(212,168,67,0.3), transparent)" }} />
                    <span className="text-[11px] text-ink-secondary">Glow = High access</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full flex-shrink-0 border-2 border-type-error" />
                    <span className="text-[11px] text-ink-secondary">Red ring = Superseded</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 flex-shrink-0 rotate-45 border border-type-task" />
                    <span className="text-[11px] text-ink-secondary">Octahedron = Procedural</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-2 flex-shrink-0 rounded-full border border-type-preference" />
                    <span className="text-[11px] text-ink-secondary">Torus = Episodic</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 flex-shrink-0 rounded-sm border border-type-decision" />
                    <span className="text-[11px] text-ink-secondary">Icosahedron = Semantic</span>
                  </div>
                </div>
              </div>

              {/* Link types */}
              <div>
                <div className="text-[10px] text-ink-faint uppercase tracking-wider mb-1.5">
                  Links
                </div>
                <div className="space-y-1">
                  {LINK_LEGEND.map(({ color, label, synth }) => (
                    <div key={label} className="flex items-center gap-2">
                      <span
                        className="w-4 h-0.5 flex-shrink-0 rounded-full"
                        style={{
                          backgroundColor: color,
                          ...(synth ? { backgroundImage: "repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(8,9,15,0.8) 2px, rgba(8,9,15,0.8) 4px)" } : {}),
                        }}
                      />
                      <span className={`text-[11px] ${synth ? "text-ink-faint italic" : "text-ink-secondary"}`}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
