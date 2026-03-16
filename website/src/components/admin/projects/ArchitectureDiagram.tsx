import { useEffect, useRef, useState, useCallback } from "react";

// Module-level flag: mermaid.initialize() is called exactly once
let mermaidInitialized = false;

interface ArchitectureDiagramProps {
  mermaidSyntax: string;
  onNodeClick?: (modulePath: string) => void;
}

export default function ArchitectureDiagram({
  mermaidSyntax,
  onNodeClick,
}: ArchitectureDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Load and render Mermaid
  useEffect(() => {
    if (!mermaidSyntax || !containerRef.current) return;

    let cancelled = false;

    async function render() {
      try {
        const mermaid = (await import("mermaid")).default;

        if (!mermaidInitialized) {
          mermaid.initialize({
            startOnLoad: false,
            theme: "base",
            look: "classic",
            themeVariables: {
              primaryColor: "#134e4a",
              primaryBorderColor: "#14b8a6",
              primaryTextColor: "#f0fdfa",
              secondaryColor: "#1e293b",
              secondaryBorderColor: "#059669",
              secondaryTextColor: "#f1f5f9",
              tertiaryColor: "#27201a",
              tertiaryBorderColor: "#d97706",
              tertiaryTextColor: "#fef3c7",
              lineColor: "#64748b",
              fontSize: "14px",
              fontFamily: "ui-sans-serif, system-ui, sans-serif",
              noteBkgColor: "#1e293b",
              noteTextColor: "#f1f5f9",
              noteBorderColor: "#fbbf24",
            },
          });
          mermaidInitialized = true;
        }

        const id = `mermaid-${Date.now()}`;
        const { svg } = await mermaid.render(id, mermaidSyntax);

        if (cancelled || !containerRef.current) return;

        // Mermaid render() returns sanitized SVG generated from our own
        // server-built syntax (not user input). This is the standard
        // Mermaid React integration pattern per mermaid.js.org.
        containerRef.current.replaceChildren();
        const template = document.createElement("template");
        template.innerHTML = svg; // eslint-disable-line -- Mermaid-sanitized SVG, safe
        if (template.content.firstChild) {
          containerRef.current.appendChild(template.content.firstChild);
        }
        setLoading(false);

        // Attach click handlers to nodes
        if (onNodeClick) {
          const nodes = containerRef.current.querySelectorAll(".node");
          nodes.forEach((node) => {
            const nodeEl = node as HTMLElement;
            nodeEl.style.cursor = "pointer";
            nodeEl.addEventListener("click", () => {
              const nodeIdAttr = nodeEl.id?.replace(/^flowchart-/, "").replace(/-\d+$/, "");
              if (nodeIdAttr) onNodeClick(nodeIdAttr.replace(/_/g, "/"));
            });
          });
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to render diagram");
          setLoading(false);
        }
      }
    }

    render();
    return () => { cancelled = true; };
  }, [mermaidSyntax, onNodeClick]);

  // Zoom with Ctrl+scroll
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setScale((s) => Math.min(3, Math.max(0.3, s + delta)));
    },
    [],
  );

  // Pan with drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setDragging(true);
    setDragStart({ x: e.clientX - translate.x, y: e.clientY - translate.y });
  }, [translate]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return;
      setTranslate({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    },
    [dragging, dragStart],
  );

  const handleMouseUp = useCallback(() => setDragging(false), []);

  const resetView = useCallback(() => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-red-400/20 bg-red-400/[0.04] p-4">
        <p className="text-xs text-red-400/70">Diagram error: {error}</p>
        <pre className="mt-2 text-[10px] text-white/30 overflow-auto max-h-32 font-mono">
          {mermaidSyntax}
        </pre>
      </div>
    );
  }

  return (
    <div className="relative rounded-xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
      {/* Zoom controls */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-1">
        <button
          onClick={() => setScale((s) => Math.min(3, s + 0.2))}
          className="h-7 w-7 rounded-md bg-white/[0.06] text-white/40 hover:text-white/70 hover:bg-white/[0.1] flex items-center justify-center text-sm transition-colors"
          title="Zoom in"
        >
          +
        </button>
        <button
          onClick={() => setScale((s) => Math.max(0.3, s - 0.2))}
          className="h-7 w-7 rounded-md bg-white/[0.06] text-white/40 hover:text-white/70 hover:bg-white/[0.1] flex items-center justify-center text-sm transition-colors"
          title="Zoom out"
        >
          -
        </button>
        <button
          onClick={resetView}
          className="h-7 rounded-md bg-white/[0.06] text-white/40 hover:text-white/70 hover:bg-white/[0.1] flex items-center justify-center text-[10px] px-2 transition-colors"
          title="Reset view"
        >
          Reset
        </button>
      </div>

      {/* Diagram container */}
      <div
        className="min-h-[300px] flex items-center justify-center p-6"
        style={{ cursor: dragging ? "grabbing" : "grab" }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {loading && (
          <div className="text-xs text-white/20 animate-pulse">Loading diagram...</div>
        )}
        <div
          ref={containerRef}
          style={{
            transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
            transformOrigin: "center center",
            transition: dragging ? "none" : "transform 0.15s ease-out",
          }}
          className="[&_.nodeLabel]:!text-white/80 [&_.edgeLabel]:!text-white/50 [&_.edgeLabel_rect]:!fill-[var(--color-canvas)]"
        />
      </div>
    </div>
  );
}
