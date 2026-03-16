import { useState } from "react";
import type { ResearchPlan } from "./types";
import { SUBSYSTEM_COLORS, PLAN_STATUS_STYLES } from "./helpers";

const SUBSYSTEMS = Object.keys(SUBSYSTEM_COLORS);

export default function PlanEditor({
  plan,
  findingContext,
  onSave,
  onCancel,
  saving,
}: {
  plan: ResearchPlan | null;
  findingContext: string | null;
  onSave: (data: {
    id?: string;
    title: string;
    content: string;
    status: "draft" | "ready" | "implemented";
    subsystem: string | null;
    finding_id: string | null;
  }) => void;
  onCancel: () => void;
  saving: boolean;
}) {
  const [title, setTitle] = useState(plan?.title ?? "");
  const [content, setContent] = useState(
    plan?.content ??
      (findingContext
        ? `> **Research context:**\n> ${findingContext.slice(0, 500).replace(/\n/g, "\n> ")}\n\n## Plan\n\n`
        : "## Plan\n\n")
  );
  const [status, setStatus] = useState<"draft" | "ready" | "implemented">(
    plan?.status ?? "draft"
  );
  const [subsystem, setSubsystem] = useState<string | null>(
    plan?.subsystem ?? null
  );
  const [preview, setPreview] = useState(false);

  const handleSubmit = () => {
    if (!title.trim() || !content.trim()) return;
    onSave({
      id: plan?.id,
      title: title.trim(),
      content: content.trim(),
      status,
      subsystem,
      finding_id: plan?.finding_id ?? null,
    });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-[20px] font-semibold text-ink">
          {plan ? "Edit Plan" : "New Plan"}
        </h3>
        <button
          onClick={() => setPreview(!preview)}
          className={`px-3 py-1.5 rounded-lg text-[14px] font-medium border transition-colors cursor-pointer ${
            preview
              ? "bg-gold/10 text-gold border-gold/25"
              : "bg-surface border-edge text-ink-tertiary hover:text-ink-secondary"
          }`}
        >
          {preview ? "Edit" : "Preview"}
        </button>
      </div>

      {/* Title */}
      <label className="text-[14px] text-ink-secondary font-medium mb-1.5">
        Title
      </label>
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Plan title..."
        className="w-full px-4 py-2.5 mb-4 rounded-lg border border-edge bg-surface-elevated text-[16px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40"
      />

      {/* Status + Subsystem */}
      <div className="flex items-center gap-3 mb-4">
        <div>
          <label className="text-[14px] text-ink-secondary font-medium mb-1.5 block">
            Status
          </label>
          <select
            value={status}
            onChange={(e) =>
              setStatus(e.target.value as "draft" | "ready" | "implemented")
            }
            className="px-3 py-2 rounded-lg border border-edge bg-surface-elevated text-[14px] text-ink-secondary focus:outline-none focus:border-gold/40 cursor-pointer"
          >
            <option value="draft">Draft</option>
            <option value="ready">Ready</option>
            <option value="implemented">Implemented</option>
          </select>
        </div>
        <div>
          <label className="text-[14px] text-ink-secondary font-medium mb-1.5 block">
            Subsystem
          </label>
          <select
            value={subsystem ?? ""}
            onChange={(e) => setSubsystem(e.target.value || null)}
            className="px-3 py-2 rounded-lg border border-edge bg-surface-elevated text-[14px] text-ink-secondary focus:outline-none focus:border-gold/40 cursor-pointer"
          >
            <option value="">No subsystem</option>
            {SUBSYSTEMS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-end h-full pt-6">
          <span
            className={`px-2.5 py-1 rounded-md text-[13px] font-medium border ${PLAN_STATUS_STYLES[status]}`}
          >
            {status}
          </span>
        </div>
      </div>

      {/* Content editor / preview */}
      <label className="text-[14px] text-ink-secondary font-medium mb-1.5">
        Content
      </label>
      <div className="flex-1 min-h-0 mb-4">
        {preview ? (
          <div className="h-full overflow-y-auto rounded-lg border border-edge bg-surface-elevated p-5">
            <pre className="text-[16px] text-ink-secondary whitespace-pre-wrap font-sans leading-relaxed">
              {content}
            </pre>
          </div>
        ) : (
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Write your implementation plan in markdown..."
            className="w-full h-full px-4 py-3 rounded-lg border border-edge bg-surface-elevated text-[15px] text-ink font-mono leading-relaxed placeholder:text-ink-faint focus:outline-none focus:border-gold/40 resize-none"
          />
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2.5 pt-3 border-t border-edge">
        <button
          onClick={handleSubmit}
          disabled={saving || !title.trim() || !content.trim()}
          className="px-5 py-2.5 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/25 hover:bg-gold/20 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? "Saving..." : plan ? "Update Plan" : "Save Plan"}
        </button>
        <button
          onClick={onCancel}
          disabled={saving}
          className="px-5 py-2.5 rounded-lg text-[14px] font-medium text-ink-tertiary border border-edge hover:bg-surface-hover transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
