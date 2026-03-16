import { useState, useCallback } from "react";
import type { GraphNode } from "./useMemoryGraph";

// All event types from the engine, grouped for the dropdown
const TYPE_GROUPS: { heading: string; types: { key: string; bg: string; text: string; label: string }[] }[] = [
  {
    heading: "Core Knowledge",
    types: [
      { key: "constraint", bg: "bg-[#ff9f43]/20", text: "text-[#ff9f43]", label: "Constraint" },
      { key: "checkpoint", bg: "bg-[#ffc048]/20", text: "text-[#ffc048]", label: "Checkpoint" },
      { key: "reminder", bg: "bg-type-reminder/20", text: "text-type-reminder", label: "Reminder" },
      { key: "decision", bg: "bg-type-decision/20", text: "text-type-decision", label: "Decision" },
      { key: "lesson_learned", bg: "bg-type-lesson/20", text: "text-type-lesson", label: "Lesson" },
      { key: "error_pattern", bg: "bg-type-error/20", text: "text-type-error", label: "Error Pattern" },
      { key: "user_preference", bg: "bg-type-preference/20", text: "text-type-preference", label: "Preference" },
    ],
  },
  {
    heading: "Reflections & Research",
    types: [
      { key: "task_completion", bg: "bg-type-task/20", text: "text-type-task", label: "Task Completion" },
      { key: "reflexion", bg: "bg-[#78dbb8]/20", text: "text-[#78dbb8]", label: "Reflexion" },
      { key: "outcome_evaluation", bg: "bg-[#68c8a8]/20", text: "text-[#68c8a8]", label: "Outcome Eval" },
      { key: "self_reflection", bg: "bg-[#88e0c8]/20", text: "text-[#88e0c8]", label: "Self Reflection" },
      { key: "advisor_insight", bg: "bg-[#a088e8]/20", text: "text-[#a088e8]", label: "Advisor Insight" },
      { key: "advisor_action_outcome", bg: "bg-[#9078d8]/20", text: "text-[#9078d8]", label: "Advisor Outcome" },
      { key: "sota_research", bg: "bg-[#60b8e8]/20", text: "text-[#60b8e8]", label: "SOTA Research" },
      { key: "research_report", bg: "bg-[#50a8d8]/20", text: "text-[#50a8d8]", label: "Research Report" },
      { key: "benchmark_update", bg: "bg-[#70c0e0]/20", text: "text-[#70c0e0]", label: "Benchmark" },
      { key: "sota_scan", bg: "bg-[#80c8e0]/20", text: "text-[#80c8e0]", label: "SOTA Scan" },
      { key: "preference_generated", bg: "bg-[#c098f0]/20", text: "text-[#c098f0]", label: "Pref Generated" },
    ],
  },
  {
    heading: "Coordination",
    types: [
      { key: "session_summary", bg: "bg-type-session/20", text: "text-type-session", label: "Session Summary" },
      { key: "session_respawn", bg: "bg-[#686890]/20", text: "text-[#686890]", label: "Session Respawn" },
      { key: "coordination_snapshot", bg: "bg-[#585878]/20", text: "text-[#585878]", label: "Coord Snapshot" },
      { key: "file_conflict", bg: "bg-[#a06868]/20", text: "text-[#a06868]", label: "File Conflict" },
      { key: "merge_claim", bg: "bg-[#887070]/20", text: "text-[#887070]", label: "Merge Claim" },
      { key: "merge_release", bg: "bg-[#887070]/20", text: "text-[#887070]", label: "Merge Release" },
      { key: "file_claimed", bg: "bg-[#787088]/20", text: "text-[#787088]", label: "File Claimed" },
      { key: "file_released", bg: "bg-[#787088]/20", text: "text-[#787088]", label: "File Released" },
      { key: "branch_claimed", bg: "bg-[#787088]/20", text: "text-[#787088]", label: "Branch Claimed" },
      { key: "branch_released", bg: "bg-[#787088]/20", text: "text-[#787088]", label: "Branch Released" },
      { key: "test", bg: "bg-[#68a088]/20", text: "text-[#68a088]", label: "Test" },
      { key: "code_chunk", bg: "bg-[#606878]/20", text: "text-[#606878]", label: "Code Chunk" },
      { key: "file_summary", bg: "bg-[#585868]/20", text: "text-[#585868]", label: "File Summary" },
    ],
  },
  {
    heading: "Say/Do Tracking",
    types: [
      { key: "public_statement", bg: "bg-[#e8c060]/20", text: "text-[#e8c060]", label: "Statement" },
      { key: "outcome_resolution", bg: "bg-[#c8d860]/20", text: "text-[#c8d860]", label: "Resolution" },
      { key: "contradiction_detected", bg: "bg-[#f07878]/20", text: "text-[#f07878]", label: "Contradiction" },
      { key: "entity_profile_update", bg: "bg-[#d0a868]/20", text: "text-[#d0a868]", label: "Entity Profile" },
    ],
  },
];

// Flat lookup for display
const TYPE_LOOKUP: Record<string, { bg: string; text: string; label: string }> = {};
for (const group of TYPE_GROUPS) {
  for (const t of group.types) {
    TYPE_LOOKUP[t.key] = { bg: t.bg, text: t.text, label: t.label };
  }
}

// Engine memory types (3 real categories, no "declarative")
const MEMORY_TYPE_LABELS: Record<string, string> = {
  procedural: "Procedural",
  episodic: "Episodic",
  semantic: "Semantic",
};

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

export function MemoryDetailPanel({
  node,
  onClose,
  onSave,
}: {
  node: GraphNode;
  onClose: () => void;
  onSave?: (updated: GraphNode) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  // Edit form state
  const [editContent, setEditContent] = useState(node.content);
  const [editPriority, setEditPriority] = useState(node.priority);
  const [editType, setEditType] = useState(node.event_type || "decision");
  const [editTags, setEditTags] = useState<string[]>(node.tags || []);
  const [tagInput, setTagInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const enterEdit = useCallback(() => {
    setEditContent(node.content);
    setEditPriority(node.priority);
    setEditType(node.event_type || "decision");
    setEditTags(node.tags || []);
    setTagInput("");
    setError(null);
    setEditing(true);
  }, [node]);

  const cancelEdit = useCallback(() => {
    setEditing(false);
    setError(null);
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(`/api/admin/memories/${node.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: editContent,
          priority: editPriority,
          event_type: editType,
          tags: editTags,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        setError(data.error || `Error: ${res.status}`);
        return;
      }

      const { memory } = await res.json();
      const updatedNode: GraphNode = {
        ...node,
        content: memory.content,
        priority: memory.priority,
        event_type: memory.event_type,
        tags: memory.tags,
        updated_at: memory.updated_at,
      };

      setEditing(false);
      onSave?.(updatedNode);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [node, editContent, editPriority, editType, editTags, onSave]);

  const addTag = useCallback(() => {
    const tag = tagInput.trim().toLowerCase();
    if (tag && !editTags.includes(tag)) {
      setEditTags([...editTags, tag]);
    }
    setTagInput("");
  }, [tagInput, editTags]);

  const removeTag = useCallback(
    (tag: string) => {
      setEditTags(editTags.filter((t) => t !== tag));
    },
    [editTags],
  );

  const typeInfo = TYPE_LOOKUP[editing ? editType : (node.event_type || "")] || {
    bg: "bg-ink-faint/20",
    text: "text-ink-secondary",
    label: (editing ? editType : node.event_type) || "Memory",
  };

  return (
    <>
      {/* Backdrop */}
      <div className="absolute inset-0 z-10" onClick={onClose} />

      {/* Panel */}
      <div className="absolute right-0 top-0 bottom-0 z-20 w-[340px] border-l border-edge bg-surface/80 backdrop-blur-xl animate-slide-in-right overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Header */}
          <div className="flex items-start justify-between gap-2">
            {editing ? (
              <select
                value={editType}
                onChange={(e) => setEditType(e.target.value)}
                className="px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-surface-elevated border border-edge-subtle text-ink focus:outline-none focus:border-gold/40"
              >
                {TYPE_GROUPS.map((group) => (
                  <optgroup key={group.heading} label={group.heading}>
                    {group.types.map((t) => (
                      <option key={t.key} value={t.key}>
                        {t.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            ) : (
              <div className="flex items-center gap-1.5 flex-wrap">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider ${typeInfo.bg} ${typeInfo.text}`}
                >
                  {typeInfo.label}
                </span>
                {node.category === "system_insight" && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#e63956]/20 text-[#e63956]">
                    System Insight
                  </span>
                )}
                {node.memory_type && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-surface-elevated text-ink-secondary border border-edge-subtle">
                    {MEMORY_TYPE_LABELS[node.memory_type] || node.memory_type}
                  </span>
                )}
              </div>
            )}
            <div className="flex items-center gap-1">
              {!editing && (
                <button
                  onClick={enterEdit}
                  className="text-ink-tertiary hover:text-gold text-[13px] p-1 -m-1"
                  title="Edit memory"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                </button>
              )}
              <button
                onClick={editing ? cancelEdit : onClose}
                className="text-ink-tertiary hover:text-ink text-[13px] p-1 -m-1"
              >
                {editing ? "Cancel" : "Esc"}
              </button>
            </div>
          </div>

          {/* Priority */}
          <div className="flex items-center gap-2 text-[12px] text-ink-tertiary">
            {editing ? (
              <div className="flex items-center gap-1">
                <span className="mr-1">Priority</span>
                {[1, 2, 3, 4, 5].map((p) => (
                  <button
                    key={p}
                    onClick={() => setEditPriority(p)}
                    className={`w-6 h-6 rounded text-[11px] font-medium transition-colors ${
                      editPriority === p
                        ? "bg-gold/20 text-gold border border-gold/30"
                        : "bg-surface-elevated text-ink-secondary border border-edge-subtle hover:border-edge-default"
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            ) : (
              <>
                <span>Priority {node.priority}/5</span>
                <span className="text-ink-faint">|</span>
                <span>{formatDate(node.created_at)}</span>
              </>
            )}
          </div>

          {/* Date (shown separately in edit mode) */}
          {editing && (
            <div className="text-[12px] text-ink-tertiary">
              {formatDate(node.created_at)}
              {node.updated_at && (
                <span className="ml-2 text-ink-faint">(edited {formatDate(node.updated_at)})</span>
              )}
            </div>
          )}

          {/* Content */}
          {editing ? (
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              rows={8}
              className="w-full text-[13px] text-ink leading-relaxed bg-surface-elevated border border-edge-subtle rounded-lg p-3 focus:outline-none focus:border-gold/40 resize-y"
            />
          ) : (
            <div className="text-[13px] text-ink leading-relaxed whitespace-pre-wrap break-words">
              {node.content}
            </div>
          )}

          {/* Superseded warning */}
          {!editing && node.is_superseded && (
            <div className="rounded-lg px-3 py-2 bg-type-error/10 border border-type-error/20">
              <div className="text-[12px] font-semibold text-type-error">Superseded</div>
              <div className="text-[11px] text-type-error/70 mt-0.5">
                This memory has been replaced by a newer version.
              </div>
            </div>
          )}

          {/* Brain Activity */}
          {!editing && (node.access_count > 0 || node.confidence != null || node.feedback_score != null) && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Brain Activity
              </div>
              <div className="space-y-2">
                {node.access_count > 0 && (
                  <div className="flex items-center justify-between text-[12px]">
                    <span className="text-ink-secondary">Access Count</span>
                    <span className="text-ink font-medium tabular-nums">{node.access_count}</span>
                  </div>
                )}
                {node.confidence != null && (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[12px]">
                      <span className="text-ink-secondary">Confidence</span>
                      <span className="text-ink font-medium tabular-nums">{Math.round(node.confidence * 100)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden bg-surface-elevated">
                      <div
                        className="h-full rounded-full transition-all duration-500 bg-gold"
                        style={{ width: `${Math.round(node.confidence * 100)}%` }}
                      />
                    </div>
                  </div>
                )}
                {node.feedback_score != null && (
                  <div className="flex items-center justify-between text-[12px]">
                    <span className="text-ink-secondary">Feedback</span>
                    <span className={`font-medium tabular-nums ${node.feedback_score >= 0 ? "text-type-lesson" : "text-type-error"}`}>
                      {node.feedback_score > 0 ? "+" : ""}{node.feedback_score}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Observation */}
          {!editing && node.observation && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
                Observation
              </div>
              <div className="text-[12px] text-ink-secondary italic leading-relaxed">
                {node.observation}
              </div>
            </div>
          )}

          {/* Extracted Facts */}
          {!editing && node.facts && node.facts.length > 0 && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
                Extracted Facts
              </div>
              <ul className="space-y-1">
                {node.facts.map((fact, i) => (
                  <li key={i} className="flex items-start gap-2 text-[12px] text-ink-secondary">
                    <span className="w-1 h-1 rounded-full bg-gold mt-1.5 flex-shrink-0" />
                    {fact}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Entity */}
          {!editing && node.entity_id && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
                Entity
              </div>
              <div className="text-[12px] text-ink-secondary font-mono">
                {node.entity_id}
              </div>
            </div>
          )}

          {/* Project (read-only in both modes) */}
          {node.project && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
                Project
              </div>
              <div className="text-[13px] text-gold font-medium">
                {node.project}
              </div>
            </div>
          )}

          {/* Tags */}
          <div className="pt-2 border-t border-edge-subtle">
            <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-2">
              Tags
            </div>
            {editing ? (
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1.5">
                  {editTags.map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-surface-elevated text-ink-secondary border border-edge-subtle"
                    >
                      {tag}
                      <button
                        onClick={() => removeTag(tag)}
                        className="text-ink-faint hover:text-type-error ml-0.5"
                      >
                        x
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-1">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        addTag();
                      }
                    }}
                    placeholder="Add tag..."
                    className="flex-1 px-2 py-1 rounded text-[11px] bg-surface-elevated border border-edge-subtle text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40"
                  />
                  <button
                    onClick={addTag}
                    className="px-2 py-1 rounded text-[11px] bg-surface-elevated border border-edge-subtle text-ink-secondary hover:text-gold hover:border-gold/30"
                  >
                    +
                  </button>
                </div>
              </div>
            ) : (
              node.tags && node.tags.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {node.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 rounded text-[11px] bg-surface-elevated text-ink-secondary border border-edge-subtle"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-[11px] text-ink-faint">No tags</span>
              )
            )}
          </div>

          {/* Session (read-only) */}
          {node.session_id && !editing && (
            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-[11px] text-ink-tertiary uppercase tracking-wider mb-1">
                Session
              </div>
              <div className="text-[12px] text-ink-secondary font-mono">
                {node.session_id.slice(0, 12)}...
              </div>
            </div>
          )}

          {/* Updated at indicator (read-only mode) */}
          {!editing && node.updated_at && (
            <div className="text-[11px] text-ink-faint">
              Edited {formatDate(node.updated_at)}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="text-[12px] text-type-error bg-type-error/10 rounded px-3 py-2">
              {error}
            </div>
          )}

          {/* Save / Cancel buttons */}
          {editing && (
            <div className="flex gap-2 pt-2">
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex-1 px-3 py-2 rounded-lg text-[12px] font-medium bg-gold/20 text-gold border border-gold/30 hover:bg-gold/30 disabled:opacity-50 transition-colors"
              >
                {saving ? "Saving..." : "Save"}
              </button>
              <button
                onClick={cancelEdit}
                disabled={saving}
                className="px-3 py-2 rounded-lg text-[12px] font-medium bg-surface-elevated text-ink-secondary border border-edge-subtle hover:border-edge-default disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
