import type { ResearchPlan } from "./types";
import { PLAN_STATUS_STYLES, timeAgo } from "./helpers";
import SubsystemPill from "./SubsystemPill";

export default function PlanDetail({
  plan,
  onEdit,
  onDelete,
  deleting,
}: {
  plan: ResearchPlan | null;
  onEdit: () => void;
  onDelete: (id: string) => void;
  deleting: boolean;
}) {
  if (!plan) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-16 text-center">
        <svg
          className="w-12 h-12 text-ink-faint/30 mb-4"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1}
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
          />
        </svg>
        <p className="text-[16px] text-ink-faint">
          Select an item to view details
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="mb-5">
        <h2 className="text-[24px] font-semibold text-ink leading-snug">
          {plan.title}
        </h2>
        <div className="flex items-center gap-2.5 mt-3 flex-wrap">
          <span
            className={`px-2.5 py-1 rounded-md text-[13px] font-medium border ${PLAN_STATUS_STYLES[plan.status]}`}
          >
            {plan.status}
          </span>
          {plan.subsystem && <SubsystemPill subsystem={plan.subsystem} />}
          <span className="text-[14px] text-ink-faint">
            {timeAgo(plan.created_at)}
          </span>
        </div>
      </div>

      {/* Content */}
      <pre className="text-[16px] text-ink-secondary whitespace-pre-wrap font-sans leading-relaxed mb-5 flex-1">
        {plan.content}
      </pre>

      {/* Actions */}
      <div className="flex items-center gap-2.5 pt-4 border-t border-edge">
        <button
          onClick={onEdit}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/25 hover:bg-gold/20 transition-colors cursor-pointer"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10"
            />
          </svg>
          Edit
        </button>
        <button
          onClick={() => onDelete(plan.id)}
          disabled={deleting}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-[14px] font-medium text-type-error/70 border border-type-error/15 hover:bg-type-error/5 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {deleting ? "Deleting..." : "Delete"}
        </button>
      </div>
    </div>
  );
}
