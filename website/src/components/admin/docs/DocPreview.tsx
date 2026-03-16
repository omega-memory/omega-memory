import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { KBItem } from "./types";

const mdComponents = {
  h1: ({ children }: any) => (
    <h1 className="font-display font-semibold text-ink text-2xl leading-tight mb-4 mt-2">
      {children}
    </h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="font-display font-semibold text-ink text-xl leading-tight mt-8 mb-3 pb-2 border-b border-edge-subtle">
      {children}
    </h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="font-display font-semibold text-ink text-lg leading-tight mt-6 mb-2">
      {children}
    </h3>
  ),
  h4: ({ children }: any) => (
    <h4 className="font-display font-semibold text-ink text-base mt-4 mb-1.5">{children}</h4>
  ),
  p: ({ children }: any) => (
    <p className="text-ink-secondary text-[14px] leading-relaxed mb-3">{children}</p>
  ),
  a: ({ href, children }: any) => (
    <a
      href={href}
      className="text-gold hover:text-gold-dim underline decoration-gold/30 underline-offset-2 transition-colors"
      target={href?.startsWith("http") ? "_blank" : undefined}
      rel={href?.startsWith("http") ? "noopener noreferrer" : undefined}
    >
      {children}
    </a>
  ),
  ul: ({ children }: any) => (
    <ul className="text-ink-secondary text-[14px] leading-relaxed mb-3 pl-5 space-y-0.5 list-disc marker:text-ink-faint">
      {children}
    </ul>
  ),
  ol: ({ children }: any) => (
    <ol className="text-ink-secondary text-[14px] leading-relaxed mb-3 pl-5 space-y-0.5 list-decimal marker:text-ink-faint">
      {children}
    </ol>
  ),
  li: ({ children }: any) => <li className="pl-0.5">{children}</li>,
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-2 border-gold/40 pl-3 my-3 text-ink-tertiary italic text-[14px]">
      {children}
    </blockquote>
  ),
  code: ({ className, children }: any) => {
    const isBlock = className?.includes("language-");
    if (isBlock) {
      return <code className={`${className} text-[13px]`}>{children}</code>;
    }
    return (
      <code className="font-mono text-[13px] text-gold bg-surface-elevated px-1 py-0.5 rounded">
        {children}
      </code>
    );
  },
  pre: ({ children }: any) => (
    <pre className="overflow-x-auto mb-4 p-3 bg-surface-elevated border border-edge-subtle rounded-lg text-[13px] leading-relaxed">
      {children}
    </pre>
  ),
  table: ({ children }: any) => (
    <div className="overflow-x-auto mb-4">
      <table className="w-full text-[13px] border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => <thead className="border-b border-edge">{children}</thead>,
  th: ({ children }: any) => (
    <th className="py-1.5 pr-3 text-left font-mono font-medium text-ink-tertiary text-[12px]">
      {children}
    </th>
  ),
  td: ({ children }: any) => (
    <td className="py-1.5 pr-3 text-ink-secondary text-[13px] border-b border-edge-subtle align-top">
      {children}
    </td>
  ),
  hr: () => <div className="border-t border-edge-subtle my-6" />,
  strong: ({ children }: any) => <strong className="font-semibold text-ink">{children}</strong>,
  img: ({ src, alt }: any) => (
    <span className="block my-4">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt || ""}
        className="rounded-lg border border-edge-subtle max-w-full"
      />
    </span>
  ),
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(ms / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

const STATUS_STYLE: Record<string, { dot: string; label: string }> = {
  pending: { dot: "bg-amber-400", label: "Pending sync" },
  processing: { dot: "bg-type-decision", label: "Processing" },
  synced: { dot: "bg-type-lesson", label: "Synced" },
  error: { dot: "bg-type-error", label: "Error" },
};

interface DocPreviewProps {
  doc: KBItem;
  breadcrumb: { id: string; name: string; type: string }[];
  onBack: () => void;
  onDelete: () => void;
}

export default function DocPreview({ doc, breadcrumb, onBack, onDelete }: DocPreviewProps) {
  const [content, setContent] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);

  const isPdf = doc.source_type === "pdf";
  const isTextReadable = !isPdf && doc.status === "synced";
  const style = STATUS_STYLE[doc.status] ?? STATUS_STYLE.pending;

  useEffect(() => {
    if (!isTextReadable) return;
    setContentLoading(true);
    setContent(null);

    (async () => {
      try {
        const res = await fetch(`/api/kb/files?path=${encodeURIComponent(doc.storage_path)}`);
        if (!res.ok) throw new Error("Download failed");
        const { content: text } = await res.json();
        setContent(text || null);
      } catch {
        setContent("[error] Failed to load file content");
      } finally {
        setContentLoading(false);
      }
    })();
  }, [doc.id, doc.storage_path, isTextReadable]);

  const contentIsError = content?.startsWith("[error]") ?? false;

  return (
    <div className="doc-detail-enter p-5">
      <button
        onClick={onBack}
        className="md:hidden flex items-center gap-1.5 text-[13px] text-ink-tertiary hover:text-ink mb-4 touch-manipulation min-h-[44px]"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
        </svg>
        Back
      </button>

      {breadcrumb.length > 0 && (
        <div className="flex items-center gap-1 text-[12px] text-ink-faint mb-3 flex-wrap">
          {breadcrumb.map((crumb, i) => (
            <span key={crumb.id} className="flex items-center gap-1">
              {i > 0 && <span className="text-ink-faint/50">/</span>}
              <span className={i === breadcrumb.length - 1 ? "text-ink-secondary" : ""}>
                {crumb.name}
              </span>
            </span>
          ))}
        </div>
      )}

      <div className="border-b border-edge-subtle pb-4 mb-4">
        <h2 className="text-[20px] font-semibold text-ink leading-snug tracking-tight">
          {doc.title || doc.file_name}
        </h2>
        <div className="flex items-center gap-2 mt-1.5 text-[12px] text-ink-tertiary flex-wrap">
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${style.dot}`} />
            {style.label}
          </span>
          <span>&middot;</span>
          <span className="font-mono">{timeAgo(doc.created_at)}</span>
          <span>&middot;</span>
          <span>{formatSize(doc.file_size_bytes)}</span>
          <span>&middot;</span>
          <span className="uppercase font-mono text-[11px]">{doc.source_type}</span>
          <button
            onClick={onDelete}
            className="ml-auto p-1 -m-1 rounded-md text-ink-faint hover:text-type-error hover:bg-type-error/10 transition-colors"
            aria-label="Delete"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
            </svg>
          </button>
        </div>
      </div>

      {doc.status === "error" && doc.error_message && (
        <div className="p-2.5 mb-4 bg-type-error/10 border border-type-error/20 rounded-lg text-[12px] text-type-error">
          {doc.error_message}
        </div>
      )}

      {isPdf && (
        <div className="rounded-lg border border-edge-subtle bg-surface-elevated p-6 text-center">
          <svg className="w-8 h-8 mx-auto mb-2 text-ink-faint opacity-50" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
          </svg>
          <p className="text-[12px] text-ink-faint">PDF preview not available</p>
          <p className="text-[11px] text-ink-faint/60 mt-1">{formatSize(doc.file_size_bytes)}</p>
        </div>
      )}

      {isTextReadable && (
        <div className="rounded-lg border border-edge-subtle bg-surface-elevated overflow-hidden">
          {contentLoading ? (
            <div className="flex items-center justify-center py-8 gap-2">
              <div className="w-4 h-4 border-2 border-ink-faint border-t-ink-tertiary rounded-full animate-spin" />
              <span className="text-[12px] text-ink-tertiary">Loading content...</span>
            </div>
          ) : contentIsError ? (
            <div className="p-4 text-center text-[12px] text-type-error/80">
              Failed to load file content.
            </div>
          ) : content ? (
            <div className="p-5 max-h-[70vh] overflow-y-auto">
              {doc.source_type === "markdown" ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[]}
                  components={mdComponents}
                >
                  {content}
                </ReactMarkdown>
              ) : (
                <pre className="text-[13px] font-mono text-ink-secondary leading-relaxed whitespace-pre-wrap break-words">
                  {content}
                </pre>
              )}
            </div>
          ) : (
            <div className="p-4 text-center text-[12px] text-ink-faint">
              Content not available
            </div>
          )}
        </div>
      )}

      {doc.status === "pending" && (
        <div className="rounded-lg border border-edge-subtle bg-surface p-4 text-center text-[13px] text-ink-tertiary">
          This document is pending sync. Content will be available once processing completes.
        </div>
      )}
    </div>
  );
}
