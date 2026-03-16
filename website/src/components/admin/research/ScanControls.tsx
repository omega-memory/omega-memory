import type { ScanReport } from "./types";

export default function ScanControls({
  scanning,
  onScan,
  scanReport,
  onDismissReport,
  scanError,
  onDismissScanError,
  ingestUrl,
  onIngestUrlChange,
  onIngest,
  ingesting,
  ingestError,
  loading,
  onRefresh,
}: {
  scanning: boolean;
  onScan: () => void;
  scanReport: ScanReport | null;
  onDismissReport: () => void;
  scanError: string | null;
  onDismissScanError: () => void;
  scanHistory: unknown[];
  ingestUrl: string;
  onIngestUrlChange: (url: string) => void;
  onIngest: () => void;
  ingesting: boolean;
  ingestError: string | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="space-y-3">
      {/* Action bar */}
      <div className="flex items-center gap-2">
        <input
          type="url"
          placeholder="Paste a URL to research..."
          value={ingestUrl}
          onChange={(e) => onIngestUrlChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onIngest();
          }}
          className="flex-1 px-4 py-2.5 rounded-lg text-[15px] bg-surface border border-edge text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40 transition-colors"
        />
        <button
          onClick={onIngest}
          disabled={ingesting || !ingestUrl.trim()}
          className="px-4 py-2.5 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/25 hover:bg-gold/20 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          {ingesting ? "..." : "Ingest"}
        </button>
        <div className="w-px h-6 bg-edge" />
        <button
          onClick={onScan}
          disabled={scanning}
          className="px-4 py-2.5 rounded-lg text-[14px] font-medium text-ink-secondary border border-edge hover:bg-surface-hover transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          {scanning ? "Scanning..." : "Scan"}
        </button>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="p-2.5 rounded-lg text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors cursor-pointer disabled:opacity-50 shrink-0"
          title="Refresh"
        >
          <svg
            className={`w-5 h-5 ${loading ? "animate-spin" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182"
            />
          </svg>
        </button>
      </div>

      {/* Banners */}
      {ingestError && (
        <div className="rounded-lg border border-type-error/20 bg-type-error/5 px-4 py-3 text-[14px] text-type-error">
          {ingestError}
        </div>
      )}
      {scanError && (
        <div className="rounded-lg border border-type-error/20 bg-type-error/5 px-4 py-3 text-[14px] text-type-error flex items-center justify-between">
          <span>{scanError}</span>
          <button
            onClick={onDismissScanError}
            className="text-ink-faint hover:text-ink-secondary ml-3 cursor-pointer"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      )}
      {scanReport && (
        <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-[14px] text-emerald-400 flex items-center justify-between">
          <span>
            Scan complete: {scanReport.kept} kept, {scanReport.discarded}{" "}
            discarded ({(scanReport.duration_ms / 1000).toFixed(1)}s)
          </span>
          <button
            onClick={onDismissReport}
            className="text-ink-faint hover:text-ink-secondary ml-3 cursor-pointer"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}
