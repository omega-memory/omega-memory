import { useState, useEffect, useCallback } from "react";
import type {
  ResearchData,
  ResearchFinding,
  ResearchPlan,
  ScanReport,
  CurationStatus,
} from "./research/types";
import ScanControls from "./research/ScanControls";
import FindingsList from "./research/FindingsList";
import FindingDetail from "./research/FindingDetail";
import PlanDetail from "./research/PlanDetail";
import PlanEditor from "./research/PlanEditor";

type View = "detail" | "editor";

export default function Research() {
  const [data, setData] = useState<ResearchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<"finding" | "plan">("finding");
  const [view, setView] = useState<View>("detail");

  const [ingestUrl, setIngestUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [scanning, setScanning] = useState(false);
  const [scanReport, setScanReport] = useState<ScanReport | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // ─── Data fetching ───────────────────────────────────────

  const fetchData = useCallback(async (fetchOffset = 0) => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        limit: "50",
        offset: String(fetchOffset),
      });
      const res = await fetch(`/api/admin/research?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: ResearchData = await res.json();
      if (fetchOffset === 0) {
        setData(json);
      } else {
        setData((prev) =>
          prev
            ? { ...json, findings: [...prev.findings, ...json.findings] }
            : json
        );
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // ─── Curation handler ─────────────────────────────────────

  const handleCurate = async (id: string, curationStatus: CurationStatus) => {
    // Optimistic update
    setData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        findings: prev.findings.map((f) =>
          f.id === id ? { ...f, curation_status: curationStatus } : f
        ),
      };
    });

    try {
      const res = await fetch("/api/admin/research", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, curation_status: curationStatus }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `HTTP ${res.status}`);
      }
    } catch (err) {
      // Revert on failure
      setError(err instanceof Error ? err.message : "Curation failed");
      fetchData(0);
    }
  };

  // ─── Plan handlers ─────────────────────────────────────

  const handleCreatePlan = (finding: ResearchFinding) => {
    setSelectedId(finding.id);
    setSelectedType("finding");
    setView("editor");
  };

  const handleSavePlan = async (planData: {
    id?: string;
    title: string;
    content: string;
    status: "draft" | "ready" | "implemented";
    subsystem: string | null;
    finding_id: string | null;
  }) => {
    try {
      setSaving(true);
      const res = await fetch("/api/admin/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "plan", ...planData }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `HTTP ${res.status}`);
      }
      const json = await res.json();
      setView("detail");
      setSelectedId(json.id);
      setSelectedType("plan");
      setOffset(0);
      fetchData(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleDeletePlan = async (id: string) => {
    try {
      setDeleting(true);
      const res = await fetch(`/api/admin/research?id=${id}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `HTTP ${res.status}`);
      }
      setSelectedId(null);
      setOffset(0);
      fetchData(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  };

  // ─── Ingest handler ──────────────────────────────────────

  const handleIngest = async () => {
    if (!ingestUrl.trim()) return;
    try {
      setIngesting(true);
      setIngestError(null);
      const res = await fetch("/api/admin/ingest-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: ingestUrl }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `HTTP ${res.status}`);
      }
      setIngestUrl("");
      setOffset(0);
      fetchData(0);
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setIngesting(false);
    }
  };

  // ─── Scan handler ────────────────────────────────────────

  const handleScan = async () => {
    try {
      setScanning(true);
      setScanError(null);
      setScanReport(null);
      const res = await fetch("/api/admin/research/scan", { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `HTTP ${res.status}`);
      }
      const json = await res.json();
      setScanReport(json.report);
      setOffset(0);
      fetchData(0);
    } catch (err) {
      setScanError(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setScanning(false);
    }
  };

  // ─── Derived state ───────────────────────────────────────

  const findings = data?.findings ?? [];
  const plans = data?.plans ?? [];
  const hasMore = data != null && offset + 50 < data.total;

  const selectedFinding: ResearchFinding | null =
    selectedId && selectedType === "finding"
      ? findings.find((f) => f.id === selectedId) ?? null
      : null;

  const selectedPlan: ResearchPlan | null =
    selectedId && selectedType === "plan"
      ? plans.find((p) => p.id === selectedId) ?? null
      : null;

  const editorFinding: ResearchFinding | null =
    view === "editor" && selectedId
      ? findings.find((f) => f.id === selectedId) ?? null
      : null;

  const editorPlan: ResearchPlan | null =
    view === "editor" && selectedType === "plan" && selectedId
      ? plans.find((p) => p.id === selectedId) ?? null
      : null;

  const handleSelect = (id: string, type: "finding" | "plan") => {
    setSelectedId(id);
    setSelectedType(type);
    setView("detail");
  };

  const handleLoadMore = () => {
    const next = offset + 50;
    setOffset(next);
    fetchData(next);
  };

  const handleRefresh = () => {
    setOffset(0);
    fetchData(0);
  };

  // ─── Render ──────────────────────────────────────────────

  return (
    <div className="px-5 pt-6 pb-8 space-y-4">
      {error && (
        <div className="rounded-xl border border-type-error/20 bg-type-error/5 px-4 py-3 text-[14px] text-type-error flex items-center justify-between">
          <span>{error}</span>
          <button
            onClick={() => {
              setError(null);
              handleRefresh();
            }}
            className="ml-3 shrink-0 px-3 py-1.5 rounded-lg text-[14px] font-medium bg-type-error/10 border border-type-error/20 hover:bg-type-error/20 transition-colors cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      <ScanControls
        scanning={scanning}
        onScan={handleScan}
        scanReport={scanReport}
        onDismissReport={() => setScanReport(null)}
        scanError={scanError}
        onDismissScanError={() => setScanError(null)}
        scanHistory={data?.scan_history ?? []}
        ingestUrl={ingestUrl}
        onIngestUrlChange={setIngestUrl}
        onIngest={handleIngest}
        ingesting={ingesting}
        ingestError={ingestError}
        loading={loading}
        onRefresh={handleRefresh}
      />

      {/* Two-column layout */}
      {loading && !data ? (
        <div className="grid grid-cols-2 gap-4">
          {[1, 2].map((i) => (
            <div key={i} className="h-64 rounded-xl skeleton" />
          ))}
        </div>
      ) : !data || (findings.length === 0 && plans.length === 0) ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <svg
            className="w-14 h-14 text-ink-faint/40 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25"
            />
          </svg>
          <p className="text-[18px] text-ink-secondary font-medium">
            No research findings yet
          </p>
          <p className="text-[15px] text-ink-faint mt-1.5 max-w-xs">
            Run the scanner or paste a URL to get started.
          </p>
          <button
            onClick={handleScan}
            disabled={scanning}
            className="mt-5 px-5 py-2.5 rounded-lg text-[14px] font-medium bg-gold/10 text-gold border border-gold/25 hover:bg-gold/20 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {scanning ? "Scanning..." : "Run Research Scanner"}
          </button>
        </div>
      ) : (
        <div
          className="grid grid-cols-5 gap-4"
          style={{ minHeight: "calc(100vh - 320px)" }}
        >
          {/* Left: findings + plans list (2/5 width) */}
          <div className="col-span-2 rounded-xl border border-edge bg-surface p-4 overflow-hidden flex flex-col">
            <FindingsList
              findings={findings}
              plans={plans}
              stats={data.stats}
              total={data.total}
              selectedId={selectedId}
              onSelect={handleSelect}
              onCurate={handleCurate}
              loading={loading}
              hasMore={hasMore}
              onLoadMore={handleLoadMore}
            />
          </div>

          {/* Right: detail / editor (3/5 width) */}
          <div className="col-span-3 rounded-xl border border-edge bg-surface p-5 overflow-hidden flex flex-col">
            {view === "editor" ? (
              <PlanEditor
                plan={editorPlan}
                findingContext={editorFinding?.content ?? null}
                onSave={handleSavePlan}
                onCancel={() => setView("detail")}
                saving={saving}
              />
            ) : selectedType === "plan" && selectedPlan ? (
              <PlanDetail
                plan={selectedPlan}
                onEdit={() => setView("editor")}
                onDelete={handleDeletePlan}
                deleting={deleting}
              />
            ) : (
              <FindingDetail
                finding={selectedFinding}
                onCreatePlan={handleCreatePlan}
                onCurate={handleCurate}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
