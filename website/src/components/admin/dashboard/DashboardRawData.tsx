import { useState } from "react";

interface DashboardRawDataProps {
  data: Record<string, unknown>;
}

export default function DashboardRawData({ data }: DashboardRawDataProps) {
  const [copied, setCopied] = useState(false);

  const jsonString = JSON.stringify(data, null, 2);

  const handleExport = () => {
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `omega-dashboard-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-3 card-enter">
      <div className="flex items-center justify-between">
        <h3 className="text-[14px] font-semibold text-ink">Raw Dashboard Data</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="text-[13px] px-3 py-1.5 rounded-lg text-ink-tertiary hover:text-ink-secondary hover:bg-surface-elevated transition-colors"
          >
            {copied ? "Copied" : "Copy"}
          </button>
          <button
            onClick={handleExport}
            className="text-[13px] px-3 py-1.5 rounded-lg bg-gold/[0.08] text-gold hover:bg-gold/[0.14] transition-colors font-medium"
          >
            Export JSON
          </button>
        </div>
      </div>
      <pre className="text-[12px] font-mono text-ink-secondary bg-surface-elevated border border-edge rounded-xl p-4 overflow-x-auto max-h-[400px] overflow-y-auto scrollbar-hide leading-relaxed">
        {jsonString}
      </pre>
    </div>
  );
}
