import { useCallback, useEffect, useState } from "react";
import CollapsibleSection from "./CollapsibleSection";

interface Campaign {
  id: string;
  account: string;
  name: string;
  voice_description: string | null;
  tone_keywords: string[] | null;
  forbidden_patterns: string[] | null;
  content_type_pool: { type: string; weight: number }[] | null;
  theme_bank: string[] | null;
  product_mention_ratio: number;
  min_quality_score: number;
  max_generation_attempts: number;
}

interface CampaignEditorProps {
  account: "all" | "jasonsosa" | "omega_memory";
}

export default function CampaignEditor({ account: rawAccount }: CampaignEditorProps) {
  const account = rawAccount === "all" ? "omega_memory" : rawAccount;
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Editable fields
  const [voiceDescription, setVoiceDescription] = useState("");
  const [toneKeywords, setToneKeywords] = useState("");
  const [forbiddenPatterns, setForbiddenPatterns] = useState("");
  const [themeBank, setThemeBank] = useState("");
  const [productRatio, setProductRatio] = useState(0.3);
  const [minScore, setMinScore] = useState(7);
  const [maxAttempts, setMaxAttempts] = useState(3);

  const fetchCampaign = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/admin/campaigns?account=${account}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      const c = data.campaign as Campaign | null;
      setCampaign(c);
      if (c) {
        setVoiceDescription(c.voice_description || "");
        setToneKeywords((c.tone_keywords || []).join(", "));
        setForbiddenPatterns((c.forbidden_patterns || []).join(", "));
        setThemeBank((c.theme_bank || []).join("\n"));
        setProductRatio(c.product_mention_ratio);
        setMinScore(c.min_quality_score);
        setMaxAttempts(c.max_generation_attempts);
      }
    } catch {
      setError("Failed to load campaign");
    }
    setLoading(false);
  }, [account]);

  useEffect(() => {
    fetchCampaign();
  }, [fetchCampaign]);

  const handleSave = async () => {
    if (!campaign) return;
    setSaving(true);
    setError(null);
    setSuccess(false);

    try {
      const res = await fetch("/api/admin/campaigns", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: campaign.id,
          voice_description: voiceDescription || null,
          tone_keywords: toneKeywords ? toneKeywords.split(",").map((s) => s.trim()).filter(Boolean) : null,
          forbidden_patterns: forbiddenPatterns ? forbiddenPatterns.split(",").map((s) => s.trim()).filter(Boolean) : null,
          theme_bank: themeBank ? themeBank.split("\n").map((s) => s.trim()).filter(Boolean) : null,
          content_type_pool: campaign.content_type_pool, // preserve — no UI editor yet
          product_mention_ratio: productRatio,
          min_quality_score: minScore,
          max_generation_attempts: maxAttempts,
        }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch {
      setError("Failed to save campaign");
    }
    setSaving(false);
  };

  if (loading) {
    return (
      <div className="p-4 admin-card">
        <div className="skeleton h-4 w-32 rounded mb-3" />
        <div className="skeleton h-24 w-full rounded-lg" />
      </div>
    );
  }

  const handleCreate = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/campaigns", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      await fetchCampaign();
    } catch {
      setError("Failed to create campaign");
    }
    setSaving(false);
  };

  if (!campaign) {
    return (
      <div className="p-4 admin-card">
        <p className="text-[14px] text-ink-tertiary mb-3">No active campaign for @{account}</p>
        {error && (
          <div className="p-3 mb-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[13px] text-type-error">
            {error}
          </div>
        )}
        <button
          onClick={handleCreate}
          disabled={saving}
          className="px-4 py-2.5 bg-gold text-surface-base text-[13px] font-semibold rounded-lg hover:bg-gold/90 transition-colors disabled:opacity-50 min-h-[44px]"
        >
          {saving ? "Creating..." : "Create Campaign"}
        </button>
      </div>
    );
  }

  return (
    <CollapsibleSection label="Campaign" summary={`${campaign.name} (${campaign.account})`}>
      <div className="space-y-4">
        {error && (
          <div className="p-3 bg-type-error/10 border border-type-error/20 rounded-lg text-[13px] text-type-error">
            {error}
          </div>
        )}
        {success && (
          <div className="p-3 bg-type-success/10 border border-type-success/20 rounded-lg text-[13px] text-type-success">
            Campaign saved
          </div>
        )}

        {/* Voice Description */}
        <div>
          <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Voice Description</label>
          <textarea
            value={voiceDescription}
            onChange={(e) => setVoiceDescription(e.target.value)}
            placeholder="Override voice description (leave empty to use defaults)"
            rows={3}
            className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink placeholder:text-ink-tertiary focus:border-gold focus:outline-none resize-y"
          />
        </div>

        {/* Tone Keywords */}
        <div>
          <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Tone Keywords</label>
          <input
            value={toneKeywords}
            onChange={(e) => setToneKeywords(e.target.value)}
            placeholder="direct, technical, warm (comma-separated)"
            className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink placeholder:text-ink-tertiary focus:border-gold focus:outline-none"
          />
        </div>

        {/* Forbidden Patterns */}
        <div>
          <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Forbidden Patterns</label>
          <input
            value={forbiddenPatterns}
            onChange={(e) => setForbiddenPatterns(e.target.value)}
            placeholder="em dashes, emoji spam (comma-separated)"
            className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink placeholder:text-ink-tertiary focus:border-gold focus:outline-none"
          />
        </div>

        {/* Theme Bank */}
        <div>
          <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Theme Bank (one per line)</label>
          <textarea
            value={themeBank}
            onChange={(e) => setThemeBank(e.target.value)}
            placeholder="Override themes (leave empty to use defaults)"
            rows={4}
            className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink font-mono placeholder:text-ink-tertiary focus:border-gold focus:outline-none resize-y"
          />
        </div>

        {/* Guardrails */}
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Product Ratio</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={productRatio}
              onChange={(e) => setProductRatio(parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink font-mono focus:border-gold focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Min Score</label>
            <input
              type="number"
              min="1"
              max="10"
              value={minScore}
              onChange={(e) => setMinScore(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink font-mono focus:border-gold focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-secondary mb-1.5">Max Attempts</label>
            <input
              type="number"
              min="1"
              max="10"
              value={maxAttempts}
              onChange={(e) => setMaxAttempts(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-surface border border-edge rounded-lg text-[13px] text-ink font-mono focus:border-gold focus:outline-none"
            />
          </div>
        </div>

        {/* Save */}
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2.5 bg-gold text-surface-base text-[13px] font-semibold rounded-lg hover:bg-gold/90 transition-colors disabled:opacity-50 min-h-[44px]"
        >
          {saving ? "Saving..." : "Save Campaign"}
        </button>
      </div>
    </CollapsibleSection>
  );
}
