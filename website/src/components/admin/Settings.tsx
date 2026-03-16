import { useCallback, useEffect, useState } from "react";
import SharedToggleSwitch from "./shared/ToggleSwitch";

// ─── Sub-nav ───────────────────────────────────────────────

type SettingsSection = "profile" | "agent" | "memory" | "projects" | "integrations" | "notifications";

const SECTIONS: { key: SettingsSection; label: string }[] = [
  { key: "profile", label: "Profile" },
  { key: "agent", label: "Agent" },
  { key: "memory", label: "Memory" },
  { key: "projects", label: "Projects" },
  { key: "integrations", label: "Integrations" },
  { key: "notifications", label: "Notifications" },
];

// ─── Types ──────────────────────────────────────────────────

interface ProfileSettings {
  name: string;
  timezone: string;
  avatar_url: string | null;
  preferences: Record<string, unknown>;
}

interface AgentSettings {
  default_model: string;
  autonomy: string;
  approval_gates: string[];
  max_concurrent_agents: number;
  auto_capture: string[];
}

interface MemorySettings {
  retention_days: number;
  auto_capture: boolean;
  blocklist_tags: string[];
  cloud_sync: boolean;
  sync_freq: string;
  max_memory_count: number;
  auto_maintenance: boolean;
}

interface Project {
  id: string;
  name: string;
  path: string | null;
  description: string | null;
  is_paused: boolean;
  launch_status: string | null;
  session_aliases?: string[];
}

interface Integration {
  id: string;
  service_name: string;
  account_identifier: string | null;
  status: string;
  config: Record<string, unknown>;
  last_verified_at: string | null;
  created_at: string;
}

interface NotificationSettings {
  email: string;
  digest_enabled: boolean;
  digest_time: string;
  alerts_enabled: boolean;
  alert_categories: string[];
  digest_categories: string[];
}

// ─── Constants ──────────────────────────────────────────────

const MODEL_OPTIONS = [
  { value: "claude-opus-4-6", label: "Opus 4.6" },
  { value: "claude-sonnet-4-6", label: "Sonnet 4.6" },
  { value: "claude-haiku-4-5-20251001", label: "Haiku 4.5" },
];

const AUTONOMY_OPTIONS = [
  { value: "supervised", label: "Supervised", desc: "Confirm before every action" },
  { value: "semi", label: "Semi-autonomous", desc: "Confirm risky actions only" },
  { value: "autonomous", label: "Autonomous", desc: "Act freely within gates" },
];

const GATE_OPTIONS = [
  { key: "deploy", label: "Deploy" },
  { key: "email", label: "Send email" },
  { key: "tweet", label: "Post tweet" },
  { key: "force_push", label: "Force push" },
  { key: "delete_branch", label: "Delete branch" },
  { key: "db_migration", label: "DB migration" },
];

const CAPTURE_OPTIONS = [
  { key: "decision", label: "Decisions" },
  { key: "lesson_learned", label: "Lessons" },
  { key: "error_pattern", label: "Error patterns" },
  { key: "user_preference", label: "Preferences" },
  { key: "task_completion", label: "Task completions" },
];

const SYNC_OPTIONS = [
  { value: "realtime", label: "Realtime" },
  { value: "hourly", label: "Hourly" },
  { value: "daily", label: "Daily" },
  { value: "manual", label: "Manual" },
];

const ALERT_OPTIONS = [
  { key: "reminder", label: "Due reminders" },
  { key: "schedule_failure", label: "Schedule failures" },
  { key: "publish_failure", label: "Publish failures" },
];

const DIGEST_OPTIONS = [
  { key: "decision", label: "Decisions" },
  { key: "lesson_learned", label: "Insights" },
  { key: "task_completion", label: "Task completions" },
  { key: "error_pattern", label: "Error patterns" },
  { key: "reminder", label: "Reminders" },
  { key: "checkpoint", label: "Stale checkpoints" },
];

// ─── Main Component ─────────────────────────────────────────

export default function Settings() {
  const [section, setSection] = useState<SettingsSection>("profile");

  return (
    <div className="admin-tab-enter">
      {/* Sub-nav pills */}
      <div className="flex gap-1 p-3 pb-0 overflow-x-auto">
        {SECTIONS.map((s) => (
          <button
            key={s.key}
            onClick={() => setSection(s.key)}
            className={`px-3 py-1.5 rounded-lg text-[12px] font-medium whitespace-nowrap transition-colors ${
              section === s.key
                ? "bg-gold/10 text-gold"
                : "text-ink-tertiary hover:text-ink-secondary hover:bg-surface-elevated"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Section content */}
      {section === "profile" && <ProfileSection />}
      {section === "agent" && <AgentSection />}
      {section === "memory" && <MemorySection />}
      {section === "projects" && <ProjectsSection />}
      {section === "integrations" && <IntegrationsSection />}
      {section === "notifications" && <NotificationsSection />}
    </div>
  );
}

// ─── Shared hooks ───────────────────────────────────────────

function useSaveState() {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  async function doSave(url: string, body: unknown) {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      const res = await fetch(url, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json();
        setError(data.error || "Save failed");
        return false;
      }
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      return true;
    } catch {
      setError("Failed to save");
      return false;
    } finally {
      setSaving(false);
    }
  }

  return { saving, saved, error, setError, doSave };
}

// ─── Profile Section ────────────────────────────────────────

function ProfileSection() {
  const [profile, setProfile] = useState<ProfileSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const { saving, saved, error, doSave } = useSaveState();

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/settings/profile");
      if (res.ok) setProfile(await res.json());
      else console.error("Failed to load profile settings:", res.status);
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) return <Spinner />;
  if (!profile) return <ErrorMsg msg="Failed to load profile" />;

  return (
    <div className="p-5 space-y-6">
      <section>
        <h3 className="admin-section-label mb-3">Profile</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-4">
          <Field label="Display name">
            <input
              type="text"
              value={profile.name}
              onChange={(e) => setProfile({ ...profile, name: e.target.value })}
              className={INPUT_CLASS}
              placeholder="Your name"
            />
          </Field>
          <Field label="Timezone">
            <select
              value={profile.timezone}
              onChange={(e) => setProfile({ ...profile, timezone: e.target.value })}
              className={INPUT_CLASS}
            >
              {["UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "America/Sao_Paulo", "Europe/London", "Europe/Berlin", "Europe/Paris", "Europe/Moscow", "Asia/Dubai", "Asia/Kolkata", "Asia/Singapore", "Asia/Shanghai", "Asia/Tokyo", "Asia/Seoul", "Australia/Sydney", "Pacific/Auckland"].map((tz) => (
                <option key={tz} value={tz}>{tz}</option>
              ))}
            </select>
          </Field>
        </div>
      </section>
      <SaveButton saving={saving} saved={saved} onClick={async () => { if (await doSave("/api/settings/profile", profile)) await load(); }} />
      {error && <ErrorBanner msg={error} />}
    </div>
  );
}

// ─── Agent Section ──────────────────────────────────────────

function AgentSection() {
  const [agent, setAgent] = useState<AgentSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const { saving, saved, error, doSave } = useSaveState();

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/settings/agent");
      if (res.ok) setAgent(await res.json());
      else console.error("Failed to load agent settings:", res.status);
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) return <Spinner />;
  if (!agent) return <ErrorMsg msg="Failed to load agent settings" />;

  function toggleGate(key: string) {
    setAgent(prev => {
      if (!prev) return prev;
      const gates = prev.approval_gates.includes(key)
        ? prev.approval_gates.filter((g) => g !== key)
        : [...prev.approval_gates, key];
      return { ...prev, approval_gates: gates };
    });
  }

  function toggleCapture(key: string) {
    setAgent(prev => {
      if (!prev) return prev;
      const caps = prev.auto_capture.includes(key)
        ? prev.auto_capture.filter((c) => c !== key)
        : [...prev.auto_capture, key];
      return { ...prev, auto_capture: caps };
    });
  }

  return (
    <div className="p-5 space-y-6">
      {/* Model */}
      <section>
        <h3 className="admin-section-label mb-3">Default Model</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4">
          <div className="grid grid-cols-3 gap-2">
            {MODEL_OPTIONS.map((m) => (
              <button
                key={m.value}
                onClick={() => setAgent({ ...agent, default_model: m.value })}
                className={`py-2 px-3 rounded-lg text-[12px] font-medium border transition-colors ${
                  agent.default_model === m.value
                    ? "border-gold bg-gold/10 text-gold"
                    : "border-edge text-ink-tertiary hover:border-ink-faint"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Autonomy */}
      <section>
        <h3 className="admin-section-label mb-3">Autonomy Level</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-2">
          {AUTONOMY_OPTIONS.map((opt) => (
            <label
              key={opt.value}
              className={`flex items-start gap-3 p-2.5 rounded-lg cursor-pointer transition-colors ${
                agent.autonomy === opt.value ? "bg-gold/[0.06]" : "hover:bg-surface-elevated/50"
              }`}
            >
              <input
                type="radio"
                name="autonomy"
                value={opt.value}
                checked={agent.autonomy === opt.value}
                onChange={() => setAgent(prev => prev ? { ...prev, autonomy: opt.value } : prev)}
                className="sr-only"
              />
              <span className={`mt-0.5 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-colors ${
                agent.autonomy === opt.value ? "border-gold" : "border-edge"
              }`}>
                {agent.autonomy === opt.value && <span className="w-2 h-2 rounded-full bg-gold" />}
              </span>
              <span>
                <span className="block text-[13px] text-ink font-medium">{opt.label}</span>
                <span className="block text-[11px] text-ink-tertiary">{opt.desc}</span>
              </span>
            </label>
          ))}
        </div>
      </section>

      {/* Approval Gates */}
      <section>
        <h3 className="admin-section-label mb-3">Approval Gates</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-2">
          <p className="text-[12px] text-ink-tertiary mb-2">Require confirmation before:</p>
          {GATE_OPTIONS.map((opt) => (
            <Checkbox
              key={opt.key}
              label={opt.label}
              checked={agent.approval_gates.includes(opt.key)}
              onChange={() => toggleGate(opt.key)}
            />
          ))}
        </div>
      </section>

      {/* Max Concurrent Agents */}
      <section>
        <h3 className="admin-section-label mb-3">Concurrency</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4">
          <Field label="Max concurrent agents">
            <input
              type="number"
              min={1}
              max={20}
              value={agent.max_concurrent_agents}
              onChange={(e) => setAgent({ ...agent, max_concurrent_agents: parseInt(e.target.value) || 4 })}
              className={INPUT_CLASS}
            />
          </Field>
        </div>
      </section>

      {/* Auto-capture */}
      <section>
        <h3 className="admin-section-label mb-3">Auto-capture</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-2">
          <p className="text-[12px] text-ink-tertiary mb-2">Automatically capture these memory types:</p>
          {CAPTURE_OPTIONS.map((opt) => (
            <Checkbox
              key={opt.key}
              label={opt.label}
              checked={agent.auto_capture.includes(opt.key)}
              onChange={() => toggleCapture(opt.key)}
            />
          ))}
        </div>
      </section>

      <SaveButton saving={saving} saved={saved} onClick={async () => { if (await doSave("/api/settings/agent", agent)) await load(); }} />
      {error && <ErrorBanner msg={error} />}
    </div>
  );
}

// ─── Memory Section ─────────────────────────────────────────

function MemorySection() {
  const [mem, setMem] = useState<MemorySettings | null>(null);
  const [loading, setLoading] = useState(true);
  const { saving, saved, error, doSave } = useSaveState();
  const [newTag, setNewTag] = useState("");

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/settings/memory");
      if (res.ok) setMem(await res.json());
      else console.error("Failed to load memory settings:", res.status);
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) return <Spinner />;
  if (!mem) return <ErrorMsg msg="Failed to load memory settings" />;

  function addTag() {
    if (!mem || !newTag.trim()) return;
    if (!mem.blocklist_tags.includes(newTag.trim())) {
      setMem({ ...mem, blocklist_tags: [...mem.blocklist_tags, newTag.trim()] });
    }
    setNewTag("");
  }

  function removeTag(tag: string) {
    if (!mem) return;
    setMem({ ...mem, blocklist_tags: mem.blocklist_tags.filter((t) => t !== tag) });
  }

  return (
    <div className="p-5 space-y-6">
      {/* Cloud Sync */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="admin-section-label">Cloud Sync</h3>
          <SharedToggleSwitch checked={mem.cloud_sync} onChange={() => setMem({ ...mem, cloud_sync: !mem.cloud_sync })} />
        </div>
        {mem.cloud_sync && (
          <div className="bg-surface-elevated border border-edge rounded-xl p-4">
            <Field label="Sync frequency">
              <div className="grid grid-cols-4 gap-2">
                {SYNC_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setMem({ ...mem, sync_freq: opt.value })}
                    className={`py-1.5 rounded-lg text-[12px] font-medium border transition-colors ${
                      mem.sync_freq === opt.value
                        ? "border-gold bg-gold/10 text-gold"
                        : "border-edge text-ink-tertiary hover:border-ink-faint"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </Field>
          </div>
        )}
      </section>

      {/* Auto-capture & Maintenance */}
      <section>
        <h3 className="admin-section-label mb-3">Automation</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[13px] text-ink-secondary">Auto-capture memories</span>
            <SharedToggleSwitch checked={mem.auto_capture} onChange={() => setMem({ ...mem, auto_capture: !mem.auto_capture })} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[13px] text-ink-secondary">Auto-maintenance</span>
            <SharedToggleSwitch checked={mem.auto_maintenance} onChange={() => setMem({ ...mem, auto_maintenance: !mem.auto_maintenance })} />
          </div>
        </div>
      </section>

      {/* Retention & Limits */}
      <section>
        <h3 className="admin-section-label mb-3">Limits</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-4">
          <Field label="Retention (days)">
            <input
              type="number"
              min={30}
              max={3650}
              value={mem.retention_days}
              onChange={(e) => setMem({ ...mem, retention_days: parseInt(e.target.value) || 365 })}
              className={INPUT_CLASS}
            />
          </Field>
          <Field label="Max memory count">
            <input
              type="number"
              min={1000}
              max={1000000}
              step={1000}
              value={mem.max_memory_count}
              onChange={(e) => setMem({ ...mem, max_memory_count: parseInt(e.target.value) || 100000 })}
              className={INPUT_CLASS}
            />
          </Field>
        </div>
      </section>

      {/* Blocklist Tags */}
      <section>
        <h3 className="admin-section-label mb-3">Blocklist Tags</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-3">
          <p className="text-[12px] text-ink-tertiary">Never capture memories with these tags:</p>
          {mem.blocklist_tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {mem.blocklist_tags.map((tag) => (
                <span key={tag} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-type-error/[0.06] text-type-error text-[11px] font-mono">
                  {tag}
                  <button onClick={() => removeTag(tag)} className="hover:text-type-error/70">&times;</button>
                </span>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <input
              type="text"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addTag()}
              className={INPUT_CLASS + " flex-1"}
              placeholder="Add tag..."
            />
            <button
              onClick={addTag}
              className="px-3 py-2 rounded-lg text-[12px] font-medium border border-edge text-ink-secondary hover:bg-surface-elevated/50 transition-colors"
            >
              Add
            </button>
          </div>
        </div>
      </section>

      <SaveButton saving={saving} saved={saved} onClick={async () => { if (await doSave("/api/settings/memory", mem)) await load(); }} />
      {error && <ErrorBanner msg={error} />}
    </div>
  );
}

// ─── Projects Section ───────────────────────────────────────

function ProjectsSection() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/projects/registry");
      if (res.ok) {
        const data = await res.json();
        // Map registry records to the local Project shape
        setProjects((data.projects || []).filter((r: Record<string, unknown>) => r.launch_status !== "archived").map((r: Record<string, unknown>) => ({
          id: r.id as string,
          name: r.name as string,
          path: r.path as string | null,
          description: r.description as string | null,
          is_paused: r.is_paused as boolean || false,
          launch_status: r.launch_status as string | null,
        })));
      }
    } catch { setError("Failed to load projects"); } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function addProject() {
    setError("");
    const id = `project-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const res = await fetch("/api/admin/projects/registry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, name: "New Project", session_aliases: [id] }),
    });
    if (!res.ok) {
      const data = await res.json();
      setError(data.error || "Failed to create project");
      return;
    }
    await load();
  }

  async function updateProject(p: Project) {
    setError("");
    const res = await fetch("/api/admin/projects/registry", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: p.id, name: p.name, path: p.path, description: p.description, is_paused: p.is_paused, session_aliases: p.session_aliases }),
    });
    if (!res.ok) {
      const data = await res.json();
      setError(data.error || "Save failed");
      return;
    }
    setEditing(null);
    await load();
  }

  async function deleteProject(id: string) {
    const project = projects.find((p) => p.id === id);
    const name = project?.name || id;
    if (!window.confirm(`Delete project "${name}"? This cannot be undone.`)) return;
    setError("");
    const res = await fetch(`/api/admin/projects/registry?id=${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (res.ok) setProjects((prev) => prev.filter((p) => p.id !== id));
    else {
      const data = await res.json();
      setError(data.error || "Delete failed");
    }
  }

  if (loading) return <Spinner />;

  return (
    <div className="p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="admin-section-label">Projects</h3>
        <button
          onClick={addProject}
          className="px-3 py-1.5 rounded-lg text-[12px] font-medium bg-gold/10 text-gold hover:bg-gold/20 transition-colors"
        >
          + Add
        </button>
      </div>

      {projects.length === 0 && (
        <div className="text-center py-8 text-ink-faint text-[13px]">No projects configured</div>
      )}

      <div className="space-y-2">
        {projects.map((p) => (
          <div key={p.id} className="bg-surface-elevated border border-edge rounded-xl p-4">
            {editing === p.id ? (
              <ProjectEditForm
                project={p}
                onSave={updateProject}
                onCancel={() => setEditing(null)}
              />
            ) : (
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-ink">{p.name}</span>
                    {p.is_paused && <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-type-error/10 text-type-error">Paused</span>}
                  </div>
                  {p.path && <span className="text-[11px] text-ink-faint font-mono">{p.path}</span>}
                </div>
                <div className="flex gap-1.5">
                  <button onClick={() => setEditing(p.id)} className="px-2.5 py-1 rounded-lg text-[11px] text-ink-tertiary hover:text-ink-secondary hover:bg-surface-elevated/50 border border-edge transition-colors">Edit</button>
                  <button onClick={() => deleteProject(p.id)} className="px-2.5 py-1 rounded-lg text-[11px] text-type-error/70 hover:text-type-error hover:bg-type-error/[0.06] border border-edge transition-colors">Delete</button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {error && <ErrorBanner msg={error} />}
    </div>
  );
}

function ProjectEditForm({ project, onSave, onCancel }: { project: Project; onSave: (p: Project) => void; onCancel: () => void }) {
  const [p, setP] = useState(project);

  return (
    <div className="space-y-3">
      <Field label="Name">
        <input type="text" value={p.name} onChange={(e) => setP({ ...p, name: e.target.value })} className={INPUT_CLASS} />
      </Field>
      <Field label="Path">
        <input type="text" value={p.path || ""} onChange={(e) => setP({ ...p, path: e.target.value || null })} className={INPUT_CLASS + " font-mono"} placeholder="~/Projects/..." />
      </Field>
      <Field label="Description">
        <input type="text" value={p.description || ""} onChange={(e) => setP({ ...p, description: e.target.value || null })} className={INPUT_CLASS} />
      </Field>
      <div className="flex gap-3">
        <div className="flex items-end gap-4 pb-0.5">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={p.is_paused} onChange={(e) => setP({ ...p, is_paused: e.target.checked })} className="accent-gold" />
            <span className="text-[12px] text-ink-secondary">Paused</span>
          </label>
        </div>
      </div>
      <div className="flex gap-2 pt-1">
        <button onClick={() => onSave(p)} className="px-4 py-2 rounded-lg text-[12px] font-semibold bg-gold text-canvas hover:bg-gold-dim transition-colors">Save</button>
        <button onClick={onCancel} className="px-4 py-2 rounded-lg text-[12px] font-medium border border-edge text-ink-tertiary hover:text-ink-secondary transition-colors">Cancel</button>
      </div>
    </div>
  );
}

// ─── Integrations Section ───────────────────────────────────

function IntegrationsSection() {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [oauthNotice, setOauthNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/settings/integrations");
      if (res.ok) {
        const data = await res.json();
        setIntegrations(data.integrations || []);
      }
    } catch { setError("Failed to load integrations"); } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  function showOauthNotice(serviceName: string) {
    setOauthNotice(serviceName);
    setError("");
  }

  async function revokeIntegration(id: string) {
    setError("");
    const res = await fetch("/api/settings/integrations", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, status: "revoked" }),
    });
    if (res.ok) await load();
    else {
      const data = await res.json();
      setError(data.error || "Revoke failed");
    }
  }

  async function deleteIntegration(id: string) {
    setError("");
    const res = await fetch("/api/settings/integrations", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    if (res.ok) setIntegrations((prev) => prev.filter((i) => i.id !== id));
    else {
      const data = await res.json();
      setError(data.error || "Delete failed");
    }
  }

  if (loading) return <Spinner />;

  const AVAILABLE_SERVICES = ["github", "slack", "linear", "vercel", "supabase", "x_twitter"];
  const connected = new Set(integrations.filter((i) => i.status === "active").map((i) => i.service_name));

  return (
    <div className="p-5 space-y-6">
      {/* Connected */}
      <section>
        <h3 className="admin-section-label mb-3">Connected</h3>
        {integrations.length === 0 ? (
          <div className="text-center py-6 text-ink-faint text-[13px]">No integrations connected</div>
        ) : (
          <div className="space-y-2">
            {integrations.map((i) => (
              <div key={i.id} className="bg-surface-elevated border border-edge rounded-xl p-4 flex items-center justify-between">
                <div>
                  <span className="text-[13px] font-medium text-ink capitalize">{i.service_name.replace("_", " ")}</span>
                  {i.account_identifier && <span className="ml-2 text-[11px] text-ink-faint font-mono">{i.account_identifier}</span>}
                  <span className={`ml-2 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                    i.status === "active" ? "bg-type-lesson/10 text-type-lesson" :
                    i.status === "revoked" ? "bg-type-error/10 text-type-error" :
                    "bg-ink/5 text-ink-tertiary"
                  }`}>{i.status}</span>
                </div>
                <div className="flex gap-1.5">
                  {i.status === "active" && (
                    <button onClick={() => revokeIntegration(i.id)} className="px-2.5 py-1 rounded-lg text-[11px] text-ink-tertiary hover:text-type-error border border-edge transition-colors">Revoke</button>
                  )}
                  <button onClick={() => deleteIntegration(i.id)} className="px-2.5 py-1 rounded-lg text-[11px] text-type-error/70 hover:text-type-error border border-edge transition-colors">Remove</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Available to connect */}
      <section>
        <h3 className="admin-section-label mb-3">Available</h3>
        <div className="grid grid-cols-3 gap-2">
          {AVAILABLE_SERVICES.filter((s) => !connected.has(s)).map((service) => (
            <button
              key={service}
              onClick={() => showOauthNotice(service)}
              className="py-3 rounded-xl text-[12px] font-medium border border-dashed border-edge text-ink-tertiary hover:border-gold/40 hover:text-gold transition-colors capitalize"
            >
              + {service.replace("_", " ")}
            </button>
          ))}
        </div>
        {oauthNotice && (
          <div className="mt-3 p-3 rounded-xl text-[12px] text-gold bg-gold/[0.06] border border-gold/[0.15] flex items-start justify-between gap-2">
            <span>OAuth setup required: configure credentials in environment variables for <strong className="capitalize">{oauthNotice.replace("_", " ")}</strong>.</span>
            <button onClick={() => setOauthNotice(null)} className="text-ink-tertiary hover:text-ink-secondary shrink-0">&times;</button>
          </div>
        )}
      </section>

      {error && <ErrorBanner msg={error} />}
    </div>
  );
}

// ─── Notifications Section (preserved from original) ────────
// TODO: Notifications loads from the general /api/settings endpoint instead of
// a dedicated /api/settings/notifications route. This couples notification
// preferences to the general settings fetch. Consider extracting to a dedicated
// route for consistency with /api/settings/profile, /api/settings/agent, etc.

const DIGEST_OPTION_KEYS = new Set(DIGEST_OPTIONS.map((o) => o.key));

function NotificationsSection() {
  const [settings, setSettings] = useState<NotificationSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const { saving, saved, error, doSave } = useSaveState();
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  // Track backend-only digest categories so we don't drop them on save
  const [extraDigestCategories, setExtraDigestCategories] = useState<string[]>([]);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        const data = await res.json();
        // Separate backend-only categories from frontend-known ones
        const backendOnly = (data.digest_categories || []).filter(
          (cat: string) => !DIGEST_OPTION_KEYS.has(cat)
        );
        setExtraDigestCategories(backendOnly);
        setSettings(data);
      } else {
        console.error("Failed to load notification settings:", res.status);
      }
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function sendTestDigest() {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/digest?test=true", { method: "POST" });
      const data = await res.json();
      if (res.ok) setTestResult({ ok: true, message: `Digest sent to ${data.to}` });
      else setTestResult({ ok: false, message: data.error || "Failed" });
    } catch (err) {
      setTestResult({ ok: false, message: err instanceof Error ? err.message : "Failed" });
    } finally {
      setTesting(false);
      setTimeout(() => setTestResult(null), 4000);
    }
  }

  function toggle(field: "digest_enabled" | "alerts_enabled") {
    if (!settings) return;
    setSettings({ ...settings, [field]: !settings[field] });
  }

  function toggleCategory(field: "alert_categories" | "digest_categories", key: string) {
    setSettings(prev => {
      if (!prev) return prev;
      const current = prev[field];
      const next = current.includes(key) ? current.filter((k) => k !== key) : [...current, key];
      return { ...prev, [field]: next };
    });
  }

  if (loading) return <Spinner />;
  if (!settings) return <ErrorMsg msg="Failed to load notification settings" />;

  return (
    <div className="p-5 space-y-6">
      {/* Email */}
      <section>
        <h3 className="admin-section-label mb-3">Email</h3>
        <div className="bg-surface-elevated border border-edge rounded-xl p-4">
          <Field label="Notification email">
            <input
              type="email"
              value={settings.email}
              onChange={(e) => setSettings({ ...settings, email: e.target.value })}
              className={INPUT_CLASS}
              placeholder="you@example.com"
            />
          </Field>
        </div>
      </section>

      {/* Daily Digest */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="admin-section-label">Daily Digest</h3>
          <SharedToggleSwitch checked={settings.digest_enabled} onChange={() => toggle("digest_enabled")} />
        </div>
        {settings.digest_enabled && (
          <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-3">
            <p className="text-[12px] text-ink-tertiary">Sent daily at 8:00 AM in your configured timezone. Includes:</p>
            <div className="space-y-2">
              {DIGEST_OPTIONS.map((opt) => (
                <Checkbox key={opt.key} label={opt.label} checked={settings.digest_categories.includes(opt.key)} onChange={() => toggleCategory("digest_categories", opt.key)} />
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Real-time Alerts */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="admin-section-label">Real-time Alerts</h3>
          <SharedToggleSwitch checked={settings.alerts_enabled} onChange={() => toggle("alerts_enabled")} />
        </div>
        {settings.alerts_enabled && (
          <div className="bg-surface-elevated border border-edge rounded-xl p-4 space-y-3">
            <p className="text-[12px] text-ink-tertiary">Checked every 15 minutes. Alert on:</p>
            <div className="space-y-2">
              {ALERT_OPTIONS.map((opt) => (
                <Checkbox key={opt.key} label={opt.label} checked={settings.alert_categories.includes(opt.key)} onChange={() => toggleCategory("alert_categories", opt.key)} />
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Actions */}
      <section className="space-y-3">
        <SaveButton saving={saving} saved={saved} onClick={async () => {
          // Merge back any backend-only digest categories before saving
          const payload = {
            ...settings,
            digest_categories: [
              ...new Set([...(settings?.digest_categories || []), ...extraDigestCategories]),
            ],
          };
          if (await doSave("/api/settings", payload)) await load();
        }} />
        <button
          onClick={sendTestDigest}
          disabled={testing}
          className="w-full py-2.5 rounded-xl text-[13px] font-medium border border-edge text-ink-secondary hover:bg-surface-elevated/50 transition-colors disabled:opacity-50"
        >
          {testing ? "Sending..." : "Send Test Digest"}
        </button>
        {testResult && (
          <div className={`p-3 rounded-xl text-[12px] card-enter ${
            testResult.ok ? "text-type-lesson bg-type-lesson/[0.06] border border-type-lesson/[0.1]" : "text-type-error bg-type-error/[0.06] border border-type-error/[0.1]"
          }`}>{testResult.message}</div>
        )}
      </section>

      {error && <ErrorBanner msg={error} />}
    </div>
  );
}

// ─── Shared sub-components ──────────────────────────────────

const INPUT_CLASS = "w-full bg-canvas border border-edge rounded-lg px-3 py-2 text-[13px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-gold/40 focus:ring-1 focus:ring-gold/20 transition-colors";

function Spinner() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="w-5 h-5 border-2 border-gold/30 border-t-gold rounded-full animate-spin" />
    </div>
  );
}

function ErrorMsg({ msg }: { msg: string }) {
  return <div className="p-5 text-center text-ink-faint text-[13px]">{msg}</div>;
}

function ErrorBanner({ msg }: { msg: string }) {
  return <div className="p-3 rounded-xl text-[12px] text-type-error bg-type-error/[0.06] border border-type-error/[0.1]">{msg}</div>;
}

function SaveButton({ saving, saved, onClick }: { saving: boolean; saved: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={saving}
      className="w-full py-2.5 rounded-xl text-[13px] font-semibold bg-gold text-canvas hover:bg-gold-dim transition-colors disabled:opacity-50"
    >
      {saving ? "Saving..." : saved ? "Saved" : "Save Settings"}
    </button>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[12px] text-ink-tertiary mb-1.5">{label}</label>
      {children}
    </div>
  );
}

function Checkbox({ label, checked, onChange }: { label: string; checked: boolean; onChange: () => void }) {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer group">
      <input type="checkbox" checked={checked} onChange={onChange} className="sr-only" />
      <span className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
        checked ? "bg-emerald-500 border-emerald-500" : "border-edge group-hover:border-ink-faint"
      }`}>
        {checked && (
          <svg className="w-2.5 h-2.5 text-canvas" fill="none" viewBox="0 0 24 24" strokeWidth={3} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
        )}
      </span>
      <span className="text-[13px] text-ink-secondary">{label}</span>
    </label>
  );
}
