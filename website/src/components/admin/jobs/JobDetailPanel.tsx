import { useEffect, useState } from "react";
import SlideDrawer from "../SlideDrawer";
import {
  type Schedule,
  type ScheduleMetadata,
  type AccountInfo,
  computeStatus,
  humanSchedule,
  isReadOnly,
  sourceLabel,
} from "./jobUtils";
import { StatusPill, OwnerBadge } from "./atoms";
import { RunsTab } from "./RunsTab";
import { CostTab } from "./CostTab";
import { AuditTab } from "./AuditTab";

// ─── Cowork Section ─────────────────────────────────────

function CoworkSection({ metadata }: { metadata: ScheduleMetadata }) {
  const tools = [
    "omega_store",
    "omega_query",
    "omega_task_create",
    "omega_goal",
  ];
  return (
    <div className="space-y-4">
      {/* Runtime badge */}
      <div className="flex items-center gap-2">
        <span className="text-[14px] font-semibold px-2 py-0.5 rounded bg-type-lesson/15 text-type-lesson">
          {metadata.runtime ?? "cowork"}
        </span>
        {metadata.cadence && (
          <span className="text-[15px] text-ink-tertiary">{metadata.cadence}</span>
        )}
      </div>

      {/* OMEGA tools */}
      <div>
        <h4 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-2">
          OMEGA Tools
        </h4>
        <div className="flex gap-1.5 flex-wrap">
          {tools.map((t) => (
            <span
              key={t}
              className="text-[13px] font-mono text-gold/70 bg-gold/[0.06] px-2 py-0.5 rounded"
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* Sensors */}
      {metadata.sensors && metadata.sensors.length > 0 && (
        <div>
          <h4 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-2">
            Sensors
          </h4>
          <div className="flex gap-1.5 flex-wrap">
            {metadata.sensors.map((s) => (
              <span
                key={s}
                className="text-[13px] font-mono text-type-decision/70 bg-type-decision/[0.06] px-2 py-0.5 rounded"
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Rules */}
      {metadata.rules && metadata.rules.length > 0 && (
        <div>
          <h4 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-2">
            Rules
          </h4>
          <ul className="space-y-1.5">
            {metadata.rules.map((rule, i) => (
              <li
                key={i}
                className="text-[16px] text-ink-secondary flex gap-2 leading-relaxed"
              >
                <span className="text-ink-faint shrink-0">-</span>
                {rule}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Paths */}
      <div className="space-y-1 text-[14px] text-ink-faint font-mono">
        {metadata.prompt_file && <div>Prompt: {metadata.prompt_file}</div>}
        {metadata.output_dir && <div>Output: {metadata.output_dir}</div>}
        {metadata.omega_goal_id && <div>Goal: {metadata.omega_goal_id}</div>}
      </div>
    </div>
  );
}

// ─── Schedule Editor ───────────────────────────────────

const WEEKDAY_OPTIONS = [
  { value: -1, label: "Every day" },
  { value: 0, label: "Sunday" },
  { value: 1, label: "Monday" },
  { value: 2, label: "Tuesday" },
  { value: 3, label: "Wednesday" },
  { value: 4, label: "Thursday" },
  { value: 5, label: "Friday" },
  { value: 6, label: "Saturday" },
];

function ScheduleEditor({
  schedule,
  onSave,
}: {
  schedule: Schedule;
  onSave: (fields: Record<string, unknown>) => Promise<void>;
}) {
  const [type, setType] = useState(schedule.schedule_type);
  const [hour, setHour] = useState(schedule.calendar_hour ?? 7);
  const [minute, setMinute] = useState(schedule.calendar_minute ?? 0);
  const [weekday, setWeekday] = useState<number | null>(schedule.calendar_weekday);
  const [intervalMin, setIntervalMin] = useState(
    schedule.interval_seconds ? Math.round(schedule.interval_seconds / 60) : 5,
  );
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setType(schedule.schedule_type);
    setHour(schedule.calendar_hour ?? 7);
    setMinute(schedule.calendar_minute ?? 0);
    setWeekday(schedule.calendar_weekday);
    setIntervalMin(
      schedule.interval_seconds
        ? Math.round(schedule.interval_seconds / 60)
        : 5,
    );
    setDirty(false);
  }, [schedule.id, schedule.schedule_type, schedule.calendar_hour, schedule.calendar_minute, schedule.calendar_weekday, schedule.interval_seconds]);

  const handleSave = async () => {
    setSaving(true);
    const fields: Record<string, unknown> = { schedule_type: type };
    if (type === "calendar") {
      fields.calendar_hour = hour;
      fields.calendar_minute = minute;
      fields.calendar_weekday = weekday;
      fields.interval_seconds = null;
    } else {
      fields.interval_seconds = intervalMin * 60;
      fields.calendar_hour = null;
      fields.calendar_minute = null;
      fields.calendar_weekday = null;
    }
    await onSave(fields);
    setSaving(false);
    setDirty(false);
  };

  const inputCls =
    "bg-surface-elevated border border-edge rounded-lg px-3 py-2.5 text-[16px] text-ink focus:border-gold/40 focus:outline-none transition-colors w-full min-h-[44px]";

  return (
    <div className="space-y-3">
      {/* Type toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => {
            setType("calendar");
            setDirty(true);
          }}
          className={`flex-1 text-[15px] font-medium py-2.5 rounded-lg border transition-all min-h-[44px] ${
            type === "calendar"
              ? "bg-gold/[0.12] text-gold border-gold/20"
              : "text-ink-tertiary border-edge hover:border-edge-strong"
          }`}
        >
          Daily time
        </button>
        <button
          onClick={() => {
            setType("interval");
            setDirty(true);
          }}
          className={`flex-1 text-[15px] font-medium py-2.5 rounded-lg border transition-all min-h-[44px] ${
            type === "interval"
              ? "bg-gold/[0.12] text-gold border-gold/20"
              : "text-ink-tertiary border-edge hover:border-edge-strong"
          }`}
        >
          Interval
        </button>
      </div>

      {type === "calendar" ? (
        <div className="space-y-3">
          {/* Weekday selector */}
          <div>
            <label className="text-[14px] text-ink-faint uppercase tracking-wider mb-1.5 block">
              Day
            </label>
            <select
              value={weekday ?? -1}
              onChange={(e) => {
                const v = parseInt(e.target.value);
                setWeekday(v === -1 ? null : v);
                setDirty(true);
              }}
              className={inputCls}
            >
              {WEEKDAY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <div className="flex-1">
              <label className="text-[14px] text-ink-faint uppercase tracking-wider mb-1.5 block">
                Hour
              </label>
              <select
                value={hour}
                onChange={(e) => {
                  setHour(parseInt(e.target.value));
                  setDirty(true);
                }}
                className={inputCls}
              >
                {Array.from({ length: 24 }, (_, i) => {
                  const ampm = i >= 12 ? "PM" : "AM";
                  const h12 = i === 0 ? 12 : i > 12 ? i - 12 : i;
                  return (
                    <option key={i} value={i}>
                      {h12} {ampm}
                    </option>
                  );
                })}
              </select>
            </div>
            <div className="flex-1">
              <label className="text-[14px] text-ink-faint uppercase tracking-wider mb-1.5 block">
                Minute
              </label>
              <select
                value={minute}
                onChange={(e) => {
                  setMinute(parseInt(e.target.value));
                  setDirty(true);
                }}
                className={inputCls}
              >
                {[0, 5, 10, 15, 20, 30, 45].map((m) => (
                  <option key={m} value={m}>
                    :{m.toString().padStart(2, "0")}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <label className="text-[14px] text-ink-faint uppercase tracking-wider mb-1.5 block">
            Interval (minutes)
          </label>
          <input
            type="number"
            min={1}
            max={1440}
            value={intervalMin}
            onChange={(e) => {
              setIntervalMin(Math.max(1, parseInt(e.target.value) || 1));
              setDirty(true);
            }}
            className={inputCls}
          />
        </div>
      )}

      {dirty && (
        <div className="flex gap-2 justify-end">
          <button
            onClick={() => {
              setType(schedule.schedule_type);
              setHour(schedule.calendar_hour ?? 7);
              setMinute(schedule.calendar_minute ?? 0);
              setWeekday(schedule.calendar_weekday);
              setIntervalMin(
                schedule.interval_seconds
                  ? Math.round(schedule.interval_seconds / 60)
                  : 5,
              );
              setDirty(false);
            }}
            disabled={saving}
            className="text-[15px] font-medium px-4 py-2 rounded-lg text-ink-tertiary hover:text-ink-secondary transition-colors min-h-[44px]"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="text-[15px] font-semibold px-5 py-2 rounded-lg bg-gold text-canvas hover:bg-gold-dim transition-colors disabled:opacity-50 min-h-[44px]"
          >
            {saving ? (
              <span className="flex items-center gap-1.5">
                <span className="w-4 h-4 border-2 border-canvas/40 border-t-canvas rounded-full animate-spin" />
                Saving
              </span>
            ) : (
              "Save"
            )}
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Account Section ───────────────────────────────────

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function AccountSection({
  handle,
  info,
}: {
  handle: string;
  info: AccountInfo;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full text-left py-1.5 group min-h-[44px]"
      >
        <svg
          className={`w-4 h-4 text-ink-faint transition-transform ${open ? "rotate-0" : "-rotate-90"}`}
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="m19.5 8.25-7.5 7.5-7.5-7.5"
          />
        </svg>
        <span className="text-[18px] font-semibold text-ink">@{handle}</span>
        <span className="text-[15px] text-ink-tertiary">{info.role}</span>
      </button>

      {open && (
        <div className="pl-6 mt-1.5 space-y-3">
          <p className="text-[15px] text-ink-faint italic leading-relaxed">{info.voice}</p>
          <div className="flex gap-2 flex-wrap">
            {info.hashtags.map((h) => (
              <span
                key={h}
                className="text-[14px] font-mono text-gold/70 bg-gold/[0.06] px-2 py-0.5 rounded"
              >
                {h}
              </span>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[15px]">
              <thead>
                <tr className="text-ink-faint">
                  <th className="text-left py-1.5 pr-3 font-medium w-12">Day</th>
                  <th className="text-left py-1.5 px-1 font-medium">Activity</th>
                  <th className="text-right py-1.5 pl-1 font-medium w-24">
                    Replies
                  </th>
                </tr>
              </thead>
              <tbody>
                {DAYS.map((day) => {
                  const d = info.schedule[day];
                  if (!d) return null;
                  const isWeekend = day === "Sat" || day === "Sun";
                  return (
                    <tr
                      key={day}
                      className={
                        isWeekend ? "text-ink-faint" : "text-ink-secondary"
                      }
                    >
                      <td className="py-1 pr-3 font-medium">{day}</td>
                      <td className="py-1 px-1">{d.activity}</td>
                      <td className="py-1 pl-1 text-right font-mono tabular-nums">
                        {d.reply_quota}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Tier Badge ────────────────────────────────────────

function TierBadge({ tier }: { tier: string }) {
  const cls =
    tier === "tier_1"
      ? "bg-gold/[0.12] text-gold"
      : tier === "tier_2"
        ? "bg-type-decision/15 text-type-decision"
        : "bg-surface-elevated text-ink-tertiary";
  const label = tier.replace("_", " ").toUpperCase();
  return (
    <span className={`text-[14px] font-semibold px-2 py-0.5 rounded ${cls}`}>
      {label}
    </span>
  );
}

// ─── Metadata Section ──────────────────────────────────

function MetadataSection({ metadata }: { metadata: ScheduleMetadata }) {
  // Cowork tasks get their own section
  if (metadata.runtime === "cowork") {
    return <CoworkSection metadata={metadata} />;
  }

  const accounts = metadata.accounts ? Object.entries(metadata.accounts) : [];
  const targets = metadata.target_accounts;
  const rules = metadata.rules;

  return (
    <div className="space-y-5">
      {/* Per-account schedules */}
      {accounts.map(([handle, info]) => (
        <div
          key={handle}
          className="bg-surface-elevated/50 rounded-lg p-4 border border-edge/50"
        >
          <AccountSection handle={handle} info={info} />
        </div>
      ))}

      {/* Target accounts */}
      {targets && (
        <div>
          <h4 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-2">
            Targets
          </h4>
          <div className="space-y-2">
            {(["tier_1", "tier_2", "tier_3"] as const).map((tier) => {
              const handles = targets[tier];
              if (!handles || handles.length === 0) return null;
              return (
                <div key={tier} className="flex items-center gap-2 flex-wrap">
                  <TierBadge tier={tier} />
                  <span className="text-[16px] text-ink-secondary">
                    {handles.join(", ")}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Rules */}
      {rules && rules.length > 0 && (
        <div>
          <h4 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-2">
            Rules
          </h4>
          <ul className="space-y-1.5">
            {rules.map((rule, i) => (
              <li
                key={i}
                className="text-[16px] text-ink-secondary flex gap-2 leading-relaxed"
              >
                <span className="text-ink-faint shrink-0">-</span>
                {rule}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* File paths */}
      {(metadata.queue_file || metadata.briefing_dir) && (
        <div className="flex gap-4 text-[14px] text-ink-faint font-mono">
          {metadata.queue_file && <span>Queue: {metadata.queue_file}</span>}
          {metadata.briefing_dir && (
            <span>Briefings: {metadata.briefing_dir}</span>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Run Now ────────────────────────────────────────────

function RunNowButton({ schedule }: { schedule: Schedule }) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<"ok" | "error" | null>(null);

  const handleRunNow = async () => {
    setRunning(true);
    setResult(null);
    try {
      const res = await fetch("/api/admin/schedule-runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: schedule.label, action: "trigger" }),
      });
      setResult(res.ok ? "ok" : "error");
    } catch {
      setResult("error");
    }
    setRunning(false);
    // Clear result after 3s
    setTimeout(() => setResult(null), 3000);
  };

  return (
    <button
      onClick={handleRunNow}
      disabled={running || !schedule.enabled}
      title={!schedule.enabled ? "Enable the job first" : "Trigger this job immediately"}
      className={`text-[15px] font-medium px-4 py-2 rounded-lg border transition-all min-h-[44px] flex items-center gap-2 ${
        result === "ok"
          ? "border-type-lesson/30 text-type-lesson bg-type-lesson/[0.08]"
          : result === "error"
            ? "border-type-error/30 text-type-error bg-type-error/[0.08]"
            : "border-gold/20 text-gold hover:bg-gold/[0.08] disabled:opacity-40 disabled:cursor-not-allowed"
      }`}
    >
      {running ? (
        <>
          <span className="w-4 h-4 border-2 border-gold/40 border-t-gold rounded-full animate-spin" />
          Triggering...
        </>
      ) : result === "ok" ? (
        "Triggered"
      ) : result === "error" ? (
        "Failed"
      ) : (
        <>
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
          </svg>
          Run Now
        </>
      )}
    </button>
  );
}

// ─── Delete Confirm ────────────────────────────────────

function DeleteSection({
  name,
  onDelete,
}: {
  name: string;
  onDelete: () => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  if (!confirming) {
    return (
      <button
        onClick={() => setConfirming(true)}
        className="text-[15px] font-medium text-type-error/60 hover:text-type-error transition-colors min-h-[44px]"
      >
        Disable job
      </button>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-[16px] text-ink-secondary leading-relaxed">
        Disable <span className="font-semibold text-ink">{name}</span>? This
        will disable the schedule. It can be re-enabled later.
      </p>
      <div className="flex gap-2 justify-end">
        <button
          onClick={() => setConfirming(false)}
          disabled={deleting}
          className="text-[15px] font-medium px-4 py-2 rounded-lg text-ink-tertiary hover:text-ink-secondary transition-colors min-h-[44px]"
        >
          Cancel
        </button>
        <button
          onClick={async () => {
            setDeleting(true);
            await onDelete();
            setDeleting(false);
          }}
          disabled={deleting}
          className="text-[15px] font-semibold px-5 py-2 rounded-lg bg-type-error text-white hover:bg-type-error/90 transition-colors disabled:opacity-50 min-h-[44px]"
        >
          {deleting ? (
            <span className="flex items-center gap-1.5">
              <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
              Disabling
            </span>
          ) : (
            "Disable"
          )}
        </button>
      </div>
    </div>
  );
}

// ─── Main Panel ────────────────────────────────────────

interface JobDetailPanelProps {
  schedule: Schedule;
  onClose: () => void;
  onUpdate: (id: string, fields: Record<string, unknown>) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  healthDots?: string[];
}

const JOB_TABS = [
  { key: "overview", label: "Overview" },
  { key: "runs", label: "Runs" },
  { key: "cost", label: "Cost" },
  { key: "audit", label: "Audit" },
];

export function JobDetailPanel({
  schedule,
  onClose,
  onUpdate,
  onDelete,
  healthDots,
}: JobDetailPanelProps) {
  const status = computeStatus(schedule, healthDots);
  const [activeTab, setActiveTab] = useState<"overview" | "runs" | "cost" | "audit">("overview");

  // Reset tab when schedule changes
  useEffect(() => setActiveTab("overview"), [schedule.id]);

  return (
    <SlideDrawer
      open={true}
      onClose={onClose}
      title={schedule.name}
      width="md"
      tabs={JOB_TABS}
      activeTab={activeTab}
      onTabChange={(key) => setActiveTab(key as typeof activeTab)}
    >
      <div className="p-6 space-y-6">
        {/* Sub-header: owner + status */}
        <div className="flex items-center gap-2">
          <OwnerBadge label={schedule.label} />
          <StatusPill status={status} />
        </div>

        {/* Run Now */}
        <RunNowButton schedule={schedule} />

        {/* Description */}
        {schedule.description && (
          <p className="text-[18px] text-ink-secondary leading-relaxed">
            {schedule.description}
          </p>
        )}

        {/* Overview tab */}
        {activeTab === "overview" && (
          <>
            {/* Current schedule display */}
            <div className="text-[16px] text-ink-tertiary font-mono">
              {humanSchedule(schedule)}
            </div>

            {/* Schedule editor or read-only notice */}
            <section>
              <h3 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-3">
                Schedule
              </h3>
              {isReadOnly(schedule) ? (
                <div className="text-[15px] text-ink-tertiary bg-surface-elevated rounded-lg p-4 border border-edge/50">
                  <div className="font-mono mb-2">{humanSchedule(schedule)}</div>
                  <div className="text-[14px] text-ink-faint">
                    {sourceLabel(schedule)}. To change the schedule, edit the source file directly.
                  </div>
                </div>
              ) : (
                <ScheduleEditor
                  schedule={schedule}
                  onSave={(fields) => onUpdate(schedule.id, fields)}
                />
              )}
            </section>

            {/* Metadata details */}
            {schedule.metadata && (
              <section>
                <h3 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-3">
                  Details
                </h3>
                <MetadataSection metadata={schedule.metadata} />
              </section>
            )}

            {/* Command */}
            {schedule.command && (
              <section>
                <h3 className="text-[14px] text-ink-faint uppercase tracking-wider font-medium mb-2">
                  Command
                </h3>
                <div className="text-[14px] text-ink-faint font-mono bg-canvas rounded-lg p-4 break-all leading-relaxed">
                  {schedule.command}
                </div>
              </section>
            )}

            {/* Delete (only for editable jobs) */}
            {!isReadOnly(schedule) && (
              <div className="pt-4 border-t border-edge">
                <DeleteSection
                  name={schedule.name}
                  onDelete={() => onDelete(schedule.id)}
                />
              </div>
            )}
          </>
        )}

        {/* Runs tab */}
        {activeTab === "runs" && (
          <RunsTab scheduleLabel={schedule.label} />
        )}

        {/* Cost tab */}
        {activeTab === "cost" && (
          <CostTab scheduleLabel={schedule.label} />
        )}

        {/* Audit tab */}
        {activeTab === "audit" && (
          <AuditTab scheduleLabel={schedule.label} />
        )}
      </div>
    </SlideDrawer>
  );
}
