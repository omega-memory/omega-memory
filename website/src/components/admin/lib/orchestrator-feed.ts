/** Orchestrator feed: types + CRUD helpers for the unified communication channel.
 *  Pattern follows lib/campaign.ts. */

import { supabaseServer } from "./supabase";

// ── Types ────────────────────────────────────────────────────────

export type OrchestratorItemType = "update" | "insight" | "report" | "alert" | "proposal" | "plan";
export type RiskLevel = "low" | "medium" | "high";
export type FeedItemStatus = "active" | "read" | "approved" | "rejected" | "expired" | "executed";

export interface OrchestratorFeedItem {
  id: string;
  account: string | null;
  item_type: OrchestratorItemType;
  risk_level: RiskLevel;
  status: FeedItemStatus;
  title: string;
  body: string | null;
  metadata: Record<string, unknown>;
  pending_event_id: string | null;
  reviewed_at: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  source: string | null;
  source_ref: string | null;
  expires_at: string | null;
  created_at: string;
  updated_at: string;
}

// ── Risk classification ──────────────────────────────────────────

export const ITEM_TYPE_RISK: Record<OrchestratorItemType, RiskLevel> = {
  update: "low",
  insight: "low",
  report: "low",
  alert: "medium",
  proposal: "high",
  plan: "high",
};

// ── Create ───────────────────────────────────────────────────────

export interface CreateFeedItemParams {
  account?: string | null;
  item_type: OrchestratorItemType;
  title: string;
  body?: string | null;
  metadata?: Record<string, unknown>;
  source?: string | null;
  source_ref?: string | null;
  expires_at?: string | null;
}

export async function createFeedItem(params: CreateFeedItemParams): Promise<OrchestratorFeedItem> {
  const db = supabaseServer();
  const risk_level = ITEM_TYPE_RISK[params.item_type] || "low";

  const { data, error } = await db
    .from("orchestrator_feed")
    .insert({
      account: params.account ?? null,
      item_type: params.item_type,
      risk_level,
      status: "active",
      title: params.title,
      body: params.body ?? null,
      metadata: params.metadata ?? {},
      source: params.source ?? null,
      source_ref: params.source_ref ?? null,
      expires_at: params.expires_at ?? null,
    })
    .select("*")
    .single();

  if (error || !data) throw new Error(`Failed to create feed item: ${error?.message}`);
  return data as OrchestratorFeedItem;
}

// ── List with keyset pagination ──────────────────────────────────

export interface ListFeedItemsParams {
  account?: string;
  item_type?: OrchestratorItemType;
  status?: FeedItemStatus;
  risk_level?: RiskLevel;
  limit?: number;
  cursor?: string; // created_at ISO string for keyset pagination
}

export async function listFeedItems(params: ListFeedItemsParams = {}): Promise<{
  items: OrchestratorFeedItem[];
  hasMore: boolean;
  nextCursor: string | null;
}> {
  const db = supabaseServer();
  const limit = params.limit || 30;

  let query = db
    .from("orchestrator_feed")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit + 1); // fetch one extra to detect hasMore

  if (params.account) query = query.eq("account", params.account);
  if (params.item_type) query = query.eq("item_type", params.item_type);
  if (params.status) query = query.eq("status", params.status);
  if (params.risk_level) query = query.eq("risk_level", params.risk_level);
  if (params.cursor) query = query.lt("created_at", params.cursor);

  const { data, error } = await query;
  if (error) throw new Error(`Failed to list feed items: ${error.message}`);

  const items = (data || []) as OrchestratorFeedItem[];
  const hasMore = items.length > limit;
  if (hasMore) items.pop();

  return {
    items,
    hasMore,
    nextCursor: items.length > 0 ? items[items.length - 1].created_at : null,
  };
}

// ── Count pending proposals ──────────────────────────────────────

export async function countPendingProposals(): Promise<number> {
  const db = supabaseServer();
  const { count, error } = await db
    .from("orchestrator_feed")
    .select("id", { count: "exact", head: true })
    .eq("status", "active")
    .eq("risk_level", "high");

  if (error) return 0;
  return count ?? 0;
}

// ── Approve proposal ─────────────────────────────────────────────

export interface EventParams {
  account?: string | null;
  event_type: string;
  scheduled_for?: string;
  payload?: Record<string, unknown>;
  recurrence?: string | null;
  source?: string;
}

export async function approveProposal(
  id: string,
  eventParams: EventParams,
  reviewNote?: string
): Promise<OrchestratorFeedItem> {
  const db = supabaseServer();
  const now = new Date().toISOString();

  // 1. Create pending_event from eventParams
  const { data: event, error: eventErr } = await db
    .from("pending_events")
    .insert({
      account: eventParams.account ?? null,
      event_type: eventParams.event_type,
      status: "pending",
      scheduled_for: eventParams.scheduled_for || now,
      payload: eventParams.payload || {},
      recurrence: eventParams.recurrence ?? null,
      source: eventParams.source || "orchestrator_proposal",
      max_attempts: 3,
    })
    .select("id")
    .single();

  if (eventErr || !event) throw new Error(`Failed to create event: ${eventErr?.message}`);

  // 2. Update feed item: status=approved, link to event
  const { data, error } = await db
    .from("orchestrator_feed")
    .update({
      status: "approved",
      pending_event_id: event.id,
      reviewed_at: now,
      reviewed_by: "admin",
      review_note: reviewNote || null,
      updated_at: now,
    })
    .eq("id", id)
    .select("*")
    .single();

  if (error || !data) throw new Error(`Failed to approve proposal: ${error?.message}`);
  return data as OrchestratorFeedItem;
}

// ── Reject proposal ──────────────────────────────────────────────

export async function rejectProposal(
  id: string,
  reviewNote?: string
): Promise<OrchestratorFeedItem> {
  const db = supabaseServer();
  const now = new Date().toISOString();

  const { data, error } = await db
    .from("orchestrator_feed")
    .update({
      status: "rejected",
      reviewed_at: now,
      reviewed_by: "admin",
      review_note: reviewNote || null,
      updated_at: now,
    })
    .eq("id", id)
    .select("*")
    .single();

  if (error || !data) throw new Error(`Failed to reject proposal: ${error?.message}`);
  return data as OrchestratorFeedItem;
}

// ── Mark as read ─────────────────────────────────────────────────

export async function markAsRead(id: string): Promise<OrchestratorFeedItem> {
  const db = supabaseServer();
  const now = new Date().toISOString();

  const { data, error } = await db
    .from("orchestrator_feed")
    .update({ status: "read", updated_at: now })
    .eq("id", id)
    .select("*")
    .single();

  if (error || !data) throw new Error(`Failed to mark as read: ${error?.message}`);
  return data as OrchestratorFeedItem;
}

// ── Dedup check ──────────────────────────────────────────────────

/** Returns true if a feed item with the same (source, source_ref) exists within the time window.
 *  Used by automation modules to prevent feed spam. Fails open (returns false on error). */
export async function hasSimilarRecentItem(params: {
  source: string;
  source_ref: string;
  hours?: number;
}): Promise<boolean> {
  try {
    const db = supabaseServer();
    const cutoff = new Date(Date.now() - (params.hours ?? 24) * 3600000).toISOString();

    const { count, error } = await db
      .from("orchestrator_feed")
      .select("id", { count: "exact", head: true })
      .eq("source", params.source)
      .eq("source_ref", params.source_ref)
      .gte("created_at", cutoff);

    if (error) return false; // fail open
    return (count ?? 0) > 0;
  } catch {
    return false; // fail open
  }
}

// ── Update status (used by event handlers) ───────────────────────

export async function updateFeedItemStatus(
  id: string,
  status: FeedItemStatus
): Promise<void> {
  const db = supabaseServer();
  const { error } = await db
    .from("orchestrator_feed")
    .update({ status, updated_at: new Date().toISOString() })
    .eq("id", id);

  if (error) throw new Error(`Failed to update feed item status: ${error.message}`);
}
