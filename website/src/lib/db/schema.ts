/**
 * Drizzle ORM schema — single source of truth for D1 (Cloudflare) and PostgreSQL.
 *
 * All column types use TEXT for UUIDs/datetimes to stay compatible with both
 * SQLite (D1) and PostgreSQL. This mirrors the Python SQLAlchemy models.
 */
import { sqliteTable, text, integer, real, uniqueIndex, index } from "drizzle-orm/sqlite-core";

// ===================================================================
// Core tables (schema v14)
// ===================================================================

export const memories = sqliteTable("memories", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  nodeId: text("node_id").notNull().unique(),
  content: text("content").notNull(),
  metadata: text("metadata"),
  createdAt: text("created_at").notNull(),
  lastAccessed: text("last_accessed"),
  accessCount: integer("access_count").default(0),
  ttlSeconds: integer("ttl_seconds"),
  sessionId: text("session_id"),
  eventType: text("event_type"),
  project: text("project"),
  contentHash: text("content_hash"),
  priority: integer("priority").default(3),
  referencedDate: text("referenced_date"),
  entityId: text("entity_id"),
  agentType: text("agent_type"),
  canonicalHash: text("canonical_hash"),
  endDate: text("end_date"),
  extractedKeywords: text("extracted_keywords"),
  retrievalCount: integer("retrieval_count").default(0),
  memoryType: text("memory_type").default("semantic"),
  validFrom: text("valid_from"),
  validUntil: text("valid_until"),
  derivedFrom: text("derived_from"),
  sourceUri: text("source_uri"),
  status: text("status").default("active"),
  // Multi-tenant
  userId: text("user_id"),
}, (t) => [
  index("idx_memories_event_type").on(t.eventType),
  index("idx_memories_project").on(t.project),
  index("idx_memories_session_id").on(t.sessionId),
  index("idx_memories_created_at").on(t.createdAt),
  index("idx_memories_user_id").on(t.userId),
]);

export const edges = sqliteTable("edges", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  sourceId: text("source_id").notNull(),
  targetId: text("target_id").notNull(),
  edgeType: text("edge_type").notNull(),
  weight: real("weight").default(1.0),
  metadata: text("metadata"),
  createdAt: text("created_at"),
}, (t) => [
  uniqueIndex("uq_edges_src_tgt_type").on(t.sourceId, t.targetId, t.edgeType),
  index("idx_edges_source_id").on(t.sourceId),
  index("idx_edges_target_id").on(t.targetId),
]);

export const forgettingLog = sqliteTable("forgetting_log", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  nodeId: text("node_id").notNull(),
  contentPreview: text("content_preview"),
  eventType: text("event_type"),
  reason: text("reason").notNull(),
  deletedAt: text("deleted_at").notNull(),
  metadata: text("metadata"),
}, (t) => [
  index("idx_forgetting_log_deleted_at").on(t.deletedAt),
  uniqueIndex("idx_forgetting_log_node_reason").on(t.nodeId, t.reason),
]);

export const cloudDeleteQueue = sqliteTable("cloud_delete_queue", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  localId: integer("local_id").notNull(),
  deletedAt: text("deleted_at").notNull(),
});

export const entityIndex = sqliteTable("entity_index", {
  entityName: text("entity_name").primaryKey(),
  entityType: text("entity_type").default("person"),
  statementCount: integer("statement_count").default(0),
  outcomeCount: integer("outcome_count").default(0),
  contradictionScore: real("contradiction_score").default(0.0),
  followThroughRate: real("follow_through_rate").default(0.0),
  firstSeen: text("first_seen").notNull(),
  lastUpdated: text("last_updated").notNull(),
  metadata: text("metadata"),
});

export const memoryClusters = sqliteTable("memory_clusters", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  clusterId: integer("cluster_id").notNull(),
  label: text("label").notNull(),
  memberCount: integer("member_count").notNull(),
  centroid: text("centroid"), // base64 encoded for D1 compat
  representativeKeywords: text("representative_keywords"),
  representativeMemoryIds: text("representative_memory_ids"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
  superseded: integer("superseded").default(0),
});

export const thompsonArms = sqliteTable("thompson_arms", {
  armId: text("arm_id").primaryKey(),
  armType: text("arm_type").notNull(),
  alpha: real("alpha").default(1.0),
  beta: real("beta").default(1.0),
  totalTrials: integer("total_trials").default(0),
  totalSuccesses: integer("total_successes").default(0),
  lastUpdated: text("last_updated").notNull(),
  context: text("context"),
});

export const maintenanceDlq = sqliteTable("maintenance_dlq", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  stageName: text("stage_name").notNull(),
  errorClass: text("error_class").default("transient"),
  errorMessage: text("error_message"),
  remediationAttempts: integer("remediation_attempts").default(0),
  maxRemediation: integer("max_remediation").default(3),
  status: text("status").default("pending"),
  nextRetryAt: text("next_retry_at"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

// ===================================================================
// Auth tables
// ===================================================================

export const users = sqliteTable("users", {
  id: text("id").primaryKey(), // uuid
  email: text("email").notNull().unique(),
  passwordHash: text("password_hash"),
  displayName: text("display_name"),
  createdAt: text("created_at"),
  updatedAt: text("updated_at"),
  isActive: integer("is_active").default(1),
});

export const apiTokens = sqliteTable("api_tokens", {
  id: text("id").primaryKey(),
  userId: text("user_id").notNull().references(() => users.id),
  tokenHash: text("token_hash").notNull(),
  name: text("name").notNull(),
  scopes: text("scopes"), // JSON
  expiresAt: text("expires_at"),
  createdAt: text("created_at"),
  lastUsedAt: text("last_used_at"),
  isRevoked: integer("is_revoked").default(0),
}, (t) => [
  index("idx_api_tokens_user_id").on(t.userId),
  index("idx_api_tokens_token_hash").on(t.tokenHash),
]);

export const agentCertificates = sqliteTable("agent_certificates", {
  id: text("id").primaryKey(),
  userId: text("user_id").references(() => users.id),
  agentName: text("agent_name").notNull(),
  fingerprint: text("fingerprint").notNull().unique(),
  certPem: text("cert_pem").notNull(),
  serialNumber: text("serial_number").notNull().unique(),
  issuedAt: text("issued_at"),
  expiresAt: text("expires_at").notNull(),
  revokedAt: text("revoked_at"),
  isRevoked: integer("is_revoked").default(0),
});

export const userSessions = sqliteTable("user_sessions", {
  id: text("id").primaryKey(),
  userId: text("user_id").references(() => users.id),
  tokenHash: text("token_hash").notNull(),
  deviceInfo: text("device_info"),
  ipAddress: text("ip_address"),
  createdAt: text("created_at"),
  expiresAt: text("expires_at").notNull(),
  isRevoked: integer("is_revoked").default(0),
}, (t) => [
  index("idx_user_sessions_token_hash").on(t.tokenHash),
  index("idx_user_sessions_user_id").on(t.userId),
]);

export const oidcProviders = sqliteTable("oidc_providers", {
  id: text("id").primaryKey(),
  name: text("name").notNull().unique(),
  discoveryUrl: text("discovery_url").notNull(),
  clientId: text("client_id").notNull(),
  clientSecretEncrypted: text("client_secret_encrypted"),
  isEnabled: integer("is_enabled").default(1),
  createdAt: text("created_at"),
});

// ===================================================================
// Multi-tenant tables
// ===================================================================

export const workspaces = sqliteTable("workspaces", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  slug: text("slug").notNull().unique(),
  ownerId: text("owner_id").notNull().references(() => users.id),
  createdAt: text("created_at"),
});

export const workspaceMembers = sqliteTable("workspace_members", {
  id: text("id").primaryKey(),
  workspaceId: text("workspace_id").notNull().references(() => workspaces.id),
  userId: text("user_id").notNull().references(() => users.id),
  role: text("role").default("member"),
  joinedAt: text("joined_at"),
}, (t) => [
  uniqueIndex("uq_workspace_members").on(t.workspaceId, t.userId),
]);

export const syncState = sqliteTable("sync_state", {
  id: text("id").primaryKey(),
  tableName: text("table_name").notNull(),
  userId: text("user_id").references(() => users.id),
  lastLocalId: integer("last_local_id").default(0),
  lastSyncAt: text("last_sync_at"),
  syncCount: integer("sync_count").default(0),
}, (t) => [
  uniqueIndex("uq_sync_state_table_user").on(t.tableName, t.userId),
]);

export const documents = sqliteTable("documents", {
  id: text("id").primaryKey(),
  localId: integer("local_id"),
  sourcePath: text("source_path").notNull(),
  sourceType: text("source_type").notNull(),
  title: text("title"),
  checksum: text("checksum"),
  chunkCount: integer("chunk_count").default(0),
  userId: text("user_id").references(() => users.id),
  createdAt: text("created_at"),
  updatedAt: text("updated_at"),
  syncedAt: text("synced_at"),
}, (t) => [
  index("idx_documents_user_id").on(t.userId),
]);

export const documentChunks = sqliteTable("document_chunks", {
  id: text("id").primaryKey(),
  documentId: text("document_id").references(() => documents.id),
  localId: integer("local_id"),
  chunkIndex: integer("chunk_index").notNull(),
  content: text("content").notNull(),
  chunkType: text("chunk_type"),
  pageNumber: integer("page_number"),
  tokenCount: integer("token_count"),
  userId: text("user_id").references(() => users.id),
  createdAt: text("created_at"),
}, (t) => [
  index("idx_document_chunks_user_id").on(t.userId),
]);

export const secureProfiles = sqliteTable("secure_profiles", {
  id: text("id").primaryKey(),
  localId: integer("local_id"),
  category: text("category").notNull(),
  fieldName: text("field_name").notNull(),
  fieldValueEncrypted: text("field_value_encrypted").notNull(),
  metadata: text("metadata"),
  userId: text("user_id").references(() => users.id),
  createdAt: text("created_at"),
  updatedAt: text("updated_at"),
  syncedAt: text("synced_at"),
}, (t) => [
  index("idx_secure_profiles_user_id").on(t.userId),
]);

// ===================================================================
// Licenses & releases
// ===================================================================

export const licenses = sqliteTable("licenses", {
  id: text("id").primaryKey(),
  key: text("key").notNull().unique(),
  email: text("email").notNull(),
  status: text("status").default("active"),
  stripeCustomerId: text("stripe_customer_id"),
  stripeSubscriptionId: text("stripe_subscription_id"),
  createdAt: text("created_at"),
  lastValidatedAt: text("last_validated_at"),
});

export const proReleases = sqliteTable("pro_releases", {
  id: text("id").primaryKey(),
  version: text("version").notNull(),
  url: text("url").notNull(),
  filename: text("filename").notNull(),
  fileSize: integer("file_size"),
  changelog: text("changelog"),
  createdAt: text("created_at"),
});

// ===================================================================
// Admin & coordination
// ===================================================================

export const adminAlerts = sqliteTable("admin_alerts", {
  id: text("id").primaryKey(),
  type: text("type").notNull(),
  severity: text("severity").default("warning"),
  message: text("message").notNull(),
  metadata: text("metadata"),
  status: text("status").default("active"),
  recurrenceCount: integer("recurrence_count").default(0),
  createdAt: text("created_at"),
  resolvedAt: text("resolved_at"),
});

export const coordSessions = sqliteTable("coord_sessions", {
  id: text("id").primaryKey(),
  userId: text("user_id").references(() => users.id),
  agentType: text("agent_type"),
  startedAt: text("started_at"),
  endedAt: text("ended_at"),
  status: text("status").default("active"),
  metadata: text("metadata"),
});

export const coordFileClaims = sqliteTable("coord_file_claims", {
  id: text("id").primaryKey(),
  sessionId: text("session_id").references(() => coordSessions.id),
  filePath: text("file_path").notNull(),
  claimType: text("claim_type").default("write"),
  claimedAt: text("claimed_at"),
  releasedAt: text("released_at"),
});

// ===================================================================
// Engagement & content
// ===================================================================

export const schedules = sqliteTable("schedules", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  cronExpr: text("cron_expr"),
  handler: text("handler").notNull(),
  enabled: integer("enabled").default(1),
  lastRunAt: text("last_run_at"),
  nextRunAt: text("next_run_at"),
  metadata: text("metadata"),
});

export const scheduleRuns = sqliteTable("schedule_runs", {
  id: text("id").primaryKey(),
  scheduleId: text("schedule_id").references(() => schedules.id),
  status: text("status").default("running"),
  startedAt: text("started_at"),
  finishedAt: text("finished_at"),
  error: text("error"),
  metadata: text("metadata"),
});

export const schemaVersion = sqliteTable("schema_version", {
  version: integer("version").notNull(),
});
