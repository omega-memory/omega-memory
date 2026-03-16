import type { ContentCategory, ParsedContent, GrantData, LeaderboardEntry } from "./types";
import { TYPE_VERBS } from "./types";

// ─── Parse Cache ────────────────────────────────────────────

const PARSE_CACHE_MAX = 256;
const _parseCache = new Map<string, ParsedContent>();
const _classifyCache = new Map<string, ContentCategory>();

function cachedGet<T>(cache: Map<string, T>, key: string, compute: () => T): T {
  const cached = cache.get(key);
  if (cached !== undefined) return cached;
  const result = compute();
  if (cache.size >= PARSE_CACHE_MAX) {
    const first = cache.keys().next().value;
    if (first !== undefined) cache.delete(first);
  }
  cache.set(key, result);
  return result;
}

// ─── Confidential Content Detection ────────────────────────

const CONFIDENTIAL_ADDRESS = /\b\d+\s+[\w\s]+\b(Dr|St|Ave|Rd|Blvd|Ln|Ct|Way|Pl|Drive|Street|Avenue|Road|Boulevard|Lane|Court|Place)\b/i;
const CONFIDENTIAL_SSN = /\b\d{3}-\d{2}-\d{4}\b/;
const CONFIDENTIAL_EIN = /\b\d{2}-\d{7}\b/;
const CONFIDENTIAL_EMAIL = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b/i;
const CONFIDENTIAL_PHONE = /\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/;
const CONFIDENTIAL_MONEY_LARGE = /\$\s?\d{3,}(?:,\d{3})*(?:\.\d{2})?(?:K|M|B)?\b/;
const CONFIDENTIAL_LEGAL = /\b(trust|estate|attorney|beneficiary|probate|back\s*pay|settlement|power\s+of\s+attorney|last\s+will)\b/i;

export function isConfidentialContent(
  content: string,
  metadata: Record<string, unknown> | null,
): boolean {
  if (metadata?.confidential === true) return true;

  if (CONFIDENTIAL_ADDRESS.test(content)) return true;
  if (CONFIDENTIAL_SSN.test(content)) return true;
  if (CONFIDENTIAL_EIN.test(content)) return true;
  if (CONFIDENTIAL_PHONE.test(content)) return true;
  if (CONFIDENTIAL_MONEY_LARGE.test(content)) return true;
  if (CONFIDENTIAL_LEGAL.test(content)) return true;

  const emailMatches = content.match(CONFIDENTIAL_EMAIL);
  if (emailMatches && emailMatches.length > 0) {
    const nonPersonal = /noreply@|no-reply@|notifications@|support@|github\.com|vercel\.com/i;
    if (emailMatches.some(e => !nonPersonal.test(e))) return true;
  }

  return false;
}

// ─── Grant Data Extraction ──────────────────────────────────

const GRANT_FUNDERS: Record<string, string> = {
  nlnet: "NLnet",
  "ngi zero": "NGI Zero",
  "ngi0": "NGI Zero",
  "goose grant": "Goose Grant",
  "sovereign tech": "Sovereign Tech Fund",
  stf: "STF",
  raais: "RAAIS",
};

export function parseGrantData(content: string): GrantData {
  const lower = content.toLowerCase();

  let status: GrantData["status"] = null;
  if (/\bsubmitted\b/i.test(content)) status = "submitted";
  else if (/\bapproved\b/i.test(content)) status = "approved";
  else if (/\brejected\b|declined\b/i.test(content)) status = "rejected";
  else if (/\bpending\b/i.test(content)) status = "pending";
  else if (/\bdraft\b/i.test(content)) status = "draft";

  let amount: string | null = null;
  const amtMatch = content.match(
    /(?:amount|funding|budget|requesting|grant)[:\s]*[$€]([\d,]+(?:\.\d+)?)\s*(K|M|B|USD|EUR)?/i
  ) || content.match(
    /[$€]\s?([\d,]+(?:\.\d+)?)\s*(K|M|B|USD|EUR)?/i
  );
  if (amtMatch) {
    const num = amtMatch[1].replace(/,/g, "");
    const suffix = amtMatch[2]?.toUpperCase() || "";
    let val = parseFloat(num);
    if (suffix === "K") val *= 1000;
    else if (suffix === "M") val *= 1000000;

    if (val >= 1000000) amount = `$${(val / 1000000).toFixed(val % 1000000 === 0 ? 0 : 1)}M`;
    else if (val >= 1000) amount = `$${(val / 1000).toFixed(val % 1000 === 0 ? 0 : 1)}K`;
    else amount = `$${val.toFixed(0)}`;

    if (suffix === "EUR" || content.includes("€")) amount = amount.replace("$", "€");
    if (suffix === "USD") amount += " USD";
  }

  let score: GrantData["score"] = null;
  const scoreMatch = content.match(/(?:score)[:\s]*(\d+(?:\.\d+)?)\s*\/\s*(\d+)/i)
    || content.match(/\b(\d+(?:\.\d+)?)\s*\/\s*(10)\b/);
  if (scoreMatch) {
    score = { value: parseFloat(scoreMatch[1]), max: parseInt(scoreMatch[2]) };
  }

  let funder: string | null = null;
  for (const [key, label] of Object.entries(GRANT_FUNDERS)) {
    if (lower.includes(key)) { funder = label; break; }
  }

  let duration: string | null = null;
  const durMatch = content.match(/(?:duration|period|timeline)[:\s]*([\d]+\s*(?:months?|years?|weeks?))/i);
  if (durMatch) duration = durMatch[1];

  let deadline: string | null = null;
  const dlMatch = content.match(/(?:deadline|due)[:\s]*(\w+\s+\d{1,2}(?:,?\s*\d{4})?)/i);
  if (dlMatch) deadline = dlMatch[1];

  return { status, amount, score, funder, duration, deadline };
}

// ─── Content Classification ────────────────────────────────

export function classifyContent(raw: string): ContentCategory {
  return cachedGet(_classifyCache, raw, () => _classifyContentImpl(raw));
}

function _classifyContentImpl(raw: string): ContentCategory {
  const lower = raw.toLowerCase();

  const isBenchmarkReport =
    /\b(longmemeval|benchmark\s+(?:v\d|run|results?|scores?|update|report))\b/i.test(raw) ||
    /^##\s+.*benchmark/im.test(raw);

  const hasStructuredMetrics =
    /\d+(?:\.\d+)?%\s*\(\d+\/\d+\)/.test(raw) ||
    /(?:overall|task[- ]averaged|accuracy|total)[:\s]+\d+(?:\.\d+)?%/i.test(raw) ||
    /\d+(?:\.\d+)?%\s+(?:overall|task[- ]averaged|accuracy|total)/i.test(raw);

  if (isBenchmarkReport && hasStructuredMetrics)
    return "benchmark";

  if (
    /## checkpoint/i.test(raw) ||
    (/^##\s/m.test(raw) &&
      /^###\s+(plan|progress|next\s*steps?|decisions|context|key)/im.test(raw))
  )
    return "checkpoint";

  if (
    /\d+\s*MCP\s*tools/i.test(raw) ||
    /~?[\d,]+K?\s*(?:lines?\s*(?:of\s*)?source|LOC)/i.test(raw) ||
    /[\d,]+\+?\s*passing\s*tests?/i.test(raw) ||
    /\d+\s*tools?\s*across/i.test(raw)
  )
    return "stats";

  if (
    /\bKvK\s*\d/i.test(raw) ||
    /\bRSIN\s*\d/i.test(raw) ||
    /^Entity:/im.test(raw) ||
    (/\b(stichting|zaidan)\b/i.test(lower) && /\b(registration|founded|jurisdiction)\b/i.test(lower))
  )
    return "entity";

  if (
    /\b(nlnet|ngi\s*zero|goose\s*grant|sovereign\s*tech|raais|stf)\b/i.test(raw) ||
    (/\bgrant\b/i.test(lower) && /\b(application|funding|proposal|deadline|submit|funder)\b/i.test(lower))
  )
    return "grant";

  if (/[$€]\s?\d[\d,]*(\.\d+)?/.test(raw) && /\bbudget\b/i.test(lower))
    return "financial";

  if (/\[due:/i.test(raw) || /\b(deadline|due date|due by)\b/i.test(lower))
    return "deadline";

  return "general";
}

// ─── Context-aware Dutch->English replacement ───────────────

export function contextAwareReplace(s: string, pattern: RegExp, english: string): string {
  return s.replace(pattern, (match, offset) => {
    const window = s.slice(Math.max(0, offset - 25), offset + match.length + 25).toLowerCase();
    const re = new RegExp(`\\b${english}\\b`, 'gi');
    const matches = window.match(re);
    return matches && matches.length > 0 ? '' : english;
  });
}

// ─── Humanize Bullet ────────────────────────────────────────

export function humanizeBullet(text: string): string {
  let s = text;
  s = s.replace(/^\s*[-*•]\s+/, "");
  s = s.replace(/^\|[\s|`\w%.,-]+\|\s*$/, "").trim();
  s = s.replace(/`([^`]+)`/g, "$1");
  s = s.replace(/\*\*([^*]+)\*\*/g, "$1");
  s = s.replace(/\b([a-z]+(?:_[a-z]+){1,})\b/g, (_m, id: string) => id.replace(/_/g, " "));
  s = s.replace(/(?:~\/|\/Users\/\S+\/|\/home\/\S+\/)(\S+)/g, (_m, p) => {
    const parts = p.split("/");
    return parts[parts.length - 1] || p;
  });
  s = s.replace(/\bsrc\/[\w/]+\/(\w+\.\w+)/g, "$1");
  s = s.replace(/\b\w+(?:\.\w+)*\(\)/g, "");
  s = s.replace(/\s--\w+(?:=\S+)?/g, "");
  s = s.replace(/\[(?:CAPTURED|BLOCKED|SKIPPED|TODO|FIXME)\]\s*/gi, "");
  s = s.replace(/\bpeer agent\b/gi, "collaborator");
  s = s.replace(/\bpy prompts?\b/gi, "test prompts");
  s = s.replace(/\bhandelsnaam\s*\(trade name\)/gi, "trade name");
  s = s.replace(/\bhandelsnaam\b/gi, "trade name");
  s = contextAwareReplace(s, /\bstichting\b/gi, "foundation");
  s = contextAwareReplace(s, /\bzaidan\b/gi, "foundation");
  s = contextAwareReplace(s, /\bkeish[oō]\b/gi, "succession");
  s = contextAwareReplace(s, /\bbestuur\b/gi, "board");
  s = contextAwareReplace(s, /\boprichting\b/gi, "incorporation");
  s = s.replace(/\blongmemeval[_\s]?official\b/gi, "memory benchmark");
  s = s.replace(/\bcoord_sessions?\b/gi, "sessions");
  s = s.replace(/\bhook_server\b/gi, "automation server");
  s = s.replace(/\btask_utils\b/gi, "task utilities");
  s = s.replace(/\b(?:embedding\s+)?cache\b/gi, "search index");
  s = s.replace(/\bvector\s+(?:store|index)\b/gi, "search index");
  s = s.replace(/\bsupabase\b/gi, "database");
  s = s.replace(/\bCRUD\b/g, "data operations");
  s = s.replace(/\bAPI\s+endpoint\b/gi, "service");
  s = s.replace(/\bwebhook\b/gi, "automation trigger");
  s = s.replace(/\bPR\b(?!\w)/g, "code review");
  s = s.replace(/\brefactored?\b/gi, "restructured");
  s = s.replace(/\bdeprecated\b/gi, "retired");
  s = s.replace(/\bmigration\b/gi, "upgrade");
  s = s.replace(/\bsingleton\b/gi, "shared instance");
  s = s.replace(/\bmutex\b/gi, "lock");
  s = s.replace(/\bCI\/CD\b/g, "build pipeline");
  s = s.replace(/\bpipeline\b/gi, "workflow");
  s = s.replace(/\bschema\b/gi, "structure");
  s = s.replace(/\bpayload\b/gi, "data");
  s = s.replace(/\bmiddleware\b/gi, "handler");
  s = s.replace(/\breg(?:ular\s+)?exp(?:ression)?\b/gi, "pattern");
  s = s.replace(/\bserialization\b/gi, "conversion");
  s = s.replace(/\bdeserialization\b/gi, "loading");
  s = s.replace(/\bdeployment\b/gi, "release");
  s = s.replace(/\bdeployed\b/gi, "published");
  s = s.replace(/\bdeploy\b/gi, "publish");
  s = s.replace(/\brebase\b/gi, "reorganize");
  s = s.replace(/\bstash(?:ed)?\b/gi, "set aside");
  s = s.replace(/\bupstream\b/gi, "source");
  s = s.replace(/\bdownstream\b/gi, "dependent");
  s = s.replace(/\bhotfix\b/gi, "urgent fix");
  s = s.replace(/\blinting?\b/gi, "code check");
  s = s.replace(/\bboilerplate\b/gi, "template code");
  s = s.replace(/\btech\s*debt\b/gi, "cleanup needed");
  s = s.replace(/\bbacklog\b/gi, "to-do list");
  s = s.replace(/\bblocke?r\b/gi, "obstacle");
  s = s.replace(/\bregression\b/gi, "broken behavior");
  s = s.replace(/\bedge\s*case\b/gi, "rare scenario");
  s = s.replace(/\brace\s*condition\b/gi, "timing conflict");
  s = s.replace(/\bdeadlock\b/gi, "stuck process");
  s = s.replace(/\bparsing\b/gi, "reading");
  s = s.replace(/\btokenization\b/gi, "processing");
  s = s.replace(/\bsanitiz(?:e|ing|ation)\b/gi, "cleaning");
  s = s.replace(/\bpolyfill\b/gi, "compatibility fix");
  s = s.replace(/\bshim\b/gi, "adapter");
  s = s.replace(/\bmonkeypatch(?:ed|ing)?\b/gi, "patched");
  s = s.replace(/\bstubbed\b/gi, "placeholder");
  s = s.replace(/\bmocking\b/gi, "simulating");
  s = s.replace(/\bmocked\b/gi, "simulated");
  s = s.replace(/\bORM\b/g, "data layer");
  s = s.replace(/\bCSRF\b/g, "security token");
  s = s.replace(/\bXSS\b/g, "injection risk");
  s = s.replace(/\bSSR\b/g, "server rendering");
  s = s.replace(/\bSSG\b/g, "static generation");
  s = s.replace(/\bISR\b/g, "incremental updates");
  s = s.replace(/\bJWT\b/g, "auth token");
  // LLM/AI jargon
  s = s.replace(/\bLLM\b/g, "AI model");
  s = s.replace(/\btoken\s+limit\b/gi, "size limit");
  s = s.replace(/\bcode\s+diffs?\b/gi, "code changes");
  s = s.replace(/\bJSON\s+strings?\b/gi, "data");
  s = s.replace(/\bJSON\b/g, "data");
  s = s.replace(/\btruncated\s+mid-string\b/gi, "cut off partway");
  s = s.replace(/\btruncated\b/gi, "cut off");
  s = s.replace(/\bomega_store\s+calls?\b/gi, "memory saves");
  s = s.replace(/\bworkflow_dispatch\b/gi, "automation trigger");
  // Strip commit hashes from detail text
  s = s.replace(/\bCommits?:\s*[a-f0-9]{6,10}\s*/gi, "");
  s = s.replace(/\b[a-f0-9]{7,10}\b/g, "");
  // DNS/email auth
  s = s.replace(/\bSPF\/DKIM\/DMARC\b/gi, "email authentication");
  s = s.replace(/\bDNS\s+fully\s+configured\b/gi, "domain verified");
  s = s.replace(/\bDNS\b/gi, "domain settings");
  // Session/coordination jargon
  s = s.replace(/\bsession\s+summary\b/gi, "work summary");
  s = s.replace(/\bauto-captured\s+session\b/gi, "automated session");
  s = s.replace(/\bGitHub\s+Actions\b/gi, "automation");

  s = s.replace(/\s*\(mem-[a-f0-9]+\)/gi, "");
  s = s.replace(/\bmem-[a-f0-9]{6,}\b/gi, "");
  s = s.replace(/\s*\([a-f0-9]{8,}\)/g, "");
  s = s.replace(/\b[a-f0-9]{12,}\b/g, "");
  s = s.replace(/\btrade name\s*\(trade name\)/gi, "trade name");
  s = s.replace(/\b(\w+)\s*\(\1\)/gi, "$1");
  s = s.replace(/\s+via\s+\S+\s+in\s+\S+\.py\b/gi, "");
  s = s.replace(/\(\s*\)/g, "");
  s = s.replace(/\s{2,}/g, " ");
  s = s.replace(/\s+([,.])/g, "$1");
  return s.trim();
}

// ─── Enhanced Content Parsing ──────────────────────────────

export function parseContentRich(content: string): ParsedContent {
  return cachedGet(_parseCache, content, () => _parseContentRichImpl(content));
}

function _parseContentRichImpl(content: string): ParsedContent {
  let text = content;
  let dueDate: Date | null = null;

  const dueMatch = text.match(/\[due:\s*([^\]]+)\]/i);
  if (dueMatch) {
    const d = new Date(dueMatch[1].trim());
    if (!isNaN(d.getTime())) dueDate = d;
    text = text.replace(dueMatch[0], "").trim();
  }

  text = text.replace(/\[[\w_]+:\s*[^\]]*\]/g, "").trim();

  const category = classifyContent(content);

  // Key-value extraction
  const keyValues: { key: string; value: string }[] = [];
  const kvkMatch = content.match(/KvK\s*(\d[\d\s]*\d)/i);
  if (kvkMatch) keyValues.push({ key: "KvK", value: kvkMatch[1].trim() });
  const rsinMatch = content.match(/RSIN\s*(\d[\d\s]*\d)/i);
  if (rsinMatch) keyValues.push({ key: "RSIN", value: rsinMatch[1].trim() });
  for (const m of content.matchAll(
    /([$€])([\d,]+[KMB]?(?:\s*[-–]\s*(?:[$€])?[\d,]+[KMB]?)?(?:\+)?)/gi
  )) {
    keyValues.push({ key: "Amount", value: `${m[1]}${m[2]}` });
  }

  // Metrics extraction
  const metrics: { label: string; value: string; unit?: string }[] = [];
  const seenMetrics = new Set<string>();

  for (const m of content.matchAll(
    /([\w][\w\s-]{0,28}?):\s*(\d+(?:\.\d+)?)\s*%/g
  )) {
    const label = m[1].trim();
    if (/^\d/.test(label) || label.length < 2) continue;
    const key = `${label.toLowerCase()}:${m[2]}`;
    if (!seenMetrics.has(key)) {
      metrics.push({ label, value: m[2], unit: "%" });
      seenMetrics.add(key);
    }
  }

  for (const m of content.matchAll(
    /\b([a-zA-Z][\w-]{2,}(?:\s+[\w-]+){0,2}?)\s+(\d+(?:\.\d+)?)\s*%/g
  )) {
    const label = m[1].trim();
    const labelLower = label.toLowerCase();
    const isMetricLabel =
      /\b(accuracy|precision|recall|pass|score|rate|coverage|abstention|overall|total|average|task|retrieval|knowledge|temporal|preference|multi|session|ss-|su-)\b/i.test(labelLower) ||
      /^[A-Z]{2,}[-_]?/.test(label);
    if (!isMetricLabel) continue;
    const key = `${labelLower}:${m[2]}`;
    if (!seenMetrics.has(key)) {
      metrics.push({ label, value: m[2], unit: "%" });
      seenMetrics.add(key);
    }
  }

  for (const m of content.matchAll(
    /(\d+(?:\.\d+)?)\s*%\s+(overall|total|average|task[- ]averaged|accuracy|recall|precision|passing|abstention)/gi
  )) {
    const label = m[2].charAt(0).toUpperCase() + m[2].slice(1);
    const key = `${label.toLowerCase()}:${m[1]}`;
    if (!seenMetrics.has(key)) {
      metrics.push({ label, value: m[1], unit: "%" });
      seenMetrics.add(key);
    }
  }

  for (const m of content.matchAll(
    /(\d+(?:\.\d+)?)\s*%\s*\((\d+)\s*\/\s*(\d+)\)/g
  )) {
    const key = `score:${m[1]}`;
    if (!seenMetrics.has(key)) {
      metrics.push({ label: `Score (${m[2]}/${m[3]})`, value: m[1], unit: "%" });
      seenMetrics.add(key);
    }
  }

  const toolsMatch = content.match(/(\d+)\s*MCP\s*tools/i);
  if (toolsMatch) metrics.push({ label: "MCP Tools", value: toolsMatch[1] });
  const acrossMatch = content.match(/(\d+)\s*tools?\s*across\s*(\d+)\s*categor/i);
  if (acrossMatch) {
    metrics.push({ label: "Tools", value: acrossMatch[1] });
    metrics.push({ label: "Categories", value: acrossMatch[2] });
  }
  const srcMatch = content.match(/~?([\d,]+K?)\s*lines?\s*(?:of\s*)?source/i);
  if (srcMatch) metrics.push({ label: "Source Lines", value: srcMatch[1] });
  const testLinesMatch = content.match(/~?([\d,]+K?)\s*lines?\s*tests?/i);
  if (testLinesMatch) metrics.push({ label: "Test Lines", value: testLinesMatch[1] });
  if (!srcMatch) {
    const locMatch = content.match(/~?([\d,]+K)\s*(?:src|source)?\s*(?:\+\s*~?[\d,]+K\s*test\s*)?LOC/i);
    if (locMatch) metrics.push({ label: "Lines of Code", value: locMatch[1] });
  }
  const testsMatch = content.match(/([\d,]+)\+?\s*passing\s*tests?/i);
  if (testsMatch) metrics.push({ label: "Passing Tests", value: testsMatch[1] });

  // Section extraction (### headings)
  const sections: { heading: string; body: string }[] = [];
  const sectionParts = content.split(/^###\s+/m);
  if (sectionParts.length > 1) {
    for (let i = 1; i < sectionParts.length; i++) {
      const nlIdx = sectionParts[i].indexOf("\n");
      if (nlIdx > 0) {
        sections.push({
          heading: sectionParts[i].slice(0, nlIdx).trim(),
          body: sectionParts[i].slice(nlIdx + 1).trim(),
        });
      } else {
        sections.push({ heading: sectionParts[i].trim(), body: "" });
      }
    }
  }

  // Bullet extraction
  const bullets: string[] = [];
  for (const line of content.split("\n")) {
    const bm = line.match(/^\s*[-*]\s+(.+)/);
    if (bm) bullets.push(bm[1].trim());
  }

  // Title extraction (priority order)
  let title = text;
  let detail: string | null = null;

  const h2Match = text.match(/^##\s+(.+)/m);
  if (h2Match) {
    title = h2Match[1].trim();
    const afterHeader = text.slice(text.indexOf(h2Match[0]) + h2Match[0].length).trim();
    if (afterHeader) detail = afterHeader;
  } else {
    let found = false;
    for (const sep of [" \u2014 ", " \u2013 "]) {
      const idx = text.indexOf(sep);
      if (idx > 0 && idx < 140) {
        title = text.slice(0, idx).trim();
        detail = text.slice(idx + sep.length).trim();
        found = true;
        break;
      }
    }
    if (!found) {
      const colonIdx = text.indexOf(":");
      if (colonIdx > 0 && colonIdx < 60) {
        const candidate = text.slice(0, colonIdx).trim();
        if (candidate.length >= 12 || !text.slice(colonIdx + 1).trim()) {
          title = candidate;
          detail = text.slice(colonIdx + 1).trim();
          found = true;
        }
      }
    }
    if (!found && text.length > 120) {
      const sentEnd = text.search(/[.\n]/);
      if (sentEnd > 0 && sentEnd < 120) {
        title = text.slice(0, sentEnd).trim();
        detail = text.slice(sentEnd + 1).trim();
      }
    }
  }

  // Title cleanup: git commit hashes
  title = title.replace(/^Committed\s+[a-fA-F0-9]{6,10}:\s*/i, "");
  title = title.replace(/\b[a-fA-F0-9]{7,10}\b:\s*/, "");
  // Strip conventional-commit prefix (feat:, fix:, chore:, etc.)
  title = title.replace(/^(?:feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert)(?:\([^)]*\))?:\s*/i, "");
  // Strip "- N files modified/changed" suffixes
  title = title.replace(/\s*[-–—]\s*\d+\s+(?:files?\s+(?:modified|changed|updated)|tasks?\s+completed)\s*$/i, "");
  // Strip "Fact:" prefix
  title = title.replace(/^Fact:\s*/i, "");
  // Strip raw XML tags
  title = title.replace(/<[^>]+>/g, "").trim();
  // Strip "Audit X - N files modified" → just "Audit X"
  title = title.replace(/\s*[-–—]\s*\d+\s+files?\b.*$/i, "");

  // Title cleanup: bracketed prefixes like [auth/oauth]
  title = title.replace(/^\[[\w/.-]+\]\s*/, "");

  // Title cleanup: consolidated memory prefix
  title = title.replace(/^\[Consolidated from (\d+) memories?\]\s*/i, (_m, n) => {
    // We'll append the note after title is finalized
    return "";
  });
  const consolidatedMatch = content.match(/^\[Consolidated from (\d+) memories?\]/i);

  if (title.trimStart().startsWith("{") || title.includes('"filePath"')) {
    const oldStringMatch = content.match(/"oldString"\s*:\s*"([^"]{1,60})/);
    if (oldStringMatch) {
      title = `Code edit: ${oldStringMatch[1].trim()}...`;
    } else {
      title = "Code change captured";
    }
    detail = null;
  }

  if (/^\/(?:Users|home|tmp|var)\//i.test(title)) {
    const filename = title.split("/").pop() || title;
    title = `File: ${filename}`;
  }

  if (/_/.test(title) && !/\s/.test(title.trim())) {
    title = title.replace(/_/g, " ");
  }

  title = title.replace(/^Check if\s+/i, "Review: ");
  title = title.replace(/^(?:DEADLINE|REMINDER|NOTE|ACTION|TODO|UPDATE|IMPORTANT|URGENT|FYI|TASK|DECISION|LESSON|ERROR|WARNING|INFO|MEMORY)\s*[:\-–—]\s*/i, "");
  // Strip coordination noise prefixes
  title = title.replace(/^(?:Pre-edit guard claim|Auto-claimed on edit|File claim|Hook invocation)\s*[-–—:]\s*/i, "");
  // Strip raw URLs, keep just the path context
  title = title.replace(/https?:\/\/[^\s]+/g, (url) => {
    try {
      const u = new URL(url);
      const params = u.searchParams.get("tab");
      return params ? params : u.pathname.split("/").pop() || "";
    } catch { return ""; }
  });
  title = title.replace(/^\s*[-*•]\s+/, "");
  title = title.replace(/\s*[(\[]\s*P\d\s*[)\]]\s*/gi, " ");
  title = title.replace(/^\|[\s|`\w%.,-]+\|$/, "").trim();
  title = title.replace(/`/g, "");

  title = title.replace(/\blongmemeval[_\s]?official\b/gi, "memory benchmark");
  title = title.replace(/\bLongMemEval\s*benchmark\b/i, "Memory Benchmark");
  title = title.replace(/\bhandelsnaam\s*\(trade name\)/gi, "trade name");
  title = title.replace(/\bhandelsnaam\b/gi, "trade name");
  title = contextAwareReplace(title, /\bstichting\b/gi, "foundation");
  title = contextAwareReplace(title, /\bzaidan\b/gi, "foundation");
  title = contextAwareReplace(title, /\bkeish[oō]\b/gi, "succession");
  title = contextAwareReplace(title, /\bbestuur\b/gi, "board");
  title = contextAwareReplace(title, /\boprichting\b/gi, "incorporation");
  title = title.replace(/\bpeer agent\b/gi, "collaborator");
  title = title.replace(/\bpy prompts?\b/gi, "test prompts");
  title = title.replace(/\btrade name\s*\(trade name\)/gi, "trade name");
  title = title.replace(/\b(\w+)\s*\(\1\)/gi, "$1");

  title = title.replace(/\brefactored?\b/gi, "restructured");
  title = title.replace(/\bdeprecated\b/gi, "retired");
  title = title.replace(/\bCI\/CD\b/g, "build workflow");
  title = title.replace(/\bhotfix\b/gi, "urgent fix");
  title = title.replace(/\blinting?\b/gi, "code check");
  title = title.replace(/\btech\s*debt\b/gi, "cleanup");
  title = title.replace(/\bblocke?r\b/gi, "obstacle");
  title = title.replace(/\bregression\b/gi, "broken behavior");
  title = title.replace(/\bedge\s*case\b/gi, "rare scenario");
  title = title.replace(/\brace\s*condition\b/gi, "timing conflict");
  title = title.replace(/\bdeadlock\b/gi, "stuck process");
  title = title.replace(/\bsingleton\b/gi, "shared instance");
  title = title.replace(/\bAPI\s+endpoint\b/gi, "service");
  title = title.replace(/\bwebhook\b/gi, "trigger");
  title = title.replace(/\bmiddleware\b/gi, "handler");
  title = title.replace(/\bsupabase\b/gi, "database");
  title = title.replace(/\bpayload\b/gi, "data");
  title = title.replace(/\bschema\b/gi, "structure");
  title = title.replace(/\bboilerplate\b/gi, "template");
  title = title.replace(/\bPR\b(?!\w)/g, "review");
  // More dev → human translations
  title = title.replace(/\baudit\s+entities\b/gi, "Review contacts");
  title = title.replace(/\bentities\b/gi, "contacts");
  title = title.replace(/\bsession\s+recap\b/gi, "Work summary");
  title = title.replace(/\bsession\s+summary\b/gi, "Work summary");
  title = title.replace(/\bcron\s+job\b/gi, "scheduled task");
  title = title.replace(/\benv\s+var(?:iable)?s?\b/gi, "settings");

  title = title.replace(/\s*\(mem-[a-f0-9]+\)/gi, "");
  title = title.replace(/\bmem-[a-f0-9]{6,}\b/gi, "");
  title = title.replace(/\s*\([a-f0-9]{8,}\)/g, "");
  title = title.replace(/\b[a-f0-9]{12,}\b/g, "");

  title = title.replace(/^Review:\s*/i, "");
  title = title.replace(/^(?:Pick|Choose|Select)\s+(?:a\s+)?/i, "");
  title = title.replace(/^Entity:\s*/i, "");
  title = title.replace(/^OMEGA\s+/i, (match) => {
    return category !== "general" ? "" : match;
  });

  if (/^\w+_\w+/.test(title) && title.indexOf(" ") === -1) {
    title = title.replace(/_/g, " ");
  }

  if (title.length > 0 && title[0] === title[0].toLowerCase() && /^[a-z]/.test(title)) {
    title = title[0].toUpperCase() + title.slice(1);
  }

  if (title === title.toUpperCase() && title.length > 4) {
    title = title
      .toLowerCase()
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }
  title = title.replace(/^([A-Z]{4,})\b/, (_m, word: string) => word.charAt(0) + word.slice(1).toLowerCase());

  const KNOWN_ACRONYMS = new Set(["OMEGA", "AI", "MCP", "API", "KB", "RSIN", "KVK", "OIDC", "CLI", "UI", "PR"]);
  title = title.replace(/\b([A-Z]{4,})\b/g, (_m, word: string) =>
    KNOWN_ACRONYMS.has(word) ? word : word.charAt(0) + word.slice(1).toLowerCase()
  );

  if (title === title.toLowerCase() && title.length > 4) {
    title = title.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  const GENERIC_TITLES = new Set([
    // Event type names (existing)
    'decision', 'lesson', 'insight', 'preference', 'error', 'session',
    'task', 'reminder', 'note', 'checkpoint', 'benchmark',
    'user preference', 'error pattern', 'lesson learned',
    'session summary', 'task completion', 'session completed',
    // Internal coordination noise
    'coding', 'pre-edit guard claim', 'auto-claimed on edit',
    'file claim', 'hook invocation', 'assistant decision',
    'files modified', 'task notification',
    'fact', 'status update', 'status',
    // Short/vague titles that need detail promotion
    'assistant fix', 'assistant decision', 'fix', 'update',
    'improvement', 'change', 'changes', 'bug fix', 'bugfix',
    'work session', 'work completed', 'task done',
    // Section headings from X Briefs / reports
    'ai agent memory & coordination',
    'ai agent memory and coordination',
    'omega / @jasonsosa mentions',
    'x brief',
    // Project names
    'omegamax', 'element1', 'kokyo', 'omega',
  ]);
  if (GENERIC_TITLES.has(title.toLowerCase().trim()) && detail) {
    // Humanize the detail before promoting to title
    const humanDetail = humanizeBullet(detail);
    const sentEnd = humanDetail.search(/[.\n]/);
    if (sentEnd > 0 && sentEnd < 60) {
      title = humanDetail.slice(0, sentEnd).trim();
      detail = humanDetail.slice(sentEnd + 1).trim() || null;
    } else if (humanDetail.length <= 55) {
      title = humanDetail;
      detail = null;
    } else {
      const cut = humanDetail.slice(0, 55);
      const ls = cut.lastIndexOf(' ');
      title = (ls > 20 ? cut.slice(0, ls) : cut).trimEnd() + '...';
    }
  }
  if (GENERIC_TITLES.has(title.toLowerCase().trim()) && !detail && bullets.length > 0) {
    const firstBullet = humanizeBullet(bullets[0]);
    if (firstBullet.length <= 55) {
      title = firstBullet;
    } else {
      const cut = firstBullet.slice(0, 55);
      const ls = cut.lastIndexOf(' ');
      title = (ls > 20 ? cut.slice(0, ls) : cut).trimEnd() + '...';
    }
  }

  if (title.length > 55) {
    const truncated = title.slice(0, 55);
    const lastSpace = truncated.lastIndexOf(" ");
    title = (lastSpace > 20 ? truncated.slice(0, lastSpace) : truncated).trimEnd() + "...";
  }

  if (detail) {
    detail = detail.replace(/^\s*[-*•]\s+/, "");
    detail = detail.replace(/^\|[^|]+(?:\|[^|]+)*\|\s*/, "");
    detail = detail.replace(/^(?:DEADLINE|REMINDER|NOTE|ACTION|TODO|UPDATE|STATUS)\s*[:\-–—]\s*/i, "");
    // Strip DNS/email auth jargon
    detail = detail.replace(/\bSPF\/DKIM\/DMARC\b/gi, "email authentication");
    detail = detail.replace(/\bDNS\s+fully\s+configured\b/gi, "domain verified");
    detail = detail.replace(/\bDNS\b/gi, "domain");
    detail = detail.replace(/\bSPF\b/g, "email auth");
    detail = detail.replace(/\bDKIM\b/g, "email signing");
    detail = detail.replace(/\bDMARC\b/g, "email policy");
    // If detail just repeats the title, drop it
    if (detail && title && detail.toLowerCase().startsWith(title.toLowerCase().replace(/\.{3}$/, ""))) {
      detail = null;
    }
    detail = detail?.trim() || null;
  }

  title = title.replace(/\s{2,}/g, " ").trim();

  // Append consolidated note
  if (consolidatedMatch && title.length < 50) {
    title = `${title} (consolidated from ${consolidatedMatch[1]})`;
  }

  return { title, detail, dueDate, category, keyValues, metrics, sections, bullets };
}

// ─── Summary Detail ─────────────────────────────────────────

export function summarizeDetail(
  text: string,
  parsed: ParsedContent
): string[] {
  let cleaned = text;
  cleaned = cleaned.replace(/\[[\w_]+:\s*[^\]]*\]/g, "");
  cleaned = cleaned.replace(/^##\s+.+$/gm, "");

  const sentences = cleaned
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .map((s) => s.replace(/^\s*[-*•]\s+/, ""))
    .filter((s) => !/^\|[\s|`\w%.,-]+\|$/.test(s))
    .map((s) => s.replace(/\.+$/, ""))
    .filter((s) => s.length > 15);

  const titleLower = parsed.title.toLowerCase();
  return sentences.filter((s) => {
    const sLower = s.toLowerCase();
    if (sLower === titleLower || titleLower.includes(sLower)) return false;
    if (/^\s*(cd |python3? |npm |git |bash |sh |curl |wget )/i.test(s)) return false;
    if (/\s&&\s/.test(s) && /\.(py|sh|js)\b/.test(s)) return false;
    if (/^\s*\/(?:Users|home|tmp|var|etc)\//i.test(s)) return false;
    if (/^\s*src\/[\w/]+\.\w+/i.test(s)) return false;
    if ((s.match(/--\w+/g) || []).length >= 2) return false;
    if (/^(Do NOT|NEVER|DON'T|Always check|Make sure to)\b/i.test(s.trim())) return false;
    if (/^\(\d+\)\s/.test(s.trim()) && /\b(splitter|parser|extractor|handler)\b/i.test(s)) return false;
    if (/\b(coord_sessions|hook_server|task_utils|clean_task_text)\b/.test(s)) return false;
    if (/\d+\s+files?\s+(changed|modified|added|deleted|updated)/i.test(s)) return false;
    if (/^\s*(import |from |class |def |export |const |function |async function )/i.test(s)) return false;
    if (/\b(merge conflict|rebase|cherry[- ]pick|force[- ]push|git pull|git merge)\b/i.test(s)) return false;
    if (/\bupdated?\s+\S+\s+to\s+v?\d+\.\d+/i.test(s)) return false;
    if (/\b(cache rebuild|index rebuild|rebuilding\s+(cache|index|embeddings?))\b/i.test(s)) return false;
    if (/\b(mutex|semaphore|thread[- ]?pool|garbage\s+collect|heap\s+alloc)\b/i.test(s)) return false;
    if (/\b(docker|kubernetes|k8s|helm|terraform|ansible)\b/i.test(s) && s.length < 60) return false;
    if (/^\s*\w+\s*=\s*\S+/.test(s)) return false;
    if (/\b(SIGTERM|SIGKILL|SIGHUP|errno|exit\s+code)\b/i.test(s)) return false;
    return true;
  });
}

// ─── Rich Content Helpers ────────────────────────────────

export function detectKeyValueBullets(text: string): { key: string; value: string }[] {
  const lines = text.split('\n');
  const kvItems: { key: string; value: string }[] = [];
  for (const line of lines) {
    const match = line.match(/^\s*[-*•]\s*([^:]{2,35}):\s+(.+)/);
    if (match) {
      kvItems.push({ key: humanizeBullet(match[1].trim()), value: humanizeBullet(match[2].trim()) });
    }
  }
  return kvItems;
}

export function splitNumberedContent(text: string): string[] {
  const matches = text.match(/\(\d+\)\s+[^(]+/g);
  if (matches && matches.length >= 2) {
    return matches.map(m => humanizeBullet(m.replace(/^\(\d+\)\s+/, '').replace(/[;.]\s*$/, '').trim()));
  }
  return [];
}

export function detectSections(text: string): { heading: string | null; lines: string[] }[] {
  const allLines = text.split('\n');
  const sections: { heading: string | null; lines: string[] }[] = [];
  let current: { heading: string | null; lines: string[] } = { heading: null, lines: [] };

  for (const line of allLines) {
    const trimmed = line.trim();
    if (!trimmed) {
      current.lines.push(line);
      continue;
    }

    const isSectionHeader =
      trimmed.length < 80 &&
      /:\s*$/.test(trimmed) &&
      !/^\s*[-*•]/.test(trimmed) &&
      !/^##/.test(trimmed) &&
      !/^[^:]{2,35}:\s+\S/.test(trimmed);

    if (isSectionHeader) {
      if (current.heading !== null || current.lines.some(l => l.trim())) {
        sections.push(current);
      }
      const heading = trimmed.replace(/:\s*$/, '').trim();
      current = { heading, lines: [] };
    } else {
      current.lines.push(line);
    }
  }

  if (current.heading !== null || current.lines.some(l => l.trim())) {
    sections.push(current);
  }

  return sections;
}

// ─── Content Stats ──────────────────────────────────────────

export function extractContentStats(content: string, parsed: ParsedContent): string | null {
  const bulletCount = parsed.bullets.length;
  const sectionCount = parsed.sections.length;
  const metricCount = parsed.metrics.length;

  const contactMatches = content.match(/\b[A-Z][a-z]+\s+[A-Z][a-z]+\s*\([^)]+\)/g);
  const contactCount = contactMatches?.length || 0;

  const tierMatches = content.match(/\bTIER\s*\d/gi);
  const tierCount = tierMatches?.length || 0;

  const fileMatches = content.match(/\b[\w/]+\.(py|ts|tsx|js|md|json|yaml|sql)\b/g);
  const fileCount = fileMatches ? new Set(fileMatches).size : 0;

  const actionMatches = content.match(/\s—\s/g);
  const actionCount = actionMatches?.length || 0;

  const detectedSections = detectSections(content).filter(s => s.heading !== null);
  const kvCount = detectKeyValueBullets(content).length;

  const parts: string[] = [];
  if (contactCount >= 3) parts.push(`${contactCount} contacts`);
  if (tierCount >= 2) parts.push(`${tierCount} tiers`);
  if (fileCount >= 3) parts.push(`${fileCount} files`);
  if (actionCount >= 3 && !contactCount) parts.push(`${actionCount} action items`);
  if (metricCount >= 2 && parts.length === 0) parts.push(`${metricCount} metrics`);

  if (bulletCount > 8 && parts.length === 0) {
    const detailCount = kvCount > 0 ? kvCount : bulletCount;
    parts.push(`${detailCount} key details`);
    if (detectedSections.length >= 2) parts.push(`${detectedSections.length} sections`);
  } else {
    if (sectionCount >= 2 && parts.length === 0) parts.push(`${sectionCount} sections`);
    if (detectedSections.length >= 2 && parts.length === 0) parts.push(`${detectedSections.length} sections`);
    if (bulletCount > 5 && parts.length === 0) parts.push(`${bulletCount} items`);
  }

  return parts.length > 0 ? parts.join(", ") : null;
}

// ─── Executive Summary Generator ────────────────────────────

export function generateExecutiveSummary(
  parsed: ParsedContent,
  eventType: string | null,
  content: string
): string | null {
  const verb = eventType ? TYPE_VERBS[eventType] || null : null;
  const detailText = parsed.detail || content;

  if (eventType === 'benchmark_update' || parsed.category === 'benchmark') {
    const metricMatch = content.match(/(\d+(?:\.\d+)?)\s*%\s*(overall|accuracy|total|task[- ]averaged)/i);
    if (metricMatch) {
      const changeMatch = content.match(/([+-]\d+(?:\.\d+)?)\s*(?:pp|percentage)/i);
      const change = changeMatch ? `, ${changeMatch[1]}pp` : '';
      return `${metricMatch[1]}% ${metricMatch[2].toLowerCase()}${change}`;
    }
  }

  if (eventType === 'decision') {
    const decisionPatterns = [
      /(?:decided?|decision)[:\s]+(.{10,120}?)(?:\.|$)/im,
      /(?:will|going to|chose to|opting for)[:\s]+(.{10,120}?)(?:\.|$)/im,
    ];
    for (const pattern of decisionPatterns) {
      const match = content.match(pattern);
      if (match) {
        let statement = humanizeBullet(match[1].trim());
        if (statement.length > 100) {
          const cut = statement.slice(0, 100);
          const ls = cut.lastIndexOf(' ');
          statement = (ls > 40 ? cut.slice(0, ls) : cut).trimEnd() + '...';
        }
        return statement;
      }
    }
  }

  if (eventType === 'task_completion') {
    const taskPatterns = [
      /(?:completed?|finished|done)[:\s]+(.{10,120}?)(?:\.|$)/im,
      /(?:implemented|added|fixed|resolved|created|built)[:\s]+(.{10,120}?)(?:\.|$)/im,
    ];
    for (const pattern of taskPatterns) {
      const match = content.match(pattern);
      if (match) {
        let statement = humanizeBullet(match[1].trim());
        if (statement.length > 100) {
          const cut = statement.slice(0, 100);
          const ls = cut.lastIndexOf(' ');
          statement = (ls > 40 ? cut.slice(0, ls) : cut).trimEnd() + '...';
        }
        return statement;
      }
    }
  }

  const cleanContent = detailText
    .replace(/^[-*•]\s*[^:]{2,35}:\s+.+$/gm, '')
    .replace(/\[[\w_]+:\s*[^\]]*\]/g, '')
    .replace(/^##?\s+.+$/gm, '')
    .trim();

  const cleanDetail = cleanContent.replace(/^\s*[-*•]\s+/, "").trim();
  if (cleanDetail.length > 5 && cleanDetail.length <= 100 && !/\n/.test(cleanDetail) && !/^\s*[-|#]/.test(cleanDetail)) {
    const humanized = humanizeBullet(cleanDetail);
    if (humanized.length > 5) {
      const titleWords = new Set(parsed.title.toLowerCase().split(/\s+/));
      const dWords = humanized.toLowerCase().split(/\s+/);
      const overlap = dWords.filter((w) => titleWords.has(w)).length / Math.max(dWords.length, 1);
      if (overlap < 0.6) {
        const prefixed = verb ? `${verb}: ${humanized}` : humanized;
        return prefixed.length > 120 ? prefixed.slice(0, 117).trimEnd() + '...' : prefixed;
      }
    }
  }

  const sentences = summarizeDetail(detailText, parsed);
  if (sentences.length === 0) return null;

  const titleWords = new Set(parsed.title.toLowerCase().split(/\s+/));
  let best: string | null = null;
  for (const s of sentences.slice(0, 3)) {
    const sWords = s.toLowerCase().split(/\s+/);
    const overlap = sWords.filter((w) => titleWords.has(w)).length / Math.max(sWords.length, 1);
    if (overlap < 0.6) {
      best = s;
      break;
    }
  }
  if (!best) best = sentences.length > 1 ? sentences[1] : sentences[0];

  let summary = humanizeBullet(best);

  if (summary.length > 100) {
    const truncated = summary.slice(0, 100);
    const lastSpace = truncated.lastIndexOf(" ");
    summary = (lastSpace > 40 ? truncated.slice(0, lastSpace) : truncated).trimEnd() + "...";
  }

  if (verb) summary = `${verb}: ${summary}`;

  return summary.length > 120 ? summary.slice(0, 117).trimEnd() + '...' : summary;
}

// ─── Leaderboard Extraction ─────────────────────────────────

const KNOWN_COMPETITORS: Omit<LeaderboardEntry, "rank">[] = [
  { name: "Mastra", score: 94.87, isOmega: false },
  { name: "Zep / Graphiti", score: 71.2, isOmega: false },
  { name: "No Memory Baseline", score: 49.6, isOmega: false },
];

export function parseLeaderboard(raw: string, omegaScore: number | null): LeaderboardEntry[] | null {
  if (!/longmemeval|memory\s*benchmark/i.test(raw) && !/benchmark\s+v\d/i.test(raw)) return null;
  if (omegaScore === null) return null;

  const entries: LeaderboardEntry[] = [];
  const seen = new Set<string>();

  const competitorPattern = /([\w]+(?:\s+[\w]+)?)\s*#(\d+)\s*(?:at|with|:)?\s*(\d+(?:\.\d+)?)%/gi;
  let match;
  while ((match = competitorPattern.exec(raw)) !== null) {
    const name = match[1].trim();
    if (name.toLowerCase() === "omega") continue;
    if (!seen.has(name.toLowerCase())) {
      entries.push({ rank: 0, name, score: parseFloat(match[3]), isOmega: false });
      seen.add(name.toLowerCase());
    }
  }

  for (const known of KNOWN_COMPETITORS) {
    if (!seen.has(known.name.toLowerCase()) && !seen.has(known.name.split(" ")[0].toLowerCase())) {
      entries.push({ rank: 0, name: known.name, score: known.score, isOmega: false });
    }
  }

  entries.push({ rank: 0, name: "OMEGA", score: omegaScore, isOmega: true });

  entries.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
  entries.forEach((e, i) => { e.rank = i + 1; });

  return entries.length >= 2 ? entries : null;
}
