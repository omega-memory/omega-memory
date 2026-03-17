"""
OMEGA Protocol — JIT coordination instructions served dynamically.

Instead of static CLAUDE.md rules loaded every turn, this module provides
context-sensitive protocol sections that are served on-demand via the
omega_protocol() MCP tool.

Architecture:
- Base protocol sections live here as versioned, structured data
- OMEGA memories tagged event_type="protocol" augment with learned lessons
- The get_protocol() function assembles the right sections based on context
"""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("omega.protocol")

# Protocol version — bump when sections change materially
PROTOCOL_VERSION = "1.8.0"

# ---------------------------------------------------------------------------
# Protocol Sections — each is a (title, content) pair
# ---------------------------------------------------------------------------

SECTIONS: Dict[str, Dict[str, str]] = {
    "memory": {
        "title": "Memory Usage",
        "content": """\
- **Load user profile**: Call `omega_profile()` at session start (after welcome/protocol) to load working style preferences. Update via `omega_profile(action="update", update={"key": "value"})` when you learn new preferences.
- `[OMEGA MEMORY]` blocks from hooks = ground truth, use immediately
- ALWAYS call `omega_query()` before non-trivial tasks — check for prior context, decisions, gotchas
- **Before suggesting**: Before recommending any action, URL, tool, or "next step", query OMEGA for prior decisions on that topic. If you don't, you WILL repeat rejected ideas and fabricate details.
- **Reconcile at session start**: When `[CONTEXT]` blocks from hooks contradict MEMORY.md, newest-timestamped OMEGA source wins. Reconcile BEFORE starting work.
- **After** completing tasks: `omega_store(content, "decision"|"lesson_learned")` for key outcomes
- Sessions with 0 `omega_store` calls trigger an auto-safety-net at session end — manual stores produce higher-quality captures. Aim for minimum 1 manual store per session with meaningful work.
- **On errors**: check OMEGA for prior solutions before debugging from scratch
- **User says "remember"**: `omega_store(text, "user_preference")`
- When asked about preferences/history: query OMEGA FIRST, don't say "I don't know"
- **Feedback loop**: After `omega_query` surfaces memories with IDs, use `omega_feedback(memory_id, "helpful"|"unhelpful"|"outdated")` to train the ranker
- **Graph intelligence**: Use `omega_traverse(memory_id, max_hops=2)` to walk relationship graphs. Use `omega_similar(memory_id)` to discover related clusters
- **Cross-session lessons**: Auto-surface as [LESSON] cards on file edits. No manual action needed.
- **Memory editing**: Prefer `omega_edit_memory(memory_id, new_content)` over delete+recreate
- **Always tag stores**: Pass `project` (cwd path) and `entity_id` when calling `omega_store`. Without tags, memories can't be scoped or filtered later. Entity mappings: personal-assistant=personal, omega=omega, element1=element1
- **Build the graph**: After `omega_store` returns a memory ID, call `omega_memory(action="similar", memory_id=<id>)`. If related memories exist (score > 0.7), call `omega_memory(action="link", memory_id=<id>, target_id=<similar_id>)` with edge_type: `evolves` (same topic, updated understanding), `related` (cross-topic), or `supersedes` (replacement). Skip for `session_summary` and `checkpoint` types.
- **Use event_type precisely**: decision (choices made), lesson_learned (insights), error_pattern (bugs), user_preference (user requests). Default "memory" is low-signal.""",
    },
    "coordination": {
        "title": "Multi-Agent Coordination",
        "content": """\
- **Assume multi-agent**: Even if you see `Mode: solo`, other agents may start at any time. Shared-repo discipline applies always: never `git add .`, never commit files you didn't modify, never assume uncommitted files are yours.
- `[HANDOFF]` blocks = predecessor's work summary, continue where they left off
- `[INBOX]` alerts = peer messages waiting, check `omega_inbox` for details
- `[DEADLOCK]` alerts = circular wait, release a file with `omega_file_release`
- `[COORD]` peer roster = ground truth on active work — don't overlap
- **Inbox response**: When `[INBOX]` alert appears, call `omega_inbox()` IMMEDIATELY before continuing other work. Unread messages may contain blockers, requests, or task completions from peers.
- **Session start inbox**: In multi-agent mode, call `omega_inbox()` within first 3 tool calls.
- **Session intent**: At session start with peers > 0, announce your planned work: `omega_intent_announce(description="<goal>", target_files=[...])` within first 5 tool calls.
- **Before offering to take ANY task**: `omega_coord_status` + `omega_tasks_list` to check what peers are working on. If a peer's task/description overlaps, ASK the user instead of starting.
- Before editing shared files: `omega_file_check(file_path=...)` for conflicts
- Multi-step work: `omega_task_create` with `omega_task_deps` for cross-agent visibility
- After significant work: `omega_task_complete(task_id=..., result="summary")`
- If task cannot be completed: `omega_task_fail(task_id, reason)` — never silently abandon
- Before multi-file changes: `omega_intent_announce(description=..., target_files=[...])`
- Before git branch operations: `omega_branch_check` then `omega_branch_claim`
- After merging/abandoning: `omega_branch_release` to free the claim
- **Peer expertise**: When task requires expertise outside your current work, check `omega_find_agents(capability="<needed>")` before creating new tasks or asking the user.
- **Delegation**: To delegate, `omega_task_create` + `omega_send_message` to the capable peer.
- **Health check**: After resolving a conflict or when coordination feels sluggish: `omega_coord_metrics` to check conflict rates and gate check counts.
- **Audit trail**: When debugging "who did what": `omega_audit(event_type="...", limit=10)` — not git log.
- **Session safety**: `omega_session_snapshot` before risky ops; `omega_session_recover` after crashes
- **Config files** (package.json, pyproject.toml, Dockerfile, etc.): Always `omega_intent_announce` + `omega_file_check` before editing. Notify peers after changes.
- **Lock files** (package-lock.json, Cargo.lock): Never manually edit. On conflicts: delete, regenerate, commit.
- **Env files**: Never commit `.env`. Use `.env.example` for templates.
- **Parallel agent isolation**: When dispatching 2+ agents via Task tool that will edit files, prefer `isolation: "worktree"` to give each agent its own copy of the repo. This prevents the shared-filesystem problem where agents see each other's unstaged changes.""",
    },
    "coordination_gate": {
        "title": "Coordination Gate (Risk-Tiered)",
        "content": """\
| Risk | Actions | Gate |
|------|---------|------|
| **EXTERNAL** | Deploy, MCP submit, email, tweet, API call | Action claim (below) |
| **HIGH** | Force-push, delete branch, rm -rf | Full gate (all 3 steps below) |
| **MEDIUM** | git commit/push, create branch, install deps | `omega_coord_status` only |
| **LOW** | Edit claimed files, run tests, read | No gate — hooks handle it |

### EXTERNAL action gate (atomic — prevents duplicate execution):
1. `omega_action_check(action_type="...", action_target="...")` — already done?
2. `omega_action_claim(session_id="...", action_type="...", action_target="...")` — atomic claim
3. If claim fails: STOP — another agent owns it
4. Perform the external action
5. `omega_action_complete(action_id=..., session_id="...", result="...")` — record outcome

### HIGH-risk gate (all 3 required):
1. `omega_query(event_type="decision", query="<target area>")` — check prior decisions
2. `omega_coord_status` — check peer activity and claimed files
3. `git log --oneline -10 -- <target_dir>` — verify recent changes
4. For architecture/design decisions: `omega_reflect(action="evolution", topic=<domain>)` — see how understanding has changed. Prevents repeating abandoned approaches.

### MEDIUM-risk gate:
1. `omega_coord_status` — check peer activity. Warn on overlaps, don't block.

"Just deploying" and "routine task" are NOT reasons to skip HIGH-tier checks.""",
    },
    "teamwork": {
        "title": "Proactive Teamwork",
        "content": """\
### Task Lifecycle
- After completing any task: `omega_task_next` to claim next work. Don't ask user "what should I do?" when tasks exist.
- Every 15 tool calls on a claimed task: `omega_task_progress(task_id, progress=<estimate>)`
- Before asking user "what's next?": `omega_tasks_list` + `omega_task_next` FIRST
- When creating tasks for others: `omega_send_message` to notify them

### Task Completion Ritual (every task)
1. Check `omega_tasks_list` — does your output unblock a peer?
2. If yes: `omega_send_message` them with what they need
3. Leave a structured handoff: `omega_handoff(action="create", completed_tasks=[...], next_steps=[...], decisions_made=[...], files_modified=[...])` — NOT omega_store (loses structure)
4. Pre-stage next step: `omega_task_create` if follow-up is clear

### Mid-Task Awareness (every commit or 15+ min)
1. `omega_coord_status` — peer intersecting your work?
2. `omega_intent_check` — file list conflict with new intent?
3. Discoveries affecting peers → `omega_send_message` NOW

### Conflict Resolution
1. Later intent yields — earlier claim has priority
2. Propose a split via `omega_send_message` with concrete division
3. One exchange max — escalate to user if unresolved

### Pipeline Thinking
- Ask: "What happens after I finish? Who's waiting on me?"
- When blocked: check `omega_tasks_list` for unblocked work
- When idle: `omega_task_next` — pick up work, don't ask permission
- When switching focus mid-session: `omega_update_task(description="<new focus>")` to update dashboard""",
    },
    "goals_and_drift": {
        "title": "Goals & Drift Detection",
        "content": """\
### When to create goals
- Multi-session work (features, refactors, migrations): `omega_goal(action="create", title="...", description="...")`
- User states a high-level objective spanning multiple steps
- Plan files exist with 3+ phases

### Drift detection
- If goals exist, run `omega_drift_check` every 20 tool calls or when switching subtasks
- Drift check is fast (<50ms, pure SQL) — no reason to skip it
- If drift score is high: pause, re-read the goal, realign before continuing
- Goals auto-link to coord tasks via `omega_goal_link` when both are active

### When NOT to create goals
- Single-task sessions (bug fix, quick edit, question)
- Goals already exist for this work (check `omega_goal(action="list")` first)""",
    },
    "context": {
        "title": "Context Management",
        "content": """\
- When context window is getting full (>70%): `omega_checkpoint` to save task state
- **Tool-call heuristic**: If you have made 30+ tool calls in a session, call `omega_checkpoint` proactively — \
you are likely past 50% context usage. Don't wait for the 70% signal.
- When starting a session for an ongoing task: `omega_resume_task` first
- When `[CHECKPOINT]` appears at session start: offer to resume with `omega_resume_task`
- Checkpoints save: plan, progress, files changed, decisions, key context, next steps""",
    },
    "reminders": {
        "title": "Reminders",
        "content": """\
- When user says "remind me" or task has a future deadline: `omega_remind(text, duration)`
- Duration examples: '1h', '30m', '2d', '1w', '1d12h'
- At session start: check `omega_remind_list` for pending/fired reminders
- After acknowledging a reminder: `omega_remind_dismiss(reminder_id)`""",
    },
    "diagnostics": {
        "title": "Diagnostics & Maintenance",
        "content": """\
- Health/status/audit: `omega_health`, `omega_type_stats`, `omega_weekly_digest`, `omega_timeline`, `omega_forgetting_log`
- **Coordination health**: `omega_coord_metrics` to check conflict rates, gate checks, message throughput
- **Coordination audit**: `omega_audit(event_type="...", limit=10)` for debugging "who changed what when"
- Before risky bulk operations: `omega_backup(filepath="~/.omega/backups/omega-<date>.json")`
- Periodic maintenance: `omega_compact`, `omega_consolidate`""",
    },
    "entity": {
        "title": "Entity & Knowledge",
        "content": """\
- **People/orgs**: Use entity tools (`omega_entity_create/get/list/update/relationships/tree`) with `entity_id` scoping
- **Documents**: `omega_search_documents(query)` before web search. Ingest: `omega_ingest_document(path)`
- **Profile data**: `omega_profile_set/get/search/list` for structured user data. Prefer profile tools over flat memories""",
    },
    "domain_boundaries": {
        "title": "Domain Boundaries",
        "content": """\
Different domains require different reasoning models. Applying the wrong model is the #1 source of wrong-approach errors.

### Domain-Model Mapping
| Domain | Reasoning Model | DO NOT apply to this domain |
|--------|----------------|----------------------------|
| **Entity/people** | Credibility scoring, reputation signals, social proof | Market prices, funding rates, court rulings |
| **Market prices** | Quantitative models, order flow, liquidity analysis | Entity reputation, legal outcomes |
| **Legal/regulatory** | Precedent analysis, statutory interpretation, jurisdiction | Market predictions, entity credibility |
| **Technical/infra** | Root cause analysis, dependency tracing, failure modes | Business strategy, content quality |
| **Content/copy** | Audience fit, tone analysis, engagement patterns | Technical debugging, market analysis |

### Domain Drift Detection
Before applying a reasoning model, check:
1. **Identify source domain**: Where did this model come from? (e.g., "entity credibility")
2. **Identify target domain**: What am I analyzing? (e.g., "market event")
3. **Flag mismatches**: If source ≠ target, STOP and choose the correct model
4. **Proxy endpoint rule**: Never reverse-engineer undocumented APIs. If an endpoint isn't documented, ask — don't probe.""",
    },
    "source_verification": {
        "title": "Source Verification",
        "content": """\
**Memory is a cache, not truth.** OMEGA memories reflect what was true when stored. Files change.

### Mandatory Verification Table
| What to verify | Verify against | NOT against |
|---------------|---------------|-------------|
| Schema version | `pyproject.toml` | Remembered value |
| Tool count | `tool_schemas.py` (count definitions) | Last known number |
| Test assertions | Run `pytest` | Recalled pass/fail |
| File contents | `Read` the file | "I recall it contains..." |
| API endpoints | Source code + docs | Cached response |
| Config values | `.env` / config files | Previous session memory |
| Git state | `git status` / `git log` | "Last time I checked..." |

### Anti-Patterns → Corrections
| Anti-pattern | Correction |
|-------------|-----------|
| "I recall this file contains..." | READ the file now |
| "The schema version is..." | CHECK `pyproject.toml` |
| "There are N tools..." | COUNT in `tool_schemas.py` |
| "This test was passing..." | RUN the test |
| "The config is set to..." | READ the config file |

### When Memory Conflicts with Source
1. **Source wins** — always
2. Mark the memory outdated: `omega_feedback(memory_id, "outdated")`
3. Store the correction: `omega_store("Corrected: <what changed>", "lesson_learned")`""",
    },
    "heuristics": {
        "title": "Decision Heuristics",
        "content": """\
- **Reversibility test**: Reversible → proceed. Irreversible → ask first.
- **Friction is signal**: Harder than expected? Investigate — usually means missing context.
- **Learn, don't just complete**: On mistakes, `omega_store` the lesson before moving on.
- **Push back from care**: If user's approach will cause problems, say so directly.
- **When in doubt, narrow scope**: Do less correctly rather than more with assumptions.

### Anti-Rationalization
| Thought | Do instead |
|---|---|
| "Just deploying" | Run the coordination gate |
| "This is routine" | Run the coordination gate |
| "I'll check after" | Check before — that's the point |
| "No one else is working" | Verify with `omega_coord_status` |
| "No one is working on this" | `omega_coord_status` + `omega_tasks_list` before claiming |
| "I know this area" | `omega_query` for decisions you missed |
| "It's a small change" | Small shared-file changes cause the biggest conflicts |
| "I know what this file contains" | Read it. Memory is stale. |
| "This model applies here too" | Check domain boundaries first. |
| "I'll update tests after" | Stage tests WITH the code change. |
| "The approach is obvious" | State it explicitly. Obvious approaches cause 35% of rework. |
| "I'll add .gitignore after" | Adjacent concerns BEFORE push. Scan for secrets, add .gitignore, LICENSE. |
| "Just creating the repo" | `omega_checkpoint()` BEFORE `gh repo create`. State your recovery plan. |
| "I remember the URL" | No you don't. Read it from a file or verify it exists. LLMs fabricate URLs. |
| "The subagent will figure it out" | Subagents can't call OMEGA. Inject context into the prompt. |
| "User asked this before" | `omega_query` to verify — your recall of past conversations is unreliable. |
| "I already suggested this and it worked" | Check if user rejected it in a later session you don't have context for. |""",
    },
    "git": {
        "title": "Git Rules",
        "content": """\
- Commit only files YOU modified. `git add <files>` -- never `git add .`
- **Session scope**: Only recommend committing, staging, or acting on files you personally created or edited in THIS session. Uncommitted files visible in `git status` may belong to other agents or prior sessions. If you didn't touch it, it's not yours to commit or recommend.
- **Multi-agent commit discipline**: With peers active, commit early and often. Uncommitted changes on disk are visible to all agents — another agent may inadvertently stage and commit YOUR in-progress work. Small, frequent commits prevent mixed-author accidents.
- **Peer-claimed files**: The pre-commit guard BLOCKS commits that stage files claimed by other sessions. If blocked, unstage the peer's files with `git reset HEAD <file>` before retrying.
- After every commit: `omega_store("Committed <hash>: <message>. Files: <list>", "decision")`
- Before "what's next": `omega_coord_status` + `omega_git_events` + `omega_query(event_type="decision")`
- **Never force push protected branches** (main/master). Warn user if requested.
- **Rebase vs merge**: Feature branches rebase onto main before PR. Shared branches merge only.
- **Commit format**: `type: Brief description (under 72 chars)`. Types: feat, fix, refactor, docs, test, chore.
- **Before push**: (1) `omega_branch_check` + claim if unclaimed, (2) `git pull --rebase origin <branch>`, (3) run tests, (4) push.
- **Merge conflicts with peers**: Check `omega_coord_status` for file owner, coordinate via `omega_intent_announce`, one agent resolves.""",
    },
    "alignment": {
        "title": "Decision Alignment",
        "content": """\
- **Before** starting domain work: `omega_decision_query(project=..., domain="<area>")` to check active decisions
- **After** making a decision: `omega_decision_register(session_id, project, domain, decision, rationale)` to make it authoritative
- **On `[ALIGNMENT]` warnings**: read the decisions, comply or explicitly supersede with `omega_decision_register`
- **On `[ALIGNMENT-BLOCK]`**: STOP. Comply with the decision or supersede it with rationale.
- **On `[DECISION]` inbox messages**: acknowledge and adjust your work if relevant
- **Decision domains**: use hierarchical keys ("auth", "deploy/vercel", "testing/e2e"). Same-domain decisions auto-supersede.
- **On contradiction_warnings**: review both decisions, revoke the outdated one with `omega_decision_revoke`
- **Never ignore alignment signals** — they exist to prevent contradictory work across agents
- **`[DECISIONS]` at session start**: shows active decisions. Check before starting work in those domains.""",
    },
    "verification": {
        "title": "Verification",
        "content": """\
### Build-Verify-Fix Loop
1. **Plan**: Read the task, scan relevant code, build a plan with verification criteria.
2. **Build**: Implement with testability in mind.
3. **Verify**: Run tests or manual checks. Compare against what was asked, not your own code.
4. **Fix**: If verification fails, revisit the original spec before patching.

### Pre-Completion Checklist
Before claiming "done" or "fixed":
- [ ] Run the verification step (tests, lint, build, or manual check)
- [ ] Show evidence of success (test output, counts, screenshots)
- [ ] Compare result against original request, not your own assessment
- [ ] **Wired check** (3-level): (1) Artifact **exists** — file/function/route is present, \
(2) Artifact is **substantive** — not a stub or placeholder, \
(3) Artifact is **connected** — called from the right place, wired end-to-end. \
Tests passing ≠ feature wired. A helper function with tests but no caller is not done.

### Loop Detection
If you have edited the same file 5+ times without progress:
- Stop and reconsider your approach
- Re-read the original task specification
- Consider an alternative strategy
- Ask the user if stuck

### Completion Gate (Hard Rule)
If the plan or task includes verification steps, they are MANDATORY — not optional.
The words "done", "complete", "all set" are blocked until:
1. Run at least ONE verification command (test, build, lint, curl, `gh repo view`, manual check)
2. Show the output or a summary in your response
3. Map result to original request: "You asked X. Evidence of X: [output]"

If you cannot verify automatically: "Cannot auto-verify. Changed [what] because [why]. Manual check: [suggestion]." """,
    },
    "efficiency": {
        "title": "Efficiency",
        "content": """\
### Direct Access
- When a file path, memory ID, or URL is explicitly provided, access it directly. Do not search/glob first.
- When `omega_query` returns a memory ID, use it directly in follow-up calls. Do not re-search.
- Prefer `Read` over `Grep` when you already know the file path.

### Context Budget
- Before large tool calls, estimate if the result will exceed your context window.
- For `omega_query`: use `limit=3` for quick checks, `limit=10` for thorough research. Avoid `limit=50+` unless explicitly needed.
- After receiving large results, extract what you need and move on. Do not re-query the same content.

### Redundancy Avoidance
- Track which files you have already read this session. Do not re-read unchanged files.
- If you searched for X and found nothing, do not search again with minor variations. Ask the user or try a different approach.""",
    },
    "what_next": {
        "title": "What's Next Protocol",
        "content": """\
Before recommending work or answering "what's next":
1. `omega_coord_status` — check active peers
1b. `omega_tasks_list` — check in_progress tasks owned by peers. Never suggest work a peer is already doing.
2. `omega_git_events` — check recent commits
3. `omega_query(event_type="decision")` — check recent decisions
4. `ps aux | grep -E "python.*(benchmark|harness)"` — check running processes
5. Cross-reference each pending task/reminder against decisions — if a decision confirms \
completion (committed, shipped, deployed, verified, configured), mark it done and dismiss the reminder.
6. **Scope your recommendations**: Only suggest committing files YOU modified. Do not present \
uncommitted files from `git status` as actionable items — they may belong to other agents or prior \
sessions. Do not fabricate a menu of options from stale git state. If you have no OMEGA tasks and \
no clear next step, simply ask: "What's next?"

CRITICAL: `[TASKS]` and reminders are stale until verified. Items tagged `[DONE?]` at session start \
have a matching decision and should be dismissed, not re-suggested. Never parrot a task list \
without checking decisions first.""",
    },
    "intelligence_cards": {
        "title": "Intelligence Cards",
        "content": """\
You have OMEGA intelligence cards. When you see `[OMEGA]` blocks in your context (from hooks or tool results), you MUST include them in your response to the user.

Rules for surfacing cards:
- Copy the `[OMEGA]` block into your response verbatim. Do not rephrase or editorialize.
- Place cards near the relevant part of your response, not all at the end.
- If multiple cards appear, include all of them.
- Cards are compact by design. Do not expand or explain them unless the user asks.
- If a card contains a warning, acknowledge it before proceeding with the action.
- If a card shows prior decisions, ensure your current action is consistent with them.

Card types you may see:
- `[OMEGA] Used:` -- a memory you are drawing on. Show the user what informed your thinking.
- `[OMEGA] Prior decisions:` -- past decisions on this topic. Stay consistent or explain why you diverge.
- `[OMEGA] Learned:` -- something auto-captured from your response. Confirms OMEGA is learning.
- `[OMEGA] Warning:` -- known issues in the area you are working on. Proceed with caution.
- `[OMEGA] Session intelligence:` -- end-of-session summary. Always surface this.

**Outcome tracking**: OMEGA tracks which surfaced memories you reference in your responses. \
Memories you use are promoted; memories surfaced but never referenced decay faster. \
This feedback loop is automatic and requires no action from you.""",
    },
    "session_awareness": {
        "title": "Session Awareness",
        "content": """\
- **Check home first**: Before web research, run `omega_query("<topic>")` for existing context
- **Narrate scope pivots**: If your action differs from the literal request, say so: "You asked X. I'm interpreting as Y because Z. Correct?"
- **Checkpoint before synthesis**: Before large outputs (>2K chars), summarize inputs and ask "Anything else before I synthesize?"
- **Evidence before assertion**: Never claim "done" or "fixed" without showing evidence first. "Tests pass (23/23)" not just "fixed."
- **Announce session shape**: For multi-phase tasks, state upfront: "~N phases: [list]. I'll check in after [phase]."
- **User feedback is ground truth**: When user contradicts your assessment, investigate immediately. Never argue.
- **Respect exit intent**: When user signals stop, acknowledge and cease tool calls.""",
    },
    "agent_discipline": {
        "title": "Agent Discipline",
        "content": """\
- **Pre-flight explore**: Before "match X" tasks, dispatch a scoped Explore sub-agent to characterize the target FIRST
- **Verify every 2 phases**: Pause after every 2 phases. Summarize changes. For code: run tests. For visual: offer preview. Never skip.
- **Context budget warning**: Before plans with 5+ phases or 50+ edits, warn user and create checkpoints
- **Persist at boundaries**: `omega_store` after every significant milestone. Minimum 1 store per completed phase.
- **Scope sub-agent briefs tightly**: Always include (1) specific file paths, (2) 2-4 specific questions, (3) what NOT to explore. No vague "be thorough".
- **Inject memory into sub-agents**: Sub-agents CANNOT call OMEGA tools. Before spawning any agent that will make decisions, suggest actions, or generate content: (1) `omega_query()` for task-relevant decisions/preferences/constraints, (2) include key results in the agent prompt, (3) add explicit instruction: "Do NOT fabricate URLs, tool names, or project details not in this prompt." This is the #1 source of memory failures.
- **Checkpoint before irreversible**: Before `gh repo create`, `vercel deploy`, `rm -rf`, `npm publish`, or any hard-to-undo action: (1) `omega_checkpoint()`, (2) state recovery plan if it fails, (3) proceed.
- **Adjacent awareness scan**: After completing the literal task, ask: "What did I NOT do that a careful human would?" Check for: .gitignore, LICENSE, secrets scan, repo description, branch protection. Surface findings even if not acting.
- **Store after external actions**: After any external action (repo create, deploy, collaborator invite, email, tweet): immediately `omega_store()` the outcome. No exceptions.
- **Minimum 1 store per session**: Every session with meaningful work (decisions, corrections, completions) MUST call `omega_store()` at least once. Sessions with 0 stores lose institutional knowledge and degrade future sessions.
- **No fabricated URLs/links**: NEVER generate a URL from memory or inference. Read from a file, query OMEGA, or verify via web fetch. Fabricated URLs are the top user complaint.""",
    },
    "consultation": {
        "title": "GPT Consultation",
        "content": """\
Use `omega_consult_gpt` to get a second opinion from GPT on genuinely hard problems.

### DO consult when:
- Stuck 10+ minutes or tried 3+ approaches without progress
- Architecture decisions that are hard to reverse (DB schema, API contracts, auth flow)
- Debugging dead ends — you have error + context but can't find root cause
- Cross-validating a solution that feels fragile or over-engineered
- Domain expertise gap (crypto, ML, networking, legal, etc.)

### Do NOT consult when:
- Simple tasks (formatting, renaming, CRUD, config changes)
- Speed-sensitive work (adds 10-60s latency per call)
- Already working — tests pass, don't second-guess success
- User asked for Claude's opinion specifically
- Routine OMEGA operations (store, query, checkpoint)

### Usage tips:
- Set `temperature` low (0.0-0.3) for factual/debugging, higher (0.5-0.7) for design/architecture
- Provide `context` with code snippets, error traces, or constraints — GPT can't see your files
- Use a specific `system` prompt for domain framing (e.g., "You are an expert in distributed systems")
- The response header shows which GPT model answered — include this when reporting to user
- Store the consultation result as a `decision` memory with metadata={"source": "cross_model_consult"} — this ensures the second opinion is retrievable in future sessions facing similar problems.""",
    },
    "system_insights": {
        "title": "System Insights",
        "content": """\
Architectural insights from prior development sessions. These are concrete, reusable lessons \
discovered during debugging and feature work. They surface automatically here and as [INSIGHT] \
warnings before editing relevant files.

{system_insights_placeholder}

### Capturing New Insights
When you discover a non-obvious architectural lesson during debugging or feature work (failure modes, \
ordering dependencies, silent regressions, config gotchas), store it:
```
omega_store(
  content="<concrete insight with root cause and fix>",
  event_type="advisor_insight",
  metadata={{"category": "system_insight", "tags": ["<subsystem>", "<topic>"]}},
  entity_id="<current_entity>"
)
```
- Use the current project's `entity_id` (not always "omega" — personal-assistant, element1, etc. all benefit)
- Good insights are: specific (not generic advice), rooted in a real incident, and include the *why* not just the *what*
- Tags should name the subsystem (`hooks`, `coordination`, `bridge`, `alerting`, etc.) — the edit-time hook uses these to match files
- These insights have permanent TTL and compound across sessions — every agent that works on this subsystem in the future will see them""",
    },
    "council": {
        "title": "System Health Council",
        "content": """\
Use `omega_council` for periodic system health audits. Three domains available:

| Domain | What it checks | When to run |
|--------|---------------|-------------|
| `platform_health` | Error rates, stale sessions, hook failures | Maintenance sessions, after incidents |
| `security` | Credential exposure, permission gaps, secrets in code | Before releases, after auth changes |
| `innovation` | Unused capabilities, feature adoption gaps | Planning sessions, quarterly reviews |

### When to run council:
- User asks for system health check or audit
- Session is explicitly for maintenance (`omega_maintain` context)
- After resolving a production incident (run `platform_health`)
- Before a release (run `security`)

### When NOT to run council:
- Normal feature development sessions
- Quick bug fixes or edits""",
    },
    "critical_tools": {
        "title": "Critical Tools Checklist",
        "content": """\
MANDATORY TOOLS — enforcement active. Protocol gate checks these within first 20 tool calls:

| Tool | When | Enforcement |
|------|------|-------------|
| `omega_reflect()` | Before major decisions | Auto-triggered at session end (stale detection). Manual calls for contradiction checks. |
| `omega_decision_query(domain="<area>")` | Before starting work in any domain | Prevents contradicting prior decisions. |
| `omega_file_check(file_path="...")` | Before editing any file | Detects conflicts before they happen. |
| `omega_checkpoint()` | When context > 70% or before risky ops | Auto-generated at session end for sessions with 3+ captures. Manual calls for mid-session saves. |
| `omega_coord_status` | Before "what's next" and before taking tasks | Gate enforced in multi-agent mode. |
| `omega_store()` | After external actions, milestones, and completed phases | Nudge at 15+ edits (existing). New: nudge after `gh`/`vercel`/`npm publish` without subsequent store. |
| `omega_goal()` | Multi-session work (3+ phases) | Creates anchor for drift detection. |
| `omega_drift_check` | Every 20 tool calls when goals exist | Fast (<50ms). Detects scope drift early. |

**Compliance is scored at session end.** Missing mandatory tools lowers your adherence score. \
Scores are stored for trend analysis across sessions.""",
    },
}

# Section groups -- named bundles for common scenarios
SECTION_GROUPS: Dict[str, List[str]] = {
    "solo": ["memory", "critical_tools", "system_insights", "intelligence_cards", "session_awareness", "consultation", "council", "context", "reminders", "heuristics", "domain_boundaries", "efficiency", "verification", "source_verification", "git", "what_next"],
    "multi_agent": [
        "memory", "critical_tools", "system_insights", "intelligence_cards", "session_awareness", "agent_discipline",
        "coordination", "coordination_gate", "teamwork", "goals_and_drift", "alignment", "consultation", "council",
        "context", "reminders", "heuristics", "domain_boundaries", "efficiency", "verification", "source_verification", "git", "what_next",
    ],
    "full": list(SECTIONS.keys()),
    "minimal": ["memory", "context", "git"],
}


def _provider_notes(provider: str) -> str:
    """Return provider-specific notes for non-Anthropic agents."""
    if provider == "anthropic":
        return ""

    from omega.llm import get_model_map

    models = get_model_map().get(provider, {})
    model_info = ", ".join(f"{tier}={name}" for tier, name in models.items())

    lines = [
        "## Provider Notes",
        f"You are running on the **{provider}** provider ({model_info}).\n",
        "- MCP tool names are unchanged — they are protocol-level, not model-level.",
        "- `omega_consult_gpt` is **unavailable** (you are GPT). Use `omega_consult_claude` for a second opinion from Claude, or skip consultation.",
        "- All other OMEGA tools work identically regardless of provider.",
        "",
    ]
    return "\n".join(lines)


def _provider_consultation(provider: str) -> Dict[str, str]:
    """Return provider-adapted consultation section content."""
    if provider in ("openai", "openai_compat"):
        has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        if has_anthropic_key:
            consult_tool = "omega_consult_claude"
            consult_name = "Claude"
        else:
            # No Anthropic key — consultation unavailable
            return {
                "title": "Cross-Model Consultation",
                "content": """\
Cross-model consultation is unavailable (no ANTHROPIC_API_KEY set).
If you need a second opinion, present the problem to the user directly.""",
            }
        return {
            "title": "Claude Consultation",
            "content": f"""\
Use `{consult_tool}` to get a second opinion from {consult_name} on genuinely hard problems.

### DO consult when:
- Stuck 10+ minutes or tried 3+ approaches without progress
- Architecture decisions that are hard to reverse (DB schema, API contracts, auth flow)
- Debugging dead ends — you have error + context but can't find root cause
- Cross-validating a solution that feels fragile or over-engineered
- Domain expertise gap (crypto, ML, networking, legal, etc.)

### Do NOT consult when:
- Simple tasks (formatting, renaming, CRUD, config changes)
- Speed-sensitive work (adds 10-60s latency per call)
- Already working — tests pass, don't second-guess success
- User asked for your opinion specifically
- Routine OMEGA operations (store, query, checkpoint)

### Usage tips:
- Set `temperature` low (0.0-0.3) for factual/debugging, higher (0.5-0.7) for design/architecture
- Provide `context` with code snippets, error traces, or constraints — {consult_name} can't see your files
- Use a specific `system` prompt for domain framing (e.g., "You are an expert in distributed systems")
- The response header shows which model answered — include this when reporting to user""",
        }

    # Default: Anthropic provider — keep existing GPT consultation
    return SECTIONS["consultation"]


def get_protocol(
    section: Optional[str] = None,
    project: Optional[str] = None,
    include_lessons: bool = True,
    peer_count: int = 0,
    session_id: Optional[str] = None,
) -> str:
    """Assemble the protocol playbook dynamically.

    Args:
        section: Specific section name, group name, or None for auto-detect.
        project: Current project path for context-sensitive rules.
        include_lessons: Whether to append relevant lessons from OMEGA.
        peer_count: Number of active peers (0 = solo mode).
        session_id: Current session ID for role assignment.

    Returns:
        Formatted protocol text ready for agent consumption.
    """
    # Determine which sections to include
    if section and section in SECTIONS:
        # Single section requested
        selected = [section]
    elif section and section in SECTION_GROUPS:
        # Named group requested
        selected = SECTION_GROUPS[section]
    elif section == "all" or section == "full":
        selected = list(SECTIONS.keys())
    elif peer_count > 0:
        # Multi-agent mode — include coordination sections
        selected = SECTION_GROUPS["multi_agent"]
    else:
        # Solo mode — skip coordination overhead
        selected = SECTION_GROUPS["solo"]

    # Build output
    lines = [f"# OMEGA Protocol v{PROTOCOL_VERSION}\n"]

    if peer_count > 0:
        lines.append(f"_Mode: multi-agent ({peer_count} peer{'s' if peer_count != 1 else ''} active)_\n")
    else:
        lines.append("_Mode: solo_\n")

    llm_provider = os.environ.get("OMEGA_LLM_PROVIDER", "anthropic")
    if llm_provider != "anthropic":
        lines.append(f"_Provider: {llm_provider}_\n")

    # Provider-specific notes (only for non-Anthropic)
    provider_notes = _provider_notes(llm_provider)
    if provider_notes:
        lines.append(provider_notes)

    # Session role assignment (behavioral diversity)
    session_role = None
    if peer_count > 0 and session_id:
        session_role = _get_session_role(session_id)
        if session_role and session_role.get("role_instruction"):
            lines.append("## Session Role")
            lines.append(f"**Role: {session_role['role'].upper()}**\n")
            lines.append(session_role["role_instruction"])
            lines.append("")

    # Build provider-adapted consultation section
    adapted_consultation = _provider_consultation(llm_provider)

    # Pre-resolve dynamic content for system_insights section
    _insights_content: Optional[str] = None
    if "system_insights" in selected:
        _insights_content = _get_system_insights(project)

    for key in selected:
        if key == "consultation":
            # Use provider-adapted consultation instead of static SECTIONS entry
            lines.append(f"## {adapted_consultation['title']}")
            lines.append(adapted_consultation["content"])
            lines.append("")
        elif key == "system_insights":
            # Build system_insights section with insights injected directly
            sec = SECTIONS.get(key)
            if sec:
                lines.append(f"## {sec['title']}")
                content = sec["content"]
                if _insights_content:
                    content = content.replace("{system_insights_placeholder}", _insights_content)
                else:
                    # Remove the placeholder line entirely
                    content = "\n".join(
                        line for line in content.split("\n")
                        if "{system_insights_placeholder}" not in line
                    )
                lines.append(content)
                lines.append("")
        else:
            sec = SECTIONS.get(key)
            if sec:
                lines.append(f"## {sec['title']}")
                lines.append(sec["content"])
                lines.append("")

    # --- Cache breakpoint: static protocol sections above, dynamic appendices below ---
    # The static SECTIONS are deterministic and identical across sessions.
    # Lessons, skills, and insights are dynamic and session-dependent.
    has_dynamic = False

    # Append relevant lessons from OMEGA if requested
    if include_lessons:
        lessons_text = _get_protocol_lessons(project)
        if lessons_text:
            if not has_dynamic:
                lines.append("<!-- omega:cache_breakpoint -->")
                has_dynamic = True
            lines.append("## Learned Protocol Lessons")
            lines.append(lessons_text)
            lines.append("")

    # Surface distilled skill templates from prior sessions
    skills_text = _get_relevant_skills(project)
    if skills_text:
        if not has_dynamic:
            lines.append("<!-- omega:cache_breakpoint -->")
            has_dynamic = True
        lines.append("## Relevant Skill Templates")
        lines.append("Patterns distilled from successful prior sessions:")
        lines.append(skills_text)
        lines.append("")

    return "\n".join(lines)


def _get_session_role(session_id: str) -> Optional[Dict]:
    """Get session role from coordination manager. Returns default role on failure."""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        return mgr.assign_session_role(session_id)
    except Exception as e:
        logger.warning("Session role assignment failed: %s", e)
        return {"role": "default", "note": "Coordination unavailable — using default role"}


def _get_system_insights(project: Optional[str] = None, context: Optional[str] = None) -> str:
    """Fetch system insights relevant to the current subsystem from OMEGA memory."""
    try:
        from omega.bridge import query_structured

        # Build a subsystem hint from project path or explicit context
        hint = context or ""
        if not hint and project:
            # Extract meaningful path segments as search hint
            parts = project.rstrip("/").split("/")
            # Use last 2 meaningful segments (e.g., "omega/website" or "omega/hooks")
            meaningful = [p for p in parts if p and not p.startswith(".") and p != "Users"]
            hint = " ".join(meaningful[-3:]) if meaningful else ""
        if not hint:
            hint = "omega coordination hooks memory alerting"

        results = query_structured(
            query_text=hint,
            limit=10,
            event_type="advisor_insight",
        )
        if not results:
            return ""

        # Filter to system_insight category
        items = []
        for r in results:
            meta = r.get("metadata") or {}
            if meta.get("category") != "system_insight":
                continue
            content = r.get("content", "")[:500]
            tags = r.get("tags") or []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            items.append(f"- {content}{tag_str}")
            if len(items) >= 5:
                break

        return "\n".join(items)
    except Exception as e:
        logger.debug("System insights fetch failed: %s", e)
        return ""


def _get_protocol_lessons(project: Optional[str] = None) -> str:
    """Fetch relevant lessons about coordination/protocol from OMEGA memory."""
    try:
        from omega.bridge import query_structured

        results = query_structured(
            query_text="coordination protocol gate deployment gotcha",
            limit=5,
            event_type="lesson_learned",
        )
        if not results:
            return ""

        items = []
        for r in results[:5]:
            content = r.get("content", "")[:400]
            items.append(f"- {content}")
        return "\n".join(items)
    except Exception as e:
        logger.debug("Protocol lessons fetch failed: %s", e)
        return ""


def _get_relevant_skills(project: Optional[str] = None) -> str:
    """Surface top-2 matching skill templates distilled from prior sessions.

    Queries the memory store for skill_template events and returns the most
    relevant ones based on recency and strength decay.
    """
    try:
        from omega.bridge import _get_store

        db = _get_store()
        results = db.query(
            "skill_template",
            limit=2,
            min_strength=0.3,
        )
        if not results:
            return ""

        items = []
        for r in results[:2]:
            content = r.get("content", "")[:300]
            items.append(f"- {content}")
        return "\n".join(items)
    except Exception as e:
        logger.debug("Relevant skills fetch failed: %s", e)
        return ""


def list_sections() -> List[Dict[str, str]]:
    """List all available protocol sections with titles."""
    return [
        {"key": key, "title": sec["title"]}
        for key, sec in SECTIONS.items()
    ]
