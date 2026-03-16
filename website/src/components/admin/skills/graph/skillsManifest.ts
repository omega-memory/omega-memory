// IMPORTANT: When adding new skills to ~/.claude/skills/, also add them here.
// This manifest is the source of truth for the admin skills graph visualization.
// Run: ls ~/.claude/skills/ to see all installed skills and verify nothing is missing.

export interface SkillDef {
  id: string;
  name: string;
  plugin: string;
  description: string;
  type: "rigid" | "flexible";
  invokes?: string[];
}

export const SKILLS_MANIFEST: SkillDef[] = [
  // superpowers plugin (14 skills)
  { id: "superpowers:brainstorming", name: "Brainstorming", plugin: "superpowers", description: "Explore user intent, requirements, and design before implementation", type: "flexible", invokes: ["superpowers:writing-plans", "personal:deep-research"] },
  { id: "superpowers:writing-plans", name: "Writing Plans", plugin: "superpowers", description: "Create detailed implementation plans from specs or requirements", type: "rigid", invokes: ["superpowers:executing-plans", "superpowers:subagent-driven-development"] },
  { id: "superpowers:executing-plans", name: "Executing Plans", plugin: "superpowers", description: "Execute implementation plans task-by-task with review checkpoints", type: "rigid", invokes: ["superpowers:verification-before-completion"] },
  { id: "superpowers:subagent-driven-development", name: "Subagent-Driven Dev", plugin: "superpowers", description: "Execute plans using fresh subagents per task in current session", type: "rigid", invokes: ["superpowers:requesting-code-review"] },
  { id: "superpowers:systematic-debugging", name: "Systematic Debugging", plugin: "superpowers", description: "Structured approach to bugs, test failures, and unexpected behavior", type: "rigid" },
  { id: "superpowers:test-driven-development", name: "Test-Driven Development", plugin: "superpowers", description: "Write failing tests before implementation code", type: "rigid" },
  { id: "superpowers:verification-before-completion", name: "Verification", plugin: "superpowers", description: "Run verification commands before claiming work is complete", type: "rigid", invokes: ["superpowers:requesting-code-review"] },
  { id: "superpowers:requesting-code-review", name: "Requesting Code Review", plugin: "superpowers", description: "Verify work meets requirements before merging", type: "rigid", invokes: ["personal:refactoring", "personal:security-audit"] },
  { id: "superpowers:receiving-code-review", name: "Receiving Code Review", plugin: "superpowers", description: "Process code review feedback with technical rigor", type: "rigid", invokes: ["personal:refactoring", "superpowers:systematic-debugging"] },
  { id: "superpowers:dispatching-parallel-agents", name: "Parallel Agents", plugin: "superpowers", description: "Dispatch 2+ independent tasks to parallel subagents", type: "flexible" },
  { id: "superpowers:using-git-worktrees", name: "Git Worktrees", plugin: "superpowers", description: "Create isolated git worktrees for feature work", type: "flexible" },
  { id: "superpowers:finishing-a-development-branch", name: "Finishing Branch", plugin: "superpowers", description: "Guide completion of development work: merge, PR, or cleanup", type: "flexible" },
  { id: "superpowers:using-superpowers", name: "Using Superpowers", plugin: "superpowers", description: "Entry point: find and use skills at conversation start", type: "rigid", invokes: ["superpowers:brainstorming"] },
  { id: "superpowers:writing-skills", name: "Writing Skills", plugin: "superpowers", description: "Create, edit, and verify skills before deployment", type: "flexible" },
  // frontend-design
  { id: "frontend-design:frontend-design", name: "Frontend Design", plugin: "frontend-design", description: "Create distinctive, production-grade frontend interfaces", type: "flexible" },
  // claude-md-management
  { id: "claude-md-management:claude-md-improver", name: "CLAUDE.md Improver", plugin: "claude-md-management", description: "Audit and improve CLAUDE.md files in repositories", type: "flexible" },
  // claude-code-setup
  { id: "claude-code-setup:claude-automation-recommender", name: "Automation Recommender", plugin: "claude-code-setup", description: "Analyze codebase and recommend Claude Code automations", type: "flexible" },
  // pr-review-toolkit
  { id: "pr-review-toolkit:review-pr", name: "PR Review", plugin: "pr-review-toolkit", description: "Comprehensive PR review using specialized agents", type: "rigid" },
  // vercel
  { id: "vercel:deploy", name: "Deploy", plugin: "vercel", description: "Deploy applications to Vercel", type: "flexible" },
  { id: "vercel:setup", name: "Setup", plugin: "vercel", description: "Set up Vercel CLI and project configuration", type: "flexible" },
  { id: "vercel:logs", name: "Logs", plugin: "vercel", description: "View Vercel deployment logs", type: "flexible" },
  // firecrawl
  { id: "firecrawl:firecrawl-cli", name: "Firecrawl", plugin: "firecrawl", description: "Web scraping, search, and content extraction", type: "flexible" },
  // commit-commands
  { id: "commit-commands:commit", name: "Commit", plugin: "commit-commands", description: "Create a git commit", type: "rigid" },
  { id: "commit-commands:commit-push-pr", name: "Commit Push PR", plugin: "commit-commands", description: "Commit, push, and open a PR", type: "rigid" },
  { id: "commit-commands:clean_gone", name: "Clean Gone", plugin: "commit-commands", description: "Clean up git branches marked as gone on remote", type: "rigid" },
  // feature-dev
  { id: "feature-dev:feature-dev", name: "Feature Dev", plugin: "feature-dev", description: "Guided feature development with architecture focus", type: "flexible" },
  // code-review
  { id: "code-review:code-review", name: "Code Review", plugin: "code-review", description: "Code review a pull request", type: "rigid" },
  // ralph-loop
  { id: "ralph-loop:ralph-loop", name: "Ralph Loop", plugin: "ralph-loop", description: "Start Ralph Loop in current session", type: "rigid" },
  { id: "ralph-loop:cancel-ralph", name: "Cancel Ralph", plugin: "ralph-loop", description: "Cancel active Ralph Loop", type: "rigid" },
  { id: "ralph-loop:help", name: "Ralph Help", plugin: "ralph-loop", description: "Explain Ralph Loop plugin and available commands", type: "flexible" },
  // agent-sdk-dev
  { id: "agent-sdk-dev:new-sdk-app", name: "New SDK App", plugin: "agent-sdk-dev", description: "Create and setup a new Claude Agent SDK application", type: "flexible" },
  // firecrawl (additional)
  { id: "firecrawl:skill-gen", name: "Skill Gen", plugin: "firecrawl", description: "Generate a complete Agent Skill from a documentation URL", type: "flexible" },
  // claude-md-management (additional)
  { id: "claude-md-management:revise-claude-md", name: "Revise CLAUDE.md", plugin: "claude-md-management", description: "Update CLAUDE.md with learnings from this session", type: "flexible" },
  // ── personal skills (~/. claude/skills/) ─────────
  { id: "personal:audit", name: "Audit", plugin: "personal", description: "Scoped investigation of a system or feature. Report only.", type: "rigid", invokes: ["personal:design-audit", "personal:security-audit"] },
  { id: "personal:audit-session", name: "Audit Session", plugin: "personal", description: "Review an agent session transcript for coordination quality", type: "rigid" },
  { id: "personal:benchmark", name: "Benchmark", plugin: "personal", description: "Run LongMemEval benchmark and compare against previous best score", type: "rigid" },
  { id: "personal:cmo", name: "CMO", plugin: "personal", description: "Consult your CMO on OMEGA growth strategy, content, and X/Twitter", type: "flexible", invokes: ["personal:growth-marketing"] },
  { id: "personal:commit", name: "Quick Commit", plugin: "personal", description: "Quick commit with pre-flight checks, lighter than ship", type: "rigid" },
  { id: "personal:design-audit", name: "Design Audit", plugin: "personal", description: "Review an existing frontend against design standards. Report only.", type: "rigid", invokes: ["personal:seo", "personal:design-process", "personal:accessibility"] },
  { id: "personal:design-process", name: "Design Process", plugin: "personal", description: "Mandatory entry point for all frontend/UX/IA work. Multi-phase pipeline.", type: "rigid", invokes: ["frontend-design:frontend-design"] },
  { id: "personal:growth-marketing", name: "Growth Marketing", plugin: "personal", description: "Expert AI growth marketing advisor for developer tools and OSS", type: "flexible", invokes: ["personal:seo", "personal:design-process"] },
  { id: "personal:healthcheck", name: "Healthcheck", plugin: "personal", description: "Run a scoped system health audit across all subsystems. Report only.", type: "rigid" },
  { id: "personal:pre-edit-reasoning", name: "Pre-Edit Reasoning", plugin: "personal", description: "Deep pre-edit analysis: surface context, constraints, and history", type: "rigid" },
  { id: "personal:ship", name: "Ship", plugin: "personal", description: "Pre-ship readiness checklist: tests, lint, benchmarks, and commit", type: "rigid", invokes: ["superpowers:verification-before-completion", "personal:release-management", "personal:codebase-docs", "personal:e2e-testing"] },
  { id: "personal:x-brief", name: "X Brief", plugin: "personal", description: "Scan X/Twitter for AI news, morning brief, or feed review", type: "flexible", invokes: ["personal:growth-marketing"] },
  { id: "personal:seo", name: "SEO", plugin: "personal", description: "Post-build SEO checklist and standalone SEO audit", type: "rigid", invokes: ["frontend-design:frontend-design", "personal:performance"] },
  { id: "personal:release-management", name: "Release Management", plugin: "personal", description: "Version bump, changelog, and publish workflow", type: "rigid", invokes: ["superpowers:finishing-a-development-branch"] },
  { id: "personal:security-audit", name: "Security Audit", plugin: "personal", description: "OWASP Top 10 review, secret scanning, dependency CVEs, auth pattern audit", type: "rigid" },
  { id: "personal:refactoring", name: "Refactoring", plugin: "personal", description: "Structured safe refactoring with characterization tests and behavior verification", type: "rigid", invokes: ["superpowers:test-driven-development", "superpowers:verification-before-completion", "personal:commit"] },
  { id: "personal:deep-research", name: "Deep Research", plugin: "personal", description: "Multi-step research with source verification and structured synthesis", type: "flexible", invokes: ["personal:architecture"] },
  { id: "personal:codebase-docs", name: "Codebase Docs", plugin: "personal", description: "Generate architecture overviews, API docs, onboarding guides, and dependency maps", type: "flexible" },
  { id: "personal:architecture", name: "Architecture", plugin: "personal", description: "ADRs, API contract design, data models, and tradeoff analysis", type: "flexible", invokes: ["superpowers:writing-plans", "frontend-design:frontend-design", "superpowers:test-driven-development", "personal:qa-test-design", "personal:db-migration"] },
  { id: "personal:keybindings-help", name: "Keybindings Help", plugin: "personal", description: "Customize keyboard shortcuts, rebind keys, add chord bindings", type: "flexible" },
  { id: "personal:accessibility", name: "Accessibility", plugin: "personal", description: "WCAG 2.2 AA compliance audit: semantic HTML, ARIA, keyboard nav, color contrast", type: "rigid", invokes: ["frontend-design:frontend-design"] },
  { id: "personal:qa-test-design", name: "QA Test Design", plugin: "personal", description: "Test quality engineering: boundary analysis, assertion quality, flaky test prevention", type: "rigid", invokes: ["superpowers:test-driven-development"] },
  { id: "personal:db-migration", name: "DB Migration", plugin: "personal", description: "Zero-downtime database migrations with reversibility enforcement", type: "rigid", invokes: ["superpowers:verification-before-completion"] },
  { id: "personal:performance", name: "Performance", plugin: "personal", description: "Data-driven performance profiling: measure first, optimize second", type: "rigid", invokes: ["personal:refactoring"] },
  { id: "personal:e2e-testing", name: "E2E Testing", plugin: "personal", description: "Playwright browser testing: POM, visual regression, smoke suites", type: "rigid", invokes: ["superpowers:systematic-debugging"] },
  { id: "personal:cold-outreach", name: "Cold Outreach", plugin: "personal", description: "Systematic warm outreach to target developers with research and tracking", type: "rigid", invokes: ["personal:competitor-monitor", "personal:x-brief", "personal:cmo", "personal:content-atomizer"] },
  { id: "personal:blog-writing", name: "Blog Writing", plugin: "personal", description: "Standalone blog posts with authentic voice and rigorous fact-checking", type: "rigid", invokes: ["personal:seo", "personal:content-atomizer"] },
  { id: "personal:content-atomizer", name: "Content Atomizer", plugin: "personal", description: "Turn one content piece into platform-specific variants (X, LinkedIn, newsletter, video)", type: "flexible", invokes: ["personal:growth-marketing", "personal:cmo"] },
  { id: "personal:competitor-monitor", name: "Competitor Monitor", plugin: "personal", description: "Scan competitor activity (Mem0, Letta, Zep) for intelligence briefs", type: "flexible", invokes: ["personal:cmo", "personal:growth-marketing", "personal:content-page-builder", "personal:content-atomizer"] },
  { id: "personal:content-page-builder", name: "Content Page Builder", plugin: "personal", description: "Build blog posts, comparison pages, and landing sections for OMEGA website", type: "rigid", invokes: ["personal:seo"] },
  { id: "personal:skill-create", name: "Skill Create", plugin: "personal", description: "Analyze patterns to auto-generate new skills from repeated workflows", type: "flexible" },
  { id: "personal:cloud-sync", name: "Cloud Sync", plugin: "personal", description: "Debug and extend the OMEGA cloud sync pipeline (SQLite to Supabase)", type: "rigid" },
  { id: "personal:coordination-internals", name: "Coordination Internals", plugin: "personal", description: "Modify the OMEGA multi-agent coordination system (sessions, claims, tasks)", type: "rigid" },
  { id: "personal:hook-development", name: "Hook Development", plugin: "personal", description: "Create and modify hooks in the OMEGA hook system (fast_hook, guards)", type: "rigid" },
  { id: "personal:feature-pipeline", name: "Feature Pipeline", plugin: "personal", description: "Build full-stack OMEGA features across Python, DB, API, and admin UI", type: "rigid", invokes: ["personal:architecture", "personal:admin-dashboard", "personal:db-migration"] },
  { id: "personal:memory-engine", name: "Memory Engine", plugin: "personal", description: "Modify the OMEGA memory engine (bridge.py, store/query pipeline)", type: "rigid" },
  { id: "personal:debugging-triage", name: "Debugging Triage", plugin: "personal", description: "Structured bug diagnosis: reproduce, classify, isolate, root-cause, fix, guard", type: "rigid", invokes: ["superpowers:systematic-debugging"] },
  { id: "personal:admin-dashboard", name: "Admin Dashboard", plugin: "personal", description: "Build and modify the OMEGA admin dashboard (Next.js, 9 tabs, dark theme)", type: "rigid" },
  { id: "personal:status", name: "Status", plugin: "personal", description: "Quick project status check across all subsystems", type: "flexible" },
  { id: "personal:visual-explainer", name: "Visual Explainer", plugin: "personal", description: "Generate self-contained HTML pages that visually explain systems and data", type: "flexible" },
  { id: "personal:session-focus", name: "Session Focus", plugin: "personal", description: "Prevent goal drift with keyword contracts and drift checks", type: "rigid" },
  { id: "personal:post-incident", name: "Post-Incident", plugin: "personal", description: "Structured post-mortem with 5-whys and blast radius assessment", type: "rigid", invokes: ["personal:debugging-triage"] },
  { id: "personal:audit-fix", name: "Audit Fix", plugin: "personal", description: "Audit a system, report findings, then fix approved items in one session", type: "rigid", invokes: ["personal:audit"] },
  { id: "personal:pushback", name: "Pushback", plugin: "personal", description: "Strategic challenge before high-stakes decisions. The mirror agents don't provide.", type: "flexible" },
  { id: "personal:gtm", name: "GTM", plugin: "personal", description: "Go-to-market planning for product launches and feature releases", type: "flexible", invokes: ["personal:cmo", "personal:growth-marketing"] },
  { id: "personal:roadmap", name: "Roadmap", plugin: "personal", description: "Product roadmap planning with RICE and Now-Next-Later frameworks", type: "flexible", invokes: ["personal:architecture"] },
  { id: "personal:marketing-advisor", name: "Marketing Advisor", plugin: "personal", description: "Active marketing execution advisor for content calendar and streak tracking", type: "flexible", invokes: ["personal:cmo", "personal:growth-marketing"] },
  { id: "personal:env-preflight", name: "Env Preflight", plugin: "personal", description: "Validate environment configuration before feature work", type: "rigid" },
  { id: "personal:deploy-preflight", name: "Deploy Preflight", plugin: "personal", description: "Pre-deploy checklist for Vercel deployments", type: "rigid" },
  { id: "personal:cross-project-sync", name: "Cross-Project Sync", plugin: "personal", description: "Sync code between omega (private) and omega-public repos with tier safety", type: "rigid" },
  { id: "personal:api-contract-testing", name: "API Contract Testing", plugin: "personal", description: "Test MCP tool schemas, API contracts, and request/response shapes", type: "rigid", invokes: ["personal:qa-test-design"] },
  { id: "personal:dependency-audit", name: "Dependency Audit", plugin: "personal", description: "Audit dependency health, scan CVEs, check outdated packages", type: "rigid" },
  { id: "personal:ship-deploy", name: "Ship Deploy", plugin: "personal", description: "Full ship-to-production pipeline with gated phases", type: "rigid", invokes: ["personal:ship", "personal:deploy-preflight"] },
];

export const PLUGIN_COLORS: Record<string, string> = {
  superpowers: "#6b9fff",
  "frontend-design": "#e8a040",
  "claude-md-management": "#b088e8",
  "claude-code-setup": "#40c8c8",
  "pr-review-toolkit": "#f06060",
  vercel: "#5ec9a0",
  firecrawl: "#ff6b6b",
  "commit-commands": "#7878a0",
  "feature-dev": "#e63956",
  "code-review": "#ff9f43",
  "ralph-loop": "#c084fc",
  "agent-sdk-dev": "#38bdf8",
  personal: "#e63956",
};
