# omega-public — Project Instructions

> Public core-only repo for `omega-memory` PyPI package. GitHub: `omega-memory/core`

## Pro-Only Blocklist (NEVER sync these from private repo)

These files and features are PRO-ONLY and must NEVER exist in this repo:

| Category | Blocked patterns |
|----------|-----------------|
| **Coordination** | `coordination.py`, `coord_handlers.py`, `coord_schemas.py`, any `coord_*` file |
| **Entity** | `entity/` directory, entity tools/handlers |
| **Cloud** | `cloud/` directory, cloud sync tools/handlers |
| **Pro tools** | Any tool schema or handler not in the 26 core tools |
| **Protocol sections** | Coordination-related sections in `protocol.py` (coordination, coordination_gate, teamwork, external action tools) |

**Before syncing ANY file from `~/Projects/omega/`:**
1. Check this blocklist
2. If the file references `coord_`, entity, or cloud features — do NOT sync
3. If `protocol.py` — strip coordination/teamwork sections before syncing
4. When in doubt — ASK the user

A pre-commit hook enforces this. Do not bypass it with `--no-verify`.

## OMEGA Memory Integration

- This repo's work is NOT auto-captured by OMEGA hooks into the main project scope.
- **After every commit**: run `omega_store("Committed <hash>: <message>. Files: <list>", "decision")` — this is critical since agents in ~/Projects/omega/ won't see your work otherwise.
- **After version bumps**: run `omega_store("omega-memory bumped to v<X.Y.Z> — <reason>", "decision")` explicitly.
- **Before editing**: `omega_query("omega-public <topic>")` to check for prior decisions.

## Key Facts

- PyPI package name: `omega-memory`
- Trusted publisher (OIDC) via GitHub Actions (`publish.yml`)
- This is a SUBSET of `~/Projects/omega/` — core memory tools only, no pro features
- 26 core MCP tools (out of 83 total in the private monorepo)

## Git

- Commit only files YOU modified. Use `git add <files>` — never `git add .`
- Do not push without user approval.
