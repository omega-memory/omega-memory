<!-- OMEGA:BEGIN — managed by omega setup, do not edit this block -->
## Memory (OMEGA)

You have OMEGA persistent memory. At session start:
1. Call `omega_welcome()` for context briefing
2. Call `omega_protocol()` for your operating instructions — it's your coordination playbook
3. Follow the protocol it returns

Quick reference (protocol has full details):
- `[MEMORY]`/`[HANDOFF]`/`[COORD]` blocks from hooks = ground truth
- Before non-trivial tasks: `omega_query()` for prior context
- Before spawning subagents: `omega_query()` first, inject results into agent prompt (subagents can't call OMEGA)
- After completing tasks: `omega_store(content, "decision")` for key outcomes — minimum 1 store per session
- User says "remember": `omega_store(text, "user_preference")`
- Context getting full: `omega_checkpoint` to save state
- Load user context: `omega_profile()` after welcome/protocol
- Before architecture decisions: `omega_reflect(action="evolution", topic=<domain>)` to check prior thinking
- After `omega_store`: check `omega_memory(similar)` and link related memories to build the knowledge graph
- NEVER fabricate URLs — read from files, query OMEGA, or verify via web fetch

If OMEGA is unavailable, use basic coordination:
- Before state changes: check `git log` and ask before deploying
- After tasks: store decisions with `omega_store()`
<!-- OMEGA:END -->
