# Releasing OMEGA Core

Core (`omega-memory`) is the public, Apache-2.0 package on PyPI. Pro
(`omega-platform`) is a separate private wheel behind a licence. This document
covers Core releases only; nothing here should ever push Pro to PyPI.

Most of this is enforced by `scripts/preflight.py` and by `scripts/release.py`,
which refuses to publish when a gate fails. The checklist below is the part
that needs a human.

## The product lines, and why the version matters

| Line | Package | Repo | Tags | Audience |
|---|---|---|---|---|
| Core | `omega-memory` | `omega-memory/omega-memory` (public) | `v*` | everyone |
| Pro | `omega-platform` | private monorepo | `pro-v*` | licensed |
| Internal | — | private branches | never tagged | nobody outside |

Three rules that are easy to get wrong:

1. **Core versions stay below 1.6.** The Pro wheel pins
   `omega-memory>=1.5.13,<1.6`, so publishing Core 1.6.0 breaks dependency
   resolution for every Pro install. The 1.6 and 1.7 lines are also reserved
   as internal milestone labels and must never appear as public Core versions.
2. **Never use a `v*` tag in the private repo, or a `pro-v*` tag here.**
3. **Version axes are independent.** Core version, Pro version, schema version
   and internal milestone are four different numbers. Never infer one from
   another, and never print a bare version without saying which one it is.

## Before you release

```bash
python3.11 scripts/preflight.py <version>
```

It builds the artifacts and runs every automated gate: version agreement, the
1.6 ceiling, changelog entry, unused tag, branch hygiene, the artifact privacy
and Core/Pro boundary scan, the free-tier cap, the paywall, lint and tests.
Add `--fast` to skip the clean-venv gates while iterating; never skip them for
a real release.

Then confirm the things a script cannot judge:

- [ ] **The changelog entry describes the user-visible effect**, not the diff.
      Someone reading it should know whether it affects them.
- [ ] **Behaviour changes are called out as such.** A ranking change or a new
      exception is not merely a "fix" to someone depending on the old shape.
- [ ] **Reporters are credited by GitHub handle**, not the display name shown in
      a notification email. Check the handle resolves.
- [ ] **Nothing in the diff came from the private tree without checking it
      applies.** Core and Pro have diverged: Pro requires `cryptography` and
      has a writer-daemon architecture Core does not. A fix that is correct
      there can be wrong or meaningless here.
- [ ] **No new required dependency** unless that is the point of the release.

## Releasing

```bash
python3.11 scripts/release.py <version> --dry-run   # build + verify only
python3.11 scripts/release.py <version>             # publish
```

`release.py` re-runs the artifact boundary scan itself, so a privacy or
boundary violation blocks the publish even if preflight was skipped.

## After

- [ ] Install from PyPI into a clean venv and exercise the actual fix. "Tests
      pass" is not the same as "the published artifact works."
- [ ] Reply to any issue or discussion the release closes.

## Branch and worktree practice

- **Branch from `origin/main`, not from whatever is checked out.** Verify with
  `git log --oneline origin/main..HEAD` before you start.
- **One branch, one concern.** The commit message says why, not what.
- **Never `git add .`.** Stage by path, every time.
- **Check `git worktree list` before releasing.** Releases here are cut from
  worktrees, and a second checkout of `main` means another session may be
  committing to the ref you are about to tag. Preflight warns about this.
- **Building from a worktree used to leak.** In a worktree `.git` is a *file*
  containing an absolute `gitdir:` path. Hatchling packaged it, so every sdist
  from 1.5.11 to 1.5.15 shipped a maintainer's home directory and internal
  worktree names. `[tool.hatch.build.targets.sdist]` now excludes it and the
  scanner fails closed if it ever returns.
- **Do not leave commits stranded on a local branch.** Push or delete it.
- **Verify a push actually succeeded.** `git push … | tail` reports *tail's*
  exit status, not the push's. Check the command's own exit code.

## Writing fixtures and docs

The sdist ships `tests/` and `docs/`, so anything written there is published.

- Use a placeholder home directory: `/Users/me`, `/Users/test`, `/Users/dev`,
  `/Users/maintainer`. The scanner treats any other name as a real account and
  fails the release.
- Never paste a path from your own machine, even in a comment.
- Fake credentials in `tests/` are fine and expected. Outside `tests/`, a
  credential-shaped string fails the release.

## What the scanner will not catch

It reads text. It cannot tell you whether a changelog entry is honest, whether
a fix is correct, or whether a behaviour change deserves a major version. That
is what the human checklist above is for.
