"""OMEGA Milestones -- Centralized milestone and streak tracking.

Single source of truth for milestone checks (previously duplicated across
bridge.py, hook_server.py, and surface_memories.py).
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from omega import json_compat as json

logger = logging.getLogger("omega.milestones")

OMEGA_HOME = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))
MILESTONES_DIR = OMEGA_HOME / "milestones"

# Capture count thresholds that trigger milestones.
# Above 100 the cadence tracks the Free-tier cap layout:
#   500   - one-quarter of the soft cap
#   1000  - halfway
#   1500  - approaching the soft cap, retrieval still full quality
#   2000  - soft cap hit; Free-tier retrieval degradation kicks in
CAPTURE_THRESHOLDS = [1, 10, 50, 100, 500, 1000, 1500, 2000]

CAPTURE_MESSAGES = {
    1: "First memory captured! Your knowledge graph has begun.",
    10: "10 memories stored! Your knowledge graph is growing.",
    50: "50 memories! You're building a rich knowledge base.",
    100: "100 memories! A substantial personal knowledge graph.",
    500: "500 memories! You have a deep knowledge archive.",
    1000: "1,000 memories! Impressive long-term knowledge base.",
    1500: "1,500 memories — approaching the Free-tier soft cap (2,000).",
    2000: "2,000 memories — Free-tier soft cap reached.",
}

# Free-tier upgrade CTA appended after the base message. Suppressed when
# is_pro_user=True so paid customers see a clean milestone celebration.
# Wired through `check_capture_milestones(count, is_pro_user=...)`.
FREE_UPGRADE_CTA = {
    500: " Pro removes the 2,000 soft cap + adds full retrieval quality: "
         "https://omegamax.co/pro?ref=milestone-500",
    1000: " Halfway to the soft cap. Pro removes it: "
          "https://omegamax.co/pro?ref=milestone-1000",
    1500: " Past 2,000, retrieval quality drops (no reranker, no LLM "
          "expansion, top-3 results) until you upgrade: "
          "https://omegamax.co/pro?ref=milestone-1500",
    2000: " Retrieval is now reduced; hard write cap at 5,000. Upgrade "
          "for instant full quality + unlimited memories: "
          "https://omegamax.co/pro?ref=milestone-2000",
}

STREAK_THRESHOLDS = [7, 30, 100, 365]

STREAK_MESSAGES = {
    7: "7-day streak! Building a consistent memory habit.",
    30: "30-day streak! A full month of continuous memory.",
    100: "100-day streak! Remarkable long-term commitment.",
    365: "365-day streak! A full year of continuous memory.",
}


def _check_milestone(name: str) -> bool:
    """Return True if milestone not yet achieved (first time). Creates marker with metadata."""
    marker = MILESTONES_DIR / name
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"achieved_at": datetime.now(timezone.utc).isoformat(), "name": name}
    marker.write_text(json.dumps(metadata))
    return True


def check_capture_milestones(count: int, *, is_pro_user: bool = False) -> Optional[str]:
    """Check if the current capture count crosses a milestone threshold.

    Returns a milestone message if a new threshold is reached, None otherwise.
    Iterates descending so the highest crossed threshold fires first.
    Uses >= so thresholds crossed between checks are still caught.

    `is_pro_user` suppresses the Free-tier upgrade CTA suffix. The base
    celebratory message stays the same so Pro users still see "1,500
    memories — approaching the Free-tier soft cap" without the upgrade
    URL. Callers pass `is_pro()` from omega_platform.license.
    """
    for threshold in reversed(CAPTURE_THRESHOLDS):
        if count >= threshold:
            name = f"capture-{threshold}"
            if _check_milestone(name):
                # Mark all lower thresholds as achieved too
                for lower in CAPTURE_THRESHOLDS:
                    if lower < threshold:
                        _check_milestone(f"capture-{lower}")
                base = CAPTURE_MESSAGES.get(threshold, f"{threshold} memories captured!")
                if not is_pro_user and threshold in FREE_UPGRADE_CTA:
                    return base + FREE_UPGRADE_CTA[threshold]
                return base
    return None


def check_streak_milestones(streak_days: int) -> Optional[str]:
    """Check if the current streak crosses a milestone threshold.

    Iterates descending so the highest crossed threshold fires first.
    Uses >= so thresholds crossed between sessions are still caught.
    """
    for threshold in reversed(STREAK_THRESHOLDS):
        if streak_days >= threshold:
            name = f"streak-{threshold}"
            if _check_milestone(name):
                for lower in STREAK_THRESHOLDS:
                    if lower < threshold:
                        _check_milestone(f"streak-{lower}")
                return STREAK_MESSAGES.get(threshold, f"{threshold}-day streak!")
    return None


def get_streak(store) -> dict:
    """Calculate the current and longest usage streaks from memory timestamps.

    Returns {"current": N, "longest": N, "today_active": bool}.
    """
    try:
        rows = store._conn.execute(
            "SELECT DISTINCT DATE(created_at) as day FROM memories ORDER BY day DESC"
        ).fetchall()
    except Exception as e:
        logger.debug("Streak query failed: %s", e)
        return {"current": 0, "longest": 0, "today_active": False}

    if not rows:
        return {"current": 0, "longest": 0, "today_active": False}

    days = [r[0] for r in rows]
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_active = days[0] == today_str

    # Walk consecutive days from most recent
    current = 1
    for i in range(len(days) - 1):
        try:
            d1 = datetime.strptime(days[i], "%Y-%m-%d")
            d2 = datetime.strptime(days[i + 1], "%Y-%m-%d")
            if (d1 - d2).days == 1:
                current += 1
            else:
                break
        except (ValueError, TypeError):
            break

    # If most recent day isn't today or yesterday, streak is 0
    if days[0] != today_str:
        try:
            most_recent = datetime.strptime(days[0], "%Y-%m-%d")
            today_dt = datetime.strptime(today_str, "%Y-%m-%d")
            if (today_dt - most_recent).days > 1:
                current = 0
        except (ValueError, TypeError):
            pass

    # Calculate longest streak across all days
    longest = 1
    run = 1
    for i in range(len(days) - 1):
        try:
            d1 = datetime.strptime(days[i], "%Y-%m-%d")
            d2 = datetime.strptime(days[i + 1], "%Y-%m-%d")
            if (d1 - d2).days == 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        except (ValueError, TypeError):
            run = 1

    if not days:
        longest = 0

    return {"current": current, "longest": longest, "today_active": today_active}


def list_milestones() -> list[dict]:
    """List all achieved milestones from the milestones directory.

    Returns [{"name": str, "achieved_at": str}] sorted by achieved_at.
    """
    if not MILESTONES_DIR.exists():
        return []

    milestones = []
    for f in MILESTONES_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            try:
                data = json.loads(f.read_text())
                milestones.append({
                    "name": data.get("name", f.name),
                    "achieved_at": data.get("achieved_at", ""),
                })
            except Exception as e:
                logger.debug("Milestone metadata parse failed for %s: %s", f.name, e)
                # Legacy empty marker files
                milestones.append({
                    "name": f.name,
                    "achieved_at": datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                })

    milestones.sort(key=lambda m: m.get("achieved_at", ""))
    return milestones
