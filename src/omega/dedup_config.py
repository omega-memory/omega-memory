"""Validated content-dedup threshold configuration.

``OMEGA_DEDUP_THRESHOLDS`` is read once when the bridge process imports this
module. Restart the MCP process after changing it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from omega.json_compat import loads
from omega.types import AutoCaptureEventType


DEFAULT_DEDUP_THRESHOLDS: dict[str, float] = {
    AutoCaptureEventType.ERROR_PATTERN: 0.70,
    AutoCaptureEventType.SESSION_SUMMARY: 0.75,
    AutoCaptureEventType.TASK_COMPLETION: 0.85,
    AutoCaptureEventType.DECISION: 0.80,
    AutoCaptureEventType.LESSON_LEARNED: 0.85,
    AutoCaptureEventType.CHECKPOINT: 0.90,
    AutoCaptureEventType.CONSTRAINT: 0.90,
    AutoCaptureEventType.ADVISOR_INSIGHT: 0.75,
    AutoCaptureEventType.USER_FACT: 0.80,
    AutoCaptureEventType.SKILL_TEMPLATE: 0.85,
    AutoCaptureEventType.PROJECT_STATUS: 0.85,
    "memory": 0.80,
}


def load_dedup_thresholds(
    env: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Return defaults overlaid with a validated JSON environment mapping."""
    source = os.environ if env is None else env
    raw = source.get("OMEGA_DEDUP_THRESHOLDS")
    thresholds = dict(DEFAULT_DEDUP_THRESHOLDS)
    if raw is None or not raw.strip():
        return thresholds

    try:
        overrides = loads(raw)
    except Exception as exc:
        raise ValueError("OMEGA_DEDUP_THRESHOLDS must be valid JSON") from exc
    if not isinstance(overrides, dict):
        raise ValueError("OMEGA_DEDUP_THRESHOLDS must be a JSON object")

    for raw_key, raw_value in overrides.items():
        key = str(raw_key).lower()
        if key not in thresholds:
            raise ValueError(f"unknown dedup threshold key: {key}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"dedup threshold {key} must be a number")
        value = float(raw_value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"dedup threshold {key} must be between 0.0 and 1.0")
        thresholds[key] = value

    return thresholds
