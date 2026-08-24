"""Self-audit council system.

Runs domain-specific analysis (platform health, security, innovation)
by gathering signals from OMEGA's memory and coordination systems,
then producing structured findings via LLM analysis.
"""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COUNCIL_DOMAINS = ("platform_health", "security", "innovation")

# Resolve config dir relative to package root
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "councils"


class Council:
    def __init__(self, domain: str, config_dir: str | None = None, db_dir: str | None = None):
        self.domain = domain
        self._config_dir = Path(config_dir) if config_dir else _CONFIG_DIR
        self._db_dir = db_dir

        template_path = self._config_dir / f"{domain}.md"
        if not template_path.exists():
            raise FileNotFoundError(f"No council template at {template_path}")
        self.prompt_template = template_path.read_text()

    def gather_signals(self, project: str | None = None) -> dict[str, Any]:
        """Gather input signals for this council domain."""
        signals: dict[str, Any] = {}

        if self.domain == "platform_health":
            signals["recent_errors"] = self._query_recent_errors(project)
            signals["health_metrics"] = self._get_health_metrics()
            signals["tool_failure_rates"] = self._get_tool_failures(project)

        elif self.domain == "security":
            signals["credential_patterns"] = self._scan_credential_patterns()
            signals["external_actions"] = self._get_external_actions(project)
            signals["recent_content_samples"] = self._get_content_samples()

        elif self.domain == "innovation":
            signals["tool_usage"] = self._get_tool_usage()
            signals["recent_decisions"] = self._get_recent_decisions(project)
            signals["recent_lessons"] = self._get_recent_lessons(project)

        return signals

    def format_prompt(self, signals: dict[str, Any]) -> str:
        """Combine template with gathered signals into a complete prompt."""
        signals_text = json.dumps(signals, indent=2, default=str)
        return f"{self.prompt_template}\n\n## Signals Data\n\n```json\n{signals_text}\n```"

    # ------------------------------------------------------------------
    # Platform Health signal gatherers
    # ------------------------------------------------------------------

    def _query_recent_errors(self, project: str | None) -> list[dict]:
        try:
            from omega.bridge import query

            result = query(
                "recent errors and failures in last 24 hours",
                event_type="error_pattern",
                limit=20,
                project=project,
            )
            if isinstance(result, str) and result.strip():
                return [{"raw": result}]
            return []
        except Exception as e:
            logger.debug("Failed to query errors: %s", e)
            return []

    def _get_health_metrics(self) -> dict:
        try:
            from omega.bridge import check_health

            result = check_health()
            # check_health returns a formatted markdown string
            if isinstance(result, str):
                return {"raw": result}
            return {}
        except Exception:
            return {}

    def _get_tool_failures(self, project: str | None) -> list:
        try:
            from omega_platform.orchestrator.coordination import get_manager

            mgr = get_manager()
            with mgr._lock:
                rows = mgr.get_read_connection().execute(
                    """SELECT tool_name, COUNT(*) as fail_count
                       FROM coord_audit
                       WHERE result_summary LIKE '%error%'
                         AND created_at > datetime('now', '-1 day')
                       GROUP BY tool_name
                       ORDER BY fail_count DESC
                       LIMIT 10"""
                ).fetchall()
            return [{"tool": r[0], "failures": r[1]} for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Security signal gatherers
    # ------------------------------------------------------------------

    def _scan_credential_patterns(self) -> list:
        try:
            import re
            from omega.bridge import query

            result = query("recent stored content", limit=50)
            patterns = [
                r"sk-[a-zA-Z0-9]{20,}",
                r"key-[a-zA-Z0-9]{20,}",
                r"Bearer [a-zA-Z0-9\-._~+/]+=*",
                r"password\s*[:=]\s*\S+",
            ]
            findings = []
            content = result if isinstance(result, str) else str(result)
            for p in patterns:
                if re.search(p, content):
                    findings.append({"pattern": p, "found": True})
            return findings
        except Exception:
            return []

    def _get_external_actions(self, project: str | None) -> list:
        try:
            from omega_platform.orchestrator.coordination import get_manager

            mgr = get_manager()
            with mgr._lock:
                rows = mgr.get_read_connection().execute(
                    """SELECT action_type, action_target, status, created_at
                       FROM coord_external_actions
                       WHERE created_at > datetime('now', '-1 day')
                       ORDER BY created_at DESC LIMIT 20"""
                ).fetchall()
            return [{"type": r[0], "target": r[1], "status": r[2], "at": r[3]} for r in rows]
        except Exception:
            return []

    def _get_content_samples(self) -> list:
        return []  # Placeholder -- intentionally empty for security reasons

    # ------------------------------------------------------------------
    # Innovation signal gatherers
    # ------------------------------------------------------------------

    def _get_tool_usage(self) -> list:
        try:
            from omega_platform.orchestrator.coordination import get_manager

            mgr = get_manager()
            with mgr._lock:
                rows = mgr.get_read_connection().execute(
                    """SELECT tool_name, COUNT(*) as calls
                       FROM coord_audit
                       WHERE created_at > datetime('now', '-7 days')
                       GROUP BY tool_name
                       ORDER BY calls DESC"""
                ).fetchall()
            return [{"tool": r[0], "calls_7d": r[1]} for r in rows]
        except Exception:
            return []

    def _get_recent_decisions(self, project: str | None) -> list:
        try:
            from omega.bridge import query

            result = query(
                "recent decisions",
                event_type="decision",
                limit=10,
                project=project,
            )
            if isinstance(result, str) and result.strip():
                return [{"raw": result}]
            return []
        except Exception:
            return []

    def _get_recent_lessons(self, project: str | None) -> list:
        try:
            from omega.bridge import query

            result = query(
                "recent lessons learned",
                event_type="lesson_learned",
                limit=10,
                project=project,
            )
            if isinstance(result, str) and result.strip():
                return [{"raw": result}]
            return []
        except Exception:
            return []
