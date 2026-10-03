"""OMEGA Plugin Interface — extensible discovery for commercial modules.

Core provides entry-point-based plugin discovery. Commercial packages
register via ``[project.entry-points."omega.plugins"]`` in their pyproject.toml.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("omega.plugins")

# How long a scan of installed plugins is reused for capability checks. A scan
# reads every installed package's entry points (about 0.7 ms) and store()
# asks for a capability several times per write, while the answer changes
# only when a package is installed or removed.
_DISCOVERY_REUSE_S = 60.0
_capability_plugins_cache: tuple[Callable, float, list[OmegaPlugin]] | None = None


class OmegaPlugin:
    """Base class for OMEGA plugins.

    Subclasses should populate these class-level attributes:

    - ``TOOL_SCHEMAS``: list of MCP tool schema dicts
    - ``HANDLERS``: dict mapping tool name → async handler function
    - ``HOOK_HANDLERS``: dict mapping hook name → sync handler function
    - ``CLI_COMMANDS``: list of (name, setup_func) tuples where setup_func(subparsers)
      registers an argparse subparser; the plugin class should also provide a
      ``cmd_{name}(args)`` method as the command handler
    - ``HOOKS_JSON``: dict matching the hooks.json manifest format (optional)
    - ``RETRIEVAL_PROFILES``: dict mapping event_type → (vec, text, word, ctx, graph) phase weights
    - ``SCORE_MODIFIERS``: list of fn(node_id, score, metadata) → score callables
    - ``CAPABILITIES``: set/list of capability strings provided by the plugin,
      for example ``{"unlimited_memory", "full_retrieval", "pro_tools"}``
    """

    TOOL_SCHEMAS: list[dict[str, Any]] = []
    HANDLERS: dict[str, Callable] = {}
    HOOK_HANDLERS: dict[str, Callable] = {}
    CLI_COMMANDS: list[tuple[str, Callable]] = []
    HOOKS_JSON: dict[str, Any] = {}
    RETRIEVAL_PROFILES: dict[str, tuple] = {}
    SCORE_MODIFIERS: list[Callable] = []
    CAPABILITIES: set[str] = set()


def discover_plugins() -> list[OmegaPlugin]:
    """Discover and instantiate all registered OMEGA plugins.

    Looks up ``omega.plugins`` entry-point group via importlib.metadata.
    Each entry point should reference a class that inherits from OmegaPlugin.
    Returns an empty list if no plugins are installed.
    """
    plugins: list[OmegaPlugin] = []
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="omega.plugins")
        for ep in eps:
            try:
                plugin_cls = ep.load()
                if isinstance(plugin_cls, type) and issubclass(plugin_cls, OmegaPlugin):
                    plugins.append(plugin_cls())
                elif isinstance(plugin_cls, OmegaPlugin):
                    plugins.append(plugin_cls)
                else:
                    logger.warning("Plugin %s is not an OmegaPlugin subclass, skipping", ep.name)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", ep.name, e)
    except Exception as e:
        logger.debug("Plugin discovery unavailable: %s", e)
    return plugins


def reset_plugin_cache() -> None:
    """Forget the plugins reused for capability checks, so the next check rescans."""
    global _capability_plugins_cache
    _capability_plugins_cache = None


def _capability_plugins() -> list[OmegaPlugin]:
    """The installed plugins, from a scan at most _DISCOVERY_REUSE_S old.

    Only the scan is reused. Each plugin's capabilities are still read on
    every call, so a license that starts or stops being valid takes effect
    at once. A swapped-in discover_plugins (an embedder, a test) is scanned
    afresh.
    """
    global _capability_plugins_cache
    now = time.monotonic()
    cached = _capability_plugins_cache
    if cached is None or cached[0] is not discover_plugins or now - cached[1] >= _DISCOVERY_REUSE_S:
        cached = (discover_plugins, now, discover_plugins())
        _capability_plugins_cache = cached
    return cached[2]


def get_capabilities() -> set[str]:
    """Return capabilities advertised by installed OMEGA plugins.

    Core uses capabilities for feature availability, not payment entitlement.
    A private extension may register capabilities; public Core never trusts a
    local license function to unlock behavior by itself.
    """
    capabilities: set[str] = set()
    for plugin in _capability_plugins():
        raw = getattr(plugin, "CAPABILITIES", None)
        if callable(raw):
            raw = raw()
        if not raw:
            raw = getattr(plugin, "capabilities", None)
            if callable(raw):
                raw = raw()
        if not raw:
            continue
        try:
            capabilities.update(str(cap) for cap in raw)
        except TypeError:
            logger.warning("Plugin %s returned invalid capabilities", plugin.__class__.__name__)
    return capabilities


def has_capability(name: str) -> bool:
    """Return True when an installed plugin provides ``name``."""
    return name in get_capabilities()
