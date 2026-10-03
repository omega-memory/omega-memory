"""Tests for omega.plugins discovery and OmegaPlugin base class."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import omega.plugins as plugins_module
from omega.plugins import (
    OmegaPlugin,
    discover_plugins,
    get_capabilities,
    has_capability,
    reset_plugin_cache,
)


@pytest.fixture(autouse=True)
def _fresh_plugin_scan():
    """Capability checks reuse a plugin scan; start and leave each test without one."""
    reset_plugin_cache()
    yield
    reset_plugin_cache()


class TestOmegaPluginBase:
    def test_default_attributes(self):
        plugin = OmegaPlugin()
        assert plugin.TOOL_SCHEMAS == []
        assert plugin.HANDLERS == {}
        assert plugin.HOOK_HANDLERS == {}
        assert plugin.CLI_COMMANDS == []
        assert plugin.HOOKS_JSON == {}
        assert plugin.RETRIEVAL_PROFILES == {}
        assert plugin.SCORE_MODIFIERS == []
        assert plugin.CAPABILITIES == set()

    def test_subclass_inherits(self):
        class MyPlugin(OmegaPlugin):
            TOOL_SCHEMAS = [{"name": "test_tool"}]
            HANDLERS = {"test_tool": lambda: None}

        p = MyPlugin()
        assert len(p.TOOL_SCHEMAS) == 1
        assert "test_tool" in p.HANDLERS
        assert isinstance(p, OmegaPlugin)


class TestDiscoverPlugins:
    @patch("importlib.metadata.entry_points")
    def test_no_plugins(self, mock_ep):
        mock_ep.return_value = []
        result = discover_plugins()
        assert result == []

    @patch("importlib.metadata.entry_points")
    def test_valid_plugin_class(self, mock_ep):
        class GoodPlugin(OmegaPlugin):
            TOOL_SCHEMAS = [{"name": "good"}]

        ep = MagicMock()
        ep.name = "good_plugin"
        ep.load.return_value = GoodPlugin
        mock_ep.return_value = [ep]

        result = discover_plugins()
        assert len(result) == 1
        assert isinstance(result[0], GoodPlugin)


class TestPluginCapabilities:
    @patch("importlib.metadata.entry_points")
    def test_get_capabilities_from_plugin_attribute(self, mock_ep):
        class CapPlugin(OmegaPlugin):
            CAPABILITIES = {"unlimited_memory", "full_retrieval"}

        ep = MagicMock()
        ep.name = "cap_plugin"
        ep.load.return_value = CapPlugin
        mock_ep.return_value = [ep]

        assert get_capabilities() == {"unlimited_memory", "full_retrieval"}
        assert has_capability("unlimited_memory") is True
        assert has_capability("pro_tools") is False

    @patch("importlib.metadata.entry_points")
    def test_get_capabilities_from_plugin_method(self, mock_ep):
        class CapPlugin(OmegaPlugin):
            def CAPABILITIES(self):
                return ["pro_tools"]

        ep = MagicMock()
        ep.name = "cap_method_plugin"
        ep.load.return_value = CapPlugin
        mock_ep.return_value = [ep]

        assert get_capabilities() == {"pro_tools"}

    @patch("importlib.metadata.entry_points")
    def test_fake_license_module_does_not_unlock_memory_cap(self, mock_ep, monkeypatch):
        """Core ignores omega_platform.license.is_pro() for cap unlocks."""
        from omega.sqlite_store import _base

        omega_platform = types.ModuleType("omega_platform")
        license_mod = types.ModuleType("omega_platform.license")
        license_mod.is_pro = lambda: True
        monkeypatch.setitem(sys.modules, "omega_platform", omega_platform)
        monkeypatch.setitem(sys.modules, "omega_platform.license", license_mod)
        monkeypatch.setenv("OMEGA_MAX_NODES", "99999")
        mock_ep.return_value = []

        assert _base._get_effective_max_nodes() == _base._CORE_HARD_LIMIT

    @patch("importlib.metadata.entry_points")
    def test_capability_plugin_unlocks_memory_cap_env(self, mock_ep, monkeypatch):
        from omega.sqlite_store import _base

        class CapPlugin(OmegaPlugin):
            CAPABILITIES = {"unlimited_memory"}

        ep = MagicMock()
        ep.name = "cap_plugin"
        ep.load.return_value = CapPlugin
        mock_ep.return_value = [ep]
        monkeypatch.setenv("OMEGA_MAX_NODES", "12345")

        assert _base._get_effective_max_nodes() == 12345

    @patch("importlib.metadata.entry_points")
    def test_invalid_plugin_skipped(self, mock_ep):
        """Non-OmegaPlugin classes are skipped with a warning."""
        ep = MagicMock()
        ep.name = "bad_plugin"
        ep.load.return_value = str  # not an OmegaPlugin

        mock_ep.return_value = [ep]
        result = discover_plugins()
        assert result == []

    @patch("importlib.metadata.entry_points")
    def test_load_failure_graceful(self, mock_ep):
        """Plugin load failures don't crash discovery."""
        ep = MagicMock()
        ep.name = "broken_plugin"
        ep.load.side_effect = ImportError("missing dependency")

        mock_ep.return_value = [ep]
        result = discover_plugins()
        assert result == []

    @patch("importlib.metadata.entry_points")
    def test_instance_accepted(self, mock_ep):
        """Pre-instantiated OmegaPlugin instances are accepted."""
        instance = OmegaPlugin()

        ep = MagicMock()
        ep.name = "instance_plugin"
        ep.load.return_value = instance

        mock_ep.return_value = [ep]
        result = discover_plugins()
        assert len(result) == 1
        assert result[0] is instance

    @patch("importlib.metadata.entry_points", side_effect=Exception("importlib broken"))
    def test_importlib_failure(self, mock_ep):
        """Total importlib failure returns empty list gracefully."""
        result = discover_plugins()
        assert result == []

    @patch("importlib.metadata.entry_points")
    def test_mixed_plugins(self, mock_ep):
        """Valid plugins are returned even when some fail."""
        class GoodPlugin(OmegaPlugin):
            pass

        good_ep = MagicMock()
        good_ep.name = "good"
        good_ep.load.return_value = GoodPlugin

        bad_ep = MagicMock()
        bad_ep.name = "bad"
        bad_ep.load.side_effect = ImportError("broken")

        mock_ep.return_value = [good_ep, bad_ep]
        result = discover_plugins()
        assert len(result) == 1
        assert isinstance(result[0], GoodPlugin)


class TestCapabilityChecksReuseTheScan:
    """store() asks for a capability several times per write; the scan is not repeated."""

    @staticmethod
    def _counting_discovery(monkeypatch, plugin):
        scans = []

        def fake_discover_plugins():
            scans.append(1)
            return [plugin]

        monkeypatch.setattr(plugins_module, "discover_plugins", fake_discover_plugins)
        return scans

    def test_repeated_checks_scan_once(self, monkeypatch):
        class Plugin(OmegaPlugin):
            CAPABILITIES = {"full_retrieval"}

        scans = self._counting_discovery(monkeypatch, Plugin())

        assert [has_capability("full_retrieval") for _ in range(5)] == [True] * 5
        assert has_capability("unlimited_memory") is False
        assert len(scans) == 1

    def test_a_capability_change_shows_without_a_rescan(self, monkeypatch):
        licensed = {"now": False}

        class Plugin(OmegaPlugin):
            def CAPABILITIES(self):
                return {"pro_tools"} if licensed["now"] else set()

        scans = self._counting_discovery(monkeypatch, Plugin())

        assert has_capability("pro_tools") is False
        licensed["now"] = True
        assert has_capability("pro_tools") is True
        licensed["now"] = False
        assert has_capability("pro_tools") is False
        assert len(scans) == 1

    def test_the_scan_is_repeated_once_it_is_old(self, monkeypatch):
        scans = self._counting_discovery(monkeypatch, OmegaPlugin())
        clock = {"now": 1000.0}
        monkeypatch.setattr(plugins_module.time, "monotonic", lambda: clock["now"])

        has_capability("pro_tools")
        clock["now"] += plugins_module._DISCOVERY_REUSE_S - 1
        has_capability("pro_tools")
        assert len(scans) == 1

        clock["now"] += 2
        has_capability("pro_tools")
        assert len(scans) == 2

    def test_reset_forces_a_rescan(self, monkeypatch):
        scans = self._counting_discovery(monkeypatch, OmegaPlugin())

        has_capability("pro_tools")
        reset_plugin_cache()
        has_capability("pro_tools")

        assert len(scans) == 2
