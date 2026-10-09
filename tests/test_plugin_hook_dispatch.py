"""Core daemon and direct hook paths consume the public plugin contract."""

import asyncio

from omega import plugins
from omega.hooks import fast_hook
from omega.server.hook_server import core


def test_plugin_lookup_accepts_class_attribute_property_and_method(monkeypatch):
    def handler(request):
        return {"output": "plugin", "error": None}

    class AttributePlugin(plugins.OmegaPlugin):
        HOOK_HANDLERS = {"hook": handler}

    class PropertyPlugin(plugins.OmegaPlugin):
        @property
        def HOOK_HANDLERS(self):
            return {"other": handler}

    class MethodPlugin(plugins.OmegaPlugin):
        def HOOK_HANDLERS(self):
            return {"third": handler}

    monkeypatch.setattr(plugins, "discover_plugins", lambda: [AttributePlugin(), PropertyPlugin(), MethodPlugin()])
    plugins.reset_plugin_cache()
    assert plugins.plugin_hook_handler("hook") is handler
    assert plugins.plugin_hook_handler("other") is handler
    assert plugins.plugin_hook_handler("third") is handler
    assert plugins.plugin_hook_handler("missing") is None
    plugins.reset_plugin_cache()


def test_plugin_lookup_skips_raising_plugin(monkeypatch, caplog):
    def handler(request):
        return {"output": "ok", "error": None}

    class BrokenPlugin(plugins.OmegaPlugin):
        @property
        def HOOK_HANDLERS(self):
            raise RuntimeError("broken")

    class GoodPlugin(plugins.OmegaPlugin):
        HOOK_HANDLERS = {"hook": handler}

    monkeypatch.setattr(plugins, "discover_plugins", lambda: [BrokenPlugin(), GoodPlugin()])
    plugins.reset_plugin_cache()
    assert plugins.plugin_hook_handler("hook") is handler
    assert "broken" in caplog.text
    plugins.reset_plugin_cache()


def test_daemon_dispatch_uses_plugin_when_not_registered(monkeypatch):
    seen = []
    monkeypatch.setattr(plugins, "plugin_hook_handler", lambda name: (lambda request: seen.append(request) or {"output": name, "error": None}))
    result = asyncio.run(core._dispatch("extension_hook", {"session_id": "s"}))
    assert result == {"output": "extension_hook", "error": None}
    assert seen == [{"session_id": "s"}]


def test_fast_hook_without_daemon_runs_plugin_and_emits_output(monkeypatch, capsys):
    monkeypatch.setattr(fast_hook, "_delegate_with_retries", lambda *args: None)
    monkeypatch.setattr(fast_hook, "_parse_payload", lambda: {"session_id": "s"})
    monkeypatch.setattr(fast_hook, "_log_timing", lambda *args: None)
    monkeypatch.setattr(fast_hook.sys, "argv", ["fast_hook.py", "extension_hook"])
    monkeypatch.setattr(plugins, "plugin_hook_handler", lambda name: lambda request: {"output": request["session_id"], "error": None})
    monkeypatch.setattr(fast_hook, "_fallback", lambda *args: (_ for _ in ()).throw(AssertionError("script fallback called")))
    fast_hook.main()
    assert capsys.readouterr().out.strip() == "s"


def test_informational_plugin_error_fails_open(monkeypatch):
    monkeypatch.setattr(plugins, "plugin_hook_handler", lambda name: lambda request: (_ for _ in ()).throw(RuntimeError("oops")))
    assert fast_hook._plugin_fallback("extension_hook", {}) is True


def test_blocking_plugin_exit_code_is_preserved(monkeypatch):
    import pytest
    monkeypatch.setattr(plugins, "plugin_hook_handler", lambda name: lambda request: {"output": "blocked", "error": None, "exit_code": 2})
    with pytest.raises(SystemExit, match="2"):
        fast_hook._plugin_fallback("pre_task_guard", {})


def test_fast_hook_without_daemon_or_plugin_is_quiet_for_an_unknown_hook(monkeypatch, capsys):
    monkeypatch.setattr(fast_hook, "_delegate_with_retries", lambda *args: None)
    monkeypatch.setattr(fast_hook, "_parse_payload", lambda: {"session_id": "s"})
    monkeypatch.setattr(fast_hook, "_log_timing", lambda *args: None)
    monkeypatch.setattr(fast_hook.sys, "argv", ["fast_hook.py", "extension_hook"])
    monkeypatch.setattr(plugins, "plugin_hook_handler", lambda name: None)
    fast_hook.main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_core_hooks_are_never_routed_to_a_plugin(monkeypatch):
    monkeypatch.setattr(plugins, "plugin_hook_handler", lambda name: (_ for _ in ()).throw(AssertionError("plugin consulted")))
    for name in fast_hook._CORE_HOOKS:
        assert fast_hook._plugin_fallback(name, {}) is False
