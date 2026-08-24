"""Core's own registry invariants for Pro-tool classification.

These are deliberately Core-only: they must pass in a standalone public Core
checkout with the private omega_platform package ABSENT. The cross-package
contract test that compares these registries against the real Pro plugin
surface lives in the Pro qualification suite, because public Core's ordinary
test run must never depend on installing private Pro.

Background: Core shipped 1.5.11 and 1.5.12 with 65 Pro tools missing from
TOOL_CATEGORIES. That made every one of them answer "Unknown tool" instead
of "requires OMEGA Pro" for an unlicensed caller.
"""

from __future__ import annotations

import pytest

from omega.server.tool_schemas import TOOL_CATEGORIES
from omega.server.mcp_server import _WRITE_TOOLS


# Representative Pro tools whose absence caused the 1.5.11/1.5.12 defect.
# Not the full set -- the exhaustive check is the Pro-side contract test.
PRO_TOOLS_REQUIRING_A_CATEGORY = [
    "omega_worker_spawns",
    "omega_dispatch",
    "omega_dispatch_template",
    "omega_global_pause",
    "omega_global_resume",
    "omega_objective_create",
    "omega_objective_get",
    "omega_trace_deposit",
    "omega_trace_check",
    "omega_project_create",
    "omega_store_create",
    "omega_typed_store",
    "omega_dream",
    "omega_audit_verify",
    "omega_federation_trust_add",
    "omega_oracle_predict",
    "omega_ingest_file",
    "omega_profile_delete",
    "omega_ask",
]

# Pro tools whose PRIMARY operation mutates state. Incidental audit or
# telemetry writes do not qualify a tool as a write.
PRO_WRITE_TOOLS = [
    "omega_worker_spawns",
    "omega_dispatch",
    "omega_global_pause",
    "omega_objective_create",
    "omega_trace_deposit",
    "omega_project_create",
    "omega_store_create",
    "omega_typed_store",
    "omega_dream_apply",
    "omega_federation_trust_add",
    "omega_ingest_file",
    "omega_profile_delete",
]

# Pro tools that only read. Putting these in the write tier would apply the
# stricter write rate limit to ordinary queries.
PRO_READ_TOOLS = [
    "omega_objective_get",
    "omega_objective_list",
    "omega_task_tree",
    "omega_task_group_results",
    "omega_trace_check",
    "omega_message_show",
    "omega_project_get",
    "omega_project_list",
    "omega_store_list",
    "omega_typed_query",
    "omega_typed_schemas",
    "omega_audit_verify",
    "omega_federation_trust_list",
    "omega_oracle_predict",
    "omega_dream_diff",
]


@pytest.mark.parametrize("name", PRO_TOOLS_REQUIRING_A_CATEGORY)
def test_pro_tool_is_categorised(name):
    """Membership is what turns 'Unknown tool' into 'requires OMEGA Pro'."""
    assert name in TOOL_CATEGORIES, (
        f"{name} is missing from TOOL_CATEGORIES, so an unlicensed caller "
        f"would be told it does not exist"
    )


@pytest.mark.parametrize("name", PRO_WRITE_TOOLS)
def test_mutating_pro_tool_is_in_the_write_tier(name):
    assert name in _WRITE_TOOLS, f"{name} mutates state but escapes the write rate tier"


@pytest.mark.parametrize("name", PRO_READ_TOOLS)
def test_read_only_pro_tool_is_not_in_the_write_tier(name):
    assert name not in _WRITE_TOOLS, (
        f"{name} only reads; the write tier would rate-limit ordinary queries"
    )


# Documented exceptions to "every write tool is categorised".
#   omega_call            -- condensed-mode meta-tool, rate-limited by the
#                            INNER tool name in call_tool, so it deliberately
#                            has no category of its own.
#   omega_track_statement -- stale. Present only in this frozenset; no schema
#   omega_resolve_outcome    or handler exists in Core or in Pro 1.5.10. Left
#                            in place rather than removed, because deleting a
#                            shipped rate-limit entry is out of scope here.
#                            Recorded as pre-existing debt for a Core cleanup.
_UNCATEGORISED_BY_DESIGN = {"omega_call"}
_STALE_WRITE_ENTRIES = {"omega_track_statement", "omega_resolve_outcome"}


def test_every_write_tool_is_also_categorised():
    """A tool Core rate-limits as a write should also be a tool Core knows.

    Guards against NEW drift: the two known-stale names are excluded
    explicitly so this stays a real assertion rather than a broad allowance.
    """
    uncategorised = sorted(
        t for t in _WRITE_TOOLS
        if t not in TOOL_CATEGORIES
        and t not in _UNCATEGORISED_BY_DESIGN
        and t not in _STALE_WRITE_ENTRIES
    )
    assert not uncategorised, f"write tools missing a category: {uncategorised}"


def test_the_known_stale_write_entries_have_not_grown():
    """If one of these ever gains a schema/handler it must gain a category."""
    from omega.server.tool_schemas import TOOL_SCHEMAS

    named = {s["name"] for s in TOOL_SCHEMAS}
    for stale in _STALE_WRITE_ENTRIES:
        assert stale not in named, (
            f"{stale} now has a schema, so it must be categorised too"
        )


def test_every_category_value_is_selectable_via_omega_tools():
    """A category string nobody can filter on is a category that lies.

    handle_omega_tools filters on the raw string, so a value missing from the
    published enum works at runtime but is rejected by input-schema
    validation on a strict MCP client.
    """
    from omega.server import tool_schemas as ts

    schema = next(
        s
        for group in (ts.CONDENSED_TOOL_SCHEMAS, ts.TOOL_SCHEMAS)
        for s in group
        if s["name"] == "omega_tools"
    )
    selectable = set(schema["inputSchema"]["properties"]["category"]["enum"])
    used = set(TOOL_CATEGORIES.values())
    assert used <= selectable, f"categories not selectable: {sorted(used - selectable)}"


def test_dispatch_consults_handlers_before_categories():
    """Classification must never be an implementation path.

    The real invariant: call_tool resolves HANDLERS first and only consults
    TOOL_CATEGORIES when no handler exists. So a category can change the
    MESSAGE an unlicensed caller sees, never whether code runs.
    """
    import asyncio

    from omega.server import mcp_server

    pro_only = [n for n in PRO_TOOLS_REQUIRING_A_CATEGORY
                if n not in mcp_server.HANDLERS]
    if not pro_only:
        pytest.skip("omega_platform is installed; nothing is Pro-gated here")

    call_tool = getattr(mcp_server.call_tool, "__wrapped__", mcp_server.call_tool)
    for name in pro_only:
        result = asyncio.run(call_tool(name, {}))
        text = result[0].text
        assert "requires OMEGA Pro" in text, (name, text[:80])
        assert "Unknown tool" not in text, (name, text[:80])


def test_a_nonexistent_name_is_still_an_unknown_tool():
    """The other half: categorisation must not swallow genuine typos."""
    import asyncio

    from omega.server import mcp_server

    call_tool = getattr(mcp_server.call_tool, "__wrapped__", mcp_server.call_tool)
    result = asyncio.run(call_tool("omega_definitely_not_a_tool", {}))
    assert "Unknown tool" in result[0].text
