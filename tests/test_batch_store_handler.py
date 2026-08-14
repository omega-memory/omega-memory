"""Tests for batch store via omega_store handler."""

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_batch_store_multiple_items():
    """omega_store(items=[...]) stores all items."""
    from omega.server.handlers import handle_omega_store

    items = [
        {"content": "memory one", "event_type": "decision"},
        {"content": "memory two", "event_type": "lesson_learned"},
    ]
    mock_result = {"ids": ["id1", "id2"], "count": 2}
    with patch("omega.bridge.batch_store", return_value=mock_result):
        result = await handle_omega_store({"items": items})
    text = result["content"][0]["text"]
    assert "id1" in text or "2" in text


@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_bridge")
async def test_batch_store_persists_per_item_event_types():
    """The MCP batch path must not collapse item types to the default."""
    from omega.server.handlers import handle_omega_store
    from omega.bridge import _get_store

    result = await handle_omega_store(
        {
            "items": [
                {
                    "content": "MCP batch decision field parity",
                    "event_type": "decision",
                    "session_id": "batch-field-parity",
                    "skip_inference": True,
                },
                {
                    "content": "MCP batch lesson field parity",
                    "event_type": "lesson_learned",
                    "session_id": "batch-field-parity",
                    "skip_inference": True,
                },
            ]
        }
    )

    assert not result.get("isError")
    rows = _get_store()._conn.execute(
        """SELECT content, event_type FROM memories
           WHERE session_id = ? ORDER BY content""",
        ("batch-field-parity",),
    ).fetchall()
    assert rows == [
        ("MCP batch decision field parity", "decision"),
        ("MCP batch lesson field parity", "lesson_learned"),
    ]


@pytest.mark.asyncio
async def test_batch_store_inherits_request_scope_without_mutating_input():
    """Request-level scope defaults apply unless an item overrides them."""
    from omega.server.handlers import handle_omega_store

    request = {
        "entity_id": "fleet",
        "project_id": "prod",
        "sensitivity": "confidential",
        "items": [
            {"content": "inherits scope"},
            {"content": "overrides scope", "entity_id": "special"},
        ],
    }
    original_items = [dict(item) for item in request["items"]]

    with patch(
        "omega.bridge.batch_store",
        return_value={"ids": ["id1", "id2"], "count": 2},
    ) as mocked_batch:
        result = await handle_omega_store(request)

    assert not result.get("isError")
    prepared = mocked_batch.call_args.args[0]
    assert prepared[0]["entity_id"] == "fleet"
    assert prepared[0]["project_id"] == "prod"
    assert prepared[0]["sensitivity"] == "confidential"
    assert prepared[1]["entity_id"] == "special"
    assert prepared[1]["project_id"] == "prod"
    assert request["items"] == original_items


@pytest.mark.asyncio
async def test_batch_store_rejects_non_object_before_writing():
    from omega.server.handlers import handle_omega_store

    with patch("omega.bridge.batch_store") as mocked_batch:
        result = await handle_omega_store(
            {"items": [{"content": "valid"}, "not an object"]}
        )

    assert result.get("isError") is True
    assert "object" in result["content"][0]["text"]
    mocked_batch.assert_not_called()


@pytest.mark.asyncio
async def test_batch_store_empty_list():
    """omega_store(items=[]) returns empty result, not error."""
    from omega.server.handlers import handle_omega_store

    result = await handle_omega_store({"items": []})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "0" in text


@pytest.mark.asyncio
async def test_batch_store_invalid_type():
    """omega_store(items="not a list") returns error."""
    from omega.server.handlers import handle_omega_store

    result = await handle_omega_store({"items": "not a list"})
    assert result.get("isError") is True


def test_omega_store_schema_documents_batch_items():
    """The MCP schema accepts items-only requests and exposes item fields."""
    from omega.server.tool_schemas import TOOL_SCHEMAS

    schema = next(tool for tool in TOOL_SCHEMAS if tool["name"] == "omega_store")["inputSchema"]
    item_schema = schema["properties"]["items"]["items"]

    assert item_schema["required"] == ["content"]
    assert {
        "content",
        "event_type",
        "metadata",
        "session_id",
        "project",
        "priority",
        "entity_id",
        "agent_type",
        "derived_from",
        "source_uri",
        "status",
        "project_id",
        "sensitivity",
    } <= item_schema["properties"].keys()
    assert "request-level scope" in schema["properties"]["items"]["description"].lower()
    assert {tuple(option["required"]) for option in schema["anyOf"]} == {
        ("content",),
        ("text",),
        ("items",),
    }
