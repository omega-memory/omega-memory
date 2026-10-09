#!/usr/bin/env python3
"""
OMEGA HTTP Bridge — Exposes OMEGA's bridge API over HTTP.

Thin FastAPI wrapper that accepts JSON-RPC 2.0 requests and delegates to
OmegaBridge. Project scoping is configured via environment variables.

Usage:
    OMEGA_BRIDGE_PROJECT=myproject OMEGA_BRIDGE_PROJECT_PATH=/path/to/project \\
        uvicorn omega.scripts.omega_http_bridge:app --port 9092
    # or directly:
    python scripts/omega_http_bridge.py
"""

import logging
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from omega.bridge import (
    auto_capture,
    check_health,
    get_cross_session_lessons,
    query,
    timeline,
    type_stats,
)

logger = logging.getLogger("omega.http_bridge")

# ---------------------------------------------------------------------------
# Project scoping — configure via env vars
# ---------------------------------------------------------------------------
PROJECT = os.getenv("OMEGA_BRIDGE_PROJECT", "default")
PROJECT_PATH = os.getenv("OMEGA_BRIDGE_PROJECT_PATH", "")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="OMEGA HTTP Bridge", version="1.0.0")


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str = 1
    method: str
    params: Dict[str, Any] = {}


class JsonRpcError(BaseModel):
    code: int
    message: str


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str = 1
    result: Optional[Dict[str, Any]] = None
    error: Optional[JsonRpcError] = None


def _text_result(text: str) -> Dict[str, Any]:
    """Wrap a string into MCP-compatible content format."""
    return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _handle_omega_store(args: Dict[str, Any]) -> str:
    content = args.get("content", "")
    event_type = args.get("event_type", "decision")
    metadata = args.get("metadata")
    return auto_capture(
        content=content,
        event_type=event_type,
        metadata=metadata,
        project=PROJECT,
    )


def _handle_omega_query(args: Dict[str, Any]) -> str:
    query_text = args.get("query", "")
    limit = args.get("limit", 10)
    event_type = args.get("event_type")
    return query(
        query_text=query_text,
        limit=limit,
        project=PROJECT,
        event_type=event_type,
    )


def _handle_omega_lessons(args: Dict[str, Any]) -> str:
    task = args.get("task")
    limit = args.get("limit", 5)
    lessons = get_cross_session_lessons(
        task=task,
        project_path=PROJECT_PATH,
        limit=limit,
    )
    if not lessons:
        return "No lessons found."
    lines = []
    for i, lesson in enumerate(lessons, 1):
        lines.append(f"{i}. [{lesson.get('event_type', 'lesson')}] {lesson.get('content', '')[:200]}")
    return "\n".join(lines)


def _handle_omega_timeline(args: Dict[str, Any]) -> str:
    days = args.get("days", 7)
    limit_per_day = args.get("limit_per_day", 10)
    return timeline(days=days, limit_per_day=limit_per_day)


def _handle_omega_type_stats(args: Dict[str, Any]) -> str:
    stats = type_stats()
    import json
    return json.dumps(stats)


def _handle_omega_health(args: Dict[str, Any]) -> str:
    return check_health()


def _handle_omega_route_prompt(args: Dict[str, Any]) -> str:
    try:
        from omega.router.engine import route_prompt
    except ImportError:
        return '{"error": "Router not available in this installation"}'

    prompt = args.get("prompt", "")
    priority = args.get("priority", "balanced")
    result = route_prompt(prompt=prompt, priority=priority)
    import json
    return json.dumps(result)


TOOL_HANDLERS = {
    "omega_store": _handle_omega_store,
    "omega_query": _handle_omega_query,
    "omega_lessons": _handle_omega_lessons,
    "omega_timeline": _handle_omega_timeline,
    "omega_type_stats": _handle_omega_type_stats,
    "omega_health": _handle_omega_health,
    "omega_route_prompt": _handle_omega_route_prompt,
}


# ---------------------------------------------------------------------------
# JSON-RPC endpoint
# ---------------------------------------------------------------------------

@app.post("/mcp")
async def mcp_endpoint(req: JsonRpcRequest) -> JsonRpcResponse:
    """Handle JSON-RPC 2.0 requests in MCP tools/call format."""
    if req.method != "tools/call":
        return JsonRpcResponse(
            id=req.id,
            error=JsonRpcError(code=-32601, message=f"Unknown method: {req.method}"),
        )

    tool_name = req.params.get("name", "")
    tool_args = req.params.get("arguments", {})

    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return JsonRpcResponse(
            id=req.id,
            error=JsonRpcError(
                code=-32602,
                message=f"Unknown tool: {tool_name}. Available: {', '.join(TOOL_HANDLERS)}",
            ),
        )

    try:
        result_text = handler(tool_args)
        return JsonRpcResponse(id=req.id, result=_text_result(result_text))
    except Exception as e:
        logger.error("Tool %s failed: %s\n%s", tool_name, e, traceback.format_exc())
        return JsonRpcResponse(
            id=req.id,
            error=JsonRpcError(code=-32000, message=str(e)),
        )


@app.get("/health")
async def health_check():
    """Quick health endpoint for monitoring."""
    return {"status": "ok", "project": PROJECT}


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9092, log_level="info")
