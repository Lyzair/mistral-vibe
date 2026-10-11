"""Provided-tool failures surface the server's own error text.

When an MCP server (or connector) returns an ``isError`` result, the failure
event's ``error.message`` must carry the server's first non-empty text block
instead of a generic placeholder, so the model sees *why* the call failed
(schema complaints, attention-gate pointers, retry hints) without a second
round trip. ``tool_error_message`` remains the fallback when no text block
carries any.
"""

from __future__ import annotations

import asyncio

from mistralai_vibe_local_harness.protocol import (
    RustProvidedToolCall,
    RustProvidedToolCallAction,
    RustToolFailedEvent,
    RustToolSucceededEvent,
)
from mistralai_vibe_local_harness.vibe._mcp_models import MCPNormalizedResult
from mistralai_vibe_local_harness.vibe._provided_tool_actions import (
    _ERROR_TEXT_LIMIT,
    execute_provided_tool_action,
)


class _RuntimeFailure(Exception):
    code = "mcp_runtime_failure"
    retryable = False


def _action() -> RustProvidedToolCallAction:
    return RustProvidedToolCallAction(
        action_id="action-1",
        turn_id="turn-1",
        call_id="call-1",
        call=RustProvidedToolCall(
            group_name="gabs", tool_name="games_call_tool", arguments={}
        ),
    )


def _run(result: MCPNormalizedResult):
    async def call(group_name: str, tool_name: str, arguments: dict):
        return result

    return asyncio.run(
        execute_provided_tool_action(
            _action(),
            call,
            failure_type=_RuntimeFailure,
            invalid_result_code="mcp_invalid_result",
            tool_error_code="mcp_tool_error",
            tool_error_message="The MCP tool returned an error",
        )
    )


def test_is_error_message_surfaces_server_text() -> None:
    event = _run(
        MCPNormalizedResult(
            content=(
                {
                    "type": "text",
                    "text": "blocked by attention; see games_get_attention",
                },
            ),
            is_error=True,
        )
    )

    assert isinstance(event, RustToolFailedEvent)
    assert event.result.error.code == "mcp_tool_error"
    assert event.result.error.message == "blocked by attention; see games_get_attention"


def test_is_error_message_strips_surrounding_whitespace() -> None:
    event = _run(
        MCPNormalizedResult(
            content=({"type": "text", "text": "  schema violation: bad arg  \n"},),
            is_error=True,
        )
    )

    assert isinstance(event, RustToolFailedEvent)
    assert event.result.error.message == "schema violation: bad arg"


def test_is_error_message_truncates_long_server_text() -> None:
    event = _run(
        MCPNormalizedResult(
            content=({"type": "text", "text": "x" * (_ERROR_TEXT_LIMIT + 500)},),
            is_error=True,
        )
    )

    assert isinstance(event, RustToolFailedEvent)
    assert event.result.error.message.endswith("… [truncated]")
    assert len(event.result.error.message) == _ERROR_TEXT_LIMIT + len("… [truncated]")


def test_is_error_falls_back_without_text_content() -> None:
    event = _run(
        MCPNormalizedResult(
            content=({"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"},), is_error=True
        )
    )

    assert isinstance(event, RustToolFailedEvent)
    assert event.result.error.code == "mcp_tool_error"
    assert event.result.error.message == "The MCP tool returned an error"


def test_is_error_falls_back_on_blank_text_content() -> None:
    event = _run(
        MCPNormalizedResult(
            content=({"type": "text", "text": "   \n  "},), is_error=True
        )
    )

    assert isinstance(event, RustToolFailedEvent)
    assert event.result.error.message == "The MCP tool returned an error"


def test_success_result_is_unchanged() -> None:
    event = _run(MCPNormalizedResult(content=({"type": "text", "text": "all good"},)))

    assert isinstance(event, RustToolSucceededEvent)
