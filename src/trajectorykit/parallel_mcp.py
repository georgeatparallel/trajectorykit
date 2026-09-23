import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import __version__

logger = logging.getLogger(__name__)

_MCP_ENDPOINT = "https://search.parallel.ai/mcp"
_MCP_PROTOCOL_VERSION = "2025-11-25"
_SUPPORTED_MCP_PROTOCOL_VERSIONS = {
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
}
_MCP_MAX_RESPONSE_BYTES = 1_000_000
_MCP_MAX_DISCOVERY_PAGES = 10
_MCP_CONNECT_TIMEOUT = 5
_MCP_IO_TIMEOUT = 25

# Identify this project for aggregate MCP usage measurement, not individual users.
_MCP_USER_AGENT = f"trajectorykit/{__version__}"


class _ParallelMCPError(Exception):
    pass


class _ExpiredMCPSession(_ParallelMCPError):
    pass


def _request_headers(
    session_id: Optional[str], protocol_version: Optional[str]
) -> Dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": _MCP_USER_AGENT,
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    if protocol_version:
        headers["MCP-Protocol-Version"] = protocol_version
    return headers


def _decode_json_rpc_message(body: bytes) -> Dict[str, Any]:
    try:
        message = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ParallelMCPError("returned invalid JSON") from exc
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        raise _ParallelMCPError("returned an invalid JSON-RPC message")
    return message


def _check_response_size(response: httpx.Response) -> None:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > _MCP_MAX_RESPONSE_BYTES:
                raise _ParallelMCPError("response exceeded the size limit")
        except ValueError:
            pass


async def _read_json_response(
    response: httpx.Response, request_id: int, deadline: float
) -> Dict[str, Any]:
    body = bytearray()
    _check_response_size(response)
    async for chunk in response.aiter_bytes(chunk_size=8192):
        if time.monotonic() >= deadline:
            raise _ParallelMCPError("request timed out")
        if not chunk:
            continue
        body.extend(chunk)
        if len(body) > _MCP_MAX_RESPONSE_BYTES:
            raise _ParallelMCPError("response exceeded the size limit")

    message = _decode_json_rpc_message(bytes(body))
    if time.monotonic() >= deadline:
        raise _ParallelMCPError("request timed out")
    if message.get("id") != request_id:
        raise _ParallelMCPError("returned a mismatched response ID")
    return message


async def _read_sse_response(
    response: httpx.Response, request_id: int, deadline: float
) -> Dict[str, Any]:
    pending = bytearray()
    data_lines: List[bytes] = []
    total_size = 0
    _check_response_size(response)

    async for chunk in response.aiter_bytes(chunk_size=4096):
        if time.monotonic() >= deadline:
            raise _ParallelMCPError("request timed out")
        if not chunk:
            continue
        total_size += len(chunk)
        if total_size > _MCP_MAX_RESPONSE_BYTES:
            raise _ParallelMCPError("response exceeded the size limit")
        pending.extend(chunk)

        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                break
            line = bytes(pending[:newline])
            del pending[:newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]

            if not line:
                if data_lines:
                    message = _decode_json_rpc_message(b"\n".join(data_lines))
                    data_lines = []
                    if time.monotonic() >= deadline:
                        raise _ParallelMCPError("request timed out")
                    if message.get("id") == request_id:
                        return message
                continue

            if line.startswith(b":"):
                continue
            field, separator, value = line.partition(b":")
            if field == b"data":
                if separator and value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)

    raise _ParallelMCPError("did not receive a matching SSE response")


def _request_timeout(remaining: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=min(_MCP_CONNECT_TIMEOUT, remaining),
        read=min(_MCP_IO_TIMEOUT, remaining),
        write=min(_MCP_IO_TIMEOUT, remaining),
        pool=min(_MCP_CONNECT_TIMEOUT, remaining),
    )


async def _post_mcp_request(
    session: httpx.AsyncClient,
    payload: Dict[str, Any],
    request_id: Optional[int],
    session_id: Optional[str],
    protocol_version: Optional[str],
    deadline: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _ParallelMCPError("request timed out")

    async def send_request() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        async with session.stream(
            "POST",
            _MCP_ENDPOINT,
            json=payload,
            headers=_request_headers(session_id, protocol_version),
            timeout=_request_timeout(remaining),
            follow_redirects=False,
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                if 300 <= response.status_code < 400:
                    raise _ParallelMCPError("redirects are not followed")
                if response.status_code == 404 and session_id:
                    raise _ExpiredMCPSession("returned HTTP 404")
                raise _ParallelMCPError(f"returned HTTP {response.status_code}")

            response_session_id = response.headers.get("Mcp-Session-Id") or session_id
            if request_id is None:
                return None, response_session_id

            content_type = response.headers.get("Content-Type", "")
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type == "application/json":
                message = await _read_json_response(response, request_id, deadline)
            elif media_type == "text/event-stream":
                message = await _read_sse_response(response, request_id, deadline)
            else:
                raise _ParallelMCPError("returned an unsupported content type")
            return message, response_session_id

    try:
        result = await asyncio.wait_for(send_request(), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise _ParallelMCPError("request timed out") from exc
    if time.monotonic() >= deadline:
        raise _ParallelMCPError("request timed out")
    return result


async def _cleanup_mcp_session(
    session: httpx.AsyncClient,
    session_id: str,
    protocol_version: Optional[str],
    deadline: float,
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    cleanup_timeout = min(2, remaining)

    async def delete_session() -> None:
        async with session.stream(
            "DELETE",
            _MCP_ENDPOINT,
            headers=_request_headers(session_id, protocol_version),
            timeout=_request_timeout(cleanup_timeout),
            follow_redirects=False,
        ):
            pass

    try:
        await asyncio.wait_for(delete_session(), timeout=cleanup_timeout)
    except Exception as exc:
        logger.debug("Parallel Search MCP session cleanup failed: %s", type(exc).__name__)


def _search_results_payload(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    structured = tool_result.get("structuredContent")
    if structured is None:
        for block in tool_result.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            try:
                structured = json.loads(block.get("text", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(structured, dict):
                break

    if not isinstance(structured, dict) or not isinstance(structured.get("results"), list):
        raise _ParallelMCPError("returned a malformed search result")
    return structured


async def _search_parallel_attempt(q: str, num_results: int, deadline: float) -> str:
    session = None
    session_id = None
    protocol_version = None
    request_number = 0

    try:
        session = httpx.AsyncClient()

        async def request(
            method: str,
            params: Optional[Dict[str, Any]] = None,
            notification: bool = False,
        ):
            nonlocal protocol_version, request_number, session_id
            request_number += 1
            request_id = None if notification else request_number
            payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                payload["params"] = params
            if request_id is not None:
                payload["id"] = request_id

            message, response_session_id = await _post_mcp_request(
                session,
                payload,
                request_id,
                session_id,
                protocol_version,
                deadline,
            )
            if response_session_id and session_id and response_session_id != session_id:
                raise _ParallelMCPError("changed its transport session")
            session_id = response_session_id
            if notification:
                return None
            if message is None:
                raise _ParallelMCPError("returned no JSON-RPC response")
            if "error" in message:
                raise _ParallelMCPError("returned a JSON-RPC error")
            result = message.get("result")
            if not isinstance(result, dict):
                raise _ParallelMCPError("returned an invalid JSON-RPC result")
            return result

        initialized = await request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "trajectorykit", "version": __version__},
            },
        )
        negotiated_version = initialized.get("protocolVersion")
        if negotiated_version not in _SUPPORTED_MCP_PROTOCOL_VERSIONS:
            raise _ParallelMCPError("negotiated an unsupported protocol version")
        protocol_version = negotiated_version

        await request("notifications/initialized", notification=True)

        cursor = None
        seen_cursors = set()
        web_search_found = False
        for _ in range(_MCP_MAX_DISCOVERY_PAGES):
            params = {"cursor": cursor} if cursor else {}
            page = await request("tools/list", params)
            tools = page.get("tools")
            if not isinstance(tools, list):
                raise _ParallelMCPError("returned an invalid tool list")
            if any(isinstance(tool, dict) and tool.get("name") == "web_search" for tool in tools):
                web_search_found = True
                break

            next_cursor = page.get("nextCursor")
            if not next_cursor:
                break
            if not isinstance(next_cursor, str) or next_cursor in seen_cursors:
                raise _ParallelMCPError("returned an invalid discovery cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        if not web_search_found:
            raise _ParallelMCPError("does not expose the web_search tool")

        tool_result = await request(
            "tools/call",
            {
                "name": "web_search",
                "arguments": {"objective": q, "search_queries": [q]},
            },
        )
        if tool_result.get("isError"):
            raise _ParallelMCPError("reported a web_search tool error")

        results = _search_results_payload(tool_result).get("results", [])
        if not results:
            if time.monotonic() >= deadline:
                raise _ParallelMCPError("request timed out")
            return f"No results found for query: {q}"

        formatted = []
        for result in results[:num_results]:
            if not isinstance(result, dict) or not isinstance(result.get("url"), str):
                continue
            excerpts = result.get("excerpts", [])
            if isinstance(excerpts, str):
                excerpts = [excerpts]
            if not isinstance(excerpts, list):
                excerpts = []
            snippet = " ".join(
                excerpt.strip()
                for excerpt in excerpts
                if isinstance(excerpt, str) and excerpt.strip()
            )[:250]
            formatted.append(
                f"{len(formatted) + 1}. {result.get('title') or 'No title'}\n"
                f"   URL: {result['url']}\n"
                f"   {snippet or 'No snippet'}"
            )

        if not formatted:
            raise _ParallelMCPError("returned no usable search results")
        if time.monotonic() >= deadline:
            raise _ParallelMCPError("request timed out")
        return f"Search Results for '{q}':\n\n" + "\n\n".join(formatted) + "\n\n"

    except _ExpiredMCPSession:
        session_id = None
        raise
    finally:
        if session is not None:
            if session_id:
                await _cleanup_mcp_session(
                    session, session_id, protocol_version, deadline
                )
            try:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    await asyncio.wait_for(session.aclose(), timeout=remaining)
            except Exception:
                pass


def _run_search_parallel_attempt(q: str, num_results: int, deadline: float) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_search_parallel_attempt(q, num_results, deadline))

    def run_attempt() -> str:
        return asyncio.run(_search_parallel_attempt(q, num_results, deadline))

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="trajectorykit-parallel-mcp"
    ) as executor:
        future = executor.submit(run_attempt)
        remaining = max(0, deadline - time.monotonic())
        try:
            return future.result(timeout=remaining)
        except FutureTimeoutError:
            return future.result()


def search_parallel(q: str, num_results: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    try:
        try:
            return _run_search_parallel_attempt(q, num_results, deadline)
        except _ExpiredMCPSession:
            return _run_search_parallel_attempt(q, num_results, deadline)
    except httpx.TimeoutException:
        return "Search error: Parallel Search MCP request timed out."
    except httpx.RequestError:
        return "Search error: Could not connect to Parallel Search MCP."
    except _ParallelMCPError as exc:
        return f"Search error: Parallel Search MCP {exc}."
    except Exception as exc:
        logger.warning("Parallel Search MCP failed: %s", type(exc).__name__)
        return "Search error: Parallel Search MCP response could not be processed."
