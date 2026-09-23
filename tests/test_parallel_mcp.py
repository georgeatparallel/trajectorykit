import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from trajectorykit import parallel_mcp, tool_store


_SESSION_ID = "fixture-session"
_RENEWED_SESSION_ID = "renewed-fixture-session"


class _MCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _record(self, payload=None):
        self.server.requests.append({
            "http_method": self.command,
            "rpc_method": payload.get("method") if payload else None,
            "headers": {name.lower(): value for name, value in self.headers.items()},
            "payload": payload,
        })

    def _send_json(self, message, add_session=False, session_id=None):
        body = json.dumps(message, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if add_session:
            self.send_header("Mcp-Session-Id", session_id or _SESSION_ID)
        self.end_headers()
        self.wfile.write(body)

    def _send_chunked_sse(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for offset in range(0, len(body), 1024):
                chunk = body[offset:offset + 1024]
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length))
        self._record(payload)

        if self.server.mode == "forbidden":
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        method = payload.get("method")
        if method == "initialize":
            session_id = _SESSION_ID
            protocol_version = (
                "2025-06-18"
                if self.server.mode == "expired_call"
                else "2025-11-25"
            )
            if self.server.mode == "expired_call":
                if self.server.initialize_count:
                    session_id = _RENEWED_SESSION_ID
                self.server.initialize_count += 1
            self._send_json({
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "serverInfo": {"name": "fixture", "version": "1"},
                },
            }, add_session=True, session_id=session_id)
            return

        if method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if method == "tools/list":
            self._send_json({
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"tools": [{"name": "web_search"}]},
            })
            return

        if method == "tools/call":
            if (
                self.server.mode == "expired_call"
                and self.headers.get("Mcp-Session-Id") == _SESSION_ID
            ):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.server.mode == "slow":
                time.sleep(5.25)
            if self.server.mode == "oversized":
                self._send_chunked_sse(b": " + b"x" * 1024 + b"\r\n")
                return

            result = {
                "structuredContent": {
                    "results": [{
                        "title": "Fixture result",
                        "url": "https://example.com/result",
                        "excerpts": ["A streamed fixture excerpt."],
                    }]
                }
            }
            message = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": result,
            }
            body = (
                b": " + b"x" * 5000 + b"\r\n"
                + b"event: message\r\ndata: "
                + json.dumps(message, separators=(",", ":")).encode()
                + b"\r\n\r\n"
            )
            self._send_chunked_sse(body)
            return

        self.send_response(400)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):
        self._record()
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _MCPFixtureServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _MCPHandler)
        self.daemon_threads = True
        self.mode = "normal"
        self.requests = []
        self.initialize_count = 0


class ParallelMCPTests(unittest.TestCase):
    def setUp(self):
        self.server = _MCPFixtureServer()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint_patch = patch.object(
            parallel_mcp,
            "_MCP_ENDPOINT",
            f"http://127.0.0.1:{self.server.server_port}/mcp",
        )
        self.endpoint_patch.start()

    def tearDown(self):
        self.endpoint_patch.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_search_accepts_delayed_fragmented_sse_with_project_identity(self):
        self.server.mode = "slow"
        started = time.monotonic()
        with patch.dict(os.environ, {"SEARCH_BACKEND": "parallel"}):
            with patch.dict(
                tool_store._SEARCH_BACKENDS,
                {"parallel": (tool_store._search_parallel, [])},
            ):
                result = tool_store.search_web("fixture query", 1)
        elapsed = time.monotonic() - started

        self.assertIn("Fixture result", result)
        self.assertIn("A streamed fixture excerpt.", result)
        self.assertGreater(elapsed, 5)
        self.assertLess(elapsed, 12)

        requests = self.server.requests
        posts = [request for request in requests if request["http_method"] == "POST"]
        self.assertEqual(
            [request["rpc_method"] for request in posts],
            ["initialize", "notifications/initialized", "tools/list", "tools/call"],
        )
        self.assertEqual(
            posts[0]["headers"]["user-agent"],
            f"trajectorykit/{parallel_mcp.__version__}",
        )
        self.assertNotIn("authorization", posts[0]["headers"])
        self.assertEqual(posts[1]["headers"]["mcp-session-id"], _SESSION_ID)
        self.assertEqual(posts[1]["headers"]["mcp-protocol-version"], "2025-11-25")
        self.assertEqual(
            posts[-1]["payload"]["params"]["arguments"],
            {"objective": "fixture query", "search_queries": ["fixture query"]},
        )
        self.assertTrue(any(request["http_method"] == "DELETE" for request in requests))
        self.assertTrue(all(
            request["headers"]["user-agent"] == f"trajectorykit/{parallel_mcp.__version__}"
            for request in requests
        ))

    def test_streamed_sse_response_without_content_length_is_bounded(self):
        self.server.mode = "oversized"
        with patch.object(parallel_mcp, "_MCP_MAX_RESPONSE_BYTES", 512):
            with patch.dict(os.environ, {"SEARCH_BACKEND": "parallel"}):
                with patch.dict(
                    tool_store._SEARCH_BACKENDS,
                    {"parallel": (tool_store._search_parallel, [])},
                ):
                    result = tool_store.search_web("fixture query", 1)

        self.assertIn("response exceeded the size limit", result)
        self.assertTrue(any(
            request["rpc_method"] == "tools/call" for request in self.server.requests
        ))

    def test_expired_session_reinitializes_and_cleans_up_only_renewed_session(self):
        self.server.mode = "expired_call"
        with patch.dict(os.environ, {"SEARCH_BACKEND": "parallel"}):
            with patch.dict(
                tool_store._SEARCH_BACKENDS,
                {"parallel": (tool_store._search_parallel, [])},
            ):
                result = tool_store.search_web("fixture query", 1)

        self.assertIn("Fixture result", result)
        posts = [
            request
            for request in self.server.requests
            if request["http_method"] == "POST"
        ]
        self.assertEqual(
            [request["rpc_method"] for request in posts],
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
            ],
        )
        self.assertNotIn("mcp-session-id", posts[0]["headers"])
        self.assertNotIn("mcp-session-id", posts[4]["headers"])
        self.assertNotIn("mcp-protocol-version", posts[4]["headers"])
        self.assertEqual(
            [request["headers"].get("mcp-protocol-version") for request in posts[1:4]],
            ["2025-06-18"] * 3,
        )
        self.assertEqual(
            [request["headers"].get("mcp-protocol-version") for request in posts[5:8]],
            ["2025-06-18"] * 3,
        )
        self.assertEqual(
            [
                request["headers"].get("mcp-session-id")
                for request in posts
                if request["rpc_method"] != "initialize"
            ],
            [
                _SESSION_ID,
                _SESSION_ID,
                _SESSION_ID,
                _RENEWED_SESSION_ID,
                _RENEWED_SESSION_ID,
                _RENEWED_SESSION_ID,
            ],
        )
        self.assertTrue(all(
            request["headers"]["user-agent"]
            == f"trajectorykit/{parallel_mcp.__version__}"
            for request in self.server.requests
        ))
        cleanup_requests = [
            request
            for request in self.server.requests
            if request["http_method"] == "DELETE"
        ]
        self.assertEqual(len(cleanup_requests), 1)
        self.assertEqual(
            cleanup_requests[0]["headers"].get("mcp-session-id"),
            _RENEWED_SESSION_ID,
        )

    def test_parallel_http_refusal_uses_configured_fallback(self):
        self.server.mode = "forbidden"
        fallback_calls = []

        def exa_fallback(_query, _num_results):
            fallback_calls.append("exa")
            return "fallback result"

        with patch.dict(os.environ, {"SEARCH_BACKEND": "parallel"}):
            with patch.dict(
                tool_store._SEARCH_BACKENDS,
                {"parallel": (tool_store._search_parallel, [exa_fallback])},
            ):
                result = tool_store.search_web("fixture query", 1)

        self.assertEqual(result, "fallback result")
        self.assertEqual(fallback_calls, ["exa"])
        self.assertEqual(self.server.requests[0]["rpc_method"], "initialize")

    def test_unset_backend_keeps_serper_default_without_parallel_request(self):
        serper_calls = []

        def serper_default(_query, _num_results):
            serper_calls.append("serper")
            return "serper result"

        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(
                tool_store._SEARCH_BACKENDS,
                {"serper": (serper_default, [])},
            ):
                result = tool_store.search_web("fixture query", 1)

        self.assertEqual(result, "serper result")
        self.assertEqual(serper_calls, ["serper"])
        self.assertEqual(self.server.requests, [])


if __name__ == "__main__":
    unittest.main()
