"""
MemPalace MCP Proxy — thin stdio MCP server forwarding calls to a remote HTTP server.

Runs on Mac/WSL. Reads MEMPALACE_REMOTE_URL and MEMPALACE_TOKEN from env vars.

Install: claude mcp add mempalace-remote -- python -m mempalace.mcp_proxy
Env:     MEMPALACE_REMOTE_URL=http://192.168.1.201:8420 MEMPALACE_TOKEN=secret
"""

import sys
import os
import json
import logging

from mempalace import __version__

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("mempalace.mcp_proxy")

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx is required for the MCP proxy.\nInstall with: pip install mempalace-gpu[serve]"
    )


_cached_client = None


def _get_client():
    """Get or create a cached httpx client with auth headers."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    url = os.environ.get("MEMPALACE_REMOTE_URL", "")
    token = os.environ.get("MEMPALACE_TOKEN", "")
    if not url:
        raise RuntimeError("MEMPALACE_REMOTE_URL environment variable is required")
    _cached_client = httpx.Client(
        base_url=url.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )
    return _cached_client


_cached_tools = None


def _fetch_tools(client):
    """Fetch and cache the tool list from the remote server."""
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools
    resp = client.get("/tools")
    resp.raise_for_status()
    _cached_tools = resp.json()
    return _cached_tools


def handle_request(request):
    """Handle a single MCP JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace-gpu-proxy", "version": __version__},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        try:
            client = _get_client()
            tools = _fetch_tools(client)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "inputSchema": t["input_schema"],
                        }
                        for t in tools
                    ]
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"Failed to fetch tools: {e}"},
            }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        try:
            client = _get_client()
            resp = client.post(f"/tool/{tool_name}", json=tool_args)
            resp.raise_for_status()
            result = resp.json().get("result", {})
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as e:
            logger.error(f"Remote tool error {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():
    logger.info("MemPalace MCP Proxy starting...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Proxy error: {e}")


if __name__ == "__main__":
    main()
