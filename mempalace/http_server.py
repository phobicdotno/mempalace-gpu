"""
MemPalace HTTP Server — remote GPU access via FastAPI.

Wraps all MCP tool handlers as REST endpoints. Designed to run on a GPU machine
so Mac/WSL clients can offload embedding computation.

Install: pip install mempalace-gpu[serve]
Run:     mempalace serve --token SECRET --port 8420
"""

import logging
import os

from . import __version__
from .config import MempalaceConfig

logger = logging.getLogger("mempalace.http_server")

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI and uvicorn are required for HTTP server mode.\n"
        "Install with: pip install mempalace-gpu[serve]"
    )

# Populated at startup
_app_token: str = ""
_config = MempalaceConfig()


def _get_tools():
    """Import TOOLS dict lazily to avoid circular imports."""
    from .mcp_server import TOOLS

    return {k: v for k, v in TOOLS.items() if k != "mempalace_self_update"}


def _check_auth(authorization: str = Header(None)):
    """Validate Bearer token."""
    if not _app_token:
        raise HTTPException(status_code=500, detail="Server misconfigured: no auth token set")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[len("Bearer ") :]
    if token != _app_token:
        raise HTTPException(status_code=401, detail="Invalid token")


def create_app(token: str = None) -> FastAPI:
    """Create the FastAPI application."""
    global _app_token
    _app_token = token or os.environ.get("MEMPALACE_TOKEN", "")

    app = FastAPI(title="MemPalace GPU Server", version=__version__)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": __version__,
        }

    @app.get("/tools")
    async def list_tools(authorization: str = Header(None)):
        _check_auth(authorization)
        tools = _get_tools()
        return [
            {
                "name": name,
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for name, t in tools.items()
        ]

    @app.post("/tool/{tool_name}")
    async def call_tool(tool_name: str, request: Request, authorization: str = Header(None)):
        _check_auth(authorization)
        tools = _get_tools()
        if tool_name not in tools:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = tools[tool_name]["handler"](**body)
            return {"result": result}
        except Exception as e:
            logger.error(f"Tool error in {tool_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def start_server(host: str = "0.0.0.0", port: int = 8420, token: str = None, device: str = "auto"):
    """Start the HTTP server with pre-warmed embeddings."""
    from .embeddings import init as init_embeddings

    init_embeddings(device)
    app = create_app(token=token)
    logger.info(f"MemPalace HTTP server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
