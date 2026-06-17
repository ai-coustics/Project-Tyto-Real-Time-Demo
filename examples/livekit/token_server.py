"""Token + static file server for the LiveKit Tyto demo frontend.

LiveKit frontends connect to a LiveKit server (Cloud or self-hosted), not to
this process: they need the server URL and a short-lived access token. This tiny
aiohttp server hands those out and serves the browser UI. The agent worker
(``agent.py``) joins the room automatically and is the actual brain.

Run::

    uv pip install -e ".[livekit]"
    # LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env.local (or .env)
    uv run examples/livekit/token_server.py        # open http://localhost:8080

This server never sees your OpenAI or ai-coustics keys; those live with the
agent worker. It only mints LiveKit room tokens.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from aiohttp import web
from livekit import api

from tyto_voice.env import load_env

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
APP_JS = HERE / "app.js"


async def index_handler(_request: web.Request) -> web.Response:
    return web.Response(text=INDEX.read_text(), content_type="text/html")


async def app_js_handler(_request: web.Request) -> web.Response:
    return web.Response(text=APP_JS.read_text(), content_type="text/javascript")


async def token_handler(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    room = f"tyto-{uuid.uuid4().hex[:12]}"
    identity = f"user-{uuid.uuid4().hex[:8]}"
    token = (
        api.AccessToken(cfg["api_key"], cfg["api_secret"])
        .with_identity(identity)
        .with_name("Tyto visitor")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return web.json_response({"url": cfg["url"], "token": token, "room": room})


def main() -> None:
    load_env(".env.local")  # LiveKit's convention; falls through to .env below
    load_env()
    cfg = {
        "url": os.environ.get("LIVEKIT_URL", ""),
        "api_key": os.environ.get("LIVEKIT_API_KEY", ""),
        "api_secret": os.environ.get("LIVEKIT_API_SECRET", ""),
    }
    if not all(cfg.values()):
        raise SystemExit(
            "Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET (see .env.example)."
        )

    app = web.Application()
    app["cfg"] = cfg
    app.add_routes(
        [
            web.get("/", index_handler),
            web.get("/app.js", app_js_handler),
            web.get("/token", token_handler),
        ]
    )
    host, port = "127.0.0.1", int(os.environ.get("PORT", "8080"))
    print(f"Tyto LiveKit token server on http://{host}:{port}  (Ctrl-C to stop)")
    print("Make sure the agent worker is running: uv run examples/livekit/agent.py dev")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
