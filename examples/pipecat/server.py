"""Pipecat web demo backend: the Tyto voice agent over a Pipecat pipeline.

Same demo and same browser UI as ``examples/web``, but the voice backend is a
Pipecat pipeline (OpenAI Realtime speech-to-speech) reached over WebRTC instead
of a hand-written WebSocket relay. The browser is still a thin client: it
captures the mic, plays the agent, and renders the UI; all scoring, the three
adaptation layers, and the keys live here.

Per browser connection, one session::

    browser mic  ── WebRTC audio ─>  SmallWebRTCTransport -> scorer.feed + agent
    agent audio  <─ WebRTC audio ──  pipeline output
    scores / room / vad / nudge  <─ WebRTC data channel ──  controller

The decision layer, scorer, and controller are shared with every other Tyto
frontend; only the provider ([pipecat_provider.py](../../src/tyto_voice/pipecat_provider.py))
is specific to this stack.

Run::

    uv pip install -e ".[pipecat]"
    # put AIC_SDK_LICENSE and OPENAI_API_KEY in .env
    uv run examples/pipecat/server.py        # then open http://localhost:8080
"""

# NOTE: no ``from __future__ import annotations`` here. FastAPI resolves route
# parameter annotations as real objects, and the request types are imported
# locally inside build_app(); stringized annotations would not resolve and the
# ``request: Request`` param would be misread as a query parameter (HTTP 422).

import os
from pathlib import Path

from tyto_voice.controller import TytoController
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.pipecat_provider import SAMPLE_RATE, PipecatRealtimeProvider
from tyto_voice.prompts import BASE_INSTRUCTIONS
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
APP_JS = HERE / "app.js"


class Session:
    """One WebRTC connection wired to a scorer, provider, and controller.

    Mirrors the ``Session`` in ``examples/web/server.py``: the brain runs here,
    the browser is a thin client. The only differences are the transport
    (WebRTC, owned by Pipecat) and that UI messages go over the data channel.
    """

    def __init__(self, connection, keys: dict):
        self.connection = connection
        self.keys = keys
        self.scorer: LiveTytoScorer | None = None
        self.provider: PipecatRealtimeProvider | None = None
        self.controller: TytoController | None = None
        self._last_tyto_state: dict | None = None

    async def start(self) -> None:
        import asyncio

        handlers = Handlers()
        scorer = LiveTytoScorer(
            self.keys["license"],
            sample_rate=SAMPLE_RATE,
            on_state=self._on_tyto_state,
        )
        provider = PipecatRealtimeProvider(
            handlers,
            api_key=self.keys["openai"],
            instructions=BASE_INSTRUCTIONS,
            scorer=scorer,
            webrtc_connection=self.connection,
            turn_detection=VAD_PROFILES["eager"],
            on_client_message=self._on_client_message,
            on_connected=self._on_connected,
            on_log=lambda k, t: self._send({"type": "log", "kind": k, "text": t}),
        )
        controller = TytoController(
            provider,
            scorer,
            on_update=self._on_update,
            on_log=lambda k, t: self._send({"type": "log", "kind": k, "text": t}),
        )
        scorer.on_scores = controller.on_scores
        provider.audio_quality_fn = controller.audio_quality_snapshot

        handlers.on_ready = controller.on_ready
        handlers.on_agent_speaking = controller.on_agent_speaking
        handlers.on_agent_audio = controller.on_agent_audio
        handlers.on_user_transcript = controller.on_user_transcript
        handlers.on_agent_transcript = controller.on_agent_transcript
        # on_tool_call is intentionally unwired: this backend answers the tool
        # with a registered Pipecat function handler instead.

        self.scorer, self.provider, self.controller = scorer, provider, controller

        # The model download + license check is blocking, so keep it off the
        # event loop. A scorer failure is non-fatal: the pipeline still runs so
        # the agent works and the error can be shown over the data channel.
        # State messages emitted here are remembered and (re)sent once the data
        # channel is up (see _on_connected); sending them now would race it.
        try:
            await asyncio.get_event_loop().run_in_executor(None, scorer.start)
        except Exception as err:  # noqa: BLE001 - surface to the browser
            self._on_tyto_state("error", str(err))

        provider.connect()  # builds and runs the pipeline on this loop
        controller.set_connected(True)

    def stop(self) -> None:
        if self.controller:
            self.controller.set_connected(False)
        if self.scorer:
            self.scorer.stop()
        if self.provider:
            self.provider.disconnect()

    # -- UI plumbing (controller -> browser over the data channel) ---------- #

    def _on_update(self, state: dict) -> None:
        if "scores" in state:
            scores = state["scores"]
            self._send(
                {"type": "scores", "scores": scores.as_dict(), "room": state.get("room", ""), "vad": state.get("vad", "eager")}
            )
        elif "transcript" in state:
            tx = state["transcript"]
            self._send({"type": "transcript", "who": tx["who"], "text": tx["text"], "final": tx["final"]})
        elif "nudge" in state:
            self._send({"type": "nudge", **state["nudge"]})

    def _on_tyto_state(self, state: str, text: str) -> None:
        self._last_tyto_state = {"type": "tyto_state", "state": state, "text": text}
        self._send(self._last_tyto_state)

    def _on_connected(self) -> None:
        # The data channel is up now, so this reliably reaches the browser even
        # if the early (pre-connection) sends were dropped.
        self._send({"type": "status", "state": "live", "label": "Live"})
        if self._last_tyto_state:
            self._send(self._last_tyto_state)

    def _on_client_message(self, message: dict) -> None:
        if message.get("type") == "nudge_threshold" and self.controller:
            self.controller.nudge_threshold = float(message.get("value", 0.5))

    def _send(self, message: dict) -> None:
        if self.provider:
            self.provider.send_ui(message)


# --------------------------------------------------------------------------- #
# FastAPI app: static files + the WebRTC offer endpoint                       #
# --------------------------------------------------------------------------- #


def build_app(keys: dict):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse
    from pipecat.transports.smallwebrtc.request_handler import (
        SmallWebRTCRequest,
        SmallWebRTCRequestHandler,
    )

    app = FastAPI()
    handler = SmallWebRTCRequestHandler(esp32_mode=False, host="127.0.0.1")
    sessions: dict[str, Session] = {}

    @app.get("/")
    async def index():
        return FileResponse(INDEX)

    @app.get("/app.js")
    async def app_js():
        return FileResponse(APP_JS, media_type="text/javascript")

    @app.get("/favicon.ico")
    async def favicon():
        from fastapi.responses import Response

        return Response(status_code=204)

    @app.post("/api/offer")
    async def offer(request: Request):
        body = await request.json()
        webrtc_request = SmallWebRTCRequest.from_dict(body)

        # The callback fires only for a brand-new connection (not for the
        # renegotiations the handler manages internally), so it's the right
        # place to spin up exactly one Tyto session per visitor.
        async def on_new_connection(connection):
            session = Session(connection, keys)

            @connection.event_handler("closed")
            async def _on_closed(_conn):
                sessions.pop(connection.pc_id, None)
                session.stop()

            await session.start()
            sessions[connection.pc_id] = session

        answer = await handler.handle_web_request(
            request=webrtc_request, webrtc_connection_callback=on_new_connection
        )
        return JSONResponse(answer)

    return app


def main() -> None:
    import uvicorn

    load_env()
    keys = {
        "license": os.environ.get("AIC_SDK_LICENSE", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
    }
    if not keys["license"] or not keys["openai"]:
        raise SystemExit("Set AIC_SDK_LICENSE and OPENAI_API_KEY (see .env.example).")

    host, port = "127.0.0.1", int(os.environ.get("PORT", "8080"))
    print(f"Tyto Pipecat demo on http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(build_app(keys), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
