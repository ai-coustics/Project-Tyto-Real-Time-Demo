"""LiveKit Agents backend for the Tyto voice agent.

Same demo and same browser UI as ``examples/web`` and ``examples/pipecat``, but
the voice backend is a LiveKit ``AgentSession`` (OpenAI Realtime speech-to-speech)
running as a LiveKit agent worker. LiveKit transports the media (browser mic and
agent audio over WebRTC) and the data channel; this worker is still the whole
brain: Tyto scoring, the three adaptation layers, and the keys all live here.

Per LiveKit room, one session::

    browser mic  ── WebRTC ─>  room ─> AgentSession ─> OpenAI Realtime
                                 └─> rtc.AudioStream ─> LiveTytoScorer.feed()
    agent audio  <─ WebRTC ──  AgentSession output (browser plays it natively)
    scores / room / vad / nudge  <─ room data channel ──  controller

The decision layer, scorer, and controller are shared with every other Tyto
frontend; only the provider ([livekit_provider.py](../../src/tyto_voice/livekit_provider.py))
is specific to this stack.

Run the worker (joins rooms automatically; no agent_name = automatic dispatch)::

    uv pip install -e ".[livekit]"
    # put LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY,
    # and AIC_SDK_LICENSE in .env.local (or .env)
    uv run --module livekit.agents download-files   # noise-cancellation model files (one-time)
    uv run examples/livekit/agent.py dev

Then serve the browser UI from a second terminal::

    uv run examples/livekit/token_server.py         # open http://localhost:8080

Or talk to it from the LiveKit Agent Console with ``console`` mode.
"""

from __future__ import annotations

import asyncio
import json

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    function_tool,
)
from livekit.plugins import openai

from tyto_voice.controller import TytoController
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.env import load_env
from tyto_voice.livekit_provider import SAMPLE_RATE, LiveKitProvider, _to_turn_detection
from tyto_voice.prompts import BASE_INSTRUCTIONS
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer

load_env(".env.local")  # LiveKit's convention; falls through to .env below
load_env()

VOICE = "alloy"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"


class Assistant(Agent):
    """The Tyto demo host. ``check_audio_quality`` is answered from the live
    Tyto snapshot, set on the agent once the controller exists."""

    def __init__(self) -> None:
        super().__init__(instructions=BASE_INSTRUCTIONS)
        # Wired to controller.audio_quality_snapshot after the controller is built.
        self.audio_quality_fn = None

    @function_tool
    async def check_audio_quality(self, context: RunContext) -> dict:
        """Get the current real-time audio quality of the user's mic input.

        Returns a summary, verdict, the Tyto Score, and the top current issue.
        Call this whenever the user asks if you can hear them, how their audio
        sounds, or about their connection/environment.
        """
        if self.audio_quality_fn is None:
            return {"status": "warming up, ask again in a few seconds"}
        return self.audio_quality_fn()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    keys = _require_keys()

    handlers = Handlers()
    agent = Assistant()
    realtime_model = openai.realtime.RealtimeModel(
        voice=VOICE,
        turn_detection=_to_turn_detection(VAD_PROFILES["eager"]),
    )
    session = AgentSession(llm=realtime_model)

    scorer = LiveTytoScorer(
        keys["license"],
        sample_rate=SAMPLE_RATE,
        on_state=lambda state, text: _send(provider, {"type": "tyto_state", "state": state, "text": text}),
    )
    provider = LiveKitProvider(
        handlers,
        session=session,
        agent=agent,
        realtime_model=realtime_model,
        room=ctx.room,
        scorer=scorer,
        turn_detection=VAD_PROFILES["eager"],
        on_client_message=lambda m: _set_threshold(controller, m),
        on_log=lambda k, t: _send(provider, {"type": "log", "kind": k, "text": t}),
    )
    controller = TytoController(
        provider,
        scorer,
        on_update=lambda state: _on_update(provider, state),
        on_log=lambda k, t: _send(provider, {"type": "log", "kind": k, "text": t}),
    )
    scorer.on_scores = controller.on_scores
    agent.audio_quality_fn = controller.audio_quality_snapshot

    handlers.on_agent_speaking = controller.on_agent_speaking
    handlers.on_agent_audio = controller.on_agent_audio
    handlers.on_user_transcript = controller.on_user_transcript
    handlers.on_agent_transcript = controller.on_agent_transcript
    # on_tool_call is intentionally unwired: the tool is answered by the
    # @function_tool above instead.

    # Forward nudge_threshold messages from the browser to the controller.
    ctx.room.on("data_received", lambda packet: _on_data(provider, packet))

    # Capture the loop and wire session events + the mic tap before any blocking
    # work, so scorer state messages have a loop to be published on.
    provider.connect()

    # The model download + license check is blocking, so keep it off the loop.
    # A scorer failure is non-fatal: the agent still runs and the error is shown
    # over the data channel.
    try:
        await asyncio.get_running_loop().run_in_executor(None, scorer.start)
    except Exception as err:  # noqa: BLE001 - surface to the browser
        _send(provider, {"type": "tyto_state", "state": "error", "text": str(err)})

    await session.start(room=ctx.room, agent=agent)
    controller.set_connected(True)
    _send(provider, {"type": "status", "state": "live", "label": "Live"})

    controller.on_ready()  # let the agent open the conversation


# --------------------------------------------------------------------------- #
# Plumbing: controller/scorer callbacks -> browser over the room data channel  #
# --------------------------------------------------------------------------- #


def _on_update(provider: LiveKitProvider, state: dict) -> None:
    if "scores" in state:
        scores = state["scores"]
        _send(
            provider,
            {"type": "scores", "scores": scores.as_dict(), "room": state.get("room", ""), "vad": state.get("vad", "eager")},
        )
    elif "transcript" in state:
        tx = state["transcript"]
        _send(provider, {"type": "transcript", "who": tx["who"], "text": tx["text"], "final": tx["final"]})
    elif "nudge" in state:
        _send(provider, {"type": "nudge", **state["nudge"]})


def _on_data(provider: LiveKitProvider, packet) -> None:
    try:
        message = json.loads(bytes(packet.data).decode("utf-8"))
    except Exception:  # noqa: BLE001 - ignore malformed packets
        return
    provider.handle_client_message(message)


def _set_threshold(controller: TytoController, message: dict) -> None:
    controller.nudge_threshold = float(message.get("value", 0.5))


def _send(provider: LiveKitProvider, message: dict) -> None:
    provider.send_ui(message)


def _require_keys() -> dict:
    import os

    keys = {
        "license": os.environ.get("AIC_SDK_LICENSE", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
    }
    if not keys["license"] or not keys["openai"]:
        raise RuntimeError("Set AIC_SDK_LICENSE and OPENAI_API_KEY (see .env.example).")
    return keys


if __name__ == "__main__":
    agents.cli.run_app(server)
