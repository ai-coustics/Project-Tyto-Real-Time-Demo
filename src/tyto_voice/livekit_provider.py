"""LiveKit Agents backend for the Tyto voice agent.

A ``VoiceProvider`` that drives a LiveKit ``AgentSession`` (with the OpenAI
Realtime model) instead of speaking the OpenAI WebSocket directly. The point of
the provider seam is that the controller and the decision layers do not change:
this is just one more subclass, exactly like ``OpenAIRealtimeProvider`` and
``PipecatRealtimeProvider`` (see [provider.py](provider.py),
[openai_realtime.py](openai_realtime.py), and
[pipecat_provider.py](pipecat_provider.py)).

The topology, per LiveKit room::

    browser mic  ── WebRTC audio ─>  LiveKit room ─>  AgentSession ─> OpenAI Realtime
                                          │
                                          └─> rtc.AudioStream ─> LiveTytoScorer.feed()
    agent audio  <─ WebRTC audio ──  AgentSession output (played natively by the browser)
    scores / room / vad / nudge  <─ room data channel ──  controller

LiveKit owns the transport and the agent loop, so the seam maps onto its session
API rather than raw protocol envelopes:

    Layer 1 Aware    set_instructions     -> Agent.update_instructions(text)
    Layer 2 Tuned    set_turn_detection   -> RealtimeModel.update_options(turn_detection=...)
                       (None disables server VAD: the listen gate during a nudge)
    Layer 3 Reactive interrupt + nudge    -> AgentSession.interrupt() + generate_reply(instructions=...)
    mic gate         set_mic_enabled      -> AgentSession.input.set_audio_enabled(on)
    greeting         request_response     -> AgentSession.generate_reply()

Provider events arrive as LiveKit session events (on the agent's event loop) and
are turned into ``Handlers`` calls:

    speech_created / SpeechHandle.done  -> on_agent_speaking (nudge tracked by handle identity)
    agent_state_changed (speaking)      -> on_agent_audio
    user_input_transcribed              -> on_user_transcript
    conversation_item_added (assistant) -> on_agent_transcript

``check_audio_quality`` is answered by a ``@function_tool`` on the agent (see
``examples/livekit/agent.py``), so ``on_tool_call`` / ``send_tool_result`` are
unused by this backend, just like the Pipecat one.

Commands arrive on the scorer thread (via the controller) and are marshaled onto
the session's event loop; the controller's re-entrant lock makes the crossing
safe.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import numpy as np

from .decision import VAD_PROFILES
from .provider import Handlers, VoiceProvider

SAMPLE_RATE = 24000  # PCM16 mono; fed to Tyto and used to resample the mic tap.


def _to_turn_detection(td: dict | None):
    """Translate a decision-layer VAD profile dict into an OpenAI Realtime model.

    ``None`` disables server turn detection (the listen gate: the agent stops
    auto-responding while a nudge is in flight).
    """
    if not td:
        return None
    from openai.types.realtime import realtime_audio_input_turn_detection as rt

    if td.get("type") == "semantic_vad":
        return rt.SemanticVad(
            type="semantic_vad",
            eagerness=td.get("eagerness", "auto"),
            create_response=td.get("create_response", True),
            interrupt_response=td.get("interrupt_response", True),
        )
    return rt.ServerVad(
        type="server_vad",
        threshold=td.get("threshold", 0.5),
        prefix_padding_ms=td.get("prefix_padding_ms", 300),
        silence_duration_ms=td.get("silence_duration_ms", 500),
        create_response=td.get("create_response", True),
    )


class LiveKitProvider(VoiceProvider):
    def __init__(
        self,
        handlers: Handlers,
        *,
        session,
        agent,
        realtime_model,
        room,
        scorer,
        turn_detection: dict | None = None,
        on_client_message: Callable[[dict], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        super().__init__(handlers)
        self._session = session
        self._agent = agent
        self._rt = realtime_model
        self._room = room
        self._scorer = scorer
        self._turn_detection = turn_detection or VAD_PROFILES["eager"]
        self._on_client_message = on_client_message
        self._on_log = on_log

        self._loop: asyncio.AbstractEventLoop | None = None
        self._nudge_handle = None  # identifies the in-flight nudge speech
        self._cancelled = False
        self._tap_tasks: list[asyncio.Task] = []

    # -- lifecycle (called from the agent's event loop) --------------------- #

    def connect(self) -> None:
        """Capture the loop, wire session events, and start the mic tap.

        Called from within the running session loop (the rtc_session entrypoint).
        Event handlers and the ``track_subscribed`` listener are registered here,
        before ``session.start``, so no early events are missed.
        """
        self._loop = asyncio.get_running_loop()
        self._wire_session_events()
        self._start_audio_tap()

    def disconnect(self) -> None:
        for task in self._tap_tasks:
            task.cancel()
        self._tap_tasks.clear()

    # -- session events (provider -> app), on the loop thread --------------- #

    def _wire_session_events(self) -> None:
        session = self._session

        @session.on("speech_created")
        def _on_speech_created(ev) -> None:
            handle = ev.speech_handle
            is_nudge = handle is self._nudge_handle
            if self.h.on_agent_speaking:
                self.h.on_agent_speaking(True, nudge=is_nudge)

            def _on_done(h) -> None:
                nudge = h is self._nudge_handle
                if nudge:
                    self._nudge_handle = None
                if self.h.on_agent_speaking:
                    self.h.on_agent_speaking(False, nudge=nudge, cancelled=h.interrupted)

            handle.add_done_callback(_on_done)

        @session.on("agent_state_changed")
        def _on_agent_state(ev) -> None:
            # Drives the "is the agent audible" signal that gates unmuting the
            # mic until the agent has actually fallen silent.
            if self.h.on_agent_audio:
                self.h.on_agent_audio(ev.new_state == "speaking")

        @session.on("user_input_transcribed")
        def _on_user_tx(ev) -> None:
            if self.h.on_user_transcript:
                self.h.on_user_transcript(ev.transcript, ev.is_final)

        @session.on("conversation_item_added")
        def _on_item(ev) -> None:
            item = ev.item
            if getattr(item, "role", None) != "assistant":
                return  # user turns come through user_input_transcribed
            text = getattr(item, "text_content", None)
            if text and self.h.on_agent_transcript:
                self.h.on_agent_transcript(text, True)

        @session.on("error")
        def _on_error(ev) -> None:
            self._log("error", str(getattr(ev, "error", ev)))

    # -- mic tap (feed the user's audio to Tyto) ---------------------------- #

    def _start_audio_tap(self) -> None:
        from livekit import rtc

        def _maybe_tap(track) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self._tap_tasks.append(self._loop.create_task(self._consume(track)))

        @self._room.on("track_subscribed")
        def _on_track_subscribed(track, _publication, _participant) -> None:
            _maybe_tap(track)

        # Catch a track that was already subscribed before we wired the handler.
        for participant in self._room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    _maybe_tap(pub.track)

    async def _consume(self, track) -> None:
        from livekit import rtc

        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            async for event in stream:
                frame = event.frame
                mono = np.frombuffer(bytes(frame.data), dtype="<i2").astype(np.float32) / 32768.0
                self._scorer.feed(mono)
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    # -- commands (app -> provider, called from the scorer thread) ---------- #

    def set_instructions(self, text: str) -> None:
        self._call_soon(lambda: self._loop.create_task(self._agent.update_instructions(text)))

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        td = _to_turn_detection(turn_detection)
        self._call_soon(lambda: self._rt.update_options(turn_detection=td))

    def set_mic_enabled(self, on: bool) -> None:
        self._call_soon(lambda: self._session.input.set_audio_enabled(on))

    def interrupt(self, clear_input: bool = False) -> None:
        self._nudge_handle = None
        self._cancelled = True
        self._call_soon(lambda: self._do_interrupt(clear_input))

    def _do_interrupt(self, clear_input: bool) -> None:
        self._session.interrupt()
        if clear_input:
            try:
                self._session.clear_user_turn()
            except Exception:  # noqa: BLE001 - nothing to clear is fine
                pass

    def nudge(self, text: str) -> None:
        self._cancelled = False
        payload = "Say exactly this in one short natural sentence and nothing else: " + json.dumps(text)
        self._call_soon(lambda: self._fire_nudge(payload))

    def _fire_nudge(self, payload: str) -> None:
        # Capture the handle so speech_created/done can tell this nudge apart
        # from a normal turn (mirrors the raw provider's _nudge_resp_id).
        self._nudge_handle = self._session.generate_reply(instructions=payload)

    def request_response(self) -> None:
        # Opening greeting: let the agent speak first. The system instructions
        # tell it to open the conversation, so no extra instructions are needed.
        self._call_soon(lambda: self._session.generate_reply())

    def send_tool_result(self, call_id: str, output: dict) -> None:
        # Unused: check_audio_quality is answered by a @function_tool on the
        # agent, not through the controller's on_tool_call/send_tool_result path.
        pass

    # -- outbound to the browser UI (room data channel) --------------------- #

    def send_ui(self, message: dict) -> None:
        data = json.dumps(message).encode("utf-8")
        self._call_soon(lambda: self._loop.create_task(self._publish(data)))

    async def _publish(self, data: bytes) -> None:
        try:
            await self._room.local_participant.publish_data(data, reliable=True, topic="tyto")
        except Exception:  # noqa: BLE001 - data channel may not be up yet
            pass

    def handle_client_message(self, message: dict) -> None:
        if message.get("type") == "nudge_threshold" and self._on_client_message:
            self._on_client_message(message)

    # -- internals ---------------------------------------------------------- #

    def _call_soon(self, fn: Callable[[], None]) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(fn)

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)
