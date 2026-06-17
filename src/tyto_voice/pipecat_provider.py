"""Pipecat backend for the Tyto voice agent.

A ``VoiceProvider`` that drives a Pipecat pipeline instead of speaking the
OpenAI Realtime WebSocket directly. The point of the provider seam is that the
controller and the decision layers do not change: this is just one more subclass
(see [provider.py](provider.py) and [openai_realtime.py](openai_realtime.py)).

The pipeline, per browser connection::

    SmallWebRTCTransport.input()   browser mic (PCM16, 24 kHz over WebRTC)
        -> TytoAudioTap            taps the mic into LiveTytoScorer.feed()
        -> user context aggregator
        -> OpenAIRealtimeLLMService (speech-to-speech, server-side)
        -> SmallWebRTCTransport.output()   agent audio back to the browser
        -> assistant context aggregator

Pipecat owns the transport and the frame flow, so a few things differ from the
raw-websocket provider. These are the documented gaps, not faked behavior:

- Agent audio is played by the transport (to the browser), not handed to a sink.
  ``on_agent_audio`` is therefore driven by ``Bot{Started,Stopped}SpeakingFrame``
  (a server-side estimate) rather than browser-reported audibility.
- ``check_audio_quality`` is answered by a registered Pipecat function handler,
  so ``on_tool_call`` / ``send_tool_result`` are not used by this backend.
- ``request_response`` (the opening greeting) queues an ``LLMRunFrame``, which is
  how the context aggregator and realtime service are driven to speak first.

The three layers still map directly to OpenAI Realtime client events, exactly as
in the raw provider:

    Layer 1 Aware   -> session.update (instructions)
    Layer 2 Tuned   -> session.update (audio.input.turn_detection)
    Layer 3 Reactive-> response.cancel + a one-shot response.create (nudge)

Commands arrive on the scorer thread (via the controller) and are marshaled onto
the pipeline's event loop; provider events arrive as pipeline frames, observed by
``TytoFrameObserver`` and turned into ``Handlers`` calls on the loop thread. The
controller's own re-entrant lock makes that crossing safe.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import numpy as np

from .decision import VAD_PROFILES
from .provider import Handlers, VoiceProvider

SAMPLE_RATE = 24000  # PCM16 mono, OpenAI Realtime's native rate; fed to Tyto too.

# The check_audio_quality tool, as a Pipecat FunctionSchema (built lazily so the
# package imports without pipecat installed).
CHECK_AUDIO_QUALITY_DESCRIPTION = (
    "Get the current real-time audio quality of the user's mic input. Returns a "
    "summary, verdict, the Tyto Score, and the top current issue. Call this whenever "
    "the user asks if you can hear them, how their audio sounds, or about their "
    "connection/environment."
)


def _turn_detection_model(td: dict | None):
    """Translate a decision-layer VAD profile dict into a Realtime event model.

    ``None`` becomes ``False`` (server turn detection disabled), which is the
    listen gate: the agent stops auto-responding while a nudge is in flight.
    """
    from pipecat.services.openai.realtime import events

    if not td:
        return False
    if td.get("type") == "semantic_vad":
        return events.SemanticTurnDetection(
            eagerness=td.get("eagerness", "auto"),
            create_response=td.get("create_response", True),
            interrupt_response=td.get("interrupt_response", True),
        )
    return events.TurnDetection(
        threshold=td.get("threshold", 0.5),
        prefix_padding_ms=td.get("prefix_padding_ms", 300),
        silence_duration_ms=td.get("silence_duration_ms", 500),
    )


class PipecatRealtimeProvider(VoiceProvider):
    def __init__(
        self,
        handlers: Handlers,
        *,
        api_key: str,
        instructions: str,
        scorer,
        webrtc_connection,
        model: str = "gpt-realtime",
        voice: str = "alloy",
        transcribe_model: str = "gpt-4o-mini-transcribe",
        turn_detection: dict | None = None,
        audio_quality_fn: Callable[[], dict] | None = None,
        on_client_message: Callable[[dict], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        super().__init__(handlers)
        self._api_key = api_key
        self._instructions = instructions
        self._scorer = scorer
        self._connection = webrtc_connection
        self._model = model
        self._voice = voice
        self._transcribe_model = transcribe_model
        self._turn_detection = turn_detection or VAD_PROFILES["eager"]
        # Settable after construction, like Handlers in the other frontends, so
        # the session can wire the controller's snapshot in once it exists.
        self.audio_quality_fn = audio_quality_fn
        self._on_client_message = on_client_message
        self._on_connected = on_connected
        self._on_log = on_log

        self._loop: asyncio.AbstractEventLoop | None = None
        self._llm = None
        self._task = None
        self._runner = None
        self._run_handle = None

        # nudge / cancel bookkeeping, mirroring the raw provider's flags.
        self._pending_nudge = False
        self._current_is_nudge = False
        self._cancelled = False

    # -- lifecycle (called from the server event loop) ---------------------- #

    def connect(self) -> None:
        """Build the pipeline and start it on the current event loop.

        Must be called from within the asyncio loop that owns the WebRTC
        connection (the FastAPI request handler), because aiortc objects are
        loop-bound. The pipeline then runs as a task on that same loop.
        """
        self._loop = asyncio.get_event_loop()
        self._build_pipeline()
        self._run_handle = self._loop.create_task(self._runner.run(self._task))

    def disconnect(self) -> None:
        if self._loop and self._task:
            asyncio.run_coroutine_threadsafe(self._task.cancel(), self._loop)

    def _build_pipeline(self) -> None:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
        )
        from pipecat.services.openai.realtime import events
        from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        tool = FunctionSchema(
            name="check_audio_quality",
            description=CHECK_AUDIO_QUALITY_DESCRIPTION,
            properties={},
            required=[],
        )
        session_properties = events.SessionProperties(
            instructions=self._instructions,
            audio=events.AudioConfiguration(
                input=events.AudioInput(
                    transcription=events.InputAudioTranscription(model=self._transcribe_model),
                    turn_detection=_turn_detection_model(self._turn_detection),
                ),
                output=events.AudioOutput(voice=self._voice),
            ),
            tools=[tool],
            tool_choice="auto",
        )
        self._llm = OpenAIRealtimeLLMService(
            api_key=self._api_key,
            settings=OpenAIRealtimeLLMService.Settings(
                model=self._model, session_properties=session_properties
            ),
        )

        async def _check_audio_quality(params):
            result = self.audio_quality_fn() if self.audio_quality_fn else {"status": "unavailable"}
            self._log("tool.check_audio_quality", result.get("summary", ""))
            await params.result_callback(result)

        self._llm.register_function("check_audio_quality", _check_audio_quality)

        transport = SmallWebRTCTransport(
            webrtc_connection=self._connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=SAMPLE_RATE,
                audio_out_sample_rate=SAMPLE_RATE,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def _on_connected(_transport, _client):
            self._log("pipecat.client", "connected")
            # Now that the data channel is up, push the initial UI state (sent
            # here rather than at session start, where it would race the channel
            # opening), then let the agent open the conversation.
            if self._on_connected:
                self._on_connected()
            if self.h.on_ready:
                self.h.on_ready()

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(_transport, _client):
            self._log("pipecat.client", "disconnected")

        @transport.event_handler("on_app_message")
        async def _on_app_message(_transport, message, _sender):
            self._handle_app_message(message)

        context = LLMContext(messages=[{"role": "system", "content": self._instructions}])
        aggregators = LLMContextAggregatorPair(context, realtime_service_mode=True)

        tap = TytoAudioTap(self._scorer)
        pipeline = Pipeline(
            [
                transport.input(),
                tap,
                aggregators.user(),
                self._llm,
                transport.output(),
                aggregators.assistant(),
            ]
        )
        self._task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            ),
            observers=[TytoFrameObserver(self)],
            idle_timeout_secs=None,
        )
        self._runner = PipelineRunner(handle_sigint=False)

    # -- commands (app -> provider, called from the scorer thread) ---------- #

    def set_instructions(self, text: str) -> None:
        from pipecat.services.openai.realtime import events

        self._send_event(events.SessionUpdateEvent(session=events.SessionProperties(instructions=text)))

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        from pipecat.services.openai.realtime import events

        session = events.SessionProperties(
            audio=events.AudioConfiguration(
                input=events.AudioInput(turn_detection=_turn_detection_model(turn_detection))
            )
        )
        self._send_event(events.SessionUpdateEvent(session=session))

    def set_mic_enabled(self, on: bool) -> None:
        self._call_soon(lambda: self._llm.set_audio_input_paused(not on))

    def interrupt(self, clear_input: bool = False) -> None:
        self._pending_nudge = False
        self._cancelled = True
        self._call_soon(lambda: self._loop.create_task(self._do_interrupt(clear_input)))

    async def _do_interrupt(self, clear_input: bool) -> None:
        from pipecat.frames.frames import InterruptionFrame
        from pipecat.services.openai.realtime import events

        await self._task.queue_frame(InterruptionFrame())
        if clear_input:
            await self._llm.send_client_event(events.InputAudioBufferClearEvent())
        await self._llm.send_client_event(events.ResponseCancelEvent())

    def nudge(self, text: str) -> None:
        from pipecat.services.openai.realtime import events

        self._pending_nudge = True
        self._cancelled = False
        payload = "Say exactly this in one short natural sentence and nothing else: " + json.dumps(text)
        self._send_event(
            events.ResponseCreateEvent(response=events.ResponseProperties(instructions=payload))
        )

    def request_response(self) -> None:
        # Kick off the opening greeting. The context aggregator emits its context
        # on an LLMRunFrame, and the realtime service responds to the first
        # context, so this is what makes the agent speak first.
        from pipecat.frames.frames import LLMRunFrame

        self._call_soon(lambda: self._loop.create_task(self._task.queue_frame(LLMRunFrame())))

    def send_tool_result(self, call_id: str, output: dict) -> None:
        # Unused: the tool is answered by the registered function handler above,
        # not through the controller's on_tool_call/send_tool_result path.
        pass

    # -- outbound to the browser UI ----------------------------------------- #

    def send_ui(self, message: dict) -> None:
        """Push a UI message to the browser over the WebRTC data channel."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._connection.send_app_message, message)

    # -- internals ---------------------------------------------------------- #

    def _handle_app_message(self, message) -> None:
        if not isinstance(message, dict):
            return
        if message.get("type") == "nudge_threshold" and self._on_client_message:
            self._on_client_message(message)

    def _send_event(self, event) -> None:
        if not self._loop or not self._llm:
            return
        asyncio.run_coroutine_threadsafe(self._llm.send_client_event(event), self._loop)

    def _call_soon(self, fn: Callable[[], None]) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(fn)

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)


class TytoAudioTap:
    """A Pipecat FrameProcessor that feeds user mic audio into the scorer.

    Sits right after the transport input, so it sees every ``InputAudioRawFrame``
    (PCM16 mono at the transport's input rate) and forwards the same audio to
    Tyto. It never consumes frames: every frame is pushed on unchanged.

    Defined as a thin wrapper so the module imports without pipecat present; the
    real base class is mixed in lazily on first construction.
    """

    def __new__(cls, scorer):
        impl = _audio_tap_class()
        return impl(scorer)


_AUDIO_TAP_CLASS = None


def _audio_tap_class():
    global _AUDIO_TAP_CLASS
    if _AUDIO_TAP_CLASS is not None:
        return _AUDIO_TAP_CLASS

    from pipecat.frames.frames import Frame, InputAudioRawFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class _TytoAudioTap(FrameProcessor):
        def __init__(self, scorer):
            super().__init__()
            self._scorer = scorer

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                mono = np.frombuffer(frame.audio, dtype="<i2").astype(np.float32) / 32768.0
                self._scorer.feed(mono)
            await self.push_frame(frame, direction)

    _AUDIO_TAP_CLASS = _TytoAudioTap
    return _AUDIO_TAP_CLASS


class TytoFrameObserver:
    """Translates pipeline frames into ``Handlers`` calls for the controller.

    Lazily subclasses ``BaseObserver``. Every frame push is observed, so we
    dedupe by frame identity to react exactly once per frame.
    """

    def __new__(cls, provider: PipecatRealtimeProvider):
        impl = _observer_class()
        return impl(provider)


_OBSERVER_CLASS = None


def _observer_class():
    global _OBSERVER_CLASS
    if _OBSERVER_CLASS is not None:
        return _OBSERVER_CLASS

    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        InterimTranscriptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TranscriptionFrame,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed

    class _TytoFrameObserver(BaseObserver):
        def __init__(self, provider: PipecatRealtimeProvider):
            super().__init__()
            self._p = provider
            self._seen: set[int] = set()
            self._agent_text = ""

        async def on_push_frame(self, data: FramePushed):
            frame = data.frame
            fid = id(frame)
            if fid in self._seen:
                return
            self._seen.add(fid)
            if len(self._seen) > 4096:
                self._seen.clear()
                self._seen.add(fid)

            h = self._p.h
            if isinstance(frame, LLMFullResponseStartFrame):
                self._p._current_is_nudge = self._p._pending_nudge
                self._p._pending_nudge = False
                self._agent_text = ""
                if h.on_agent_speaking:
                    h.on_agent_speaking(True, nudge=self._p._current_is_nudge)
            elif isinstance(frame, LLMFullResponseEndFrame):
                nudge = self._p._current_is_nudge
                cancelled = self._p._cancelled
                if self._agent_text and h.on_agent_transcript:
                    h.on_agent_transcript(self._agent_text, True)
                self._agent_text = ""
                self._p._current_is_nudge = False
                self._p._cancelled = False
                if h.on_agent_speaking:
                    h.on_agent_speaking(False, nudge=nudge, cancelled=cancelled)
            elif isinstance(frame, BotStartedSpeakingFrame):
                if h.on_agent_audio:
                    h.on_agent_audio(True)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                if h.on_agent_audio:
                    h.on_agent_audio(False)
            elif isinstance(frame, LLMTextFrame):
                self._agent_text += frame.text
                if h.on_agent_transcript:
                    h.on_agent_transcript(frame.text, False)
            elif isinstance(frame, TranscriptionFrame):
                if h.on_user_transcript:
                    h.on_user_transcript(frame.text, True)
            elif isinstance(frame, InterimTranscriptionFrame):
                if h.on_user_transcript:
                    h.on_user_transcript(frame.text, False)

    _OBSERVER_CLASS = _TytoFrameObserver
    return _OBSERVER_CLASS
