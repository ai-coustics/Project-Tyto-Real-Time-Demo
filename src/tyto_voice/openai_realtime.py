"""OpenAI Realtime provider over WebSocket (server-side).

The Python counterpart of the browser reference's ``OpenAIRealtimeProvider``.
The browser uses WebRTC and an ephemeral key; a server uses the WebSocket
transport and the standard API key. The control envelopes (``session.update`` /
``response.create``), the turn-detection profiles, and the nudge mechanism are
the same, which is the point of the provider seam.

Audio is decoupled from playback. This class speaks the OpenAI protocol and
hands agent audio to three callbacks, so the same provider drives a local
speaker (the terminal demo) or a browser over a websocket (the web demo):

    audio_out(pcm16: bytes)   one chunk of agent audio (mono PCM16, 24 kHz)
    audio_done()              the agent finished generating this response
    audio_flush()             drop any pending agent audio (on interrupt)

Whoever plays the audio owns the "is the agent audible" signal and calls
``handlers.on_agent_audio(playing)``; this class only reports generation
lifecycle via ``handlers.on_agent_speaking``.

Mic audio is sent with :meth:`send_audio`; muting just stops sending it.
Audio format is PCM16 mono at 24 kHz, the Realtime WebSocket default.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from typing import Callable

import numpy as np

from .controller import CHECK_AUDIO_QUALITY_TOOL
from .decision import VAD_PROFILES
from .provider import Handlers, VoiceProvider

REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"
SAMPLE_RATE = 24000  # Realtime WebSocket default for PCM16


class OpenAIRealtimeProvider(VoiceProvider):
    def __init__(
        self,
        handlers: Handlers,
        *,
        api_key: str,
        instructions: str,
        audio_out: Callable[[bytes], None],
        audio_done: Callable[[], None] | None = None,
        audio_flush: Callable[[], None] | None = None,
        model: str = "gpt-realtime",
        voice: str = "alloy",
        transcribe_model: str = "gpt-4o-mini-transcribe",
        turn_detection: dict | None = None,
        tools: list | None = None,
        on_log=None,
    ):
        super().__init__(handlers)
        self._api_key = api_key
        self._instructions = instructions
        self._audio_out = audio_out
        self._audio_done = audio_done
        self._audio_flush = audio_flush
        self._model = model
        self._voice = voice
        self._transcribe_model = transcribe_model
        self._turn_detection = turn_detection or VAD_PROFILES["eager"]
        self._tools = tools if tools is not None else [CHECK_AUDIO_QUALITY_TOOL]
        self._on_log = on_log

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._thread: threading.Thread | None = None
        self.closed = threading.Event()

        self._mic_enabled = True
        self._nudge_resp_id: str | None = None
        self._pending_nudge = False

    # -- lifecycle ---------------------------------------------------------- #

    def connect(self) -> None:
        self._thread = threading.Thread(target=self._run, name="openai-realtime", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        self.closed.set()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as err:  # noqa: BLE001
            self._log("error", str(err))
        finally:
            self.closed.set()

    async def _main(self) -> None:
        from websockets.asyncio.client import connect

        self._loop = asyncio.get_running_loop()
        url = REALTIME_URL.format(model=self._model)
        headers = [("Authorization", f"Bearer {self._api_key}")]
        async with connect(url, additional_headers=headers, max_size=None) as ws:
            self._ws = ws
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": self._instructions,
                    "audio": {
                        "input": {
                            "transcription": {"model": self._transcribe_model},
                            "turn_detection": self._turn_detection,
                        },
                        "output": {"voice": self._voice},
                    },
                    "tools": self._tools,
                    "tool_choice": "auto",
                },
            }))
            self._log("session.update", "sent")
            if self.h.on_ready:
                self.h.on_ready()
            async for raw in ws:
                self._receive(json.loads(raw))

    # -- commands (app -> provider) ----------------------------------------- #

    def set_instructions(self, text: str) -> None:
        self._send({"type": "session.update", "session": {"type": "realtime", "instructions": text}})

    def set_turn_detection(self, turn_detection: dict | None) -> None:
        self._send(
            {
                "type": "session.update",
                "session": {"type": "realtime", "audio": {"input": {"turn_detection": turn_detection}}},
            }
        )

    def set_mic_enabled(self, on: bool) -> None:
        self._mic_enabled = on

    def interrupt(self, clear_input: bool = False) -> None:
        self._nudge_resp_id = None
        if self._audio_flush:
            self._audio_flush()
        if clear_input:
            self._send({"type": "input_audio_buffer.clear"})
        self._send({"type": "response.cancel"})

    def nudge(self, text: str) -> None:
        self._pending_nudge = True
        self._send(
            {
                "type": "response.create",
                "response": {
                    "metadata": {"tyto_purpose": "nudge"},
                    "input": [],
                    "instructions": (
                        "Say exactly this in one short natural sentence and nothing else: "
                        + json.dumps(text)
                    ),
                },
            }
        )

    def request_response(self) -> None:
        self._send({"type": "response.create"})

    def send_tool_result(self, call_id: str, output: dict) -> None:
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                },
            }
        )
        self._send({"type": "response.create"})

    def send_audio(self, mono: np.ndarray) -> None:
        """Forward one block of mono float32 mic audio to the agent (if unmuted)."""
        if not self._mic_enabled or not self._ws:
            return
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16).decode("ascii")})

    # -- events (provider -> app) ------------------------------------------- #

    def _receive(self, msg: dict) -> None:
        t = msg.get("type", "")
        if t == "error":
            reason = (msg.get("error") or {}).get("message", "")
            if "cancel" in reason.lower() or "no active response" in reason.lower():
                return
            self._log("error", reason)
            return

        if t == "response.created":
            nudge = self._pending_nudge or self._is_nudge(msg.get("response"))
            if nudge:
                self._pending_nudge = False
                self._nudge_resp_id = (msg.get("response") or {}).get("id")
            if self.h.on_agent_speaking:
                self.h.on_agent_speaking(True, nudge=nudge)

        elif t in ("response.output_audio.delta", "response.audio.delta"):
            data = msg.get("delta", "")
            if data and self._audio_out:
                self._audio_out(base64.b64decode(data))

        elif t in ("response.done", "response.cancelled"):
            response = msg.get("response") or {}
            nudge = self._is_nudge(response)
            cancelled = t == "response.cancelled" or (response.get("status") == "cancelled")
            if nudge:
                self._nudge_resp_id = None
            if self._audio_done:
                self._audio_done()
            if self.h.on_agent_speaking:
                self.h.on_agent_speaking(False, nudge=nudge, cancelled=cancelled)

        elif t == "response.function_call_arguments.done":
            if self.h.on_tool_call:
                self.h.on_tool_call(msg.get("name", ""), msg.get("call_id", ""))

        elif t == "conversation.item.input_audio_transcription.delta":
            self._user_tx(msg.get("delta", ""), False)
        elif t == "conversation.item.input_audio_transcription.completed":
            self._user_tx(msg.get("transcript", ""), True)

        elif t in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
            self._agent_tx(msg.get("delta", ""), False)
        elif t in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
            self._agent_tx(msg.get("transcript", ""), True)

    # -- helpers ------------------------------------------------------------ #

    def _is_nudge(self, response: dict | None) -> bool:
        if not response:
            return False
        if (response.get("metadata") or {}).get("tyto_purpose") == "nudge":
            return True
        return self._nudge_resp_id is not None and response.get("id") == self._nudge_resp_id

    def _send(self, obj: dict) -> None:
        if not self._loop or not self._ws:
            return
        asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(obj)), self._loop)

    def _user_tx(self, text: str, final: bool) -> None:
        if self.h.on_user_transcript:
            self.h.on_user_transcript(text, final)

    def _agent_tx(self, text: str, final: bool) -> None:
        if self.h.on_agent_transcript:
            self.h.on_agent_transcript(text, final)

    def _log(self, kind: str, text: str) -> None:
        if self._on_log:
            self._on_log(kind, text)
