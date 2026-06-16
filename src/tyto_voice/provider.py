"""The voice provider seam.

Every backend-specific detail lives behind this interface, exactly like the
``OpenAIRealtimeProvider`` class in the browser reference. The Tyto control
layers (decision + controller) talk to a voice agent only through these methods,
so adding another backend means writing one more subclass with the same shape.

Commands (app -> provider):
    connect()                        open the session and start streaming
    disconnect()
    set_instructions(text)           Layer 1 - Aware
    set_turn_detection(td | None)    Layer 2 - Tuned (and the listen gate)
    set_mic_enabled(on)              stop/forward mic audio to the agent
    interrupt(clear_input=False)     cancel in-flight agent output
    nudge(text)                      Layer 3 - one spoken line
    request_response()               let the agent speak first (opening greeting)
    send_tool_result(call_id, out)   answer a tool call, then let the agent reply

Events (provider -> app) are delivered through a Handlers object, all optional:
    on_ready()
    on_agent_speaking(active, nudge=False, cancelled=False)
    on_agent_audio(playing)
    on_user_transcript(text, final)
    on_agent_transcript(text, final)
    on_tool_call(name, call_id)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class Handlers:
    on_ready: Callable[[], None] | None = None
    on_agent_speaking: Callable[..., None] | None = None
    on_agent_audio: Callable[[bool], None] | None = None
    on_user_transcript: Callable[[str, bool], None] | None = None
    on_agent_transcript: Callable[[str, bool], None] | None = None
    on_tool_call: Callable[[str, str], None] | None = None


class VoiceProvider(ABC):
    def __init__(self, handlers: Handlers):
        self.h = handlers

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_instructions(self, text: str) -> None: ...

    @abstractmethod
    def set_turn_detection(self, turn_detection: dict | None) -> None: ...

    @abstractmethod
    def set_mic_enabled(self, on: bool) -> None: ...

    @abstractmethod
    def interrupt(self, clear_input: bool = False) -> None: ...

    @abstractmethod
    def nudge(self, text: str) -> None: ...

    @abstractmethod
    def request_response(self) -> None: ...

    @abstractmethod
    def send_tool_result(self, call_id: str, output: dict) -> None: ...
