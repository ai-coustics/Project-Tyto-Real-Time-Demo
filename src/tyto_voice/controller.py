"""The Tyto control layer, provider-agnostic.

This is the server-side port of the ``onScores`` handler and the mute/nudge
state machine in the browser reference. It takes a live ``Scores`` stream from
the scorer and a ``VoiceProvider``, and drives the three adaptation layers:

    Layer 1 - Aware:    swap a one-sentence room note into the instructions.
    Layer 2 - Tuned:    eager vs patient turn-taking based on background noise.
    Layer 3 - Reactive: interrupt the agent with one spoken nudge, then resume.

It also answers the ``check_audio_quality`` tool, and gates scoring so Tyto only
reads the user's audio while the user is actually speaking (mic muted, scoring
paused while the agent talks).

All state is guarded by a single re-entrant lock because scores arrive on the
scorer thread while provider events arrive on the transport thread.
"""

from __future__ import annotations

import threading
from typing import Callable

from .decision import (
    COMPOSITE_TH,
    ENV_KEYS,
    LABELS,
    MIN_EXPLANATION_VALUE,
    NO_POLARITY,
    NUDGE_THRESHOLD_DEFAULT,
    NUDGE_THRESHOLD_MAX,
    NUDGE_THRESHOLD_MIN,
    THRESHOLDS,
    VAD_PROFILES,
    EnvMonitor,
    Nudge,
    Scores,
    pick_vad_profile,
    room_state_summary,
)
from .prompts import BASE_INSTRUCTIONS
from .provider import VoiceProvider

# The tool the agent calls when the user asks "how do I sound?".
CHECK_AUDIO_QUALITY_TOOL = {
    "type": "function",
    "name": "check_audio_quality",
    "description": (
        "Get the current real-time audio quality of the user's mic input. Returns a "
        "summary, verdict, the Tyto Score, and the top current issue. Call this whenever "
        "the user asks if you can hear them, how their audio sounds, or about their "
        "connection/environment."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class TytoController:
    def __init__(
        self,
        provider: VoiceProvider,
        scorer,
        *,
        nudge_threshold: float = NUDGE_THRESHOLD_DEFAULT,
        on_update: Callable[[dict], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self.provider = provider
        self.scorer = scorer
        self.on_update = on_update
        self.on_log = on_log

        self._lock = threading.RLock()
        self._monitor = EnvMonitor(min_persist=1, threshold=nudge_threshold)

        self.connected = False
        self.listening = True
        self.agent_speaking = False
        self.agent_audio_playing = False
        self.awaiting_nudge = False
        self.nudge_active = False
        self.nudge_playback_pending = False

        self._last_room = ""
        self._last_vad = "eager"
        self._last_scores: Scores | None = None
        self._last_risk: float | None = None

    # -- session ------------------------------------------------------------ #

    def set_connected(self, value: bool) -> None:
        self.connected = value
        self._sync_scoring_gate()

    @property
    def nudge_threshold(self) -> float:
        return self._monitor.threshold

    @nudge_threshold.setter
    def nudge_threshold(self, value: float) -> None:
        self._monitor.threshold = min(NUDGE_THRESHOLD_MAX, max(NUDGE_THRESHOLD_MIN, value))

    # -- Tyto scores in (scorer thread) ------------------------------------- #

    def on_scores(self, scores: Scores) -> None:
        with self._lock:
            self._last_scores = scores
            self._last_risk = scores.risk_score

            room = room_state_summary(scores)
            vad = pick_vad_profile(scores)

            if room != self._last_room:  # Layer 1 - Aware
                self._last_room = room
                self.provider.set_instructions(
                    BASE_INSTRUCTIONS + (f"\n\n{room}" if room else "")
                )
                self._log("tyto.aware.update", room or "(clean)")

            if vad != self._last_vad:  # Layer 2 - Tuned
                self._last_vad = vad
                if self.listening:
                    self.provider.set_turn_detection(VAD_PROFILES[vad])
                self._log("tyto.vad.swap", vad)

            directive = self._monitor.evaluate(scores)  # Layer 3 - Reactive
            if directive:
                self._fire_nudge(directive)

            self._push_update(room, vad, scores)

    # -- provider events (transport thread) --------------------------------- #

    def on_ready(self) -> None:
        # Once the session is configured, have the agent open the conversation.
        self.provider.request_response()

    def on_agent_speaking(self, active: bool, nudge: bool = False, cancelled: bool = False) -> None:
        with self._lock:
            self.agent_speaking = active
            if active:
                if nudge and self.awaiting_nudge:
                    self._mute_mic_for_agent()
                    self.nudge_active = True
                    self.awaiting_nudge = False
            elif nudge or self.nudge_active:
                # Generation done but audio may still be playing: stay muted
                # until playback stops.
                if cancelled or not self.agent_audio_playing:
                    self._resume_after_nudge()
                else:
                    self.nudge_playback_pending = True
            elif not self.agent_audio_playing:
                self._maybe_unmute_mic()
            self._sync_scoring_gate()

    def on_agent_audio(self, playing: bool) -> None:
        with self._lock:
            self.agent_audio_playing = playing
            if not playing:
                if self.nudge_playback_pending:
                    self._resume_after_nudge()
                else:
                    self._maybe_unmute_mic()
            self._sync_scoring_gate()

    def on_tool_call(self, name: str, call_id: str) -> None:
        if name != "check_audio_quality":
            return
        result = self.audio_quality_snapshot()
        self._log("tool.check_audio_quality", result.get("summary", ""))
        self.provider.send_tool_result(call_id, result)

    def on_user_transcript(self, text: str, final: bool) -> None:
        self._push_transcript("user", text, final)

    def on_agent_transcript(self, text: str, final: bool) -> None:
        self._push_transcript("agent", text, final)

    # -- Layer 3 internals (mirror the browser state machine) --------------- #

    def _fire_nudge(self, directive: Nudge) -> None:
        if self.awaiting_nudge or self.nudge_active or not self.listening:
            return
        self._log("tyto.nudge.trip", f"{directive.label}={directive.value:.2f}")
        # Cut the user's audio and the agent's output, then nudge immediately.
        self._set_mic_enabled(False)
        self._set_listening(False)
        self._sync_scoring_gate()
        self._interrupt_agent_speech(clear_input=True)
        self.awaiting_nudge = True
        self.provider.nudge(directive.text)
        self._log("tyto.nudge.dispatch", directive.text)
        if self.on_update:
            self.on_update({"nudge": {"label": directive.label, "value": directive.value, "text": directive.text}})

    def _resume_after_nudge(self) -> None:
        self.nudge_active = False
        self.awaiting_nudge = False
        self.nudge_playback_pending = False
        self._set_listening(True)
        self._maybe_unmute_mic()
        self._sync_scoring_gate()
        self._log("tyto.input.resumed", "")

    def _interrupt_agent_speech(self, clear_input: bool = False) -> None:
        self.awaiting_nudge = False
        self.nudge_active = False
        self.nudge_playback_pending = False
        self.provider.interrupt(clear_input=clear_input)

    def _mute_mic_for_agent(self) -> None:
        self._set_mic_enabled(False)
        self._sync_scoring_gate()

    def _maybe_unmute_mic(self) -> None:
        if self.agent_speaking or self.agent_audio_playing:
            return
        if self.awaiting_nudge or self.nudge_active or self.nudge_playback_pending or not self.listening:
            return
        self._set_mic_enabled(True)

    def _set_mic_enabled(self, on: bool) -> None:
        self.provider.set_mic_enabled(on)

    def _set_listening(self, on: bool) -> None:
        if self.listening == on:
            return
        self.listening = on
        self.provider.set_turn_detection(VAD_PROFILES.get(self._last_vad) if on else None)

    def _sync_scoring_gate(self) -> None:
        should_score = (
            self.connected and self.listening and not self.agent_speaking and not self.agent_audio_playing
        )
        if should_score and not self.scorer.scoring:
            self.scorer.resume()
        elif not should_score and self.scorer.scoring:
            self.scorer.pause()

    # -- check_audio_quality tool ------------------------------------------- #

    def _strongest_env(self, scores: Scores):
        best = None
        for key in ENV_KEYS:
            if key in NO_POLARITY:
                continue
            value = getattr(scores, key)
            if value < MIN_EXPLANATION_VALUE:
                continue
            low = THRESHOLDS.get(key, (0.30, 0.60))[0]
            severity = max(0.0, value - low) / max(1 - low, 1e-6)
            if best is None or severity > best["severity"]:
                best = {"key": key, "value": value, "severity": severity}
        return best

    def audio_quality_snapshot(self) -> dict:
        scores = self._last_scores
        if scores is None:
            return {"status": "warming up, ask again in a few seconds"}
        risk = scores.risk_score
        verdict = "degraded" if risk >= COMPOSITE_TH[1] else "marginal" if risk >= COMPOSITE_TH[0] else "good"
        result: dict = {"tyto_score": round(risk, 2), "verdict": verdict}
        for key in ENV_KEYS:
            result[LABELS[key].lower().replace(" ", "_")] = round(getattr(scores, key), 2)
        top = self._strongest_env(scores)
        if top and top["severity"] > 0:
            result["top_issue"] = {
                "key": top["key"],
                "label": LABELS[top["key"]],
                "value": round(top["value"], 2),
                "direction": "high",
            }
            result["summary"] = (
                f"Audio is {verdict}. Biggest issue: {LABELS[top['key']]} is high at {top['value']:.2f}."
            )
        else:
            result["summary"] = f"Audio is {verdict}."
        return result

    # -- UI plumbing -------------------------------------------------------- #

    def _push_update(self, room: str, vad: str, scores: Scores) -> None:
        if self.on_update:
            self.on_update({"room": room, "vad": vad, "scores": scores})

    def _push_transcript(self, who: str, text: str, final: bool) -> None:
        if self.on_update:
            self.on_update({"transcript": {"who": who, "text": text, "final": final}})

    def _log(self, kind: str, text: str) -> None:
        if self.on_log:
            self.on_log(kind, text if isinstance(text, str) else str(text))
