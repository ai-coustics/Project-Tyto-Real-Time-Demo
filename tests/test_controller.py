"""Controller wiring tests with a fake provider and scorer (no hardware).

These check the provider-agnostic glue: that scores drive the three layers and
that the mute/nudge/resume state machine gates scoring correctly.
"""

from tyto_voice.controller import TytoController
from tyto_voice.provider import VoiceProvider


class FakeProvider(VoiceProvider):
    def __init__(self):
        super().__init__(handlers=None)
        self.calls = []

    def connect(self): ...
    def disconnect(self): ...
    def set_instructions(self, text): self.calls.append(("instructions", text))
    def set_turn_detection(self, td): self.calls.append(("turn_detection", td))
    def set_mic_enabled(self, on): self.calls.append(("mic", on))
    def interrupt(self, clear_input=False): self.calls.append(("interrupt", clear_input))
    def nudge(self, text): self.calls.append(("nudge", text))
    def request_response(self): self.calls.append(("request_response", None))
    def send_tool_result(self, call_id, output): self.calls.append(("tool_result", call_id))

    def kinds(self):
        return [c[0] for c in self.calls]


class FakeScorer:
    def __init__(self):
        self.scoring = True
        self.events = []

    def pause(self):
        self.scoring = False
        self.events.append("pause")

    def resume(self):
        self.scoring = True
        self.events.append("resume")


def make(**overrides):
    from tyto_voice.decision import Scores

    base = dict(
        risk_score=0.0, noise=0.0, speaker_reverb=0.0, speaker_loudness=0.0,
        interfering_speech=0.0, media_speech=0.0, packet_loss=0.0,
    )
    base.update(overrides)
    return Scores(**base)


def build():
    provider = FakeProvider()
    scorer = FakeScorer()
    controller = TytoController(provider, scorer)
    controller.set_connected(True)
    return provider, scorer, controller


def test_aware_pushes_room_note_then_clears():
    provider, _, controller = build()
    controller.on_scores(make(risk_score=0.7, media_speech=0.8))
    assert any(k == "instructions" and "TV or radio" in v for k, v in provider.calls)
    provider.calls.clear()
    controller.on_scores(make(risk_score=0.1))  # clean again
    assert any(k == "instructions" for k in provider.kinds())  # instructions reset


def test_tuned_swaps_turn_detection_on_noise():
    provider, _, controller = build()
    # Noisy room but risk below the clear band, so Tuned acts without a nudge.
    controller.on_scores(make(risk_score=0.2, noise=0.6))  # noisy -> patient
    tds = [v for k, v in provider.calls if k == "turn_detection"]
    assert tds and tds[-1]["type"] == "server_vad"


def test_reactive_nudge_mutes_interrupts_and_dispatches():
    provider, scorer, controller = build()
    controller.on_scores(make(risk_score=0.7, media_speech=0.8))
    kinds = provider.kinds()
    assert "interrupt" in kinds and "nudge" in kinds
    assert controller.awaiting_nudge is True
    assert ("mic", False) in provider.calls  # mic muted for the nudge


def test_nudge_lifecycle_resumes_listening():
    provider, scorer, controller = build()
    controller.on_scores(make(risk_score=0.7, media_speech=0.8))
    assert controller.awaiting_nudge
    # Agent starts speaking the nudge, then finishes with no audio left to play.
    controller.on_agent_speaking(True, nudge=True)
    assert controller.nudge_active
    controller.on_agent_speaking(False, nudge=True, cancelled=False)
    # Back to listening and scoring gated back on.
    assert controller.listening is True
    assert scorer.scoring is True


def test_scoring_pauses_while_agent_speaks():
    _, scorer, controller = build()
    controller.on_agent_speaking(True)  # ordinary reply, not a nudge
    assert scorer.scoring is False
    controller.on_agent_speaking(False)
    controller.on_agent_audio(False)
    assert scorer.scoring is True


def test_audio_quality_snapshot_summarizes_top_issue():
    _, _, controller = build()
    controller.on_scores(make(risk_score=0.7, media_speech=0.8))
    snap = controller.audio_quality_snapshot()
    assert snap["verdict"] == "degraded"
    assert snap["top_issue"]["key"] == "media_speech"
