"""Tests for the scoring contract and decision layer.

Pure Python, no SDK or audio hardware required. These pin the behavior that must
stay identical across every branch of the demo.
"""

from tyto_voice.decision import (
    COMPOSITE_CLEAR,
    NUDGE_THRESHOLD_DEFAULT,
    EnvMonitor,
    Scores,
    pick_vad_profile,
    room_state_summary,
    strongest_cause,
)


def make(**overrides) -> Scores:
    base = dict(
        risk_score=0.0,
        noise=0.0,
        speaker_reverb=0.0,
        speaker_loudness=0.0,
        interfering_speech=0.0,
        media_speech=0.0,
        packet_loss=0.0,
    )
    base.update(overrides)
    return Scores(**base)


# -- scoring contract ------------------------------------------------------- #


def test_ema_no_history_returns_self():
    s = make(risk_score=0.8)
    assert s.ema(None) is s


def test_ema_blends_with_alpha_half():
    prev = make(risk_score=0.2, noise=0.2)
    cur = make(risk_score=0.8, noise=0.4)
    blended = cur.ema(prev, alpha=0.5)
    assert abs(blended.risk_score - 0.5) < 1e-9
    assert abs(blended.noise - 0.3) < 1e-9


# -- strongest_cause -------------------------------------------------------- #


def test_no_cause_when_clean():
    assert strongest_cause(make(risk_score=0.1, noise=0.1)) is None


def test_loudness_and_reverb_never_named():
    # Both are informational only, so even when high they are never the cause.
    assert strongest_cause(make(speaker_loudness=0.9, speaker_reverb=0.9)) is None


def test_most_exceeded_dimension_wins():
    # media_speech is further past its red cutoff (0.45) than noise is past 0.45.
    cause = strongest_cause(make(noise=0.5, media_speech=0.9))
    assert cause is not None and cause["key"] == "media_speech"


def test_below_min_explanation_value_is_ignored():
    # packet_loss past its 0.15 cutoff but under MIN_EXPLANATION_VALUE (0.30).
    assert strongest_cause(make(packet_loss=0.25)) is None


# -- room note (Aware) ------------------------------------------------------ #


def test_room_note_empty_when_clean():
    assert room_state_summary(make(risk_score=0.2)) == ""


def test_room_note_mentions_cause_and_severity():
    note = room_state_summary(make(risk_score=0.7, media_speech=0.8))
    assert "Audio note:" in note and "degraded" in note and "TV or radio" in note


# -- turn-taking (Tuned) ---------------------------------------------------- #


def test_vad_eager_when_quiet():
    assert pick_vad_profile(make(noise=0.1)) == "eager"


def test_vad_patient_when_noisy():
    assert pick_vad_profile(make(noise=0.6)) == "patient"


# -- nudge (Reactive) ------------------------------------------------------- #


def test_no_nudge_below_clear_band():
    m = EnvMonitor(min_persist=1, threshold=NUDGE_THRESHOLD_DEFAULT)
    assert m.evaluate(make(risk_score=COMPOSITE_CLEAR - 0.01, media_speech=0.9)) is None


def test_no_nudge_without_a_cause():
    # High risk but no dominant cause: a red score alone never fires.
    m = EnvMonitor(min_persist=1)
    assert m.evaluate(make(risk_score=0.9)) is None


def test_nudge_fires_with_risk_and_cause():
    m = EnvMonitor(min_persist=1, threshold=0.50)
    nudge = m.evaluate(make(risk_score=0.7, media_speech=0.8))
    assert nudge is not None and nudge.key == "media_speech"


def test_min_persist_one_fires_every_window():
    # A streak of one is already a full streak, so it re-fires each window.
    m = EnvMonitor(min_persist=1, threshold=0.50)
    bad = make(risk_score=0.7, media_speech=0.8)
    assert m.evaluate(bad) is not None
    assert m.evaluate(bad) is not None


def test_persistence_and_rearm_with_min_persist_two():
    m = EnvMonitor(min_persist=2, threshold=0.50)
    bad = make(risk_score=0.7, media_speech=0.8)
    assert m.evaluate(bad) is None  # window 1: not yet persistent
    assert m.evaluate(bad) is not None  # window 2: fires, then re-arms
    assert m.evaluate(bad) is None  # window 3: streak rebuilding
    assert m.evaluate(bad) is not None  # window 4: fires again


def test_nudge_gated_by_threshold():
    m = EnvMonitor(min_persist=1, threshold=0.60)
    # Cause present and over clear band, but risk under the (raised) gate.
    assert m.evaluate(make(risk_score=0.5, media_speech=0.8)) is None
