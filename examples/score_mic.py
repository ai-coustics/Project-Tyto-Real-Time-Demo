"""Score your microphone live with Tyto and watch the three layers react.

This is the smallest possible demo: no voice agent, no API keys beyond your
ai-coustics license. It runs the exact same scoring and decision layer the full
voice agent uses, and prints what each adaptation layer would do right now:

    Aware    - the one-sentence room note the agent would be told
    Tuned    - eager or patient turn-taking
    Reactive - the spoken nudge that would fire (and why)

Run:
    uv pip install -e .
    export AIC_SDK_LICENSE=...        # from https://developers.ai-coustics.com
    uv run examples/score_mic.py

Talk, then try making it worse: play a video, turn on a fan, walk to a hard
room. Watch the bars and the layer notes move.
"""

from __future__ import annotations

import os
import sys

from tyto_voice.env import load_env
from tyto_voice.decision import (
    COMPOSITE_TH,
    ENV_KEYS,
    LABELS,
    NO_POLARITY,
    THRESHOLDS,
    EnvMonitor,
    Scores,
    pick_vad_profile,
    room_state_summary,
)
from tyto_voice.scorer import LiveTytoScorer

BAR_WIDTH = 24
CLEAR = "\033[2J\033[H"  # clear screen, cursor home


def color(key: str, value: float) -> str:
    if key in NO_POLARITY:
        return "\033[36m"  # cyan: informational
    low, high = THRESHOLDS.get(key, (0.30, 0.60))
    return "\033[32m" if value < low else "\033[33m" if value < high else "\033[31m"


def bar(value: float, width: int = BAR_WIDTH) -> str:
    filled = round(min(1.0, max(0.0, value)) * width)
    return "#" * filled + "-" * (width - filled)


def render(scores: Scores, monitor: EnvMonitor) -> None:
    risk = scores.risk_score
    verdict = "GOOD" if risk < COMPOSITE_TH[0] else "WARN" if risk < COMPOSITE_TH[1] else "BAD"
    room = room_state_summary(scores)
    vad = pick_vad_profile(scores)

    lines = [CLEAR, "Tyto live mic score   (higher = worse, Ctrl-C to stop)", ""]
    lines.append(f"  Risk {risk:0.2f} [{bar(risk)}] {verdict}")
    lines.append("")
    for key in ENV_KEYS:
        value = getattr(scores, key)
        tag = " (info)" if key in NO_POLARITY else ""
        lines.append(
            f"  {LABELS[key]:<18}{color(key, value)}{value:0.2f} [{bar(value)}]\033[0m{tag}"
        )
    lines.append("")
    lines.append(f"  1 Aware    : {room or 'room sounds clean, no extra context'}")
    lines.append(f"  2 Tuned    : {vad} turn-taking")
    directive = monitor.evaluate(scores)
    if directive:
        lines.append(f"  3 Reactive : NUDGE -> \"{directive.text}\"  ({directive.label} {directive.value:0.2f})")
    else:
        lines.append("  3 Reactive : idle")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def main() -> None:
    load_env()
    license_key = os.environ.get("AIC_SDK_LICENSE")
    if not license_key:
        sys.exit("Set AIC_SDK_LICENSE (get a key at https://developers.ai-coustics.com)")

    monitor = EnvMonitor()

    def on_state(state: str, text: str) -> None:
        if state in ("loading", "warming", "error"):
            print(f"[{state}] {text}")

    scorer = LiveTytoScorer(
        license_key,
        on_scores=lambda scores: render(scores, monitor),
        on_state=on_state,
    )
    scorer.start()
    print(f"Scoring at {scorer.sample_rate} Hz. Start talking...")
    try:
        scorer.run_from_microphone()
    except KeyboardInterrupt:
        pass
    finally:
        scorer.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
