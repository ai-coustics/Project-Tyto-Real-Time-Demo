"""The full demo: a live voice agent that adapts to your acoustics.

Talk to an OpenAI Realtime agent while Tyto scores your mic in real time. The
agent adapts on three layers, exactly like the browser reference:

    1 Aware    - a room note is injected into its instructions
    2 Tuned    - turn-taking goes patient when the room is noisy
    3 Reactive - it interrupts itself to nudge you when one issue dominates

This wires three pieces from the package: the scorer (Tyto over aic-sdk), the
controller (the provider-agnostic decision logic), and the OpenAI Realtime
provider (the swappable voice backend). The microphone is owned here so the same
frames feed both Tyto and the agent.

Run:
    uv pip install -e ".[agent]"
    export AIC_SDK_LICENSE=...        # https://developers.ai-coustics.com
    export OPENAI_API_KEY=...         # https://platform.openai.com/api-keys
    uv run examples/voice_agent.py

Use headphones. There is no echo cancellation server-side, so on speakers the
agent would hear itself.
"""

from __future__ import annotations

import os
import sys

from tyto_voice.env import load_env
from tyto_voice.audio import SounddeviceSink
from tyto_voice.controller import CHECK_AUDIO_QUALITY_TOOL, TytoController
from tyto_voice.decision import VAD_PROFILES
from tyto_voice.openai_realtime import SAMPLE_RATE, OpenAIRealtimeProvider
from tyto_voice.prompts import BASE_INSTRUCTIONS
from tyto_voice.provider import Handlers
from tyto_voice.scorer import LiveTytoScorer


def make_logger():
    def log(kind: str, text: str) -> None:
        print(f"  [{kind}] {text}"[:110])

    return log


def make_updater():
    def update(state: dict) -> None:
        if "transcript" in state:
            tx = state["transcript"]
            if tx["final"] and tx["text"]:
                print(f"  {tx['who']}: {tx['text']}")
        elif "nudge" in state:
            n = state["nudge"]
            print(f"  >> NUDGE ({n['label']} {n['value']:0.2f}): {n['text']}")

    return update


def main() -> None:
    load_env()
    license_key = os.environ.get("AIC_SDK_LICENSE")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not license_key or not openai_key:
        sys.exit("Set AIC_SDK_LICENSE and OPENAI_API_KEY first.")

    log = make_logger()
    handlers = Handlers()  # filled in once the controller exists

    # The sink plays the agent locally and reports when it is audible.
    sink = SounddeviceSink(on_playing=lambda playing: controller.on_agent_audio(playing))
    provider = OpenAIRealtimeProvider(
        handlers,
        api_key=openai_key,
        instructions=BASE_INSTRUCTIONS,
        audio_out=sink.write,
        audio_done=sink.notify_done,
        audio_flush=sink.flush,
        turn_detection=VAD_PROFILES["eager"],
        tools=[CHECK_AUDIO_QUALITY_TOOL],
        on_log=log,
    )

    # Score at the agent's sample rate so one mic stream feeds both.
    scorer = LiveTytoScorer(
        license_key,
        sample_rate=SAMPLE_RATE,
        on_state=lambda state, text: log(f"tyto.{state}", text),
    )
    controller = TytoController(provider, scorer, on_update=make_updater(), on_log=log)
    scorer.on_scores = controller.on_scores  # route smoothed scores into the layers

    handlers.on_ready = controller.on_ready
    handlers.on_agent_speaking = controller.on_agent_speaking
    handlers.on_agent_audio = controller.on_agent_audio
    handlers.on_user_transcript = controller.on_user_transcript
    handlers.on_agent_transcript = controller.on_agent_transcript
    handlers.on_tool_call = controller.on_tool_call

    scorer.start()
    sink.start()
    provider.connect()
    controller.set_connected(True)

    import sounddevice as sd

    def mic_cb(indata, _frames, _time, _status):
        mono = indata[:, 0]
        scorer.feed(mono)
        provider.send_audio(mono)

    print("Connecting to the agent. Put your headphones on and say hello...\n")
    try:
        with sd.InputStream(
            samplerate=scorer.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=scorer.num_frames,
            callback=mic_cb,
        ):
            provider.closed.wait()
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_connected(False)
        scorer.stop()
        sink.stop()
        provider.disconnect()
        print("\nStopped.")


if __name__ == "__main__":
    main()
