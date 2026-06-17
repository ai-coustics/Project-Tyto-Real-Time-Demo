# Tyto voice agent, Python reference

A Python port of the [Tyto](https://docs.ai-coustics.com) acoustics-aware
voice-agent demo. You talk to a live voice agent; in parallel **Tyto scores your
microphone in real time** with the ai-coustics Python SDK, and the agent adapts
to your acoustics on three layers: it stays aware of your room, retunes
turn-taking when it gets noisy, and nudges you when something is fixable ("Could
you turn the TV down?").

This branch is the server-side sibling of the browser reference in
[index.html](index.html) (OpenAI Realtime over WebRTC, fully client-side). The
Tyto scoring contract and the tuned constants are identical to that reference so
behavior is comparable across stacks.

## What is in here

Three things you can run:

| Demo | What it shows | Needs |
| --- | --- | --- |
| [examples/web/server.py](examples/web/server.py) | The full demo with the browser UI, same as the reference. Tyto scoring, the agent, and the keys all run on the Python backend; the browser is a thin client. | ai-coustics key + OpenAI key |
| [examples/pipecat/server.py](examples/pipecat/server.py) | The same browser demo, but the voice backend is a [Pipecat](https://pipecat.ai) pipeline (OpenAI Realtime speech-to-speech) reached over WebRTC instead of a hand-written WebSocket relay. Same UI, same three layers. | ai-coustics key + OpenAI key |
| [examples/score_mic.py](examples/score_mic.py) | Live Tyto scoring of your mic in the terminal, with the three layer decisions printed. No agent. | ai-coustics key + a mic |
| [examples/voice_agent.py](examples/voice_agent.py) | The full agent in the terminal (no UI), for headless or scripting use. | ai-coustics key + OpenAI key + headphones |

The web demo is the one to start with: it is the visual UI from the browser
reference, but every key stays on the server and Tyto runs in Python.

The reusable library lives in [src/tyto_voice](src/tyto_voice). It is small and
split by job, mirroring the commented sections of the browser reference:

- `decision.py` - the scoring contract (`Scores`, tuned constants) and the
  decision layer (room note, turn-taking profile, nudge monitor). Pure Python,
  no dependencies, fully unit tested. This is the part that is identical across
  every branch.
- `prompts.py` - the agent instructions and the Tyto background.
- `scorer.py` - `LiveTytoScorer`, real-time scoring over the aic-sdk streaming
  analyzer, with the warm-up gate and pause/resume that match the browser worker.
- `provider.py` - `VoiceProvider`, the one interface every voice backend hides
  behind. Swap backends by writing one subclass.
- `controller.py` - `TytoController`, the provider-agnostic glue that turns a
  score stream into the three adaptations and answers the `check_audio_quality`
  tool.
- `openai_realtime.py` - `OpenAIRealtimeProvider`, the OpenAI Realtime WebSocket
  backend. Audio playback is delegated to a sink so the same provider drives a
  local speaker or a browser.
- `pipecat_provider.py` - `PipecatRealtimeProvider`, the same OpenAI Realtime
  backend but driven through a [Pipecat](https://pipecat.ai) pipeline with a
  WebRTC transport. A second `VoiceProvider` subclass, written without touching
  the controller or decision layers - the provider seam in action.
- `audio.py` - `SounddeviceSink`, local speaker playback for the terminal agent.

## Run it

You need an ai-coustics SDK license key from
<https://developers.ai-coustics.com> (and an OpenAI key for the agent). The
model (`tyto-l-16khz`) is downloaded from the ai-coustics CDN on first run into
`./models`. Put your keys in a `.env`; everything loads it automatically.

```bash
uv venv
cp .env.example .env    # then edit AIC_SDK_LICENSE and OPENAI_API_KEY

# the full demo with the browser UI (start here)
uv pip install -e ".[web]"
uv run examples/web/server.py        # then open http://localhost:8080

# or: the same browser demo on a Pipecat pipeline (OpenAI Realtime over WebRTC)
uv pip install -e ".[pipecat]"
uv run examples/pipecat/server.py    # then open http://localhost:8080

# or: live mic scoring in the terminal, no agent
uv pip install -e .
uv run examples/score_mic.py

# or: the agent in the terminal, no UI (use headphones)
uv pip install -e ".[agent]"
uv run examples/voice_agent.py
```

Real exported environment variables take precedence over `.env`. Install extras:
plain `.` for the mic scorer, `.[web]` for the browser demo, `.[agent]` for the
terminal agent, `.[dev]` for the tests.

## How keys and secrets are handled

Keys come from environment variables (or a `.env`) and stay on the backend. This
is the opposite of the browser reference, where each visitor pastes their own
keys: here the visitor's browser never sees a key.

- `AIC_SDK_LICENSE` runs the Tyto analyzer on the backend. Audio is scored on the
  server; nothing leaves it for scoring.
- `OPENAI_API_KEY` opens the OpenAI Realtime WebSocket connection from the
  backend. The browser only exchanges mic and agent audio with your server, never
  with OpenAI, so no key (or ephemeral secret) is ever sent to the browser.

All entry points auto-load a `.env` from the project root (see
[.env.example](.env.example)); exported environment variables override it.

## Which of the three Tyto layers are supported

All three, server-side, with the same tuned thresholds as the browser:

1. **Aware** - a one-sentence room note is swapped into the agent instructions
   via `session.update` whenever the dominant cause changes. Fully supported.
2. **Tuned** - turn-taking switches between an eager `semantic_vad` profile and a
   patient `server_vad` profile (longer end-of-speech, higher threshold) when the
   room is noisy. Fully supported: OpenAI Realtime exposes `turn_detection`
   directly, so the same profiles as the browser apply.
3. **Reactive** - when the smoothed risk crosses the threshold and one cause
   dominates, the agent interrupts itself with a single spoken nudge, then
   resumes. Fully supported via `response.cancel` plus a one-shot `response.create`.

The `check_audio_quality` tool is wired as an OpenAI function tool, so the user
can ask "how do I sound?" at any time.

### The Pipecat backend

`examples/pipecat` is the same demo on a [Pipecat](https://pipecat.ai) pipeline.
It exists to show the provider seam: the scorer, decision layer, and controller
(and their tests) are reused unchanged; only a new `VoiceProvider` subclass,
[pipecat_provider.py](src/tyto_voice/pipecat_provider.py), is added. The pipeline
per connection is `SmallWebRTCTransport.input -> TytoAudioTap (feeds the scorer)
-> OpenAIRealtimeLLMService -> SmallWebRTCTransport.output`, and the three layers
map to the same OpenAI Realtime client events as the raw provider (instructions
and turn detection via `session.update`, the nudge via `response.create`).

Because Pipecat owns the transport and the frame flow, three things differ from
the raw-websocket provider, documented rather than faked:

- Agent audio is played to the browser by the transport, so `on_agent_audio` is
  driven by `Bot{Started,Stopped}SpeakingFrame` (a server-side estimate) instead
  of browser-reported audibility.
- `check_audio_quality` is answered by a registered Pipecat function handler, so
  the controller's `on_tool_call` / `send_tool_result` path is unused here.
- The opening greeting is kicked off by `request_response` queuing an
  `LLMRunFrame` (the context aggregator emits its context on that frame, and the
  realtime service responds to it), rather than by a raw `response.create`.

The browser client ([examples/pipecat/app.js](examples/pipecat/app.js)) reuses
the websocket demo's UI verbatim and only swaps the transport: WebRTC media for
audio (so the browser's echo cancellation and jitter buffer do the work) and a
WebRTC data channel for the score / layer / transcript / log messages.

### Limitations and notes

- **Echo cancellation.** The web demo captures the mic in the browser with echo
  cancellation on, so speakers are fine. The terminal agent plays through a raw
  output device with no cancellation, so use headphones there. In both, the
  controller stops sending mic audio to the agent while the agent speaks.
- **Audio transport.** The browser reference uses WebRTC straight to OpenAI; here
  audio is PCM16 mono at 24 kHz, relayed browser to backend to OpenAI and back.
  Tyto is fed the same 24 kHz frames and resamples internally to its 16 kHz rate.
  The extra hop adds a little latency in exchange for keeping all logic and keys
  on the server.
- **Verification.** The decision layer and controller state machine are covered
  by unit tests (`pytest`), the aic-sdk calls are verified against the installed
  package (model download, config, analyzer pair), and the web server boot plus
  websocket session bridge are smoke tested. The live audio path needs your own
  keys, a mic, and a browser to exercise.

## Deploy story

The web demo is the deployable one: it is a single aiohttp process serving the
page and one websocket per visitor, with keys in its environment. Put it behind
TLS (the mic needs a secure origin, so HTTPS or localhost) and it is shareable as
a normal web app. The terminal demos are local tools.

Tyto itself runs anywhere Python runs, on CPU, with no audio leaving the host, so
it also drops into an existing server-side voice pipeline (Pipecat, LiveKit
Agents, a cascaded STT to LLM to TTS stack) by feeding the same
`LiveTytoScorer.feed` from whatever already has the user's audio.

## Develop

```bash
uv pip install -e ".[dev]"
uv run pytest -q
```

See [AGENTS.md](AGENTS.md) for an architecture map and the invariants to keep if
you extend this (it doubles as context for AI coding assistants).

## Links

- Tyto docs: <https://docs.ai-coustics.com>
- ai-coustics: <https://ai-coustics.com>
- Get an SDK key: <https://developers.ai-coustics.com>
- Python SDK: <https://github.com/ai-coustics/aic-sdk-py>
