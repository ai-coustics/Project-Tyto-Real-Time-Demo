# AGENTS.md

Context for AI coding assistants (and humans) working in this repo. Read this
before making changes. It follows the [agents.md](https://agents.md) convention
and is also useful as Claude Code / Cursor project context.

## What this repo is

The Python reference of the Tyto acoustics-aware voice-agent demo. A live voice
agent talks to the user while the ai-coustics **Tyto** model scores the user's
microphone in real time, and the agent adapts on three layers (Aware, Tuned,
Reactive). The canonical browser reference is [index.html](index.html); this
branch reproduces the same behavior server-side in Python. Keep the two
comparable.

Tyto returns, per fixed 5 second window: a **risk_score** (0..1, higher is
worse) and six dimensions (`noise`, `speaker_reverb`, `speaker_loudness`,
`interfering_speech`, `media_speech`, `packet_loss`).

## Architecture and data flow

```
 mic ──> LiveTytoScorer.feed() ──(aic-sdk Collector)──┐
                                                       │ every ~2s
                                          Analyzer.analyze_buffered()
                                                       │  (smoothed, EMA 0.5)
                                                       v
 mic ──> provider.send_audio() ──> agent      TytoController.on_scores()
            (OpenAI Realtime)                          │
                ^   │ events                           ├─ Layer 1 Aware:    set_instructions(BASE + room note)
                │   v                                  ├─ Layer 2 Tuned:    set_turn_detection(eager | patient)
            VoiceProvider <───── commands ────────────-┴─ Layer 3 Reactive: interrupt() + nudge()
```

- **scorer.py** owns the SDK analyzer and the audio buffering. It does not own
  the mic; callers push audio via `feed()`.
- **controller.py** is the brain. It is provider-agnostic and holds the
  mute/nudge state machine. It runs on two threads (scores arrive on the scorer
  thread, provider events on the transport thread), guarded by one re-entrant
  lock.
- **provider.py** is the seam. **openai_realtime.py** is the only file that
  knows about a specific backend. Audio playback is delegated to callbacks
  (`audio_out` / `audio_done` / `audio_flush`), so the same provider drives a
  local speaker or a browser.
- **audio.py** is `SounddeviceSink`, the local-speaker player for the terminal
  agent. It also owns the "is the agent audible" signal (`on_agent_audio`).
- **decision.py** is the pure scoring contract and decision functions, shared
  and identical across branches.

### Frontends

- **examples/score_mic.py** - terminal mic scorer (no agent).
- **examples/voice_agent.py** - terminal agent; uses `SounddeviceSink`.
- **examples/web/** - the browser UI. `server.py` (aiohttp) is the whole brain
  per tab; the browser is a thin client. `index.html` is generated from the root
  reference (CSS + markup reused, BYOK gate removed); `app.js` is the transport
  (mic capture, agent playback, render). Audio is relayed browser <-> backend
  <-> OpenAI; keys stay in the server env. The player (browser) owns
  `on_agent_audio`, reported back over the socket.

### Who owns "agent audible" (on_agent_audio)

The component that plays the audio reports it: `SounddeviceSink` in the terminal,
the browser in the web demo. The provider never calls `on_agent_audio`; it only
reports generation lifecycle via `on_agent_speaking`. Keep this split when adding
a backend.

## The provider seam: adding a new voice backend

This is the main extension point. To add ElevenLabs, LiveKit, a cascaded
pipeline, etc., write one subclass of `VoiceProvider` (see
[src/tyto_voice/provider.py](src/tyto_voice/provider.py)) and nothing else changes.

1. Implement the commands: `connect`, `disconnect`, `set_instructions` (Aware),
   `set_turn_detection` (Tuned and the listen gate), `set_mic_enabled`,
   `interrupt`, `nudge` (Reactive), `request_response`, `send_tool_result`.
2. Drive the controller from the backend's events through a `Handlers` object:
   `on_ready`, `on_agent_speaking(active, nudge, cancelled)`,
   `on_user_transcript`, `on_agent_transcript`, `on_tool_call`. Call
   `controller.on_agent_audio(playing)` from whatever plays the audio, not the
   provider (see below).
3. Wire it like [examples/voice_agent.py](examples/voice_agent.py) (terminal) or
   [examples/web/server.py](examples/web/server.py) (browser).

If a backend cannot support a layer (for example it manages turn-taking itself),
do not fake it. Implement what you can and document the gap in the README.

## Invariants to preserve

These keep the demo correct and comparable across branches. Do not change them
casually.

- **Tuned constants are ground truth.** Window 5 s, hop ~2 s, EMA alpha 0.5,
  the per-dimension thresholds, the nudge bands. They live in `decision.py` and
  match the browser byte for byte. If you change one, change it in every branch
  and say why.
- **Warm-up gate.** Never score until a full fresh 5 s window has been buffered
  since the last reset. On resume after the agent speaks, reset the analyzer and
  re-warm. Stale audio must never skew a reading.
- **Mute and pause while the agent speaks.** The mic is muted (no frames sent to
  the agent) and scoring is paused while the agent talks; both resume after.
- **A nudge always needs a cause.** A high risk_score alone never nudges; one
  dimension must dominate (`strongest_cause`).
- **`speaker_loudness` and `speaker_reverb` are informational only.** Never
  colored as a problem, never named as a cause, never the reason for a nudge.

## aic-sdk quick reference (verified against aic-sdk 2.4.0)

```python
import aic_sdk as aic
path = aic.Model.download("tyto-l-16khz", "./models")     # CDN, cached
model = aic.Model.from_file(path)
# streaming (live):
collector, analyzer = aic.analyzer_pair(model, license_key)
config = aic.ProcessorConfig.optimal(model, sample_rate=24000, num_channels=1)
collector.initialize(config)
collector.buffer(np.zeros((1, config.num_frames), dtype=np.float32))  # exact num_frames
result = analyzer.analyze_buffered()   # rolling window; silence-padded if short
analyzer.reset()                       # clears analyzer AND collector
# result fields: risk_score, noise, speaker_reverb, speaker_loudness,
#                interfering_speech, media_speech, packet_loss
```

This demo is real-time only. (`aic.FileAnalyzer(model, key).analyze(...)` exists
for offline batch scoring, but it is intentionally not part of this demo.)

`Model.download` and `ProcessorConfig.optimal(..., sample_rate=24000)` are
confirmed to work; only the licensed analysis steps need a real key.

## OpenAI Realtime event mapping (WebSocket, server-side)

| Concept | Outgoing / incoming |
| --- | --- |
| configure session | `session.update` (instructions, audio.input.turn_detection, transcription, audio.output.voice, tools) |
| send mic | `input_audio_buffer.append` (base64 PCM16, 24 kHz) |
| agent audio | `response.output_audio.delta` (also legacy `response.audio.delta`) |
| agent text | `response.output_audio_transcript.delta` / `.done` |
| user text | `conversation.item.input_audio_transcription.delta` / `.completed` |
| tool call | `response.function_call_arguments.done` |
| nudge | `response.create` with `metadata.tyto_purpose = "nudge"` |
| interrupt | `response.cancel` (and clear the local playback buffer) |

## Running and verifying

```bash
uv pip install -e ".[dev]"
uv run pytest -q                 # 24 tests: decision layer + controller + scorer
```

The unit tests need no SDK, key, or hardware. The end-to-end audio path needs an
ai-coustics key, an OpenAI key, a mic, and headphones.

## Conventions

- Style: KISS and DRY, clean and minimal. Match the surrounding code.
- No em dashes in prose or comments. Use hyphens or rewrite.
- Secure context: the mic needs HTTPS or localhost in any browser-facing
  extension of this.
- Verify provider APIs against current docs; they change often. Do not trust
  training memory for method or event names.

## Related ai-coustics tooling

If your assistant has the ai-coustics MCP skills available, these help when
working with audio in this repo: `tyto` (score audio with the observability
model), `enhance` (run SDK enhancement), `transcribe` (STT across providers),
and `before-after` (comparison pages). They are separate tools from this demo
but share the same model family and license.
