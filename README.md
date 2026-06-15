# Tyto · Acoustics-aware voice agent (live demo)

A self-contained browser demo of [**Tyto**](https://docs.ai-coustics.com),
audio insight for voice AI. You talk to a live voice agent; in parallel **Tyto
scores your microphone in real time** and the agent adapts to your acoustics:
slowing down on bad audio, retuning turn-taking in a noisy room, and nudging you
when something is fixable ("Could you turn the TV down?").

This is a single self-contained `index.html`. There is **no backend and no build
step**: each visitor enters their own OpenAI and ai-coustics keys, which stay in
their browser and go straight to those APIs over HTTPS.

## Try it

Open the page, click **Enter keys**, paste your own keys, then click the mic and
start talking.

- OpenAI API key: <https://platform.openai.com/api-keys>
- ai-coustics SDK license key: <https://developers.ai-coustics.com>

Your keys are stored only in your browser (`localStorage`), used directly with
OpenAI and ai-coustics, and never sent anywhere else. Clear them anytime from the
**Enter keys** button.

## What it does

Every couple of seconds Tyto scores your mic, and the agent reacts on three
levels:

1. **Aware:** a one-sentence "room note" is added to the agent's instructions, so
   it factors your acoustics into every reply.
2. **Tuned:** in a noisy room it switches to a more patient turn-detection mode so
   it stops triggering on background sound.
3. **Reactive:** when the risk is high and one cause dominates, the agent
   interrupts itself with a short spoken nudge. The **Nudge sensitivity** slider
   under the score bar tunes how eagerly that fires.

The agent also greets you with a fun prompt to get you talking, and you can ask
"how do I sound?" any time.

## Host it yourself

It is one file, so any static host works. For GitHub Pages:

1. Put `index.html` at the root of a repo (or a `docs/` folder).
2. **Settings → Pages → Deploy from a branch**, pick the branch and folder, save.
3. Share the `https://<user>.github.io/<repo>/` URL.

The page must be served over HTTPS (GitHub Pages is), because the microphone and
the in-browser WASM worker need a secure origin. Opening the file with `file://`
will not work.

## Links

- **Tyto docs:** <https://docs.ai-coustics.com>
- **ai-coustics:** <https://ai-coustics.com>
- **Get an SDK key:** <https://developers.ai-coustics.com>
