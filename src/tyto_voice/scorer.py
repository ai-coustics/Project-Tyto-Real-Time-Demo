"""Live Tyto scoring over the ai-coustics Python SDK.

This is the server-side equivalent of the browser's ``tyto-worker.js``. It owns
an ``aic_sdk`` Collector/Analyzer pair, buffers mono audio into it, and every
``hop`` seconds reads the rolling 5 s window and emits a smoothed ``Scores``.

The two behaviors worth understanding:

- Warm-up gate: ``analyze_buffered`` pads with silence if it is called before a
  full window has been buffered, which would skew the reading. So we never score
  until ``WINDOW_SECONDS`` of real audio has been buffered since the last reset.

- Pause / resume: while the agent is speaking the mic is muted, so scoring is
  paused and incoming audio is dropped. On resume we ``reset()`` the analyzer
  (clearing stale audio) and wait for a fresh full window. This matches the
  browser worker so the two demos behave identically.

The scorer does not own the microphone. Call :meth:`feed` from your own audio
callback (the voice agent does this so it can also forward the same frames to
the model), or use :meth:`run_from_microphone` for the standalone mic demo.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from .decision import HOP_SECONDS, SCORE_EMA_ALPHA, WINDOW_SECONDS, Scores

# Model published on the ai-coustics artifact CDN. Same model family as the
# browser demo (tyto-l-16khz), so scores are comparable.
DEFAULT_MODEL = "tyto-l-16khz"

ScoresCallback = Callable[[Scores], None]
StateCallback = Callable[[str, str], None]  # (state, human_text)


class LiveTytoScorer:
    def __init__(
        self,
        license_key: str,
        *,
        model_id: str = DEFAULT_MODEL,
        models_dir: str = "./models",
        sample_rate: int | None = None,
        hop_seconds: float = HOP_SECONDS,
        ema_alpha: float = SCORE_EMA_ALPHA,
        on_scores: ScoresCallback | None = None,
        on_state: StateCallback | None = None,
    ):
        self._license_key = license_key
        self._model_id = model_id
        self._models_dir = models_dir
        self._requested_rate = sample_rate
        self._hop_seconds = hop_seconds
        self._ema_alpha = ema_alpha
        self.on_scores = on_scores  # public: settable after construction
        self._on_state = on_state

        # Audio config, filled in by start().
        self.sample_rate = 0
        self.num_frames = 0
        self.num_channels = 1

        self._collector = None
        self._analyzer = None
        self._window_samples = 0

        self._lock = threading.Lock()
        self._buffered = 0  # real samples since the last reset (warm-up gate)
        self._residual = np.empty(0, dtype=np.float32)  # leftover < one block
        self._scoring = True
        self._smoothed: Scores | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Load the model, build the analyzer, and start the scoring loop."""
        import aic_sdk as aic  # imported lazily so the package loads without the SDK

        self._emit_state("loading", "loading Tyto model...")
        model_path = aic.Model.download(self._model_id, self._models_dir)
        model = aic.Model.from_file(model_path)

        rate = self._requested_rate or model.get_optimal_sample_rate()
        config = aic.ProcessorConfig.optimal(model, sample_rate=rate, num_channels=1)
        self.sample_rate = config.sample_rate
        self.num_frames = config.num_frames
        self.num_channels = config.num_channels
        self._window_samples = round(WINDOW_SECONDS * self.sample_rate)

        self._collector, self._analyzer = aic.analyzer_pair(model, self._license_key)
        self._collector.initialize(config)

        self._emit_state("warming", "warming up - keep talking")
        self._thread = threading.Thread(target=self._loop, name="tyto-scorer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- audio in ----------------------------------------------------------- #

    def feed(self, mono: np.ndarray) -> None:
        """Buffer mono float32 audio of any length. Dropped while paused.

        The SDK requires each ``buffer`` call to be exactly ``num_frames``, so we
        accumulate a residual and emit fixed-size blocks (just like the browser
        worker turns 128-sample worklet quanta into model-sized blocks).
        """
        with self._lock:
            if not self._scoring or self._collector is None:
                return
            data = np.concatenate([self._residual, np.ascontiguousarray(mono, dtype=np.float32)])
            offset, n = 0, self.num_frames
            while len(data) - offset >= n:
                self._collector.buffer(data[offset : offset + n].reshape(1, n))
                offset += n
                self._buffered += n
            self._residual = data[offset:]

    # -- gate (mirrors the browser worker's pause/resume) ------------------- #

    def pause(self) -> None:
        """Stop scoring and drop incoming audio (agent is speaking)."""
        with self._lock:
            self._scoring = False

    def resume(self) -> None:
        """Resume scoring: clear stale audio and require a fresh full window."""
        with self._lock:
            if self._analyzer is not None:
                try:
                    self._analyzer.reset()  # clears both analyzer and collector
                except Exception:
                    pass
            self._buffered = 0
            self._residual = np.empty(0, dtype=np.float32)
            self._scoring = True

    @property
    def scoring(self) -> bool:
        return self._scoring

    # -- microphone convenience (standalone demo) --------------------------- #

    def run_from_microphone(self) -> None:
        """Open the default mic and feed it until interrupted. Blocking."""
        import sounddevice as sd

        def callback(indata, _frames, _time, status):
            if status:
                pass  # over/underruns are non-fatal for scoring
            self.feed(indata[:, 0])

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.num_frames,
            callback=callback,
        ):
            self._stop.wait()

    # -- internals ---------------------------------------------------------- #

    def _loop(self) -> None:
        first = True
        while not self._stop.wait(self._hop_seconds):
            with self._lock:
                ready = self._scoring and self._buffered >= self._window_samples
            if not ready:
                continue
            try:
                result = self._analyzer.analyze_buffered()
            except Exception as err:  # noqa: BLE001 - surfaced to the UI
                self._emit_state("error", f"Tyto error: {err}")
                continue
            raw = Scores.from_result(result)
            self._smoothed = raw.ema(self._smoothed, self._ema_alpha)
            if first:
                first = False
                self._emit_state("live", "scoring")
            if self.on_scores:
                self.on_scores(self._smoothed)

    def _emit_state(self, state: str, text: str) -> None:
        if self._on_state:
            self._on_state(state, text)
