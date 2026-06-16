"""Local speaker playback for the terminal voice agent.

A small sink that plays the agent's PCM16 audio through the default output
device and reports when the agent becomes audible and falls silent, which is the
signal the controller uses to mute/unmute the mic. The web demo replaces this
with the browser, which plays audio and reports the same signal over a socket.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np


class SounddeviceSink:
    def __init__(self, on_playing: Callable[[bool], None], *, sample_rate: int = 24000):
        self._on_playing = on_playing
        self._sample_rate = sample_rate
        self._pcm = bytearray()
        self._lock = threading.Lock()
        self._playing = False
        self._gen_done = False
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16", callback=self._callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # -- audio callbacks handed to the provider ----------------------------- #

    def write(self, pcm16: bytes) -> None:
        with self._lock:
            self._pcm.extend(pcm16)
            self._gen_done = False
        if not self._playing:
            self._playing = True
            self._on_playing(True)

    def notify_done(self) -> None:
        """The agent finished generating; stop once the buffer drains."""
        with self._lock:
            self._gen_done = True
            empty = len(self._pcm) == 0
        if empty and self._playing:
            self._playing = False
            self._on_playing(False)

    def flush(self) -> None:
        """Drop pending audio immediately (interrupt)."""
        with self._lock:
            self._pcm.clear()
            self._gen_done = False
        if self._playing:
            self._playing = False
            self._on_playing(False)

    # -- audio thread ------------------------------------------------------- #

    def _callback(self, outdata, frames, _time, _status) -> None:
        want = frames * 2  # int16 bytes, mono
        with self._lock:
            take = min(len(self._pcm), want)
            chunk = bytes(self._pcm[:take])
            del self._pcm[:take]
            empty = len(self._pcm) == 0
            gen_done = self._gen_done
        n = take // 2
        if n:
            outdata[:n, 0] = np.frombuffer(chunk, dtype="<i2")
        if n < frames:
            outdata[n:, 0] = 0
        if empty and gen_done and self._playing:
            self._playing = False
            self._on_playing(False)
