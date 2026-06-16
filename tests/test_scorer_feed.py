"""Test the scorer's block accumulation without the SDK or a microphone.

feed() must emit exactly num_frames-sized buffers regardless of the caller's
block size, and must drop audio while paused.
"""

import numpy as np

from tyto_voice.scorer import LiveTytoScorer


class FakeCollector:
    def __init__(self):
        self.sizes = []

    def buffer(self, block):
        assert block.shape[0] == 1  # mono, shape (1, num_frames)
        self.sizes.append(block.shape[1])


def make_scorer(num_frames=160):
    scorer = LiveTytoScorer("dummy")
    scorer.num_frames = num_frames
    scorer._collector = FakeCollector()
    return scorer


def test_feed_emits_fixed_size_blocks():
    scorer = make_scorer(num_frames=160)
    # Feed 500 samples in odd-sized chunks; expect 3 full 160-frame blocks.
    scorer.feed(np.zeros(100, dtype=np.float32))
    scorer.feed(np.zeros(100, dtype=np.float32))
    scorer.feed(np.zeros(100, dtype=np.float32))
    scorer.feed(np.zeros(200, dtype=np.float32))
    sizes = scorer._collector.sizes
    assert sizes == [160, 160, 160]  # 500 buffered, 480 emitted, 20 residual
    assert scorer._buffered == 480


def test_feed_dropped_while_paused():
    scorer = make_scorer()
    scorer.pause()
    scorer.feed(np.zeros(320, dtype=np.float32))
    assert scorer._collector.sizes == []
    assert scorer._buffered == 0
