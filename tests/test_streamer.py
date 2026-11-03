"""Tests for live streaming buffers."""

import numpy as np

from speck.streaming.sessions import AudioBuffer


def test_audio_buffer_absolute_indexing_and_discard():
    buf = AudioBuffer(capacity=4)
    buf.append(np.arange(10, dtype=np.float32))
    assert len(buf) == 10
    np.testing.assert_array_equal(buf.slice(2, 5), [2, 3, 4])
    buf.discard_before(6)
    buf.append(np.array([10, 11], dtype=np.float32))
    assert len(buf) == 12
    np.testing.assert_array_equal(buf.slice(6, 20), [6, 7, 8, 9, 10, 11])
    np.testing.assert_array_equal(buf.slice(0, 7), [6])  # discarded samples are gone
