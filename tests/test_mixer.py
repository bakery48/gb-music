"""ミキサーのテスト"""

import numpy as np
import pytest

from gb_converter.mixer import mix, apply_gb_lowpass


class TestMixer:
    def test_mix_all_channels(self):
        n = 1000
        ch1 = np.ones(n, dtype=np.float32) * 0.5
        ch2 = np.ones(n, dtype=np.float32) * 0.5
        ch3 = np.ones(n, dtype=np.float32) * 0.5
        ch4 = np.ones(n, dtype=np.float32) * 0.5
        result = mix(ch1, ch2, ch3, ch4)
        assert result.shape == (n,)
        assert result.max() <= 1.0 + 1e-6

    def test_mix_with_none(self):
        n = 1000
        ch1 = np.ones(n, dtype=np.float32) * 0.5
        result = mix(ch1, None, None, None)
        assert result.shape == (n,)

    def test_mix_different_lengths(self):
        ch1 = np.ones(1000, dtype=np.float32)
        ch2 = np.ones(500, dtype=np.float32)
        result = mix(ch1, ch2, None, None)
        assert result.shape == (1000,)

    def test_apply_lowpass(self):
        sr = 44100
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        # 高周波ノイズを含む信号
        signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 15000 * t)
        signal = signal.astype(np.float32)
        filtered = apply_gb_lowpass(signal, sr)
        assert filtered.shape == signal.shape
        assert filtered.dtype == np.float32
