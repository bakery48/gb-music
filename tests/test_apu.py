"""APUチャンネルの基本テスト"""

import numpy as np
import pytest

from gb_converter.apu.channel1 import PulseChannel, hz_to_register, register_to_hz, quantize_4bit
from gb_converter.apu.channel3 import WaveChannel, make_wave_from_audio
from gb_converter.apu.channel4 import NoiseChannel, noise_frequency


class TestChannel1:
    def test_hz_register_roundtrip(self):
        freq = 440.0
        reg = hz_to_register(freq)
        recovered = register_to_hz(reg)
        # GBのレジスタ精度内で一致すればOK
        assert abs(recovered - freq) < 5.0

    def test_register_range(self):
        assert 0 <= hz_to_register(65.0) <= 2047
        assert 0 <= hz_to_register(2000.0) <= 2047

    def test_quantize_4bit_range(self):
        samples = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        result = quantize_4bit(samples)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_quantize_4bit_levels(self):
        samples = np.linspace(-1.0, 1.0, 10000, dtype=np.float32)
        result = quantize_4bit(samples)
        unique = np.unique(np.round(result, 4))
        assert len(unique) == 16

    def test_render_basic(self):
        ch = PulseChannel(sample_rate=44100)
        events = [(0, 440.0, 2, 0.8)]
        output = ch.render(events, n_samples=4410)
        assert output.shape == (4410,)
        assert output.dtype == np.float32

    def test_render_empty(self):
        ch = PulseChannel(sample_rate=44100)
        output = ch.render([], n_samples=1000)
        assert np.all(output == 0.0)

    def test_render_silent_freq(self):
        ch = PulseChannel(sample_rate=44100)
        events = [(0, 0.0, 2, 0.5)]
        output = ch.render(events, n_samples=1000)
        assert np.all(output == 0.0)


class TestWaveChannel:
    def test_sine_wave_ram(self):
        wave = WaveChannel.sine_wave_ram()
        assert wave.shape == (32,)
        assert wave.dtype == np.uint8
        assert wave.min() >= 0
        assert wave.max() <= 15

    def test_make_wave_from_audio(self):
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        sine = np.sin(t).astype(np.float32)
        wave = make_wave_from_audio(sine)
        assert len(wave) == 32
        assert wave.min() >= 0
        assert wave.max() <= 15

    def test_render_basic(self):
        ch = WaveChannel(sample_rate=44100)
        wave_ram = WaveChannel.sine_wave_ram()
        events = [(0, 110.0, wave_ram, 1.0)]
        output = ch.render(events, n_samples=4410)
        assert output.shape == (4410,)
        assert output.dtype == np.float32

    def test_render_empty(self):
        ch = WaveChannel(sample_rate=44100)
        output = ch.render([], n_samples=1000)
        assert np.all(output == 0.0)


class TestNoiseChannel:
    def test_noise_frequency(self):
        freq = noise_frequency(0, 0)
        assert freq > 0

    def test_estimate_params(self):
        r, s, w = NoiseChannel.estimate_params(0.9)
        assert 0 <= r <= 7
        assert 0 <= s <= 13
        assert isinstance(w, bool)

    def test_render_basic(self):
        ch = NoiseChannel(sample_rate=44100)
        events = [(0, 0, 2, False, 0.8)]
        output = ch.render(events, n_samples=4410)
        assert output.shape == (4410,)
        assert output.dtype == np.float32
        # ノイズは単調でないはず
        assert not np.all(output == output[0])

    def test_render_empty(self):
        ch = NoiseChannel(sample_rate=44100)
        output = ch.render([], n_samples=1000)
        assert np.all(output == 0.0)
