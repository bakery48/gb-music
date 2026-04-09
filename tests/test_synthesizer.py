"""synthesizer.py のテスト"""

import numpy as np
import pytest

from gb_converter.transcriber import NoteEvent
from gb_converter.scheduler import ChannelTracks, OnsetEvent
from gb_converter.note_mapper import map_to_apu_events
from gb_converter.synthesizer import synthesize, render_channel_previews, GainConfig


SR = 44100
N = SR  # 1秒分


def note(pitch: int, start: float, end: float, vel: int = 80) -> NoteEvent:
    return NoteEvent(pitch_midi=pitch, start_sec=start, end_sec=end, velocity=vel)


def make_events(notes_ch1=None, notes_ch2=None, notes_ch3=None, onsets=None):
    tracks = ChannelTracks(
        ch1=notes_ch1 or [],
        ch2=notes_ch2 or [],
        ch3=notes_ch3 or [],
        ch4=onsets or [],
    )
    return map_to_apu_events(tracks, sr=SR, duty=2)


class TestSynthesize:
    def test_returns_float32(self):
        ch1e, ch2e, ch3e, ch4e = make_events()
        out = synthesize(ch1e, ch2e, ch3e, ch4e, n_samples=N, sr=SR)
        assert out.dtype == np.float32

    def test_output_length(self):
        ch1e, ch2e, ch3e, ch4e = make_events()
        out = synthesize(ch1e, ch2e, ch3e, ch4e, n_samples=N, sr=SR)
        assert len(out) == N

    def test_output_range(self):
        """出力が -1.0〜1.0 の範囲内"""
        ch1e, ch2e, ch3e, ch4e = make_events(
            notes_ch1=[note(69, 0.0, 0.9)]
        )
        out = synthesize(ch1e, ch2e, ch3e, ch4e, n_samples=N, sr=SR)
        assert out.max() <= 1.0 + 1e-5
        assert out.min() >= -1.0 - 1e-5

    def test_silent_with_no_events(self):
        """イベントなし→ほぼ無音"""
        ch1e, ch2e, ch3e, ch4e = make_events()
        out = synthesize(ch1e, ch2e, ch3e, ch4e, n_samples=N, sr=SR)
        assert np.max(np.abs(out)) < 1e-5

    def test_ch1_note_produces_sound(self):
        """CH1にノートを入れると有音になる"""
        ch1e, ch2e, ch3e, ch4e = make_events(
            notes_ch1=[note(69, 0.0, 0.9)]
        )
        out = synthesize(ch1e, ch2e, ch3e, ch4e, n_samples=N, sr=SR)
        assert np.max(np.abs(out)) > 0.01

    def test_note_off_silences_channel(self):
        """note-off 後の後半は無音に近い"""
        ch1e, ch2e, ch3e, ch4e = make_events(
            notes_ch1=[note(69, 0.0, 0.3)]  # 0.3秒で終了
        )
        out = synthesize(
            ch1e, ch2e, ch3e, ch4e, n_samples=N, sr=SR, apply_lowpass=False
        )
        # 後半 (0.5秒〜1.0秒) はほぼ無音のはず
        late = out[SR // 2:]
        assert np.max(np.abs(late)) < 0.05

    def test_custom_gains(self):
        """カスタムゲインが反映される"""
        ch1e, ch2e, ch3e, ch4e = make_events(
            notes_ch1=[note(69, 0.0, 0.9)]
        )
        gains_loud  = GainConfig(ch1=1.0, ch2=0.0, ch3=0.0, ch4=0.0)
        gains_quiet = GainConfig(ch1=0.1, ch2=0.0, ch3=0.0, ch4=0.0)
        out_loud  = synthesize(ch1e, ch2e, ch3e, ch4e, N, SR, gains_loud,  apply_lowpass=False)
        out_quiet = synthesize(ch1e, ch2e, ch3e, ch4e, N, SR, gains_quiet, apply_lowpass=False)
        # ピーク正規化後は両者同じになる（0.95付近）
        assert np.max(np.abs(out_loud))  == pytest.approx(0.95, abs=0.05)
        assert np.max(np.abs(out_quiet)) == pytest.approx(0.95, abs=0.05)

    def test_lowpass_reduces_high_freq(self):
        """LPFで高周波エネルギーが減衰する"""
        ch1e, ch2e, ch3e, ch4e = make_events(
            notes_ch1=[note(96, 0.0, 0.9)]  # 高音域 C7
        )
        out_with    = synthesize(ch1e, ch2e, ch3e, ch4e, N, SR, apply_lowpass=True)
        out_without = synthesize(ch1e, ch2e, ch3e, ch4e, N, SR, apply_lowpass=False)
        # FFTで高周波成分を比較
        fft_with    = np.abs(np.fft.rfft(out_with))
        fft_without = np.abs(np.fft.rfft(out_without))
        freqs = np.fft.rfftfreq(N, 1 / SR)
        high_mask = freqs > 10000
        assert fft_with[high_mask].mean() < fft_without[high_mask].mean()


class TestRenderChannelPreviews:
    def test_returns_four_channels(self):
        ch1e, ch2e, ch3e, ch4e = make_events(notes_ch1=[note(69, 0.0, 0.9)])
        previews = render_channel_previews(ch1e, ch2e, ch3e, ch4e, N, SR)
        assert set(previews.keys()) == {"ch1", "ch2", "ch3", "ch4"}

    def test_each_channel_correct_length(self):
        ch1e, ch2e, ch3e, ch4e = make_events(notes_ch1=[note(69, 0.0, 0.9)])
        previews = render_channel_previews(ch1e, ch2e, ch3e, ch4e, N, SR)
        for audio in previews.values():
            assert len(audio) == N

    def test_active_channel_not_silent(self):
        ch1e, ch2e, ch3e, ch4e = make_events(notes_ch1=[note(69, 0.0, 0.9)])
        previews = render_channel_previews(ch1e, ch2e, ch3e, ch4e, N, SR)
        assert np.max(np.abs(previews["ch1"])) > 0.01
        assert np.max(np.abs(previews["ch2"])) < 0.01  # CH2は空


class TestGainConfig:
    def test_to_dict(self):
        g = GainConfig(ch1=0.5, ch2=0.3, ch3=0.2, ch4=0.1)
        d = g.to_dict()
        assert d == {"ch1": 0.5, "ch2": 0.3, "ch3": 0.2, "ch4": 0.1}

    def test_defaults(self):
        g = GainConfig()
        assert sum(g.to_dict().values()) == pytest.approx(0.95, abs=0.1)
