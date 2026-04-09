"""transcriber.py のテスト"""

import numpy as np
import pytest
import tempfile
import soundfile as sf

from gb_converter.transcriber import NoteEvent, _midi_name


class TestNoteEvent:
    def test_duration(self):
        e = NoteEvent(pitch_midi=69, start_sec=1.0, end_sec=2.5, velocity=80)
        assert e.duration_sec == pytest.approx(1.5)

    def test_duration_zero_if_reversed(self):
        e = NoteEvent(pitch_midi=69, start_sec=2.0, end_sec=1.0, velocity=80)
        assert e.duration_sec == 0.0

    def test_freq_hz_a4(self):
        """A4 (MIDI 69) = 440 Hz"""
        e = NoteEvent(pitch_midi=69, start_sec=0.0, end_sec=1.0, velocity=80)
        assert e.freq_hz == pytest.approx(440.0, rel=1e-4)

    def test_freq_hz_a5(self):
        """A5 (MIDI 81) = 880 Hz (1オクターブ上)"""
        e = NoteEvent(pitch_midi=81, start_sec=0.0, end_sec=1.0, velocity=80)
        assert e.freq_hz == pytest.approx(880.0, rel=1e-4)

    def test_freq_hz_c4(self):
        """C4 (MIDI 60) ≒ 261.63 Hz"""
        e = NoteEvent(pitch_midi=60, start_sec=0.0, end_sec=1.0, velocity=80)
        assert e.freq_hz == pytest.approx(261.63, rel=1e-3)


class TestMidiName:
    def test_a4(self):
        assert _midi_name(69) == "A4"

    def test_c4(self):
        assert _midi_name(60) == "C4"

    def test_c5(self):
        assert _midi_name(72) == "C5"


class TestHasBasicPitch:
    def test_returns_bool(self):
        from gb_converter.transcriber import has_basic_pitch
        result = has_basic_pitch()
        assert isinstance(result, bool)


class TestTranscribeIntegration:
    """実際の音声ファイルを使った統合テスト（sinewaveで代用）。両エンジンで動作確認。"""

    def _make_sine_wav(self, freq_hz: float = 440.0, duration: float = 2.0,
                       sr: int = 44100) -> str:
        """テスト用サイン波WAVを一時ファイルとして作成する。"""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wave, sr)
        return tmp.name

    def test_transcribe_returns_list(self):
        """transcribe() がリストを返すこと"""
        from gb_converter.transcriber import transcribe
        path = self._make_sine_wav(freq_hz=440.0, duration=2.0)
        events = transcribe(path)
        assert isinstance(events, list)

    def test_transcribe_sine_detects_pitch(self):
        """440Hzサイン波から A4付近のノートが検出されること（両エンジン共通）"""
        from gb_converter.transcriber import transcribe
        path = self._make_sine_wav(freq_hz=440.0, duration=2.0)
        events = transcribe(path)
        assert len(events) > 0
        pitches = [e.pitch_midi for e in events]
        # basic-pitch: ±2半音、lite: ±3半音で許容
        assert any(66 <= p <= 72 for p in pitches), f"検出ピッチ: {pitches}"

    def test_transcribe_lite_directly(self):
        """liteモードを直接呼び出して動作確認"""
        from gb_converter.transcriber import _transcribe_lite
        path = self._make_sine_wav(freq_hz=440.0, duration=2.0)
        events = _transcribe_lite(path, min_note_len_sec=0.05,
                                  min_freq_hz=65.0, max_freq_hz=2093.0)
        assert isinstance(events, list)
        if events:
            pitches = [e.pitch_midi for e in events]
            assert any(66 <= p <= 72 for p in pitches), f"検出ピッチ: {pitches}"

    def test_transcribe_events_sorted(self):
        """出力が開始時刻順にソートされていること"""
        from gb_converter.transcriber import transcribe
        path = self._make_sine_wav(freq_hz=440.0, duration=2.0)
        events = transcribe(path)
        starts = [e.start_sec for e in events]
        assert starts == sorted(starts)

    def test_transcribe_velocity_range(self):
        """ベロシティが 1〜127 の範囲内であること"""
        from gb_converter.transcriber import transcribe
        path = self._make_sine_wav(freq_hz=440.0, duration=2.0)
        events = transcribe(path)
        for e in events:
            assert 1 <= e.velocity <= 127
