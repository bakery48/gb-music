"""transcriber.py のテスト (librosaのみ・TF不要)"""

import numpy as np
import pytest
import tempfile
import soundfile as sf

from gb_converter.transcriber import NoteEvent, _midi_name, transcribe


class TestNoteEvent:
    def test_duration(self):
        e = NoteEvent(pitch_midi=69, start_sec=1.0, end_sec=2.5, velocity=80)
        assert e.duration_sec == pytest.approx(1.5)

    def test_duration_zero_if_reversed(self):
        e = NoteEvent(pitch_midi=69, start_sec=2.0, end_sec=1.0, velocity=80)
        assert e.duration_sec == 0.0

    def test_freq_hz_a4(self):
        e = NoteEvent(pitch_midi=69, start_sec=0.0, end_sec=1.0, velocity=80)
        assert e.freq_hz == pytest.approx(440.0, rel=1e-4)

    def test_freq_hz_a5(self):
        e = NoteEvent(pitch_midi=81, start_sec=0.0, end_sec=1.0, velocity=80)
        assert e.freq_hz == pytest.approx(880.0, rel=1e-4)

    def test_freq_hz_c4(self):
        e = NoteEvent(pitch_midi=60, start_sec=0.0, end_sec=1.0, velocity=80)
        assert e.freq_hz == pytest.approx(261.63, rel=1e-3)


class TestMidiName:
    def test_a4(self):
        assert _midi_name(69) == "A4"

    def test_c4(self):
        assert _midi_name(60) == "C4"

    def test_c5(self):
        assert _midi_name(72) == "C5"


class TestTranscribeIntegration:
    """サイン波WAVを使った統合テスト。"""

    def _make_sine_wav(self, freq_hz: float = 440.0, duration: float = 2.0,
                       sr: int = 22050) -> str:
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave = (0.6 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wave, sr)
        return tmp.name

    def test_returns_list(self):
        path = self._make_sine_wav()
        assert isinstance(transcribe(path), list)

    def test_sorted_by_start(self):
        path = self._make_sine_wav()
        events = transcribe(path)
        starts = [e.start_sec for e in events]
        assert starts == sorted(starts)

    def test_velocity_range(self):
        path = self._make_sine_wav()
        for e in transcribe(path):
            assert 1 <= e.velocity <= 127

    def test_pitch_midi_range(self):
        path = self._make_sine_wav()
        for e in transcribe(path):
            assert 0 <= e.pitch_midi <= 127

    def test_detects_a4(self):
        """440Hz サイン波から A4 (MIDI69) ±3半音が検出される"""
        path = self._make_sine_wav(freq_hz=440.0, duration=3.0)
        events = transcribe(path, voiced_threshold=0.2)
        assert len(events) > 0
        pitches = [e.pitch_midi for e in events]
        assert any(66 <= p <= 72 for p in pitches), f"検出: {pitches}"

    def test_no_notes_from_silence(self):
        """無音ファイルからはノートが検出されない"""
        sr = 22050
        silence = np.zeros(sr * 2, dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, silence, sr)
        events = transcribe(tmp.name)
        assert events == []
