"""scheduler.py のテスト"""

import numpy as np
import pytest

from gb_converter.transcriber import NoteEvent
from gb_converter.scheduler import (
    schedule, ChannelTracks, OnsetEvent,
    _schedule_mono_channel, _schedule_two_mono_channels,
)


def note(pitch: int, start: float, end: float, vel: int = 80) -> NoteEvent:
    return NoteEvent(pitch_midi=pitch, start_sec=start, end_sec=end, velocity=vel)


class TestScheduleMonoChannel:
    def test_no_overlap(self):
        """重複しない音符はすべて割り当てられる"""
        notes = [note(60, 0.0, 1.0), note(62, 1.0, 2.0), note(64, 2.0, 3.0)]
        result = _schedule_mono_channel(notes)
        assert len(result) == 3

    def test_overlap_keeps_earlier(self):
        """同ベロシティの重複は先の音符を優先"""
        notes = [note(60, 0.0, 2.0), note(62, 1.0, 3.0)]
        result = _schedule_mono_channel(notes)
        assert len(result) == 1
        assert result[0].pitch_midi == 60

    def test_overlap_higher_velocity_wins(self):
        """後から来た音符のベロシティが高ければ置き換える"""
        notes = [note(60, 0.0, 2.0, vel=40), note(62, 1.0, 3.0, vel=100)]
        result = _schedule_mono_channel(notes)
        # 置き換え後、60は0.0〜1.0に短縮されて62が続く
        pitches = [n.pitch_midi for n in result]
        assert 62 in pitches

    def test_no_overlaps_in_output(self):
        """出力に重複がないこと"""
        notes = [note(60 + i, i * 0.5, i * 0.5 + 0.8) for i in range(10)]
        result = _schedule_mono_channel(notes)
        for i in range(len(result) - 1):
            assert result[i].end_sec <= result[i + 1].start_sec + 1e-9

    def test_empty_input(self):
        assert _schedule_mono_channel([]) == []


class TestScheduleTwoChannels:
    def test_simultaneous_notes_split(self):
        """同時発音の2音符はCH1とCH2に分かれる"""
        notes = [note(60, 0.0, 1.0), note(64, 0.0, 1.0)]
        ch1, ch2 = _schedule_two_mono_channels(notes)
        assert len(ch1) >= 1
        assert len(ch2) >= 1

    def test_sequential_notes_all_in_ch1(self):
        """時系列が重複しない音符はすべてCH1に入る"""
        notes = [note(60, 0.0, 1.0), note(62, 1.0, 2.0), note(64, 2.0, 3.0)]
        ch1, ch2 = _schedule_two_mono_channels(notes)
        assert len(ch1) == 3
        assert len(ch2) == 0

    def test_no_overlaps_per_channel(self):
        """各チャンネル内に重複がないこと"""
        notes = [note(60 + i, i * 0.3, i * 0.3 + 0.5) for i in range(20)]
        ch1, ch2 = _schedule_two_mono_channels(notes)
        for ch in [ch1, ch2]:
            for i in range(len(ch) - 1):
                assert ch[i].end_sec <= ch[i + 1].start_sec + 1e-9

    def test_empty_input(self):
        ch1, ch2 = _schedule_two_mono_channels([])
        assert ch1 == [] and ch2 == []


class TestSchedule:
    def test_bass_goes_to_ch3(self):
        """MIDI < 48 のノートはCH3に振り分けられる"""
        notes = [note(36, 0.0, 1.0), note(40, 1.0, 2.0)]  # C2, E2
        tracks = schedule(notes)
        assert len(tracks.ch3) == 2
        assert len(tracks.ch1) == 0
        assert len(tracks.ch2) == 0

    def test_melody_goes_to_ch1_ch2(self):
        """MIDI >= 48 のノートはCH1/CH2に振り分けられる"""
        notes = [note(60, 0.0, 1.0), note(64, 0.0, 1.0)]  # C4, E4（同時）
        tracks = schedule(notes)
        assert len(tracks.ch1) + len(tracks.ch2) == 2

    def test_mixed_pitch_separation(self):
        """高音・低音が正しく分離されること"""
        notes = [
            note(36, 0.0, 0.5),   # C2 → CH3
            note(60, 0.0, 0.5),   # C4 → CH1
            note(67, 0.0, 0.5),   # G4 → CH2
        ]
        tracks = schedule(notes)
        assert len(tracks.ch3) == 1
        assert len(tracks.ch1) + len(tracks.ch2) == 2

    def test_no_audio_no_ch4(self):
        """audioなしのときCH4は空"""
        notes = [note(60, 0.0, 1.0)]
        tracks = schedule(notes, audio=None)
        assert tracks.ch4 == []

    def test_ch4_with_audio(self):
        """audioを渡すとCH4にオンセットが検出される"""
        sr = 44100
        # 周期的なクリック音を生成
        audio = np.zeros(sr * 2, dtype=np.float32)
        for i in range(0, sr * 2, sr // 4):
            audio[i] = 1.0
        tracks = schedule([], audio=audio, sr=sr)
        assert isinstance(tracks.ch4, list)
        for ev in tracks.ch4:
            assert isinstance(ev, OnsetEvent)
            assert 0.0 <= ev.strength <= 1.0

    def test_summary_format(self):
        notes = [note(60, 0.0, 1.0), note(36, 0.0, 1.0)]
        tracks = schedule(notes)
        summary = tracks.summary()
        assert "CH1" in summary and "CH4" in summary

    def test_empty_notes(self):
        tracks = schedule([])
        assert tracks.ch1 == [] and tracks.ch2 == []
        assert tracks.ch3 == [] and tracks.ch4 == []
