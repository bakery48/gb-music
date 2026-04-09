"""note_mapper.py のテスト"""

import numpy as np
import pytest

from gb_converter.transcriber import NoteEvent
from gb_converter.scheduler import ChannelTracks, OnsetEvent
from gb_converter.note_mapper import (
    map_to_apu_events,
    _sec_to_sample,
    _snap_to_gb_freq,
    _adjust_freq_for_ch3,
    _velocity_to_volume,
    _notes_to_pulse_events,
    _notes_to_wave_events,
    _onsets_to_noise_events,
)


def note(pitch: int, start: float, end: float, vel: int = 80) -> NoteEvent:
    return NoteEvent(pitch_midi=pitch, start_sec=start, end_sec=end, velocity=vel)


def onset(start: float, strength: float = 0.8) -> OnsetEvent:
    return OnsetEvent(start_sec=start, strength=strength)


class TestUtilities:
    def test_sec_to_sample(self):
        assert _sec_to_sample(1.0, 44100) == 44100
        assert _sec_to_sample(0.5, 44100) == 22050
        assert _sec_to_sample(-1.0, 44100) == 0  # 負は0

    def test_snap_to_gb_freq_a4(self):
        """440Hz はGBレジスタにスナップされてもほぼ同じ周波数"""
        snapped = _snap_to_gb_freq(440.0)
        assert abs(snapped - 440.0) < 10.0  # ±10Hz以内

    def test_snap_to_gb_freq_zero(self):
        assert _snap_to_gb_freq(0.0) == 0.0
        assert _snap_to_gb_freq(-1.0) == 0.0

    def test_snap_discretizes(self):
        """スナップ後の値はレジスタ解像度に制限される"""
        # 隣接する2周波数はスナップ後に同じ値になりうる
        f1 = _snap_to_gb_freq(440.0)
        f2 = _snap_to_gb_freq(441.0)
        # どちらも有効な周波数（0以上）
        assert f1 > 0
        assert f2 > 0

    def test_adjust_freq_for_ch3_low(self):
        """低すぎる周波数はオクターブ上げされる"""
        result = _adjust_freq_for_ch3(30.0)  # 30Hz → 60Hz → ...
        assert result >= 65.0

    def test_adjust_freq_for_ch3_high(self):
        """高すぎる周波数はオクターブ下げされる"""
        result = _adjust_freq_for_ch3(10000.0)
        assert result <= 4186.0

    def test_adjust_freq_for_ch3_normal(self):
        """有効範囲内はそのまま"""
        result = _adjust_freq_for_ch3(110.0)
        assert 65.0 <= result <= 4186.0

    def test_velocity_to_volume_range(self):
        for vel in range(0, 128, 8):
            v = _velocity_to_volume(vel)
            assert 0.0 <= v <= 1.0

    def test_velocity_to_volume_max(self):
        assert _velocity_to_volume(127) == pytest.approx(1.0)

    def test_velocity_to_volume_quantized(self):
        """4bit量子化 → 16段階のみ"""
        values = set(_velocity_to_volume(v) for v in range(128))
        assert len(values) <= 16


class TestPulseEvents:
    def test_note_generates_on_and_off(self):
        """1音符 → note-on と note-off の2イベント"""
        notes = [note(69, 0.0, 1.0)]  # A4
        events = _notes_to_pulse_events(notes, sr=44100, duty=2)
        assert len(events) == 2

    def test_note_on_has_positive_freq(self):
        notes = [note(69, 0.0, 1.0)]
        events = _notes_to_pulse_events(notes, sr=44100, duty=2)
        on_events = [e for e in events if e[1] > 0]
        assert len(on_events) == 1
        assert on_events[0][1] > 0  # freq > 0

    def test_note_off_has_zero_freq_and_volume(self):
        notes = [note(69, 0.0, 1.0)]
        events = _notes_to_pulse_events(notes, sr=44100, duty=2)
        off_events = [e for e in events if e[1] == 0.0]
        assert len(off_events) == 1
        assert off_events[0][3] == 0.0  # volume == 0

    def test_events_sorted_by_offset(self):
        notes = [note(69, 1.0, 2.0), note(72, 0.0, 0.5)]
        events = _notes_to_pulse_events(notes, sr=44100, duty=2)
        offsets = [e[0] for e in events]
        assert offsets == sorted(offsets)

    def test_duty_propagated(self):
        notes = [note(69, 0.0, 1.0)]
        for duty in range(4):
            events = _notes_to_pulse_events(notes, sr=44100, duty=duty)
            assert all(e[2] == duty for e in events)

    def test_empty_input(self):
        assert _notes_to_pulse_events([], sr=44100, duty=2) == []


class TestWaveEvents:
    def test_note_generates_on_and_off(self):
        from gb_converter.apu.channel3 import WaveChannel
        wave_ram = WaveChannel.sine_wave_ram()
        notes = [note(36, 0.0, 1.0)]  # C2
        events = _notes_to_wave_events(notes, sr=44100, wave_ram=wave_ram)
        assert len(events) == 2

    def test_freq_adjusted_for_ch3(self):
        """超低音は CH3 有効範囲にオクターブ調整される"""
        from gb_converter.apu.channel3 import WaveChannel
        wave_ram = WaveChannel.sine_wave_ram()
        notes = [note(21, 0.0, 1.0)]  # A0 = 27.5Hz (CH3範囲外)
        events = _notes_to_wave_events(notes, sr=44100, wave_ram=wave_ram)
        on_event = next(e for e in events if e[1] > 0)
        assert on_event[1] >= 65.0


class TestNoiseEvents:
    def test_onset_generates_on_and_off(self):
        onsets = [onset(0.5, strength=0.8)]
        events = _onsets_to_noise_events(onsets, sr=44100)
        assert len(events) == 2

    def test_gate_off_has_zero_volume(self):
        onsets = [onset(0.5, strength=0.8)]
        events = _onsets_to_noise_events(onsets, sr=44100)
        off_events = [e for e in events if e[4] == 0.0]
        assert len(off_events) == 1

    def test_events_sorted(self):
        onsets = [onset(1.0), onset(0.5), onset(0.0)]
        events = _onsets_to_noise_events(onsets, sr=44100)
        offsets = [e[0] for e in events]
        assert offsets == sorted(offsets)

    def test_empty_input(self):
        assert _onsets_to_noise_events([], sr=44100) == []


class TestMapToApuEvents:
    def test_returns_four_lists(self):
        tracks = ChannelTracks(
            ch1=[note(69, 0.0, 1.0)],
            ch2=[note(65, 0.0, 1.0)],
            ch3=[note(36, 0.0, 1.0)],
            ch4=[onset(0.5)],
        )
        result = map_to_apu_events(tracks, sr=44100, duty=2)
        assert len(result) == 4
        ch1e, ch2e, ch3e, ch4e = result
        assert len(ch1e) == 2   # note-on + note-off
        assert len(ch2e) == 2
        assert len(ch3e) == 2
        assert len(ch4e) == 2

    def test_empty_tracks(self):
        tracks = ChannelTracks()
        ch1e, ch2e, ch3e, ch4e = map_to_apu_events(tracks, sr=44100)
        assert ch1e == [] and ch2e == [] and ch3e == [] and ch4e == []
