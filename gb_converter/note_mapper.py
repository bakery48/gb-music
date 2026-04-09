"""
Step 3: GB ノートイベント生成

ChannelTracks (NoteEvent / OnsetEvent) を各APUチャンネルの
render() が受け取れるイベント形式に変換する。

変換内容:
  NoteEvent.pitch_midi  → freq_hz → GBレジスタ値でスナップ → freq_hz
  NoteEvent.velocity    → GB volume (0.0〜1.0、4bitスケール)
  NoteEvent.start_sec   → sample_offset
  NoteEvent.end_sec     → note-off イベントのsample_offset
  OnsetEvent            → LFSRパラメータ (r, s, width7)

出力イベント形式:
  CH1/CH2: [(sample_offset, freq_hz, duty, volume), ...]
  CH3:     [(sample_offset, freq_hz, wave_ram, volume), ...]
  CH4:     [(sample_offset, r, s, width7, volume), ...]
"""

from __future__ import annotations

import numpy as np

from gb_converter.transcriber import NoteEvent
from gb_converter.scheduler import ChannelTracks, OnsetEvent
from gb_converter.apu.channel1 import hz_to_register, register_to_hz
from gb_converter.apu.channel3 import hz_to_register_wave, WaveChannel
from gb_converter.apu.channel4 import NoiseChannel


# CH3のデフォルト波形（サイン波 → GB風のまろやかなベース音）
_DEFAULT_WAVE_RAM = WaveChannel.sine_wave_ram()

# ノイズのデフォルト持続時間 [サンプル]（ゲート長）
_NOISE_GATE_SAMPLES = 2048


# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------

PulseEvents = list[tuple[int, float, int, float]]   # CH1/CH2
WaveEvents  = list[tuple[int, float, np.ndarray, float]]  # CH3
NoiseEvents = list[tuple[int, int, int, bool, float]]     # CH4


def map_to_apu_events(
    tracks: ChannelTracks,
    sr: int,
    duty: int = 2,
    wave_ram: np.ndarray | None = None,
) -> tuple[PulseEvents, PulseEvents, WaveEvents, NoiseEvents]:
    """
    ChannelTracks → APU各チャンネルのイベントリストに変換する。

    Args:
        tracks:   scheduler.schedule() の出力
        sr:       サンプルレート [Hz]
        duty:     CH1/CH2 のデューティ比 (0〜3)
        wave_ram: CH3 のWave RAM (省略時はサイン波)

    Returns:
        (ch1_events, ch2_events, ch3_events, ch4_events)
    """
    wram = wave_ram if wave_ram is not None else _DEFAULT_WAVE_RAM

    ch1_events = _notes_to_pulse_events(tracks.ch1, sr, duty)
    ch2_events = _notes_to_pulse_events(tracks.ch2, sr, duty)
    ch3_events = _notes_to_wave_events(tracks.ch3, sr, wram)
    ch4_events = _onsets_to_noise_events(tracks.ch4, sr)

    return ch1_events, ch2_events, ch3_events, ch4_events


# ---------------------------------------------------------------------------
# 内部変換関数
# ---------------------------------------------------------------------------

def _notes_to_pulse_events(
    notes: list[NoteEvent],
    sr: int,
    duty: int,
) -> PulseEvents:
    """
    NoteEvent リスト → CH1/CH2 パルス波イベントリスト。

    各音符につき note-on と note-off の2イベントを生成する。
    note-off は (offset, 0.0, duty, 0.0) — 周波数0・音量0 で無音化。
    """
    events: PulseEvents = []

    for note in notes:
        on_offset  = _sec_to_sample(note.start_sec, sr)
        off_offset = _sec_to_sample(note.end_sec,   sr)

        # MIDIピッチ → GBレジスタ → 周波数スナップ
        freq_snapped = _snap_to_gb_freq(note.freq_hz)

        # ベロシティ → GB 4bitボリューム (0〜1)
        volume = _velocity_to_volume(note.velocity)

        events.append((on_offset,  freq_snapped, duty, volume))
        events.append((off_offset, 0.0,          duty, 0.0))

    # sample_offset で昇順ソート（同一offsetは note-on を先に）
    events.sort(key=lambda e: (e[0], 0 if e[1] > 0 else 1))
    return events


def _notes_to_wave_events(
    notes: list[NoteEvent],
    sr: int,
    wave_ram: np.ndarray,
) -> WaveEvents:
    """
    NoteEvent リスト → CH3 波形チャンネルイベントリスト。

    CH3の周波数式: f = 65536 / (2048 - reg)
    ベース音域（低音）向けのためオクターブ調整を行う。
    """
    events: WaveEvents = []

    for note in notes:
        on_offset  = _sec_to_sample(note.start_sec, sr)
        off_offset = _sec_to_sample(note.end_sec,   sr)

        # CH3の周波数範囲に収まるよう調整（最低32Hz付近〜最高131kHz）
        freq = _adjust_freq_for_ch3(note.freq_hz)
        volume = _velocity_to_volume(note.velocity)

        events.append((on_offset,  freq, wave_ram, volume))
        events.append((off_offset, 0.0,  wave_ram, 0.0))

    events.sort(key=lambda e: (e[0], 0 if e[1] > 0 else 1))
    return events


def _onsets_to_noise_events(
    onsets: list[OnsetEvent],
    sr: int,
) -> NoiseEvents:
    """
    OnsetEvent リスト → CH4 LFSRノイズイベントリスト。

    強度に応じてLFSRパラメータを選択し、
    一定時間後に音量0のイベントを追加してゲートを閉じる。
    """
    events: NoiseEvents = []

    for onset in onsets:
        on_offset  = _sec_to_sample(onset.start_sec, sr)
        off_offset = on_offset + _NOISE_GATE_SAMPLES

        r, s, width7 = NoiseChannel.estimate_params(onset.strength)
        volume = float(np.clip(onset.strength * 1.2, 0.1, 1.0))

        events.append((on_offset,  r, s, width7, volume))
        events.append((off_offset, r, s, width7, 0.0))   # gate off

    events.sort(key=lambda e: (e[0], 0 if e[4] > 0 else 1))
    return events


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _sec_to_sample(sec: float, sr: int) -> int:
    """時刻[秒] → サンプルオフセット（負にならないよう保証）"""
    return max(0, int(round(sec * sr)))


def _snap_to_gb_freq(freq_hz: float) -> float:
    """
    周波数をGBの11bitレジスタ解像度にスナップする。

    これにより「階段状ピッチ」というGB音源特有の音色が再現される。
    """
    if freq_hz <= 0:
        return 0.0
    reg = hz_to_register(freq_hz)
    return register_to_hz(reg)


def _adjust_freq_for_ch3(freq_hz: float) -> float:
    """
    CH3の有効周波数範囲 (~32Hz〜) に収まるようオクターブを調整する。

    低すぎる場合はオクターブ上げ、高すぎる場合は下げる。
    """
    if freq_hz <= 0:
        return 0.0
    # CH3の実用範囲: 65Hz〜4186Hz
    while freq_hz < 65.0:
        freq_hz *= 2.0
    while freq_hz > 4186.0:
        freq_hz /= 2.0
    return freq_hz


def _velocity_to_volume(velocity: int) -> float:
    """
    MIDIベロシティ (0〜127) → GB 4bitボリューム (0.0〜1.0)。

    GB DACは 0〜15 の16段階なので、量子化して戻す。
    最低音量を0.1に設定して完全無音を防ぐ。
    """
    raw = velocity / 127.0
    # 4bit量子化 (0〜15 → 0.0〜1.0)
    quantized = round(raw * 15) / 15.0
    return float(np.clip(quantized, 0.0667, 1.0))  # 1/15 = 最小音量
