"""
Step 1: MP3 → ノートイベント変換

basic-pitch (Spotify) を使ってMP3をポリフォニック対応でMIDIノートに変換する。
出力はシンプルな NoteEvent のリストで、後段のスケジューラーが消費する。
"""

from __future__ import annotations

import warnings
import os
import numpy as np
from dataclasses import dataclass


@dataclass
class NoteEvent:
    """1つの音符を表すイベント。"""
    pitch_midi: int      # MIDIノート番号 (0〜127)
    start_sec: float     # 開始時刻 [秒]
    end_sec: float       # 終了時刻 [秒]
    velocity: int        # ベロシティ (0〜127)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def freq_hz(self) -> float:
        """MIDIノート番号 → 周波数[Hz] (A4=440Hz基準)"""
        return 440.0 * (2.0 ** ((self.pitch_midi - 69) / 12.0))


def transcribe(
    audio_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_len_sec: float = 0.05,
    min_freq_hz: float = 65.0,
    max_freq_hz: float = 2093.0,
) -> list[NoteEvent]:
    """
    音声ファイルをノートイベントのリストに変換する。

    Args:
        audio_path:       入力音声ファイルパス (.mp3, .wav など)
        onset_threshold:  発音検出の閾値 (0〜1、高いほど厳しい)
        frame_threshold:  フレーム持続の閾値 (0〜1)
        min_note_len_sec: これより短いノートは除去する [秒]
        min_freq_hz:      この周波数以下のノートは除去する [Hz]
        max_freq_hz:      この周波数以上のノートは除去する [Hz]

    Returns:
        NoteEvent のリスト（開始時刻順）
    """
    # TFの余分なログを抑制
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore")

    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model = Model(ICASSP_2022_MODEL_PATH)

    _, midi_data, note_events = predict(
        audio_path,
        model,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_len_sec * 1000,  # basic-pitchはms単位
        minimum_frequency=min_freq_hz,
        maximum_frequency=max_freq_hz,
        multiple_pitch_bends=False,
        melodia_trick=True,   # ボーカルメロディ強調
    )

    # note_events: [(start_sec, end_sec, pitch_midi, amplitude, pitch_bends), ...]
    events = []
    for start, end, pitch, amplitude, _ in note_events:
        velocity = int(np.clip(amplitude * 127, 1, 127))
        events.append(NoteEvent(
            pitch_midi=int(pitch),
            start_sec=float(start),
            end_sec=float(end),
            velocity=velocity,
        ))

    # 開始時刻順にソート
    events.sort(key=lambda e: (e.start_sec, e.pitch_midi))
    return events


def print_summary(events: list[NoteEvent]) -> None:
    """ノートイベントの統計を表示する（デバッグ用）。"""
    if not events:
        print("ノートイベントなし")
        return

    pitches = [e.pitch_midi for e in events]
    durations = [e.duration_sec for e in events]
    total_dur = events[-1].end_sec - events[0].start_sec

    print(f"ノート数:       {len(events)}")
    print(f"音域:           MIDI {min(pitches)}〜{max(pitches)}")
    print(f"               ({_midi_name(min(pitches))}〜{_midi_name(max(pitches))})")
    print(f"長さ分布:       最短 {min(durations):.2f}s  最長 {max(durations):.2f}s  "
          f"平均 {np.mean(durations):.2f}s")
    print(f"演奏時間:       {total_dur:.1f}s")


def _midi_name(n: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{n // 12 - 1}"
