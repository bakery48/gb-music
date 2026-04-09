"""
MP3 → ノートイベント変換 (librosa のみ使用)

pop-to-8bit (IEEE 2017) の手法に基づく:
  1. HPSS で調波成分 / 打楽器成分に分離
  2. 調波成分に pYIN でピッチ追跡（有声/無声の信頼度付き）
  3. オンセット検出でノート境界を決定
  4. フレームを NoteEvent に集約

依存ライブラリ: librosa, numpy のみ（TensorFlow 不要）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NoteEvent:
    """1つの音符を表すイベント。"""
    pitch_midi: int    # MIDIノート番号 (0〜127)
    start_sec: float   # 開始時刻 [秒]
    end_sec: float     # 終了時刻 [秒]
    velocity: int      # ベロシティ (0〜127)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def freq_hz(self) -> float:
        """MIDIノート番号 → 周波数[Hz] (A4=440Hz基準)"""
        return 440.0 * (2.0 ** ((self.pitch_midi - 69) / 12.0))


def transcribe(
    audio_path: str,
    min_note_len_sec: float = 0.08,
    min_freq_hz: float = 65.0,
    max_freq_hz: float = 2093.0,
    voiced_threshold: float = 0.35,
    onset_delta: float = 0.07,
) -> list[NoteEvent]:
    """
    音声ファイルを NoteEvent リストに変換する。

    Args:
        audio_path:       入力音声ファイル (.mp3 / .wav など)
        min_note_len_sec: これより短いノートは除去する [秒]
        min_freq_hz:      検出下限周波数 [Hz]  (デフォルト: C2)
        max_freq_hz:      検出上限周波数 [Hz]  (デフォルト: C7)
        voiced_threshold: 有声判定の確率閾値 (0〜1)
        onset_delta:      オンセット検出感度（小さいほど多く検出）

    Returns:
        NoteEvent のリスト（開始時刻順）
    """
    import librosa

    # 1. 音声ロード
    samples, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop = 256

    # 2. HPSS: 調波成分と打楽器成分を分離
    #    調波成分 → ピッチ追跡 / 打楽器成分 → オンセット検出に使う
    harmonic, percussive = librosa.effects.hpss(samples, margin=3.0)

    # 3. pYIN: 調波成分から有声/無声フラグ付きでピッチを推定
    f0, voiced_flag, voiced_prob = librosa.pyin(
        harmonic,
        fmin=min_freq_hz,
        fmax=max_freq_hz,
        sr=sr,
        hop_length=hop,
        fill_na=0.0,
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    # 信頼度が低いフレームを無音扱いにする
    f0 = np.where(voiced_prob >= voiced_threshold, f0, 0.0)

    # 4. オンセット検出: 打楽器成分から発音境界を検出
    onset_env = librosa.onset.onset_strength(
        y=percussive, sr=sr, hop_length=hop
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop,
        backtrack=True,
        delta=onset_delta,
    )
    onset_set = set(onset_frames.tolist())

    # 5. フレームごとの RMS → ベロシティ推定
    rms = librosa.feature.rms(y=harmonic, frame_length=512, hop_length=hop)[0]
    rms_norm = rms / (rms.max() + 1e-8)

    # 6. フレームをノートイベントに集約
    events = _frames_to_notes(
        f0, rms_norm, onset_set,
        sr=sr, hop=hop,
        min_note_len_sec=min_note_len_sec,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
    )

    events.sort(key=lambda e: (e.start_sec, e.pitch_midi))
    return events


def _frames_to_notes(
    f0: np.ndarray,
    rms_norm: np.ndarray,
    onset_set: set[int],
    sr: int,
    hop: int,
    min_note_len_sec: float,
    min_freq_hz: float,
    max_freq_hz: float,
) -> list[NoteEvent]:
    """
    pYIN のフレーム列をノートイベントに集約する。

    ノート境界の決定ルール:
      - オンセット検出フレーム
      - 無音→有声 / 有声→無音 の切り替わり
      - ピッチが半音以上変化したとき
    """
    events: list[NoteEvent] = []

    # 現在のセグメント
    seg_start: int | None = None
    seg_pitches: list[float] = []
    seg_vels: list[float] = []

    def flush(end_frame: int) -> None:
        nonlocal seg_start, seg_pitches, seg_vels
        if seg_start is None or not seg_pitches:
            seg_start = None
            seg_pitches = []
            seg_vels = []
            return

        dur = (end_frame - seg_start) * hop / sr
        if dur >= min_note_len_sec:
            mean_freq = float(np.median(seg_pitches))
            pitch_midi = int(np.clip(
                round(librosa.hz_to_midi(mean_freq)), 0, 127
            ))
            velocity = int(np.clip(np.mean(seg_vels) * 127, 1, 127))
            events.append(NoteEvent(
                pitch_midi=pitch_midi,
                start_sec=seg_start * hop / sr,
                end_sec=end_frame * hop / sr,
                velocity=velocity,
            ))

        seg_start = None
        seg_pitches = []
        seg_vels = []

    import librosa
    prev_midi: int | None = None

    for i, (freq, vol) in enumerate(zip(f0, rms_norm)):
        is_voiced = float(freq) > 0

        # ノート境界: オンセット
        if i in onset_set and seg_start is not None:
            flush(i)

        if is_voiced:
            cur_midi = int(round(librosa.hz_to_midi(float(freq))))

            # ノート境界: 半音以上のピッチジャンプ
            if (seg_start is not None
                    and prev_midi is not None
                    and abs(cur_midi - prev_midi) >= 1):
                flush(i)

            if seg_start is None:
                seg_start = i

            seg_pitches.append(float(freq))
            seg_vels.append(float(vol))
            prev_midi = cur_midi
        else:
            if seg_start is not None:
                flush(i)
            prev_midi = None

    flush(len(f0))
    return events


def print_summary(events: list[NoteEvent]) -> None:
    """ノートイベントの統計を表示する。"""
    if not events:
        print("ノートイベントなし")
        return

    pitches   = [e.pitch_midi for e in events]
    durations = [e.duration_sec for e in events]
    total_dur = events[-1].end_sec - events[0].start_sec

    print(f"ノート数:   {len(events)}")
    print(f"音域:       MIDI {min(pitches)}〜{max(pitches)}"
          f"  ({_midi_name(min(pitches))}〜{_midi_name(max(pitches))})")
    print(f"長さ:       最短 {min(durations):.2f}s  "
          f"最長 {max(durations):.2f}s  平均 {np.mean(durations):.2f}s")
    print(f"演奏時間:   {total_dur:.1f}s")


def _midi_name(n: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{n // 12 - 1}"
