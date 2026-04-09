"""
Step 1: MP3 → ノートイベント変換

basic-pitch (Spotify) がインストールされていればそちらを使い、
なければ librosa の YIN ピッチ追跡にフォールバックする。

  basic-pitch モード: ポリフォニック対応・高精度
  lite モード:       モノフォニック・軽量（basic-pitch不要）
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


def has_basic_pitch() -> bool:
    """basic-pitch が使用可能かチェックする。"""
    try:
        import importlib
        importlib.import_module("basic_pitch")
        return True
    except ImportError:
        return False


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

    basic-pitch が使える場合はそちらを使い、
    使えない場合は自動的に lite モードにフォールバックする。
    """
    if has_basic_pitch():
        return _transcribe_basic_pitch(
            audio_path, onset_threshold, frame_threshold,
            min_note_len_sec, min_freq_hz, max_freq_hz,
        )
    else:
        return _transcribe_lite(audio_path, min_note_len_sec, min_freq_hz, max_freq_hz)


def _transcribe_basic_pitch(
    audio_path: str,
    onset_threshold: float,
    frame_threshold: float,
    min_note_len_sec: float,
    min_freq_hz: float,
    max_freq_hz: float,
) -> list[NoteEvent]:
    """basic-pitch によるポリフォニック変換。"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("root").setLevel(logging.ERROR)

    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model = Model(ICASSP_2022_MODEL_PATH)
    _, _midi, note_events = predict(
        audio_path,
        model,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_len_sec * 1000,
        minimum_frequency=min_freq_hz,
        maximum_frequency=max_freq_hz,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )

    events = []
    for start, end, pitch, amplitude, _ in note_events:
        velocity = int(np.clip(amplitude * 127, 1, 127))
        events.append(NoteEvent(
            pitch_midi=int(pitch),
            start_sec=float(start),
            end_sec=float(end),
            velocity=velocity,
        ))
    events.sort(key=lambda e: (e.start_sec, e.pitch_midi))
    return events


def _transcribe_lite(
    audio_path: str,
    min_note_len_sec: float,
    min_freq_hz: float,
    max_freq_hz: float,
) -> list[NoteEvent]:
    """
    librosa だけで動くライトモードの変換。

    YIN ピッチ追跡 + オンセット検出 でノートの開始・終了を推定する。
    モノフォニック前提だが basic-pitch 不要で軽量。
    """
    import librosa

    samples, sr = librosa.load(audio_path, sr=44100, mono=True)
    hop_length = 512

    # ピッチ検出 (YIN)
    f0 = librosa.yin(
        samples,
        fmin=min_freq_hz,
        fmax=max_freq_hz,
        sr=sr,
        hop_length=hop_length,
    )

    # オンセット検出（ノートの切れ目として使う）
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        delta=0.1,
    )
    onset_set = set(onset_frames.tolist())

    # フレームごとの音量（ベロシティ推定に使用）
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    rms_norm = rms / (rms.max() + 1e-8)

    # フレームをノートイベントにまとめる
    events: list[NoteEvent] = []
    seg_start_frame: int | None = None
    seg_pitch: float = 0.0
    seg_vel_sum: float = 0.0
    seg_frames: int = 0

    def _flush(end_frame: int) -> None:
        nonlocal seg_start_frame, seg_pitch, seg_vel_sum, seg_frames
        if seg_start_frame is None or seg_frames == 0:
            return
        dur = (end_frame - seg_start_frame) * hop_length / sr
        if dur < min_note_len_sec:
            seg_start_frame = None
            seg_frames = 0
            return
        pitch_midi = int(round(librosa.hz_to_midi(seg_pitch / seg_frames)))
        pitch_midi = int(np.clip(pitch_midi, 0, 127))
        velocity   = int(np.clip(seg_vel_sum / seg_frames * 127, 1, 127))
        start_sec  = seg_start_frame * hop_length / sr
        end_sec    = end_frame * hop_length / sr
        events.append(NoteEvent(pitch_midi, start_sec, end_sec, velocity))
        seg_start_frame = None
        seg_frames = 0

    for i, (freq, vol) in enumerate(zip(f0, rms_norm)):
        is_voiced = freq > 0 and min_freq_hz <= freq <= max_freq_hz
        is_onset  = i in onset_set

        if is_onset and seg_start_frame is not None:
            _flush(i)

        if is_voiced:
            if seg_start_frame is None:
                seg_start_frame = i
                seg_pitch = 0.0
                seg_vel_sum = 0.0
                seg_frames = 0
            seg_pitch   += freq
            seg_vel_sum += float(vol)
            seg_frames  += 1
        else:
            if seg_start_frame is not None:
                _flush(i)

    _flush(len(f0))

    events.sort(key=lambda e: (e.start_sec, e.pitch_midi))
    return events


def print_summary(events: list[NoteEvent]) -> None:
    """ノートイベントの統計を表示する（デバッグ用）。"""
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
