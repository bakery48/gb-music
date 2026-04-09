"""
Step 2: ノートイベント → GB 4チャンネルへのスケジューリング

GBの制約:
  - CH1, CH2, CH3 は同時に1音しか発音できない（モノフォニック）
  - CH4 はノイズ専用

振り分けルール:
  - MIDI >= bass_threshold (デフォルト48=C3) → CH1/CH2 候補（メロディ）
  - MIDI <  bass_threshold                   → CH3 候補（ベース）
  - 打楽器検出（オーディオのオンセット解析）→ CH4

CH1/CH2 のスケジューリングはグリーディー法:
  音符を開始時刻順に処理し、CH1が空ければCH1、次にCH2、
  両方埋まっていれば優先度（ベロシティ）の低い方を押し出して代替。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from gb_converter.transcriber import NoteEvent


@dataclass
class OnsetEvent:
    """CH4ノイズチャンネル用の打楽器イベント。"""
    start_sec: float
    strength: float      # 0.0〜1.0
    duration_sec: float = 0.05  # ノイズの持続時間


@dataclass
class ChannelTracks:
    """GB 4チャンネルに振り分けられたノートのセット。"""
    ch1: list[NoteEvent] = field(default_factory=list)   # メロディ（矩形波）
    ch2: list[NoteEvent] = field(default_factory=list)   # ハーモニー（矩形波）
    ch3: list[NoteEvent] = field(default_factory=list)   # ベース（波形）
    ch4: list[OnsetEvent] = field(default_factory=list)  # 打楽器（ノイズ）

    def summary(self) -> str:
        return (
            f"CH1 {len(self.ch1):3d}音  |  "
            f"CH2 {len(self.ch2):3d}音  |  "
            f"CH3 {len(self.ch3):3d}音  |  "
            f"CH4 {len(self.ch4):3d}打"
        )


def schedule(
    notes: list[NoteEvent],
    audio: np.ndarray | None = None,
    sr: int = 44100,
    bass_threshold_midi: int = 48,
) -> ChannelTracks:
    """
    ノートイベントをGB 4チャンネルに振り分ける。

    Args:
        notes:                NoteEvent のリスト（transcriber の出力）
        audio:                打楽器検出用のPCM波形（省略可）
        sr:                   サンプルレート
        bass_threshold_midi:  これ未満のMIDIノートはCH3（ベース）に振る

    Returns:
        ChannelTracks
    """
    if not notes:
        return ChannelTracks()

    # 1. ピッチ域でメロディ / ベースに分類
    melody_notes = [n for n in notes if n.pitch_midi >= bass_threshold_midi]
    bass_notes   = [n for n in notes if n.pitch_midi <  bass_threshold_midi]

    # 2. CH1 / CH2: グリーディースケジューリング
    ch1, ch2 = _schedule_two_mono_channels(melody_notes)

    # 3. CH3: ベース（モノフォニック・グリーディー）
    ch3 = _schedule_mono_channel(bass_notes)

    # 4. CH4: 打楽器検出
    ch4 = _detect_onsets(audio, sr) if audio is not None else []

    return ChannelTracks(ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4)


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _schedule_two_mono_channels(
    notes: list[NoteEvent],
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """
    ノートリストを2つのモノフォニックチャンネル (CH1/CH2) に振り分ける。

    アルゴリズム:
      1. 開始時刻でソート
      2. CH1 が空いていれば CH1 に割り当て
      3. CH2 が空いていれば CH2 に割り当て
      4. 両方埋まっている場合: 現在のチャンネル音符と比較し、
         ベロシティが低い方（重要度が低い音符）を押し出して新音符を挿入

    「空いている」= そのチャンネルの最後の音符が既に終わっている
    """
    ch1: list[NoteEvent] = []
    ch2: list[NoteEvent] = []
    ch1_end = 0.0  # CH1 の現在の発音終了時刻
    ch2_end = 0.0  # CH2 の現在の発音終了時刻

    for note in sorted(notes, key=lambda n: n.start_sec):
        if note.start_sec >= ch1_end:
            ch1.append(note)
            ch1_end = note.end_sec
        elif note.start_sec >= ch2_end:
            ch2.append(note)
            ch2_end = note.end_sec
        else:
            # 両チャンネルが埋まっている場合: ベロシティで競合解決
            # 最後に割り当てた音符と比較して、より重要な音符を優先
            last_ch1 = ch1[-1] if ch1 else None
            last_ch2 = ch2[-1] if ch2 else None

            # CH1の最後の音符よりベロシティが高ければ置き換え
            if last_ch1 and note.velocity > last_ch1.velocity:
                # 古い音符の終了時刻を note.start_sec に短縮
                old = last_ch1
                ch1[-1] = NoteEvent(
                    pitch_midi=old.pitch_midi,
                    start_sec=old.start_sec,
                    end_sec=note.start_sec,
                    velocity=old.velocity,
                )
                ch1.append(note)
                ch1_end = note.end_sec
            elif last_ch2 and note.velocity > last_ch2.velocity:
                old = last_ch2
                ch2[-1] = NoteEvent(
                    pitch_midi=old.pitch_midi,
                    start_sec=old.start_sec,
                    end_sec=note.start_sec,
                    velocity=old.velocity,
                )
                ch2.append(note)
                ch2_end = note.end_sec
            # どちらも重要でなければスキップ（ポリフォニーを切り捨て）

    return ch1, ch2


def _schedule_mono_channel(notes: list[NoteEvent]) -> list[NoteEvent]:
    """
    ノートリストを1つのモノフォニックチャンネルにスケジューリングする。

    重複する音符は開始が早い方を優先（後から来た音は待機/スキップ）。
    """
    result: list[NoteEvent] = []
    current_end = 0.0

    for note in sorted(notes, key=lambda n: n.start_sec):
        if note.start_sec >= current_end:
            result.append(note)
            current_end = note.end_sec
        else:
            # 重複: より重要な音符（ベロシティ高）に切り替える
            if result and note.velocity > result[-1].velocity:
                old = result[-1]
                result[-1] = NoteEvent(
                    pitch_midi=old.pitch_midi,
                    start_sec=old.start_sec,
                    end_sec=note.start_sec,
                    velocity=old.velocity,
                )
                result.append(note)
                current_end = note.end_sec

    return result


def _detect_onsets(audio: np.ndarray, sr: int) -> list[OnsetEvent]:
    """
    librosa を使って打楽器のオンセットを検出し OnsetEvent のリストを返す。
    """
    import librosa

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        delta=0.2,       # 検出感度（高いほど少なく）
    )

    if len(onset_env) == 0 or onset_env.max() == 0:
        return []

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    strengths = onset_env[onset_frames] / onset_env.max()

    return [
        OnsetEvent(start_sec=float(t), strength=float(s))
        for t, s in zip(onset_times, strengths)
    ]
