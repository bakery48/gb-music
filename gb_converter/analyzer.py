"""
音楽解析モジュール

MP3のPCM波形から:
  - ピッチ（基本周波数）を時系列で検出
  - 打楽器のオンセット（発音タイミング）を検出
  - 各チャンネルへのマッピング情報を生成する
"""

import numpy as np
import librosa


def detect_pitches(
    samples: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 2093.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ピッチ（基本周波数）を検出する。

    Returns:
        (times, freqs, magnitudes)
        times: 各フレームの時刻[s]
        freqs: 各フレームの周波数[Hz] (0=無音)
        magnitudes: 各フレームのマグニチュード (0〜1)
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # 無声区間は0Hzとする
    freqs = np.where(voiced_flag, f0, 0.0)
    freqs = np.nan_to_num(freqs, nan=0.0)

    return times, freqs.astype(np.float32), voiced_prob.astype(np.float32)


def detect_onsets(
    samples: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    打楽器のオンセット（発音タイミング）を検出する。

    Returns:
        (onset_times, onset_strengths)
        onset_times: オンセット時刻[s]
        onset_strengths: 各オンセットの強度 (0〜1)
    """
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # 正規化した強度を取得
    if len(onset_env) == 0 or onset_env.max() == 0:
        strengths = np.zeros(len(onset_times), dtype=np.float32)
    else:
        strengths = onset_env[onset_frames] / onset_env.max()
        strengths = strengths.astype(np.float32)

    return onset_times, strengths


def split_melody_lines(
    times: np.ndarray,
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    n_channels: int = 2,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    ピッチ列を複数のメロディラインに分割する（CH1/CH2への振り分け）。

    現在は単純に1本のメロディラインをそのままCH1に割り当て、
    低音域をCH2に振り分ける簡易実装。

    Returns:
        [(times, freqs, magnitudes), ...] × n_channels
    """
    if n_channels == 1:
        return [(times, freqs, magnitudes)]

    # 中央値より高い周波数 → CH1、低い周波数 → CH2
    median_freq = np.median(freqs[freqs > 0]) if np.any(freqs > 0) else 440.0

    ch1_freqs = np.where(freqs >= median_freq, freqs, 0.0)
    ch2_freqs = np.where((freqs > 0) & (freqs < median_freq), freqs, 0.0)

    return [
        (times, ch1_freqs.astype(np.float32), magnitudes),
        (times, ch2_freqs.astype(np.float32), magnitudes),
    ]


def pitch_events(
    times: np.ndarray,
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    sr: int,
    hop_length: int = 512,
    duty: int = 2,
) -> list[tuple[int, float, int, float]]:
    """
    ピッチ列をAPUチャンネルのイベントリストに変換する。

    Returns:
        [(sample_offset, freq_hz, duty, volume), ...]
    """
    events = []
    prev_freq = -1.0

    for i, (t, freq, mag) in enumerate(zip(times, freqs, magnitudes)):
        sample_offset = int(t * sr)
        volume = float(np.clip(mag * 1.5, 0.0, 1.0))

        if freq != prev_freq:
            events.append((sample_offset, float(freq), duty, volume))
            prev_freq = freq

    return events


def onset_events(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    sr: int,
) -> list[tuple[int, int, int, bool, float]]:
    """
    オンセット列をCH4ノイズチャンネルのイベントリストに変換する。

    Returns:
        [(sample_offset, r, s, width7, volume), ...]
    """
    from .apu.channel4 import NoiseChannel

    events = []
    for t, strength in zip(onset_times, onset_strengths):
        sample_offset = int(t * sr)
        r, s, width7 = NoiseChannel.estimate_params(float(strength))
        volume = float(np.clip(strength, 0.1, 1.0))
        events.append((sample_offset, r, s, width7, volume))

    return events
