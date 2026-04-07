"""MP3ファイルの読み込みと前処理"""

import numpy as np
import librosa


# GBの出力サンプルレート相当（内部処理用）
GB_SAMPLE_RATE = 44100


def load_audio(path: str, sample_rate: int = GB_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    音声ファイルを読み込んでモノラルPCMに変換する。

    Returns:
        (samples, sample_rate): float32のモノラル波形と使用サンプルレート
    """
    samples, sr = librosa.load(path, sr=sample_rate, mono=True)
    return samples.astype(np.float32), sr


def normalize(samples: np.ndarray) -> np.ndarray:
    """ピーク正規化 (-1.0 〜 1.0)"""
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    return samples / peak


def split_bands(
    samples: np.ndarray, sr: int
) -> dict[str, np.ndarray]:
    """
    周波数帯域ごとにバンド分割する。

    Returns:
        {
            "low":  低音域 (〜250Hz)     → CH3 波形チャンネル用
            "mid":  中音域 (250〜2000Hz) → CH1/CH2 矩形波用
            "high": 高音域 (2000Hz〜)    → CH1/CH2 矩形波用
        }
    """
    from scipy.signal import butter, sosfilt

    def bandpass(data: np.ndarray, low: float, high: float) -> np.ndarray:
        sos = butter(4, [low, high], btype="band", fs=sr, output="sos")
        return sosfilt(sos, data)

    def lowpass(data: np.ndarray, cutoff: float) -> np.ndarray:
        sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
        return sosfilt(sos, data)

    def highpass(data: np.ndarray, cutoff: float) -> np.ndarray:
        sos = butter(4, cutoff, btype="high", fs=sr, output="sos")
        return sosfilt(sos, data)

    return {
        "low": lowpass(samples, 250.0),
        "mid": bandpass(samples, 250.0, 2000.0),
        "high": highpass(samples, 2000.0),
    }
