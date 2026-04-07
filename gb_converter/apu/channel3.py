"""
CH3: 波形チャンネル（カスタム波形）

Wave RAM: 32ニブル (4bit × 32サンプル) のカスタム波形を繰り返し再生する。
周波数レジスタ: f_Hz = 65536 / (2048 - register_value)
"""

import numpy as np


def hz_to_register_wave(freq_hz: float) -> int:
    """周波数[Hz]をCH3の11bitレジスタ値に変換する。"""
    if freq_hz <= 0:
        return 0
    reg = round(2048 - 65536 / freq_hz)
    return int(np.clip(reg, 0, 2047))


def make_wave_from_audio(samples: np.ndarray) -> np.ndarray:
    """
    任意のオーディオ波形から32ニブルのWave RAMデータを生成する。

    Args:
        samples: 1周期分の波形 (float32, -1〜1)

    Returns:
        32要素の整数配列 (0〜15)
    """
    if len(samples) == 0:
        return np.full(32, 8, dtype=np.uint8)

    # 32点にリサンプル
    indices = np.linspace(0, len(samples) - 1, 32)
    resampled = np.interp(indices, np.arange(len(samples)), samples)

    # 0〜15の4bitに量子化
    clipped = np.clip(resampled, -1.0, 1.0)
    nibbles = np.round((clipped + 1.0) / 2.0 * 15.0)
    return nibbles.astype(np.uint8)


class WaveChannel:
    """
    CH3 カスタム波形チャンネル。

    Wave RAMの波形を指定周波数で繰り返し再生する。

    events: list of (sample_offset, freq_hz, wave_ram, volume)
        wave_ram: 32要素の uint8配列 (0〜15)
        volume: 0.0〜1.0
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def render(
        self,
        events: list[tuple[int, float, np.ndarray, float]],
        n_samples: int,
    ) -> np.ndarray:
        output = np.zeros(n_samples, dtype=np.float32)
        if not events:
            return output

        events = sorted(events, key=lambda e: e[0])

        for i, (offset, freq_hz, wave_ram, volume) in enumerate(events):
            end = events[i + 1][0] if i + 1 < len(events) else n_samples
            if offset >= n_samples:
                break
            end = min(end, n_samples)

            n = end - offset
            wave = _render_wave_ram(wave_ram, freq_hz, n, self.sample_rate)
            output[offset:end] += wave * volume

        return output.astype(np.float32)

    @staticmethod
    def sine_wave_ram() -> np.ndarray:
        """サイン波のWave RAM (デフォルト)"""
        t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        sine = np.sin(t)
        return make_wave_from_audio(sine)


def _render_wave_ram(
    wave_ram: np.ndarray, freq_hz: float, n_samples: int, sample_rate: int
) -> np.ndarray:
    """Wave RAMを指定周波数で繰り返し再生してfloat32配列を返す。"""
    if freq_hz <= 0 or len(wave_ram) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    # Wave RAMを -1〜1 に正規化
    wave_normalized = wave_ram.astype(np.float32) / 15.0 * 2.0 - 1.0

    # 1周期のサンプル数
    period_samples = sample_rate / freq_hz
    t = np.arange(n_samples, dtype=np.float64)
    # Wave RAMの32サンプルを周期に対応させてインデックスを計算
    indices = (t / period_samples * 32).astype(np.int64) % 32
    return wave_normalized[indices].astype(np.float32)
