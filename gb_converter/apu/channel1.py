"""
CH1: 矩形波チャンネル（周波数スイープ付き）

GB周波数レジスタとHzの関係:
    f_Hz = 131072 / (2048 - register_value)
    register_value = 2048 - 131072 / f_Hz   (11bit: 0〜2047)
"""

import numpy as np


# デューティ比パターン (12.5% / 25% / 50% / 75%)
DUTY_CYCLES = {
    0: 0.125,
    1: 0.25,
    2: 0.50,
    3: 0.75,
}


def hz_to_register(freq_hz: float) -> int:
    """周波数[Hz]をGBの11bitレジスタ値に変換する。範囲外はクランプ。"""
    if freq_hz <= 0:
        return 0
    reg = round(2048 - 131072 / freq_hz)
    return int(np.clip(reg, 0, 2047))


def register_to_hz(reg: int) -> float:
    """GBの11bitレジスタ値を周波数[Hz]に変換する。"""
    if reg >= 2048:
        return 0.0
    return 131072 / (2048 - reg)


class PulseChannel:
    """
    CH1 矩形波ジェネレータ（スイープ付き）。

    使い方:
        ch = PulseChannel(sample_rate=44100)
        audio = ch.render(events, n_samples)

    events: list of (sample_offset, freq_hz, duty, volume)
        duty: 0〜3
        volume: 0.0〜1.0
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def render(
        self,
        events: list[tuple[int, float, int, float]],
        n_samples: int,
    ) -> np.ndarray:
        """
        イベントリストをもとに矩形波を合成してfloat32配列を返す。

        Args:
            events: [(offset, freq_hz, duty, volume), ...]
            n_samples: 出力サンプル数
        """
        output = np.zeros(n_samples, dtype=np.float32)
        if not events:
            return output

        # イベントを時間順にソート
        events = sorted(events, key=lambda e: e[0])

        t = np.arange(n_samples, dtype=np.float64) / self.sample_rate

        for i, (offset, freq_hz, duty, volume) in enumerate(events):
            end = events[i + 1][0] if i + 1 < len(events) else n_samples
            if offset >= n_samples:
                break
            end = min(end, n_samples)

            seg_t = t[offset:end] - t[offset]
            duty_ratio = DUTY_CYCLES.get(duty, 0.5)
            wave = _pulse_wave(seg_t, freq_hz, duty_ratio)
            output[offset:end] += wave * volume

        # 4bitクランプ (GB音源は4bit DAC)
        output = quantize_4bit(output)
        return output

    @staticmethod
    def freq_to_nearest_gb(freq_hz: float) -> float:
        """GBのレジスタ解像度に合わせて最近傍の周波数に丸める。"""
        reg = hz_to_register(freq_hz)
        return register_to_hz(reg)


def _pulse_wave(t: np.ndarray, freq_hz: float, duty: float) -> np.ndarray:
    """矩形波を生成する (-1 or +1)。"""
    if freq_hz <= 0:
        return np.zeros_like(t)
    phase = (t * freq_hz) % 1.0
    return np.where(phase < duty, 1.0, -1.0).astype(np.float32)


def quantize_4bit(samples: np.ndarray) -> np.ndarray:
    """float32[-1,1]を4bit(16レベル)に量子化して再び[-1,1]に戻す。"""
    clipped = np.clip(samples, -1.0, 1.0)
    levels = 16
    quantized = np.round((clipped + 1.0) / 2.0 * (levels - 1))
    quantized = np.clip(quantized, 0, levels - 1)
    return (quantized / (levels - 1) * 2.0 - 1.0).astype(np.float32)
