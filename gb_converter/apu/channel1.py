"""
CH1: 矩形波チャンネル（周波数スイープ付き）

GB周波数レジスタとHzの関係（正確な式）:
    f_Hz = 4194304 / (8 * (2048 - register_value))
         = 524288 / (2048 - register_value)
    register_value = 2048 - 524288 / f_Hz   (11bit: 0〜2047)

GBのDAC特性:
    デジタル値 0  → アナログ +1（最大振幅）
    デジタル値 15 → アナログ −1（最小振幅）
    ※通常の符号とは逆転している
"""

import numpy as np


# GBのデューティ比パターン（8ステップの正確なビットパターン）
# 各要素はそのステップがHIGH(+1)かLOW(-1)かを示す
# HIGH時にDAC出力が下がる（反転）ため、WAVEとしてはビット0=LOW、1=HIGH
_DUTY_PATTERNS = {
    0: [0, 0, 0, 0, 0, 0, 0, 1],  # 12.5%: 00000001
    1: [1, 0, 0, 0, 0, 0, 0, 1],  # 25%:   10000001
    2: [1, 0, 0, 0, 0, 1, 1, 1],  # 50%:   10000111
    3: [0, 1, 1, 1, 1, 1, 1, 0],  # 75%:   01111110
}

# デューティ比パターンを位相計算用の閾値に変換
# (HIGHステップの割合 = duty ratio)
_DUTY_RATIOS = {0: 0.125, 1: 0.25, 2: 0.50, 3: 0.75}

# GB DAC: HIGH=LOW、LOW=HIGH なので波形を反転させる係数
_DAC_SIGN = -1.0


def hz_to_register(freq_hz: float) -> int:
    """周波数[Hz]をGBの11bitレジスタ値に変換する。範囲外はクランプ。"""
    if freq_hz <= 0:
        return 0
    reg = round(2048 - 524288 / freq_hz)
    return int(np.clip(reg, 0, 2047))


def register_to_hz(reg: int) -> float:
    """GBの11bitレジスタ値を周波数[Hz]に変換する。"""
    if reg >= 2048:
        return 0.0
    return 524288.0 / (2048 - reg)


class PulseChannel:
    """
    CH1 矩形波ジェネレータ（スイープ付き）。

    使い方:
        ch = PulseChannel(sample_rate=44100)
        audio = ch.render(events, n_samples)

    events: list of (sample_offset, freq_hz, duty, volume)
        duty: 0〜3  (0=12.5%, 1=25%, 2=50%, 3=75%)
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

        events = sorted(events, key=lambda e: e[0])
        t = np.arange(n_samples, dtype=np.float64) / self.sample_rate

        for i, (offset, freq_hz, duty, volume) in enumerate(events):
            end = events[i + 1][0] if i + 1 < len(events) else n_samples
            if offset >= n_samples:
                break
            end = min(end, n_samples)

            seg_t = t[offset:end] - t[offset]
            duty_ratio = _DUTY_RATIOS.get(duty, 0.5)
            wave = _pulse_wave(seg_t, freq_hz, duty_ratio)
            output[offset:end] += wave * volume

        output = quantize_4bit(output)
        return output

    @staticmethod
    def freq_to_nearest_gb(freq_hz: float) -> float:
        """GBのレジスタ解像度に合わせて最近傍の周波数に丸める。"""
        reg = hz_to_register(freq_hz)
        return register_to_hz(reg)


def _pulse_wave(t: np.ndarray, freq_hz: float, duty: float) -> np.ndarray:
    """
    矩形波を生成する。GB DACの反転特性を適用済み。

    duty比より低い位相 → LOW（GB DAC反転でアナログ+1）
    duty比より高い位相 → HIGH（GB DAC反転でアナログ-1）
    """
    if freq_hz <= 0:
        return np.zeros_like(t, dtype=np.float32)
    phase = (t * freq_hz) % 1.0
    # GB DAC反転: duty区間はLOW(-1)、それ以外はHIGH(+1)
    # ※ 音楽的には50%デューティは対称なので実質同じだが、
    #    12.5%/25%/75%では波形の向きが変わる
    raw = np.where(phase < duty, -1.0, 1.0)
    return raw.astype(np.float32)


def quantize_4bit(samples: np.ndarray) -> np.ndarray:
    """
    float32[-1,1]を符号付き4bit(−8〜7)に量子化して再び[-1,1]に戻す。

    符号付きにすることで 0.0 → 0.0 のマッピングを保証する。
    """
    clipped = np.clip(samples, -1.0, 1.0)
    quantized = np.round(clipped * 8.0)
    quantized = np.clip(quantized, -8.0, 7.0)
    return (quantized / 8.0).astype(np.float32)
