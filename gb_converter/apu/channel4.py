"""
CH4: ノイズチャンネル

LFSR (Linear Feedback Shift Register) ベースのノイズ。

GBの正確なLFSRアルゴリズム:
    1. ビット0とビット1をXOR
    2. レジスタ全体を1bit右シフト
    3. XOR結果をビット14（最上位）に代入
    4. 7bitモード時はさらにビット6にも代入

周波数式:
    clock_divider = r == 0 ? 4 : r * 8
    period_cycles = clock_divider * 2^(s+1)
    freq = 4194304 / period_cycles
         = 524288 / max(r, 0.5) / 2^s  (近似)

    ※ s=14,15はLFSRクロック停止（無音）
"""

import numpy as np


def noise_frequency(r: int, s: int) -> float:
    """
    LFSRパラメータから等価周波数[Hz]を計算する。

    r: クロック分周比 (0〜7)
    s: シフト量 (0〜13 が有効, 14/15は無音)
    """
    if s >= 14:
        return 0.0
    clock_divider = 4 if r == 0 else r * 8
    period_cycles = clock_divider * (2 ** (s + 1))
    return 4194304.0 / period_cycles


class NoiseChannel:
    """
    CH4 ノイズチャンネル。

    events: list of (sample_offset, r, s, width7, volume)
        r: 0〜7 クロック分周比
        s: 0〜13 シフトクロック (14/15は無音)
        width7: True=7bit LFSR (周期短い・音程感あるノイズ), False=15bit LFSR (ホワイトノイズ)
        volume: 0.0〜1.0
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def render(
        self,
        events: list[tuple[int, int, int, bool, float]],
        n_samples: int,
    ) -> np.ndarray:
        output = np.zeros(n_samples, dtype=np.float32)
        if not events:
            return output

        events = sorted(events, key=lambda e: e[0])

        for i, (offset, r, s, width7, volume) in enumerate(events):
            end = events[i + 1][0] if i + 1 < len(events) else n_samples
            if offset >= n_samples:
                break
            end = min(end, n_samples)

            freq = noise_frequency(r, s)
            if freq <= 0:
                continue

            n = end - offset
            noise = _generate_lfsr_noise(n, freq, width7, self.sample_rate)
            output[offset:end] += noise * volume

        return output.astype(np.float32)

    @staticmethod
    def estimate_params(onset_strength: float) -> tuple[int, int, bool]:
        """
        打楽器の強度からLFSRパラメータを推定する。

        onset_strength: 0.0〜1.0
        Returns: (r, s, width7)
        """
        if onset_strength > 0.7:
            return 0, 2, True    # ハイハット/スネア: 高周波・7bit（短周期ノイズ）
        elif onset_strength > 0.4:
            return 1, 4, False   # スネア: 中周波・15bit
        else:
            return 4, 6, False   # バスドラム: 低周波・15bit


def _generate_lfsr_noise(
    n_samples: int, freq_hz: float, width7: bool, sample_rate: int
) -> np.ndarray:
    """
    GB仕様に準拠したLFSR疑似ノイズを生成する。

    アルゴリズム:
        XOR = bit0 XOR bit1
        LFSR >>= 1
        LFSR |= XOR << 14
        if width7: LFSR |= XOR << 6  (bit6にも代入)
        出力 = bit0の反転（0=HIGH, 1=LOW → GB DAC反転で0→+1）
    """
    clock_period = max(1, round(sample_rate / freq_hz))

    lfsr = 0x7FFF  # 初期値（全ビット1）
    output = np.zeros(n_samples, dtype=np.float32)
    clock = 0

    for i in range(n_samples):
        if clock == 0:
            # bit0 XOR bit1
            xor = (lfsr ^ (lfsr >> 1)) & 1
            # 右シフトしてbit14に挿入
            lfsr = (lfsr >> 1) | (xor << 14)
            if width7:
                # 7bitモード: bit6にも挿入、下位7bitのみ有効
                lfsr = (lfsr & ~(1 << 6)) | (xor << 6)
                lfsr &= 0x7FFF  # 15bitマスク（bit14まで保持）
            clock = clock_period

        # bit0が0→+1 (LOW=HIGH per GB DAC inversion)、bit0が1→-1
        output[i] = 1.0 if (lfsr & 1) == 0 else -1.0
        clock -= 1

    return output
