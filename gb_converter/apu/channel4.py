"""
CH4: ノイズチャンネル

LFSR (Linear Feedback Shift Register) ベースのホワイト/ピンクノイズ。

GBの実際のLFSRパラメータ:
    - クロック分周比 (r): 0〜7
    - シフトクロックフリケンシー (s): 0〜13
    - カウンタステップ幅 (width): 15bit or 7bit
    周波数: f = 524288 / max(r, 0.5) / 2^(s+1)
"""

import numpy as np


def noise_frequency(r: int, s: int) -> float:
    """LFSRのパラメータから等価周波数[Hz]を計算する。"""
    divisor = max(r, 0.5)
    return 524288.0 / divisor / (2 ** (s + 1))


class NoiseChannel:
    """
    CH4 ノイズチャンネル。

    events: list of (sample_offset, r, s, width7, volume)
        r: 0〜7 クロック分周比
        s: 0〜13 シフトクロック
        width7: True=7bit LFSR (高音域ノイズ), False=15bit LFSR (ホワイトノイズ)
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

            n = end - offset
            freq = noise_frequency(r, s)
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
        # 強いオンセット → シャープなノイズ (高周波, 7bit LFSR)
        # 弱いオンセット → ソフトなノイズ (低周波, 15bit LFSR)
        if onset_strength > 0.7:
            return 0, 2, True    # スネア/ハイハット系
        elif onset_strength > 0.4:
            return 1, 4, False   # 中間
        else:
            return 4, 7, False   # バスドラム系


def _generate_lfsr_noise(
    n_samples: int, freq_hz: float, width7: bool, sample_rate: int
) -> np.ndarray:
    """LFSR疑似ノイズを生成する。"""
    if freq_hz <= 0:
        return np.zeros(n_samples, dtype=np.float32)

    # LFSRのクロック周期（サンプル単位）
    clock_period = max(1, round(sample_rate / freq_hz))

    # 簡略化したLFSR実装
    if width7:
        mask = 0x7F
        tap = 0x60      # bit6 XOR bit5
    else:
        mask = 0x7FFF
        tap = 0x6000    # bit14 XOR bit13

    lfsr = 0x7FFF
    output = np.zeros(n_samples, dtype=np.float32)
    clock = 0

    for i in range(n_samples):
        if clock == 0:
            feedback = bin(lfsr & tap).count("1") % 2
            lfsr = ((lfsr >> 1) | (feedback << (6 if width7 else 14))) & mask
            clock = clock_period
        output[i] = 1.0 if (lfsr & 1) else -1.0
        clock -= 1

    return output
