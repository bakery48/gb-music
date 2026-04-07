"""
4チャンネルミキサー

GB APUの4チャンネルを合成して最終的なモノラル出力を生成する。

実際のGBではチャンネルごとにボリューム設定があり、左右のパンが可能だが、
ここではモノラルミックスに簡略化する。
"""

import numpy as np


# 各チャンネルのデフォルトゲイン (GB実機に近い配分)
DEFAULT_GAINS = {
    "ch1": 0.25,
    "ch2": 0.25,
    "ch3": 0.30,
    "ch4": 0.20,
}


def mix(
    ch1: np.ndarray | None,
    ch2: np.ndarray | None,
    ch3: np.ndarray | None,
    ch4: np.ndarray | None,
    gains: dict[str, float] | None = None,
) -> np.ndarray:
    """
    4チャンネルをモノラルにミックスする。

    Args:
        ch1〜ch4: 各チャンネルのfloat32配列 (Noneは無音)
        gains: チャンネルごとのゲイン設定

    Returns:
        ミックスしたfloat32配列
    """
    if gains is None:
        gains = DEFAULT_GAINS

    channels = {"ch1": ch1, "ch2": ch2, "ch3": ch3, "ch4": ch4}

    # 最長のサンプル数を取得
    n_samples = max(
        len(c) for c in channels.values() if c is not None
    )

    output = np.zeros(n_samples, dtype=np.float32)

    for name, ch in channels.items():
        if ch is None:
            continue
        gain = gains.get(name, 0.25)
        # 長さが異なる場合は短い方に合わせる
        length = min(len(ch), n_samples)
        output[:length] += ch[:length] * gain

    # ピーク正規化
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak

    return output


def apply_gb_lowpass(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    GB実機のアナログ出力回路を模したローパスフィルタを適用する。

    GBのDACとアンプ回路は約20kHz以上をロールオフする特性を持つが、
    ここでは8kHzのローパスでチップチューン感を強調する。
    """
    from scipy.signal import butter, sosfilt

    cutoff = 8000.0
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    filtered = sosfilt(sos, samples)
    return filtered.astype(np.float32)
