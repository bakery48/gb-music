"""
CH2: 矩形波チャンネル（スイープなし）

CH1と同じ矩形波だが周波数スイープ機能を持たない。
"""

import numpy as np
from .channel1 import PulseChannel, quantize_4bit


class PulseChannel2(PulseChannel):
    """
    CH2 矩形波ジェネレータ（スイープなし）。

    CH1と同一のレンダリングロジックを使用。
    スイープが必要な場合はCH1を使うこと。
    """
    pass
