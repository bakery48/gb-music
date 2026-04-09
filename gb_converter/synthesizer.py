"""
Step 4: GB APU 合成 + ミックス

note_mapper が生成した APUイベントリストを受け取り、
各チャンネルを合成してモノラル出力を生成する。

パイプライン:
  CH1 events → PulseChannel.render()  ─┐
  CH2 events → PulseChannel2.render() ─┤
  CH3 events → WaveChannel.render()   ─┼─ mixer.mix() → LPF → WAV
  CH4 events → NoiseChannel.render()  ─┘
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gb_converter.apu import PulseChannel, PulseChannel2, WaveChannel, NoiseChannel
from gb_converter.mixer import mix, apply_gb_lowpass, DEFAULT_GAINS
from gb_converter.note_mapper import PulseEvents, WaveEvents, NoiseEvents


@dataclass
class GainConfig:
    """各チャンネルのゲイン設定。"""
    ch1: float = 0.30   # メロディ（主役）
    ch2: float = 0.22   # ハーモニー（少し控えめ）
    ch3: float = 0.25   # ベース
    ch4: float = 0.18   # 打楽器（主張しすぎない）

    def to_dict(self) -> dict[str, float]:
        return {"ch1": self.ch1, "ch2": self.ch2, "ch3": self.ch3, "ch4": self.ch4}


def synthesize(
    ch1_events: PulseEvents,
    ch2_events: PulseEvents,
    ch3_events: WaveEvents,
    ch4_events: NoiseEvents,
    n_samples: int,
    sr: int,
    gains: GainConfig | None = None,
    apply_lowpass: bool = True,
) -> np.ndarray:
    """
    4チャンネルのAPUイベントを合成してモノラルPCMを返す。

    Args:
        ch1_events〜ch4_events: note_mapper.map_to_apu_events() の出力
        n_samples:    出力サンプル数
        sr:           サンプルレート [Hz]
        gains:        チャンネルゲイン設定（省略時はデフォルト値）
        apply_lowpass: GB風ローパスフィルタを適用するか

    Returns:
        float32 モノラル波形 (-1.0〜1.0)
    """
    if gains is None:
        gains = GainConfig()

    # 各チャンネルを合成
    ch1_audio = PulseChannel(sr).render(ch1_events, n_samples)
    ch2_audio = PulseChannel2(sr).render(ch2_events, n_samples)
    ch3_audio = WaveChannel(sr).render(ch3_events, n_samples)
    ch4_audio = NoiseChannel(sr).render(ch4_events, n_samples)

    # ミックス
    output = mix(ch1_audio, ch2_audio, ch3_audio, ch4_audio, gains.to_dict())

    # GB風ローパスフィルタ
    if apply_lowpass:
        output = apply_gb_lowpass(output, sr)

    # ピーク正規化（クリッピング防止）
    peak = np.max(np.abs(output))
    if peak > 0:
        output = (output / peak * 0.95).astype(np.float32)

    return output


def render_channel_previews(
    ch1_events: PulseEvents,
    ch2_events: PulseEvents,
    ch3_events: WaveEvents,
    ch4_events: NoiseEvents,
    n_samples: int,
    sr: int,
) -> dict[str, np.ndarray]:
    """
    各チャンネルを個別に合成して返す（デバッグ・ミックス調整用）。

    Returns:
        {"ch1": array, "ch2": array, "ch3": array, "ch4": array}
    """
    return {
        "ch1": PulseChannel(sr).render(ch1_events, n_samples),
        "ch2": PulseChannel2(sr).render(ch2_events, n_samples),
        "ch3": WaveChannel(sr).render(ch3_events, n_samples),
        "ch4": NoiseChannel(sr).render(ch4_events, n_samples),
    }
