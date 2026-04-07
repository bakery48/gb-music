"""
gb-music: MP3をゲームボーイ音源に変換するCLIツール

使い方:
    python main.py input.mp3 -o output.wav
    python main.py input.mp3 -o output.wav --mode simple
    python main.py input.mp3 -o output.wav --mode full --duty 2
"""

import sys
import click
import numpy as np

from gb_converter.loader import load_audio, normalize, split_bands
from gb_converter.mixer import mix, apply_gb_lowpass
from gb_converter.exporter import export_wav


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_path",
    default=None,
    help="出力ファイルパス (.wav または .mp3)。未指定時は input_gb.wav",
)
@click.option(
    "--mode",
    type=click.Choice(["simple", "full"]),
    default="simple",
    show_default=True,
    help=(
        "simple: ビットクラッシャー方式（高速）/ "
        "full: チャンネル合成方式（高品質）"
    ),
)
@click.option(
    "--duty",
    type=click.IntRange(0, 3),
    default=2,
    show_default=True,
    help="矩形波のデューティ比 (0=12.5%%, 1=25%%, 2=50%%, 3=75%%)",
)
@click.option(
    "--ch1-gain", default=0.25, show_default=True, help="CH1 ゲイン (0.0〜1.0)"
)
@click.option(
    "--ch2-gain", default=0.25, show_default=True, help="CH2 ゲイン (0.0〜1.0)"
)
@click.option(
    "--ch3-gain", default=0.30, show_default=True, help="CH3 ゲイン (0.0〜1.0)"
)
@click.option(
    "--ch4-gain", default=0.20, show_default=True, help="CH4 ゲイン (0.0〜1.0)"
)
@click.option(
    "--no-lowpass", is_flag=True, default=False,
    help="GB風ローパスフィルタを無効にする",
)
def main(
    input_path: str,
    output_path: str | None,
    mode: str,
    duty: int,
    ch1_gain: float,
    ch2_gain: float,
    ch3_gain: float,
    ch4_gain: float,
    no_lowpass: bool,
) -> None:
    """MP3ファイルをゲームボーイ音源に変換する。"""

    if output_path is None:
        import os
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base}_gb.wav"

    click.echo(f"入力: {input_path}")
    click.echo(f"出力: {output_path}")
    click.echo(f"モード: {mode}")

    click.echo("音声ファイルを読み込み中...")
    samples, sr = load_audio(input_path)
    samples = normalize(samples)
    n_samples = len(samples)

    gains = {
        "ch1": ch1_gain,
        "ch2": ch2_gain,
        "ch3": ch3_gain,
        "ch4": ch4_gain,
    }

    if mode == "simple":
        output = _convert_simple(samples, sr)
    else:
        output = _convert_full(samples, sr, n_samples, duty, gains)

    if not no_lowpass:
        click.echo("GB風ローパスフィルタを適用中...")
        output = apply_gb_lowpass(output, sr)

    # 最終ピーク正規化
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.95

    click.echo(f"WAVファイルに書き出し中: {output_path}")
    if output_path.endswith(".mp3"):
        from gb_converter.exporter import export_mp3
        export_mp3(output, sr, output_path)
    else:
        export_wav(output, sr, output_path)

    click.echo("完了!")


def _convert_simple(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Phase 1: ビットクラッシャー方式

    4bit量子化 + ダウンサンプリングによるGB風エフェクト。
    """
    from gb_converter.apu.channel1 import quantize_4bit

    click.echo("ビットクラッシャー変換中...")

    # 4bit量子化
    quantized = quantize_4bit(samples)

    # GBのサンプリングレート相当 (約8192Hz) でダウンサンプリング → アップサンプリング
    gb_sr = 8192
    factor = sr // gb_sr
    if factor > 1:
        # ダウンサンプリング（間引き）
        downsampled = quantized[::factor]
        # アップサンプリング（最近傍補間）
        upsampled = np.repeat(downsampled, factor)
        # 長さを元に合わせる
        if len(upsampled) < len(samples):
            upsampled = np.pad(upsampled, (0, len(samples) - len(upsampled)))
        quantized = upsampled[: len(samples)]

    return quantized.astype(np.float32)


def _convert_full(
    samples: np.ndarray,
    sr: int,
    n_samples: int,
    duty: int,
    gains: dict[str, float],
) -> np.ndarray:
    """
    Phase 2: チャンネル合成方式

    ピッチ/オンセット検出 → 各チャンネル合成 → ミックス。
    """
    from gb_converter.analyzer import (
        detect_pitches,
        detect_onsets,
        split_melody_lines,
        pitch_events,
        onset_events,
    )
    from gb_converter.apu import PulseChannel, PulseChannel2, WaveChannel, NoiseChannel

    click.echo("ピッチ検出中 (時間がかかる場合があります)...")
    times, freqs, magnitudes = detect_pitches(samples, sr)

    click.echo("オンセット検出中...")
    onset_times, onset_strengths = detect_onsets(samples, sr)

    click.echo("メロディラインを分割中...")
    lines = split_melody_lines(times, freqs, magnitudes, n_channels=2)

    # CH1: 高音域メロディ
    click.echo("CH1 (矩形波) 合成中...")
    ch1_events = pitch_events(*lines[0], sr=sr, duty=duty)
    ch1 = PulseChannel(sr).render(ch1_events, n_samples)

    # CH2: 低音域メロディ
    click.echo("CH2 (矩形波) 合成中...")
    ch2_events = pitch_events(*lines[1], sr=sr, duty=duty)
    ch2 = PulseChannel2(sr).render(ch2_events, n_samples)

    # CH3: 低音域をカスタム波形で再生
    click.echo("CH3 (波形) 合成中...")
    bands = split_bands(samples, sr)
    wave_ram = WaveChannel.sine_wave_ram()
    ch3_events = _make_wave_events(bands["low"], sr, wave_ram)
    ch3 = WaveChannel(sr).render(ch3_events, n_samples)

    # CH4: ノイズ（打楽器）
    click.echo("CH4 (ノイズ) 合成中...")
    ch4_events = onset_events(onset_times, onset_strengths, sr)
    ch4 = NoiseChannel(sr).render(ch4_events, n_samples)

    click.echo("チャンネルをミックス中...")
    return mix(ch1, ch2, ch3, ch4, gains)


def _make_wave_events(
    low_band: np.ndarray,
    sr: int,
    wave_ram: np.ndarray,
    segment_sec: float = 0.1,
) -> list[tuple[int, float, np.ndarray, float]]:
    """
    低音域のエンベロープからCH3イベントを生成する。

    低音域のRMSをボリュームとして、固定周波数でWave RAMを再生する。
    """
    seg_len = int(sr * segment_sec)
    events = []
    n = len(low_band)

    for i in range(0, n, seg_len):
        seg = low_band[i: i + seg_len]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms > 0.01:
            # 低音域を代表する周波数 (固定: ベース音A2=110Hz付近)
            freq = 110.0
            volume = float(np.clip(rms * 4.0, 0.0, 1.0))
            events.append((i, freq, wave_ram, volume))

    return events


if __name__ == "__main__":
    main()
