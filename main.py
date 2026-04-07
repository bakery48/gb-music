"""
gb-music: MP3をゲームボーイ音源に変換するCLIツール

使い方:
    python main.py input.mp3 -o output.wav
    python main.py input.mp3 -o output.wav --mode simple
    python main.py input.mp3 -o output.wav --mode full --duty 2
"""

import os
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
    help="出力ファイルパス (.wav または .mp3)。未指定時は <input>_gb.wav",
)
@click.option(
    "--mode",
    type=click.Choice(["simple", "full"]),
    default="simple",
    show_default=True,
    help=(
        "simple: ピッチ追跡→矩形波合成（高速・推奨）/ "
        "full: 4チャンネル合成方式（低速・高品質）"
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
    "--ch1-gain", default=0.30, show_default=True, help="CH1 ゲイン (0.0〜1.0)"
)
@click.option(
    "--ch2-gain", default=0.25, show_default=True, help="CH2 ゲイン (0.0〜1.0)"
)
@click.option(
    "--ch3-gain", default=0.25, show_default=True, help="CH3 ゲイン (0.0〜1.0)"
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
        output = _convert_simple(samples, sr, duty)
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


def _convert_simple(samples: np.ndarray, sr: int, duty: int = 2) -> np.ndarray:
    """
    シンプルモード: ピッチ追跡 → GB矩形波再合成

    1. librosa.yin でフレームごとのピッチを高速検出
    2. RMS でボリュームエンベロープを取得
    3. GBのレジスタ値に周波数をスナップ（特徴的な階段状ピッチ）
    4. CH1矩形波として合成 + 4bit量子化
    """
    import librosa
    from gb_converter.apu.channel1 import (
        PulseChannel, quantize_4bit, hz_to_register, register_to_hz
    )

    hop_length = 512

    click.echo("ピッチ追跡中 (YIN)...")
    f0 = librosa.yin(
        samples,
        fmin=65.0,   # C2
        fmax=2093.0, # C7
        sr=sr,
        hop_length=hop_length,
    )

    click.echo("ボリュームエンベロープ取得中...")
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    rms_norm = rms / (rms.max() + 1e-8)

    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # GBレジスタ値にスナップしてイベント生成
    # 同じレジスタ値が続く場合はイベントをまとめる（無駄な重複排除）
    events = []
    prev_reg = -1

    for i, (freq, vol) in enumerate(zip(f0, rms_norm)):
        offset = int(times[i] * sr)
        if offset >= n_samples := len(samples):
            break

        # ピッチをGBレジスタ値に丸めることで「階段状ピッチ」を再現
        reg = hz_to_register(float(freq))
        gb_freq = register_to_hz(reg)
        volume = float(np.clip(vol * 1.2, 0.0, 1.0))

        if reg != prev_reg:
            events.append((offset, gb_freq, duty, volume))
            prev_reg = reg

    click.echo("矩形波合成中...")
    output = PulseChannel(sr).render(events, len(samples))
    return output


def _convert_full(
    samples: np.ndarray,
    sr: int,
    n_samples: int,
    duty: int,
    gains: dict[str, float],
) -> np.ndarray:
    """
    フルモード: 4チャンネル合成

    CH1: メインメロディ（高音域ピッチ） → 矩形波
    CH2: ハーモニー（5度下）          → 矩形波
    CH3: ベースライン（低音域）        → カスタム波形
    CH4: 打楽器（オンセット検出）      → LFSRノイズ
    """
    import librosa
    from gb_converter.analyzer import (
        detect_pitches,
        detect_onsets,
        pitch_events,
        onset_events,
    )
    from gb_converter.apu import PulseChannel, PulseChannel2, WaveChannel, NoiseChannel
    from gb_converter.apu.channel1 import hz_to_register, register_to_hz

    click.echo("ピッチ検出中 (pYIN)...")
    times, freqs, magnitudes = detect_pitches(samples, sr)

    click.echo("オンセット検出中...")
    onset_times, onset_strengths = detect_onsets(samples, sr)

    # CH1: メインメロディ
    click.echo("CH1 (矩形波 メロディ) 合成中...")
    ch1_events = pitch_events(times, freqs, magnitudes, sr=sr, duty=duty)
    ch1 = PulseChannel(sr).render(ch1_events, n_samples)

    # CH2: メロディの完全5度下（GB音楽でよく使われるハーモニー）
    click.echo("CH2 (矩形波 ハーモニー) 合成中...")
    fifth_down_ratio = 2 / 3  # 完全5度下 = 周波数を2/3倍
    ch2_events = [
        (off, max(freq * fifth_down_ratio, 65.0), duty, vol * 0.7)
        for off, freq, duty_, vol in ch1_events
        if freq > 0
    ]
    ch2 = PulseChannel2(sr).render(ch2_events, n_samples)

    # CH3: 低音域のベースライン（波形チャンネル）
    click.echo("CH3 (波形 ベース) 合成中...")
    wave_ram = WaveChannel.sine_wave_ram()
    ch3_events = _make_bass_events(times, freqs, magnitudes, sr, wave_ram)
    ch3 = WaveChannel(sr).render(ch3_events, n_samples)

    # CH4: 打楽器ノイズ
    click.echo("CH4 (ノイズ 打楽器) 合成中...")
    ch4_events = onset_events(onset_times, onset_strengths, sr)
    ch4 = NoiseChannel(sr).render(ch4_events, n_samples)

    click.echo("チャンネルをミックス中...")
    return mix(ch1, ch2, ch3, ch4, gains)


def _make_bass_events(
    times: np.ndarray,
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    sr: int,
    wave_ram: np.ndarray,
) -> list[tuple[int, float, np.ndarray, float]]:
    """
    メロディピッチの2オクターブ下をCH3ベースとして生成する。

    有音フレームのみ発音、無音区間はスキップ。
    """
    from gb_converter.apu.channel3 import hz_to_register_wave

    events = []
    prev_reg = -1

    for t, freq, mag in zip(times, freqs, magnitudes):
        offset = int(t * sr)
        if freq <= 0:
            if prev_reg != -1:
                events.append((offset, 0.0, wave_ram, 0.0))
                prev_reg = -1
            continue

        # 2オクターブ下
        bass_freq = freq / 4.0
        bass_freq = max(bass_freq, 65.0)  # CH3の最低周波数付近

        reg = hz_to_register_wave(bass_freq)
        volume = float(np.clip(mag * 0.8, 0.0, 1.0))

        if reg != prev_reg:
            events.append((offset, bass_freq, wave_ram, volume))
            prev_reg = reg

    return events


if __name__ == "__main__":
    main()
