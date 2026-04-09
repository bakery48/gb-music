"""
gb-music: MP3をゲームボーイ音源に変換するCLIツール

パイプライン:
  1. 音声ロード          (loader)
  2. 楽譜化              (transcriber / basic-pitch)
  3. チャンネル振り分け  (scheduler)
  4. GBイベント生成      (note_mapper)
  5. APU合成+ミックス    (synthesizer)
  6. WAV出力             (exporter)

使い方:
  python main.py input.mp3
  python main.py input.mp3 -o output.wav
  python main.py input.mp3 --duty 1 --bass-midi 48 --no-ch4
"""

import os
import click
import numpy as np

from gb_converter.loader import load_audio, normalize
from gb_converter.transcriber import transcribe, print_summary
from gb_converter.scheduler import schedule
from gb_converter.note_mapper import map_to_apu_events
from gb_converter.synthesizer import synthesize, GainConfig
from gb_converter.exporter import export_wav


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_path",
    default=None,
    help="出力ファイルパス (.wav)。未指定時は <input>_gb.wav",
)
# --- 変換品質 ---
@click.option(
    "--onset-threshold", default=0.5, show_default=True,
    help="basic-pitch 発音検出の閾値 (0〜1、高いほど厳しい)",
)
@click.option(
    "--frame-threshold", default=0.3, show_default=True,
    help="basic-pitch フレーム持続の閾値 (0〜1)",
)
@click.option(
    "--min-note-len", default=0.05, show_default=True,
    help="これより短いノートは除去する [秒]",
)
# --- チャンネル設定 ---
@click.option(
    "--duty", type=click.IntRange(0, 3), default=2, show_default=True,
    help="矩形波デューティ比 (0=12.5%%, 1=25%%, 2=50%%, 3=75%%)",
)
@click.option(
    "--bass-midi", default=48, show_default=True,
    help="この値未満のMIDIノートをCH3ベースに振る (デフォルト48=C3)",
)
@click.option(
    "--no-ch4", is_flag=True, default=False,
    help="打楽器チャンネル (CH4) を無効にする",
)
# --- ゲイン ---
@click.option("--ch1-gain", default=0.30, show_default=True, help="CH1ゲイン")
@click.option("--ch2-gain", default=0.22, show_default=True, help="CH2ゲイン")
@click.option("--ch3-gain", default=0.25, show_default=True, help="CH3ゲイン")
@click.option("--ch4-gain", default=0.18, show_default=True, help="CH4ゲイン")
# --- フィルタ ---
@click.option(
    "--no-lowpass", is_flag=True, default=False,
    help="GB風ローパスフィルタを無効にする",
)
# --- デバッグ ---
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="検出ノート数などの詳細を表示する",
)
def main(
    input_path: str,
    output_path: str | None,
    onset_threshold: float,
    frame_threshold: float,
    min_note_len: float,
    duty: int,
    bass_midi: int,
    no_ch4: bool,
    ch1_gain: float,
    ch2_gain: float,
    ch3_gain: float,
    ch4_gain: float,
    no_lowpass: bool,
    verbose: bool,
) -> None:
    """MP3ファイルをゲームボーイ音源 (WAV) に変換する。"""

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base}_gb.wav"

    click.echo(f"[1/6] 音声ロード: {input_path}")
    samples, sr = load_audio(input_path)
    samples = normalize(samples)
    n_samples = len(samples)
    duration = n_samples / sr
    click.echo(f"      {duration:.1f}秒  SR={sr}Hz")

    from gb_converter.transcriber import has_basic_pitch
    engine = "basic-pitch" if has_basic_pitch() else "lite (librosa YIN)"
    click.echo(f"[2/6] 楽譜化中 ({engine}) ...")
    notes = transcribe(
        input_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_note_len_sec=min_note_len,
    )
    if verbose:
        print_summary(notes)
    else:
        click.echo(f"      {len(notes)}音符 検出")

    if not notes:
        click.echo("警告: 音符が検出されませんでした。出力は無音になります。")

    click.echo("[3/6] チャンネル振り分け中 ...")
    audio_for_ch4 = None if no_ch4 else samples
    tracks = schedule(notes, audio=audio_for_ch4, sr=sr, bass_threshold_midi=bass_midi)
    click.echo(f"      {tracks.summary()}")

    click.echo("[4/6] GBイベント生成中 ...")
    ch1e, ch2e, ch3e, ch4e = map_to_apu_events(tracks, sr=sr, duty=duty)

    click.echo("[5/6] APU合成+ミックス中 ...")
    gains = GainConfig(ch1=ch1_gain, ch2=ch2_gain, ch3=ch3_gain, ch4=ch4_gain)
    output = synthesize(
        ch1e, ch2e, ch3e, ch4e,
        n_samples=n_samples,
        sr=sr,
        gains=gains,
        apply_lowpass=not no_lowpass,
    )

    click.echo(f"[6/6] 書き出し: {output_path}")
    export_wav(output, sr, output_path)

    click.echo("完了!")


if __name__ == "__main__":
    main()
