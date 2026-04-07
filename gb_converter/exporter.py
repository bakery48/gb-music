"""
出力モジュール

合成したGB音源をWAVファイルとして書き出す。
"""

import numpy as np
import soundfile as sf


def export_wav(samples: np.ndarray, sr: int, path: str) -> None:
    """
    float32のモノラル波形をWAVファイルに書き出す。

    Args:
        samples: float32配列 (-1.0 〜 1.0)
        sr: サンプルレート
        path: 出力先ファイルパス
    """
    # 16bit PCMとして書き出す (CDクオリティ)
    clipped = np.clip(samples, -1.0, 1.0)
    sf.write(path, clipped, sr, subtype="PCM_16")


def export_mp3(samples: np.ndarray, sr: int, path: str, bitrate: str = "128k") -> None:
    """
    float32波形をMP3ファイルに書き出す (soundfileはMP3未対応のためWAV経由)。

    Args:
        samples: float32配列
        sr: サンプルレート
        path: 出力先 .mp3 ファイルパス
        bitrate: ビットレート (例: "128k", "192k")
    """
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_wav(samples, sr, tmp_path)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-b:a", bitrate, path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpegによるMP3変換に失敗しました:\n{result.stderr}"
            )
    finally:
        os.unlink(tmp_path)
