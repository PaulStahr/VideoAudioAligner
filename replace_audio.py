import argparse
import logging
import numpy as np
import subprocess
import os
from audio_matcher import AudioMatcher

logger = logging.getLogger(__name__)

def replace_audio(
        input_video: str,
        input_audio: str,
        output_video: str,
        xp=np):

    audio_matcher = AudioMatcher(input_video, input_audio, xp=xp)
    audio_matcher.run()
    best_match = audio_matcher.get_best_match()

    logger.log(logging.INFO, f"Best match: {best_match}")

    if output_video is None:
        return

    offset = best_match["match_begin"]       # negative = audio earlier
    video_dur = best_match["video_duration"]

    # -----------------------------------------------------------
    # CASE 1 — Audio starts before video → trim audio at the start
    # -----------------------------------------------------------
    if offset < 0:
        audio_seek = -offset  # positive amount to skip from audio
        audio_filter = "anull"  # no filtering needed
        trim_options = ["-ss", str(audio_seek), "-t", str(video_dur)]
        before_audio_input = True

    # -----------------------------------------------------------
    # CASE 2 — Audio starts after video → we must pad silence
    # -----------------------------------------------------------
    else:
        audio_seek = 0  # no trimming; audio is already late
        delay_ms = int(offset * 1000)
        audio_filter = f"adelay={delay_ms}|{delay_ms}"
        trim_options = ["-t", str(video_dur)]
        before_audio_input = False

    # -----------------------------------------------------------
    # Build ffmpeg command
    # -----------------------------------------------------------
    cmd = [
        "ffmpeg",
        "-i", input_video
    ]

    # If trimming audio at start (-ss) must appear BEFORE audio input
    if before_audio_input:
        cmd += trim_options + ["-i", input_audio]
    else:
        cmd += ["-i", input_audio] + trim_options

    # Audio filter only if needed
    if audio_filter != "anull":
        cmd += ["-af", audio_filter]

    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "256k",
        "-ar", "48000",
        "-y",
        output_video
    ]

    logger.log(logging.INFO, "Running command: " + " ".join(cmd))

    subprocess.run(cmd, check=True)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace audio in a video file.')
    parser.add_argument('-i', '--input-video', required=True, help='Path to the input video file.')
    parser.add_argument('-a', '--input-audio', required=True, help='Path to the input audio file.')
    parser.add_argument('--library', default="numpy", choices=('numpy', 'cupy'))
    parser.add_argument('-o', '--output-video', required=False, default=None, help='Path to the output video file.')
    parser.add_argument('--log-level', default='INFO', help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))

    match args.library:
        case 'numpy':
            import numpy as xp
        case 'cupy':
            import cupy as xp
        case _:
            raise ValueError(f"Unknown library: {args.library}")
    replace_audio(args.input_video, args.input_audio, args.output_video, xp=xp)