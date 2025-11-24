import argparse
import logging
import numpy as np
import subprocess
import json
from audio_matcher import AudioMatcher

logger = logging.getLogger(__name__)

def replace_audio(
        input_video: str,
        input_audio: str,
        output_video: str,
        options: list[str] = None,
        xp=np):

    audio_matcher = AudioMatcher(input_video, input_audio, xp=xp)
    audio_matcher.run()
    best_match = audio_matcher.get_best_match()

    logger.log(logging.INFO, f"Best match: {best_match}")

    if output_video is None:
        return

    af = []
    offset = best_match["match_begin"]       # negative = audio earlier
    video_dur = best_match["video_duration"]

    # -----------------------------------------------------------
    # CASE 1 — Audio starts before video → trim audio at the start
    # -----------------------------------------------------------
    if offset < 0:
        audio_seek = -offset  # positive amount to skip from audio
        af += [f"atrim=start={audio_seek}:duration={video_dur}", f"asetpts=PTS-STARTPTS"]
        before_audio_input = True

    # -----------------------------------------------------------
    # CASE 2 — Audio starts after video → we must pad silence
    # -----------------------------------------------------------
    else:
        delay_ms = int(offset * 1000)
        af += [f"adelay={delay_ms}|{delay_ms}"]
        trim_options = ["-t", str(video_dur)]
        before_audio_input = False

    # -----------------------------------------------------------
    # Build ffmpeg command
    # -----------------------------------------------------------
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-i", input_audio
    ]

    if "loudness" in options:
        af_probe = [*af, "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json"]
        # Use ffmpeg to measure loudness and adjust to -16 LUFS
        probe_cmd = [
            "ffmpeg",
            "-i", input_audio,
            "-af", ",".join(af_probe),
            "-f", "null",
            "-"
        ]
        logger.log(logging.INFO, "Measuring loudness with command: " + " ".join(probe_cmd))

        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        # Print ffmpeg output (for debugging)
        for line in result.stderr.splitlines():
            print(line)

        # Extract multiline JSON block
        json_str = ""
        inside_json = False
        for line in result.stderr.splitlines():
            if "{" in line:
                inside_json = True
            if inside_json:
                json_str += line
            if "}" in line and inside_json:
                break  # stop after closing brace

        if not json_str:
            raise RuntimeError("Could not find loudness JSON data in ffmpeg output")

        loudness_data = json.loads(json_str)

        target_i = -18.0
        measured_i = float(loudness_data["input_i"])
        volume = target_i - measured_i

        logger.log(logging.INFO, f"Adjusting volume by {volume} dB to reach {target_i} LUFS")

        # Append volume adjustment to existing filter chain
        af += [f"volume={volume:2f}dB"]
    if "acompressor" in options:
        af += ["acompressor=threshold=-15dB:ratio=1.5:attack=20:release=250:makeup=1"]

    if not output_video.endswith(".mpg"):
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
    else:
        vf = ""
        if "minterpolate" in options:
            vf += "minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=1:search_param=6,",
        vf += "format=yuv420p"
        if "fadein" in options:
            af += ["afade=t=in:ss=0:d=2"]
        if "fadeout" in options:
            af += [f"afade=t=out:st={video_dur - 2}:d=2"]
        if len(af) > 0:
            af = ["-af", ",".join(af)]
        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-b:v", "8000k",
            "-g", "50",
            "-keyint_min", "1",
            "-sc_threshold", "40",
            "-vf", vf,
            *af,
            "-aspect", "16:9",
            "-target", "pal-dvd",
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
    parser.add_argument('--options', nargs='*', choices=("minterpolate", "fadein", "fadeout", "loudness", "acompressor"), default=[])
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
    replace_audio(args.input_video, args.input_audio, args.output_video, options=args.options, xp=xp)