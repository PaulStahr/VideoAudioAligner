import logging
import numbers

import librosa
import numpy as np
import scipy.signal
import subprocess
import sys
import os
import logging
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
import tempfile
import inspect
import argparse
from tqdm import tqdm
import io
import json
import pandas as pd
import av
import fractions
import soundfile

from util.cache import threadsafe_lru_cache

logger = logging.getLogger(__name__)

@threadsafe_lru_cache(maxsize=32)
def load_audio_from_video_stream(video_path, sr=44100, mono=True):
    """
    Extract audio from a video file using ffmpeg and load it directly into memory.

    Args:
        video_path (str): Path to the video file.
        sr (int): Sample rate to convert audio to (defaults to 44100 Hz).

    Returns:
        y (np.ndarray): Audio samples (mono).
        sr (int): Sample rate of the returned audio.
    """
    if not mono:
        raise ValueError("Mono audio extraction is required for this function.")
    cmd = [
        FFMPEG_PATH,
        "-i", video_path,
        "-vn",  # No video
        "-af", "aresample=async=1",
        "-ac", "1",  # Mono
        "-ar", str(sr),  # Set sample rate
        "-f", "wav",  # Raw WAV format
        "pipe:1"
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    y, actual_sr = soundfile.read(io.BytesIO(result.stdout), dtype="float32")
    return y, actual_sr

@threadsafe_lru_cache(maxsize=32)
def find_keyframes(video_input):
    probe_cmd = [
        "ffprobe", "-select_streams", "v", "-show_frames",
        "-show_entries", "frame=pkt_pts_time,key_frame",
        "-of", "json", video_input
    ]
    logger.log(logger.DEBUG, "Probing for keyframes...")
    result = subprocess.run(probe_cmd, capture_output=True, check=True, text=True)
    data = json.loads(result.stdout)
    keyframes = [float(f["pkt_pts_time"]) for f in data["frames"] if f["key_frame"] == 1]
    return keyframes

@threadsafe_lru_cache(maxsize=32)
def find_keyframes_fast(video_input):
    cmd = [
        "ffprobe",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "json",
        "-print_format", "json",
        video_input
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    packets = json.loads(result.stdout).get("packets", [])
    keyframe_times = [
        float(p["pts_time"])
        for p in packets
        if p.get("flags") and "K" in p["flags"]
    ]
    return keyframe_times


load_audio = threadsafe_lru_cache(librosa.load)


# ---------- Settings ----------
FFMPEG_PATH = "ffmpeg"           # Adjust if ffmpeg isn't in your PATH
# ------------------------------

class AudioMatcher:
    def __init__(self, long_audio_path, short_audio_path, xp=np):
        self.long_audio_path = long_audio_path
        self.short_audio_path = short_audio_path
        self.xp = xp
        self.correlation = None
        self.best_time = None
        self.match_score = None
        self.short_duration = None
        self.long_duration = None

    def run(self):
        logger.log(logging.INFO, f"Loading {self.long_audio_path}")
        self.y_long, self.sr = load_audio_from_video_stream(self.long_audio_path)
        logger.log(logging.INFO, f"Loading {self.short_audio_path}")
        self.y_short, _ = load_audio_from_video_stream(self.short_audio_path, sr=self.sr)
        logger.log(logging.INFO,
                   f"Convolving audios with {len(self.y_long) // self.sr} and {len(self.y_short) // self.sr} seconds at {self.sr} Hz")
        if xp is np:
            from scipy.signal import oaconvolve, butter, sosfilt
        else:
            from cupyx.scipy.signal import oaconvolve, butter, sosfilt

        self.y_long = convert(self.y_long, xp).astype(xp.float32)
        self.y_short = convert(self.y_short, xp).astype(xp.float32)

        def audio_transoform(x):
            sos = butter(N=4, Wn=200, fs=self.sr, btype='high', output='sos')
            x = sosfilt(sos, x)
            x = xp.abs(x)
            x = xp.sqrt(x)
            x = oaconvolve(x, 1 - xp.abs(xp.linspace(-1, 1, 10000)), mode='same')
            x = xp.gradient(x)
            return x

        self.y_long_transformed = audio_transoform(self.y_long)
        self.y_short_transformed = audio_transoform(self.y_short)

        # loudness = scipy.signal.oaconvolve(self.y_long_transformed, self.y_short_transformed[::-1], mode="valid")
        self.correlation = oaconvolve(self.y_long_transformed, self.y_short_transformed[::-1], mode="full")
        self.best_offset = xp.argmax(self.correlation)  # / loudness)
        self.match_score = self.correlation[self.best_offset]
        self.best_offset = self.best_offset
        self.best_time = (self.best_offset - len(self.y_short)) / self.sr

        long_audio_filename = os.path.basename(self.long_audio_path)
        logger.log(logging.INFO, f"Best match at {self.best_time:.2f}s of {long_audio_filename} with score {self.match_score:.2f}")

        self.short_duration = len(self.y_short) / self.sr
        self.long_duration = len(self.y_long) / self.sr
        #self.short_duration = librosa.get_duration(y=self.y_short, sr=self.sr)
        #self.long_duration = librosa.get_duration(y=self.y_long, sr=self.sr)

    def get_best_offset(self, tmin:None|int|float=None, tmax:None|int|float=None):
        if tmin is None and tmax is None:
            return self.best_offset, self.best_time
        short_offset = len(self.y_short)
        if tmin is None:
            tmin = 0
        elif isinstance(tmin, float):
            tmin = int(round(tmin * self.sr)) + short_offset
        if tmax is None:
            tmax = len(self.correlation)
        elif isinstance(tmax, float):
            tmax = int(round(tmax * self.sr)) + short_offset
        best_offset = xp.argmax(self.correlation[tmin:tmax]) + tmin
        best_time = (best_offset - short_offset) / self.sr
        return best_offset, best_time

    def get_best_match(self, tmin:None|int|float=None, tmax:None|int|float=None):
        best_offset, best_time = self.get_best_offset(tmin, tmax)
        best_score = self.correlation[best_offset]
        match_map = {
                "audio_duration": self.short_duration,
                "video_duration": self.long_duration,
                "match_score": best_score,
                "match_begin": best_time,
                "match_end": best_time + self.short_duration,
                "overlap": min(min(self.long_duration - max(best_time, 0), self.short_duration + best_time), self.short_duration),
                "video_input": self.long_audio_path
            }
        return match_map

    @staticmethod
    def get_plot_options():
        linewidth = 0
        marker = "."
        markersize = 0.1
        markerwidth = 0.1
        plot_options = {"linewidth": linewidth, "marker": marker, "markersize": markersize,
                        "markeredgewidth": markerwidth}
        return plot_options

    def plot(self, fig, num_figures=3):
        short_audio_filename = os.path.basename(self.short_audio_path)
        long_audio_filename = os.path.basename(self.long_audio_path)
        short_timepoints = xp.arange(len(self.y_short)) / self.sr + self.best_time
        long_timepoints = xp.arange(len(self.y_long)) / self.sr
        ax = [fig.add_subplot(num_figures, 1, i + 1) for i in range(num_figures)]
        # plot short audio
        plot_options = AudioMatcher.get_plot_options()
        l0 = ax[0].plot(convert(short_timepoints, np), convert(xp.abs(self.y_short), np), label=short_audio_filename, alpha=1,
                        **plot_options)
        # plot long audio
        slice_long = slice(max(0, self.best_offset - len(self.y_short)), min(len(self.y_long), self.best_offset))
        selected_long_timeinterval = long_timepoints[slice_long]
        l1 = ax[0].plot(convert(selected_long_timeinterval, np), convert(-xp.abs(self.y_long[slice_long]), np),
                        label=long_audio_filename, alpha=0.5, **plot_options)
        custom_legend = [
            Line2D([0], [0], color='C0', linewidth=2, marker='.', markersize=6, label=short_audio_filename),
            Line2D([0], [0], color='C1', linewidth=2, marker='.', markersize=6, label=long_audio_filename),
        ]
        ax[0].set_ylabel("Amplitude")
        ax[0].legend(handles=custom_legend, loc='upper left')
        ax[0].set_ylim(-1, 1)

        ax[1].plot(convert(short_timepoints, np), convert(self.y_short_transformed, np), label="Short Audio Transformed",
                   alpha=0.5, **plot_options)
        ax[1].plot(convert(selected_long_timeinterval, np), convert(self.y_long_transformed[slice_long], np),
                   label="Long Audio Transformed", alpha=0.5, **plot_options)
        ax[1].set_ylim(-0.5, 0.5)

        ax[1].set_ylabel("Correlated function (Loudness change")
        ax[1].legend(handles=custom_legend, loc='upper left')
        # plot correlation around best offset
        window = self.sr * 10
        correlation_times = xp.arange(-len(self.y_short), len(self.correlation) - len(self.y_short)) / self.sr
        ax[2].plot(convert(correlation_times[self.best_offset - window:self.best_offset + window], np),
                   convert(self.correlation[self.best_offset - window:self.best_offset + window], np), label="Correlation",
                   **plot_options)
        ax[2].set_ylim(-3000, 3000)
        ax[2].axvline(self.best_time, color='red', linestyle='--', label="Best Match")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Correlation")
        ax[2].set_title("Audio Correlation")
        ax[2].legend(loc='upper left')
        return ax

    @staticmethod
    def verification_plot(ax, files):
        sr = 44100
        audio = []
        filename = []
        for i in range(len(files)):
            a, _ = load_audio_from_video_stream(files[i], sr=sr)
            audio.append(a)
            filename.append(os.path.basename(files[i]))
        plot_options = AudioMatcher.get_plot_options()
        ax.plot(np.arange(len(audio[0])) / sr, np.abs(audio[0]), label=filename[0], alpha=0.5, **plot_options)
        ax.plot(np.arange(len(audio[1])) / sr, -np.abs(audio[1]), label=filename[1], alpha=0.5, **plot_options)
        custom_legend = [
            Line2D([0], [0], color='C0', linewidth=2, marker='.', markersize=6, label=filename[0]),
            Line2D([0], [0], color='C1', linewidth=2, marker='.', markersize=6, label=filename[1]),
        ]
        ax.legend(handles=custom_legend, loc='upper left')


def cut_video_av(start_time, duration, video_input, video_output, encoder="libx264"):
    assert len(start_time) == len(duration) == len(video_input), "Arrays must be same length"

    output = av.open(video_output, mode='w')

    video_stream_out = output.add_stream(encoder, rate=30)
    video_stream_out.pix_fmt = 'yuv420p'
    audio_stream_out = output.add_stream("aac", rate=48000)

    for idx, input_path in enumerate(video_input):
        container = av.open(input_path)

        video_stream = next(s for s in container.streams if s.type == 'video')
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

        start = start_time[idx]
        dur = duration[idx]
        end = start + dur

        # Seek to keyframe before start time
        container.seek(int(start * video_stream.time_base), any_frame=False, backward=True, stream=video_stream)

        frame_queue = []
        audio_queue = []

        for packet in tqdm(container.demux((video_stream, audio_stream) if audio_stream else (video_stream,))):
            for frame in packet.decode():
                t = float(frame.pts * frame.time_base)
                if t >= end + 1:  # Allow a small buffer to avoid missing frames
                    break
                if start <= t < end:
                    if packet.stream.type == 'video':
                        frame_queue.append((t, frame))
                    elif packet.stream.type == 'audio':
                        audio_queue.append(frame)

        # Generate constant frame rate frames (30 fps)
        frame_pts = start
        frame_interval = 1.0 / 30
        i = 0
        while frame_pts < end:
            while i + 1 < len(frame_queue) and frame_queue[i + 1][0] <= frame_pts:
                i += 1
            nearest_frame = frame_queue[i][1]
            logger.log(logger.DEBUG, nearest_frame)
            nearest_frame.pts = None
            nearest_frame.time_base = fractions.Fraction(1, 30)
            output.mux(video_stream_out.encode(nearest_frame))
            frame_pts += frame_interval

        # Add audio frames
        for audio_frame in audio_queue:
            logger.log(logger.DEBUG, audio_frame)
            audio_frame.pts = None
            out_pack = audio_stream_out.encode(audio_frame)
            output.mux(out_pack)

        container.close()

    # Flush encoders
    for pkt in video_stream_out.encode():
        output.mux(pkt)
    for pkt in audio_stream_out.encode():
        output.mux(pkt)

    output.close()


def cut_video(start_time, duration, video_input, video_output, mode="copy", video_duration=None, encoder="libx264", encoder_opts=()):
    start_time = [start_time] if isinstance(start_time, numbers.Number) else start_time
    duration = [duration] if isinstance(duration, numbers.Number) else duration
    video_input = [video_input] if isinstance(video_input, str) else video_input


    start_str = [f"{st:.3f}" for st in start_time]
    duration_str = [f"{d:.3f}" for d in duration]

    input_list = []
    num_videos = len(start_str)
    for i in range(num_videos):
        input_list.extend(["-i", video_input[i], "-ss", start_str[i], "-t", duration_str[i]])
    if mode == "copy":
        # Fast but only cuts at keyframes
        cmd = [
            FFMPEG_PATH,
            *input_list,
            "-c", "copy",
            video_output
        ]
        print("Running FFmpeg [copy]:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    elif mode == "reencode":
        # Slow but accurate
        extra_commands = []
        if len(start_str) > 1:
            extra_commands.append("-filter_complex")
            input_list = []
            for v in video_input:
                input_list.extend(["-i", v])
            inner_string = ""
            for i in range(num_videos):
                inner_string += f"[{i}:v]trim=start={start_str[i]}:duration={duration_str[i]},setpts=PTS-STARTPTS[v{i}];"
                inner_string += f"[{i}:a]atrim=start={start_str[i]}:duration={duration_str[i]},asetpts=PTS-STARTPTS[a{i}];"

            if encoder == "vaapi":
                hwupload = "format=nv12,hwupload"
            elif encoder == "h264_nvenc" or encoder == "hevc_nvenc":
                hwupload = "format=nv12,hwupload"
            else:
                hwupload = "format=yuv420p"
            inner_string+=''.join([f'[v{i}][a{i}]' for i in range(num_videos)])
            extra_commands.append(f"{inner_string}concat=n={num_videos}:v=1:a=1[v][a];[v]{hwupload}[vout]")
            extra_commands.extend(["-map", "[vout]", "-map", "[a]"])
        if encoder == "vaapi":
            cmd = [
                FFMPEG_PATH,
                "-hwaccel", "vaapi",
                "-hwaccel_device", "/dev/dri/renderD128",
                *input_list,
                "-y",
                *extra_commands,
                "-c:v", "h264_vaapi",
                "-b:v", "4M",  # Specify a bitrate or use "-qp" or "-crf"
                "-c:a", "aac",
                "-b:a", "192k",
                video_output
            ]
        elif encoder == "h264_amf":
            cmd = [
                FFMPEG_PATH,
                *input_list,
                "-y",
                *extra_commands,
                "-c:v", "h264_amf",
                "-b:v", "4M",
                "-c:a", "aac",
                "-b:a", "192k",
                video_output
            ]
        elif encoder == "h264_nvenc" or encoder == "hevc_nvenc":
            cmd = [
                FFMPEG_PATH,
                "-hwaccel", "cuda",
                *input_list,
                "-y",
                *extra_commands,
                "-c:v", encoder,
                "-preset", "slow",
                "-rc:v", "vbr_hq",
                "-cq:v", "15",
                *encoder_opts,
                "-c:a", "aac",
                "-b:a", "192k",
                "-fps_mode", "cfr",
                "-async", "1",
                "-r", "30",
                video_output
            ]
        elif encoder == "libx264":
            cmd = [
                FFMPEG_PATH,
                *input_list,
                "-y",
                *extra_commands,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-vsync", "2",
                video_output
            ]
        logger.log(logging.INFO, f"Running FFmpeg [reencode]: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)

    elif mode == "concatenate":
        # Smart-like: reencodes only GOP around cut, copies the rest
        #temp_dir = tempfile.mkdtemp()
        temp_dir = "/media/ramdisk/"
        pre_path = os.path.join(temp_dir, "pre.mp4")
        mid_path = os.path.join(temp_dir, "mid.mp4")
        post_path = os.path.join(temp_dir, "post.mp4")
        concat_list = os.path.join(temp_dir, "concat.txt")

        keyframes = np.asarray(find_keyframes_fast(video_input))
        logger.log(logger.DEBUG, f"Keyframes {keyframes}")

        # Step 2: Find keyframe before and after the cut
        end_time = start_time + duration
        start_kf = np.min(keyframes[keyframes >= start_time])
        end_kf = np.max(keyframes[keyframes <= end_time])

        logger.log(logger.DEBUG, f"Nearest keyframes: start_kf={start_kf:.3f}, end_kf={end_kf:.3f}")

        # Step 3: Split segments
        # Pre-cut (copy)
        # 1. Head segment (start_time to start_kf) — needs reencoding
        subprocess.run([
            FFMPEG_PATH,
            "-ss", f"{start_time:.3f}",
            "-y",
            "-i", video_input,
            "-t", f"{start_kf - start_time:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            pre_path
        ], check=True)

        # 2. Middle segment (start_kf to end_kf) — copy only
        subprocess.run([
            FFMPEG_PATH,
            "-ss", f"{start_kf:.3f}",
            "-y",
            "-i", video_input,
            "-t", f"{end_kf - start_kf:.3f}",
            "-c", "copy",
            mid_path
        ], check=True)

        # 3. Tail segment (end_kf to end_time) — needs reencoding
        subprocess.run([
            FFMPEG_PATH,
            "-ss", f"{end_kf:.3f}",
            "-y",
            "-i", video_input,
            "-t", f"{end_time - end_kf:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            post_path
        ], check=True)

        # 4. Concatenate all
        with open("concat_list.txt", "w") as f:
            f.write(f"file '{pre_path}'\n")
            f.write(f"file '{mid_path}'\n")
            f.write(f"file '{post_path}'\n")

        subprocess.run([
            FFMPEG_PATH, "-f", "concat", "-safe", "0",
            "-i", "concat_list.txt",
            "-c", "copy",
            output_file
        ], check=True)

        # Step 4: Create concat list
        with open(concat_list, 'w') as f:
            if start_kf > 0:
                f.write(f"file '{pre_path}'\n")
            f.write(f"file '{mid_path}'\n")
            if video_duration is None or end_kf < video_duration:
                f.write(f"file '{post_path}'\n")

        # Step 5: Concatenate segments
        subprocess.run([
            FFMPEG_PATH, "-f", "concat", "-safe", "0","-y",
            "-i", concat_list, "-c", "copy", video_output
        ], check=True)

        logger.log(logger.DEBUG, f"Concatenation complete. Output: {video_output}")
    elif mode == "dummy":
        # Dummy mode for testing
        logger.log(logger.DEBUG, f"Dummy mode: Would cut video from {start_time:.3f}s for {duration:.3f}s, but not actually doing it.")
        with open(video_output, 'w') as f:
            f.write("This is a dummy output file. No video was processed.")
    else:
        raise ValueError("Invalid mode. Choose from: 'copy', 'reencode', 'concatenate'")


@staticmethod
def convert(img, module):
    if isinstance(img, str):
        return img
    if module == None:
        return img
    t = type(img)
    if inspect.getmodule(t) == module:
        return img
    if logging.DEBUG >= logging.root.level:
        finfo = inspect.getouterframes(inspect.currentframe())[1]
        logger.log(logging.DEBUG,
                   F'convert {t.__module__} to {module.__name__} by {finfo.filename} line {finfo.lineno}')
    if t.__module__ == 'cupy':
        return module.array(img.get(), copy=False)
    return module.array(img, copy=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process program arguments.')
    parser.add_argument('-v', '--video-input', nargs='+', default=None, help='Path to the long video file')
    parser.add_argument('-a', '--audio-input', nargs='+', default=None, help='Path to the short audio file')
    parser.add_argument('-o', '--output-dir', default='.', help='Directory to save the output video')
    parser.add_argument('-m', '--mode', choices=['copy', 'reencode', 'concatenate', 'dummy'], default='reencode')
    parser.add_argument('-e', '--encoder', default='libx264', choices=('libx264','vaapi','h264_nvenc', 'hevc_nvenc', 'h264_amf'),)
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('-po', '--plot-output', type=str, default=None, help='Path to save debug plots')
    parser.add_argument('-l', '--loglevel', default='info', help='Set logging level (default: info)')
    parser.add_argument('--encoder-opts', type=str, default=None,)
    parser.add_argument('--library', default="numpy", choices=('numpy', 'cupy'))

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel.upper())

    match args.library:
        case 'numpy':
            import numpy as xp
        case 'cupy':
            import cupy as xp
        case _:
            raise ValueError(f"Unknown library: {args.library}")
    for audio_path in args.audio_input:
        audio_filename = os.path.basename(audio_path)
        figures = [plt.figure(figsize=(12, 6)) for _ in range(len(args.video_input))]

        run_parallel = True
        audio_matcher = [AudioMatcher(video_path, audio_path, xp=xp) for video_path in args.video_input]
        if run_parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(am.run) for am in audio_matcher}
            for future in futures:
                try:
                    future.result()  # Wait for the result and raise any exceptions
                except Exception as e:
                    logger.log(logging.ERROR, f"Error processing audio match: {e}")
                    sys.exit(1)
        else:
            for am in audio_matcher:
                am.run()

        table = pd.DataFrame([am.get_best_match() for am in audio_matcher])
        table['video_input'] = args.video_input
        table = table.reindex(columns=['audio_duration', 'video_duration', 'match_score', 'match_begin', 'match_end', 'overlap', 'video_input'])
        logger.log(logging.INFO, f"Matches\n{table.to_string()}")
        #get best match and export that one
        best_match_id = [table['match_score'].idxmax()]
        best_match = [table.loc[idx] for idx in best_match_id]

        if best_match[0]['match_begin'] < 0 and best_match_id[0] > 0:
            best_match_id = [best_match_id[0] - 1, best_match_id[0]]
            tmin = table.loc[best_match_id[0]]['video_duration'] + table.loc[best_match_id[1]]['match_begin']
            logger.log(logging.INFO, f"Realigning previous match {best_match_id[0]} with tmin={tmin:.3f}s")
            table.loc[best_match_id[0]] = audio_matcher[best_match_id[0]].get_best_match(tmin=tmin)
        elif best_match[0]['match_end'] > best_match[0]['video_duration']:
            best_match_id = [best_match_id[0], best_match_id[0] + 1]
            tmax = table.loc[best_match_id[0]]['match_begin'] - table.loc[best_match_id[0]]['video_duration']
            logger.log(logging.INFO, f"Realigning following match {best_match_id[1]} with tmax={tmax:.3f}s")
            table.loc[best_match_id[1]] = audio_matcher[best_match_id[1]].get_best_match(tmax = tmax)

        if len(best_match_id) > 1:
            logger.log(logging.INFO, f"Found multiple matches: {len(best_match_id)}")
            logger.log(logging.INFO, f"Realigned matches\n{table.to_string()}")

        best_match = [table.loc[idx] for idx in best_match_id]

        short_audio_filename = os.path.basename(audio_path)
        for i in range(len(best_match_id)):
            long_audio_filename = os.path.basename(best_match[i]["video_input"])  # Remove .mp4 extension
            plot_output = os.path.join(args.plot_output,
                                       f"{os.path.splitext(long_audio_filename)[0]}_{os.path.splitext(short_audio_filename)[0]}")
            if not os.path.exists(args.plot_output):
                os.makedirs(args.plot_output)
            plot_output = os.path.abspath(plot_output)
            audio_matcher[best_match_id[i]].plot(figures[best_match_id[i]])
            figures[best_match_id[i]].savefig(f"{plot_output}_audio_correlation.png", dpi=300, bbox_inches='tight')
        for fig in figures:
            plt.close(fig)

        if args.output_dir.endswith('.mp4'):
            output_file = args.output_dir
        else:
            output_file = f"{args.output_dir}/{os.path.splitext(os.path.basename(best_match[0]["video_input"]))[0]}_{os.path.splitext(audio_filename)[0]}.mp4"

        if len(best_match) > 1:
            #create table with columns overlap, start_time, video_input
            cut_table = pd.DataFrame(columns=["overlap", "start_time", "video_input"])

            safety_offset = 0
            #add rows to the table
            overlap0 = best_match[0]["overlap"] - safety_offset
            cut_table.loc[0] = {"overlap": overlap0,"start_time": best_match[0]["match_begin"], "video_input": best_match[0]["video_input"]}
            cut_table.loc[1] = {"overlap": -best_match[1]['match_begin'] - overlap0 + safety_offset, "start_time": safety_offset, "video_input": best_match[1]["video_input"]}
            cut_table.loc[2] = {"overlap": best_match[1]["overlap"] - safety_offset, "start_time": safety_offset, "video_input": best_match[1]["video_input"]}
            logger.log(logging.INFO, f"Final cut table \n{cut_table.to_string()}")
            logger.log(logging.INFO, f"Summing up to {cut_table["overlap"].sum():.3f}s from {len(cut_table)} videos")
            cut_video(cut_table["start_time"].to_numpy(), cut_table["overlap"].to_numpy(),cut_table["video_input"].to_numpy(), video_output = output_file, mode=args.mode, encoder=args.encoder, encoder_opts=args.encoder_opts.split(' ') if args.encoder_opts != '' else [])
            #cut_video_av(cut_table["start_time"].to_numpy(), cut_table["overlap"].to_numpy(),cut_table["video_input"].to_numpy(), video_output = output_file)
        else:
            logger.log(logging.INFO, f"Cutting {best_match[0]['audio_duration']}s from video {best_match[0]["video_input"]} from {best_match[0]["match_begin"]:.3f}s to {best_match[0]["match_end"]:.3f}s")
            cut_video(best_match[0]['match_begin'],
                      best_match[0]['audio_duration'],
                      best_match[0]["video_input"], video_output = output_file, mode=args.mode, encoder=args.encoder, encoder_opts=args.encoder_opts.split(' ') if args.encoder_opts != '' else [])

        fig, ax = plt.subplots(figsize=(12, 6))
        AudioMatcher.verification_plot(ax, [audio_path, output_file])
        fig.savefig(f"{args.plot_output}/{short_audio_filename}_verification_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.log(logging.INFO, f"Done! Saved as {output_file}")
