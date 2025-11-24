import os
import logging
import numpy as np
from matplotlib.lines import Line2D
from util.cache import threadsafe_lru_cache
from util.array_util import convert
import subprocess
import soundfile
import io

logger = logging.getLogger(__name__)

FFMPEG_PATH = "ffmpeg"


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
        if self.xp is np:
            from scipy.signal import oaconvolve, butter, sosfilt
        else:
            from cupyx.scipy.signal import oaconvolve, butter, sosfilt

        self.y_long = convert(self.y_long, self.xp).astype(self.xp.float32)
        self.y_short = convert(self.y_short, self.xp).astype(self.xp.float32)

        def audio_transoform(x):
            sos = butter(N=4, Wn=200, fs=self.sr, btype='high', output='sos')
            x = sosfilt(sos, x)
            x = self.xp.abs(x)
            x = self.xp.sqrt(x)
            x = oaconvolve(x, 1 - self.xp.abs(self.xp.linspace(-1, 1, 10000)), mode='same')
            x = self.xp.gradient(x)
            return x

        self.y_long_transformed = audio_transoform(self.y_long)
        self.y_short_transformed = audio_transoform(self.y_short)

        # loudness = scipy.signal.oaconvolve(self.y_long_transformed, self.y_short_transformed[::-1], mode="valid")
        self.correlation = oaconvolve(self.y_long_transformed, self.y_short_transformed[::-1], mode="full")
        self.best_offset = self.xp.argmax(self.correlation)  # / loudness)
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
