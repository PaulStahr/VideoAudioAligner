# Video/Audio Alignment Script

A small script for aligning video and audio files. It is intended for recordings such as live concerts, where high-quality audio is captured with professional microphones and video is recorded using one or multiple cameras.

## Input

- **Audio:** A continuous audio recording that has already been split into individual tracks.
- **Video:** One or more video files, potentially with small gaps between consecutive recordings.

## What This Tool Does

This tool finds the matching sections in the video files, then trims and concatenates them so the resulting video segments align with the segmented audio tracks.

## Usage

python3 cut_videos.py --video-input [VIDEO_INPUT ...] --audio-input [AUDIO_INPUT ...] --output-dir OUTPUT_DIR

Process program arguments.

options:\
  --video-input VIDEO_INPUT [VIDEO_INPUT ...]\
                        Path to the video file\
  --audio-input AUDIO_INPUT [AUDIO_INPUT ...]\
                        Path to the audio file\
  --output-dir OUTPUT_DIR\
                        Directory to save the output video\
  --encoder {libx264,vaapi,h264_nvenc,hevc_nvenc,h264_amf}\
  --debug           Enable debug output\
  --plot-output PLOT_OUTPUT\
                        Path to save debug plots\
  --loglevel LOGLEVEL\
                        Set logging level (default: info)\
  --encoder-opts ENCODER_OPTS\
  --library {numpy,cupy}

## Feedback

If your workflow differs or you have ideas, recommendations, or improvements, feel free to share them.
