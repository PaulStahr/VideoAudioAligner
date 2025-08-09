usage: python3 cut_videos.py --video-input [VIDEO_INPUT ...] --audio-input [AUDIO_INPUT ...] --output-dir OUTPUT_DIR

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
