# Video Background Remover

A powerful tool for removing backgrounds from videos using Robust Video Matting (RVM) with optimizations for Apple Silicon (M1/M2/M3). This tool supports various output formats including transparent video (alpha channel) and chroma key ready MP4s.

## Features

- ✅ Background removal from videos with state-of-the-art quality
- ✅ Apple Silicon optimization with Metal Performance Shaders
- ✅ Multiple output formats supported:
  - MP4 with chroma key (green screen)
  - MOV with alpha channel transparency (ProRes 4444)
  - WebM with alpha channel transparency (VP9)
  - Individual frames with transparency (PNG)
- ✅ Audio preservation in output videos
- ✅ Processing speed/quality adjustment with downsampling
- ✅ Multiple model options (mobilenet vs resnet) for speed/quality trade-offs

## Requirements

- macOS with Python 3.9+ (tested on macOS with Apple Silicon M1)
- FFmpeg (for transparent video creation)

## Installation

1. **Clone this repository**:
```bash
git clone https://github.com/yourusername/video-bg-remover.git
cd video-bg-remover
```

2. **Create a virtual environment and install dependencies**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Download the pre-trained model**:
```bash
python download_model.py
```
When prompted, select your preferred model (1-3):
- 1: MobileNetV3 (small) - Fastest, lower quality
- 2: MobileNetV3 (regular) - Good balance of speed and quality
- 3: ResNet50 - Best quality, slower processing

## Usage

### Basic Usage

Process a video with default settings (green background):

```bash
python simple_rvm.py --input your_video.mp4 --output output.mp4
```

### Output Options

#### MP4 with Green Screen (for chroma keying)

Create an MP4 with green screen background that preserves audio:

```bash
python simple_rvm.py --input your_video.mp4 --output output.mp4 --mp4-alpha
```

#### True Transparent Video (MOV with alpha channel)

Create a QuickTime video with transparency:

```bash
python simple_rvm.py --input your_video.mp4 --output output.mov --transparent
```

#### WebM with Alpha Channel (for web use)

Create a WebM video with transparency:

```bash
python simple_rvm.py --input your_video.mp4 --output output.webm --transparent
```

#### Extract Individual Transparent Frames

Extract all frames as PNG files with transparency:

```bash
python simple_rvm.py --input your_video.mp4 --output frames_folder --output-type frames
```

### Performance Optimization

For faster processing, use the downsample option (0.5 = half resolution):

```bash
python simple_rvm.py --input your_video.mp4 --output output.mp4 --downsample 0.5
```

Adjust the value based on your needs:
- 1.0: Original resolution (best quality)
- 0.5: Half resolution (good balance)
- 0.25: Quarter resolution (fastest)

### Background Color Options

Change the background color (for non-transparent output):

```bash
python simple_rvm.py --input your_video.mp4 --output output.mp4 --bg-color 0 0 255  # Blue background
```

### Device Selection

Force a specific compute device:

```bash
python simple_rvm.py --input your_video.mp4 --output output.mp4 --device mps  # Use Apple Metal
python simple_rvm.py --input your_video.mp4 --output output.mp4 --device cpu  # Force CPU
```

## Complete Command Reference

```
usage: simple_rvm.py [-h] --input INPUT --output OUTPUT [--model MODEL]
                    [--output-type {video,frames,transparent}]
                    [--bg-color BG_COLOR BG_COLOR BG_COLOR] [--downsample DOWNSAMPLE]
                    [--device {auto,cpu,cuda,mps}] [--transparent] [--mp4-alpha]

Video Background Removal with RVM

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to input video
  --output OUTPUT, -o OUTPUT
                        Path to output video or directory for frames
  --model MODEL, -m MODEL
                        Path to RVM model (default: models/rvm_resnet50.pth)
  --output-type {video,frames,transparent}, -t {video,frames,transparent}
                        Output format (video, frames, or transparent)
  --bg-color BG_COLOR BG_COLOR BG_COLOR, -b BG_COLOR BG_COLOR BG_COLOR
                        Background color as R G B values (0-255)
  --downsample DOWNSAMPLE, -d DOWNSAMPLE
                        Downsample ratio for faster processing (e.g., 0.5 for half resolution)
  --device {auto,cpu,cuda,mps}
                        Device for processing
  --transparent, -a     Create video with transparency (requires ffmpeg)
  --mp4-alpha          Create MP4 with green screen for transparency
```

## Technical Details

### How It Works

1. **Robust Video Matting (RVM)** - The tool uses RVM, a state-of-the-art algorithm for video matting that produces high-quality alpha mattes for videos without requiring green screens in the input.

2. **Recurrent Architecture** - Unlike frame-by-frame processing, RVM uses a recurrent neural network to maintain temporal consistency across frames.

3. **Output Process**:
   - For transparent videos: Frames are processed individually and then combined with FFmpeg
   - For MP4 with alpha: Background is replaced with pure green for chroma keying
   - For standard output: Background is replaced with the specified color

### Supported Models

- **MobileNetV3 Small**: Lightweight model, fastest processing
- **MobileNetV3**: Medium model, good balance of speed/quality
- **ResNet50**: Heavy model, best quality but slower processing

### File Formats and Transparency

- **MOV (QuickTime)**: Uses ProRes 4444 codec with alpha channel
- **WebM**: Uses VP9 codec with alpha channel
- **MP4**: Uses standard H.264 codec with green screen background (no native alpha support)
- **PNG Frames**: Full RGBA transparency support

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found**: Ensure FFmpeg is installed and available in your PATH
   ```bash
   brew install ffmpeg
   ```

2. **Model Not Found**: Run `python download_model.py` to download the required model

3. **Memory Issues**: Try reducing the resolution with `--downsample 0.5` or `--downsample 0.25`

4. **Poor Quality**: Use the ResNet50 model and avoid downsampling for best quality

## Credits

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) - Original RVM implementation
- This project builds upon RVM to provide an easy-to-use interface for background removal