# MotionHeatmapGenerator

![Motion Heatmap Example](https://github.com/ylp1455/MotionHeatmapGenerator/assets/115799462/417c8a9c-0a27-4e82-b44a-6cf34980f99a)

A Python package for generating motion heatmaps from video sequences. This package allows you to analyze areas of motion within a sequence of images by highlighting these areas in a color-coded heatmap overlay.

## Features

- Generate motion heatmaps from a sequence of images
- Highlight areas of motion within the images using temporal intensity analysis
- Customize the appearance of the heatmap, including color intensity and smoothing
- Configurable grid resolution for motion detection
- Deterministic output with optional random seed
- Built on high-pass Butterworth filtering and Gaussian smoothing

## Installation

### From PyPI (when published)
```bash
pip install MotionHeatmapGenerator
```

### From Source
```bash
git clone https://github.com/ylp1455/MotionHeatmapGenerator.git
cd MotionHeatmapGenerator
pip install -e .
```

## Requirements

- Python >= 3.6
- opencv-python >= 4.0.0
- numpy >= 1.19.0
- scipy >= 1.5.0

## Usage

### Basic Example

```python
from MotionHeatmapGenerator import MotionHeatmapGenerator

# Initialize the generator with the desired number of divisions and a list of images
generator = MotionHeatmapGenerator(
    num_vertical_divisions=4,
    num_horizontal_divisions=4,
    images=["frame001.jpg", "frame002.jpg", "frame003.jpg"]
)

# Generate the motion heatmap
generator.generate_motion_heatmap("output_heatmap.jpg")
```

This will generate a motion heatmap from the provided images and save it as `output_heatmap.jpg`.

### Advanced Usage

```python
from MotionHeatmapGenerator import MotionHeatmapGenerator

# More control over parameters
generator = MotionHeatmapGenerator(
    num_vertical_divisions=8,           # Higher resolution grid
    num_horizontal_divisions=8,
    images=image_paths,
    use_average_image_overlay=True,     # Use averaged frame as background
    sigma=2.0,                           # More smoothing
    color_intensity_factor=5,            # Subtle color overlay
    print_debug=True,                    # Show progress
    random_seed=42                       # Reproducible results
)

generator.generate_motion_heatmap("motion_heatmap.jpg")
```

### Processing Video Files

```python
import cv2
import glob
from MotionHeatmapGenerator import MotionHeatmapGenerator

# Extract frames from video
video = cv2.VideoCapture("input_video.mp4")
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imwrite(f"frames/frame_{frame_count:04d}.jpg", frame)
    frame_count += 1
video.release()

# Get all frame paths
frame_paths = sorted(glob.glob("frames/*.jpg"))

# Generate heatmap
generator = MotionHeatmapGenerator(
    num_vertical_divisions=10,
    num_horizontal_divisions=10,
    images=frame_paths
)
generator.generate_motion_heatmap("video_motion_heatmap.jpg")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_vertical_divisions` | int | *required* | Number of vertical blocks in heatmap grid |
| `num_horizontal_divisions` | int | *required* | Number of horizontal blocks in heatmap grid |
| `images` | list[str] | *required* | Ordered list of image file paths |
| `use_average_image_overlay` | bool | `True` | If True, overlay on averaged frame; if False, use first frame |
| `sigma` | float | `1.5` | Gaussian smoothing standard deviation |
| `color_intensity_factor` | int | `7` | Multiplier for color overlay intensity |
| `print_debug` | bool | `True` | Print progress messages during processing |
| `random_seed` | int | `None` | Random seed for deterministic pixel sampling |

## How It Works

The algorithm detects motion through temporal intensity variance analysis:

1. Divides the image into a grid of blocks
2. Samples pixel intensities across all frames for each block
3. Applies high-pass Butterworth filtering to remove slow trends (lighting changes, camera drift)
4. Computes standard deviation of filtered intensities as the motion metric
5. Applies Gaussian smoothing for visual appeal
6. Overlays color-coded heatmap (red = high motion, blue = low motion)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter any problems or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research or project, please cite:

```
@software{motionheatmapgenerator,
  author = {Yasiru Perera},
  title = {MotionHeatmapGenerator: A Python Package for Video Motion Analysis},
  year = {2024},
  url = {https://github.com/ylp1455/MotionHeatmapGenerator}
}
```

## Documentation

For comprehensive documentation including algorithm details, performance analysis, and troubleshooting, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
