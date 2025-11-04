# MotionHeatmapGenerator - Comprehensive Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Architecture](#core-architecture)
4. [Implementation Details](#implementation-details)
5. [Algorithm Explanation](#algorithm-explanation)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Dependencies](#dependencies)
9. [Known Issues and Bugs](#known-issues-and-bugs)
10. [Limitations](#limitations)
11. [Testing Status](#testing-status)
12. [Improvement Suggestions](#improvement-suggestions)
13. [Performance Analysis](#performance-analysis)
14. [Edge Cases](#edge-cases)

---

## Project Overview

**Project Name:** MotionHeatmapGenerator  
**Repository:** https://github.com/ylp1455/MotionHeatmapGenerator  
**Owner:** ylp1455 (Yasiru Perera)  
**License:** MIT License  
**Python Version:** >= 3.6  
**Current Version:** 0.1.0 (Alpha)

### Purpose
MotionHeatmapGenerator is a Python library designed to analyze motion patterns in video sequences by generating visual heatmaps. It processes a series of image frames and highlights regions with significant temporal intensity variations, making it useful for:
- Video surveillance analysis
- Traffic flow pattern detection
- Sports movement analysis
- User interaction studies
- Any application requiring temporal motion visualization

### Key Features
- Motion detection through temporal intensity analysis
- Customizable grid-based block division for resolution control
- High-pass Butterworth filtering to isolate motion components
- Gaussian smoothing for visually appealing heatmaps
- Overlay on average frame or first frame
- Adjustable color intensity scaling

---

## Repository Structure

```
MotionHeatmapGenerator/
├── LICENSE                          # MIT License
├── README.md                        # User-facing documentation (has issues)
├── pyproject.toml                   # Empty (not configured)
├── setup.py                         # Package configuration (misaligned)
├── MotionHeatmapGenerator/          # Main package directory
│   ├── __init__.py                  # Package initialization (has import bug)
│   └── motion_heatmap_generator.py  # Core implementation
└── tests/                           # Test directory
    └── test_motion_heatmap_generator.py  # Placeholder tests (broken)
```

### File Analysis

#### `MotionHeatmapGenerator/motion_heatmap_generator.py`
- **Lines:** ~115
- **Purpose:** Contains the main `MotionHeatmapGenerator` class
- **Dependencies:** cv2, numpy, scipy, collections, itertools, math, random
- **Status:** Functional but has performance and robustness issues

#### `MotionHeatmapGenerator/__init__.py`
- **Lines:** 3 (plus docstring)
- **Purpose:** Package entry point
- **Issue:** Import statement has syntax/formatting problem: `from.motion_heatmap_generator` (missing space)
- **Should be:** `from .motion_heatmap_generator import MotionHeatmapGenerator`

#### `setup.py`
- **Purpose:** Package installation configuration
- **Issue:** References `src/` directory layout which doesn't exist
- **Specified package location:** `packages=find_packages(where="src"), package_dir={"": "src"}`
- **Actual location:** Package is at repository root under `MotionHeatmapGenerator/`
- **Result:** Installation will fail

#### `tests/test_motion_heatmap_generator.py`
- **Purpose:** Unit tests
- **Issue:** Imports from `src.motion_heatmap_generator` which doesn't match actual structure
- **Issue:** References non-existent test images `["test1.jpg", "test2.jpg"]`
- **Status:** Placeholder only; tests are incomplete

#### `README.md`
- **Issue:** Content is duplicated (entire sections appear twice)
- **Issue:** Code examples have formatting problems
- **Status:** Needs cleanup

#### `pyproject.toml`
- **Status:** Empty file (not configured for modern Python packaging)

---

## Core Architecture

### Design Pattern
The project uses a **single-class procedural design** where all computation happens in the constructor, with a separate method for output generation.

### Class: `MotionHeatmapGenerator`

```
Constructor (__init__)
    ├── Validate image dimensions
    ├── Generate random pixel sample locations per block
    ├── Process all frames
    │   ├── Read each image
    │   ├── Accumulate average image (optional)
    │   └── Sample pixel intensities per block
    ├── Apply high-pass Butterworth filter to time series
    ├── Compute standard deviation per block (motion metric)
    └── Apply Gaussian spatial smoothing to heatmap

Method: generate_motion_heatmap()
    ├── Choose base image (average or first frame)
    ├── Compute color offsets from heatmap
    ├── Apply color overlay to each block
    └── Write output image to disk
```

### Data Flow

```
Input: List of image file paths
    ↓
Step 1: Read frames and sample pixel intensities
    ↓
Step 2: Build time series per block (list of intensity values)
    ↓
Step 3: Apply high-pass filter to each time series
    ↓
Step 4: Compute std(filtered_time_series) per block → unfiltered heatmap
    ↓
Step 5: Apply Gaussian smoothing → final heatmap
    ↓
Step 6: Map heatmap values to color offsets
    ↓
Step 7: Overlay colors on base image
    ↓
Output: Motion heatmap image file
```

---

## Implementation Details

### Constructor Parameters

```python
MotionHeatmapGenerator(
    num_vertical_divisions: int,      # Number of vertical grid divisions
    num_horizontal_divisions: int,    # Number of horizontal grid divisions
    images: List[str],                # List of image file paths (ordered frames)
    use_average_image_overlay: bool = True,  # Use averaged frame as base
    sigma: float = 1.5,               # Gaussian smoothing parameter
    color_intensity_factor: int = 7,  # Color overlay scaling factor
    print_debug: bool = True          # Enable debug output
)
```

### Internal State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `self.pixel_locations` | `dict[(row, col)] -> (pixel_row, pixel_col)` | Random sample pixel per block |
| `self.block_intensities` | `defaultdict(list)` | Time series of intensities per block |
| `self.average_image` | `numpy.ndarray` (H×W×3, float) | Accumulated average frame |
| `self.heatmap` | `numpy.ndarray` (divisions×divisions, float) | Final smoothed motion heatmap |
| `self.height` | `int` | Image height in pixels |
| `self.width` | `int` | Image width in pixels |

### Key Methods

#### `__init__(self, ...)`
**Purpose:** Processes all input frames and generates the internal heatmap.

**Steps:**
1. **Dimension extraction:**
   ```python
   sample_image = cv2.imread(self.images[0])
   self.height = len(sample_image)      # Should use sample_image.shape[0]
   self.width = len(sample_image[0])    # Should use sample_image.shape[1]
   ```
   
2. **Block pixel sampling:**
   - For each block (row, col), randomly selects one pixel within that block region
   - Formula: `row_position = row * height / divisions + random() * height / divisions`
   - **Issue:** Non-deterministic; repeated runs produce different results

3. **Frame processing loop:**
   ```python
   for frame in images:
       frame_data = cv2.imread(frame)
       for each block:
           sample pixel intensity (mean of BGR values)
           append to block_intensities[(row, col)]
   ```

4. **High-pass filtering:**
   ```python
   b, a = scipy.signal.butter(5, 0.2, 'high')  # 5th order, cutoff 0.2
   for each block:
       filtered = scipy.signal.filtfilt(b, a, intensities, padlen=0)
   ```
   - **Purpose:** Remove low-frequency trends (camera motion, lighting changes)
   - **Issue:** `padlen=0` can cause errors with short sequences

5. **Motion metric computation:**
   ```python
   for each block:
       unfiltered_heatmap[row][col] = np.std(filtered_intensities)
   ```
   - Standard deviation quantifies temporal variation

6. **Spatial smoothing:**
   ```python
   heatmap = scipy.ndimage.filters.gaussian_filter(unfiltered_heatmap, sigma=1.5)
   ```
   - **Issue:** Uses deprecated API (should use `scipy.ndimage.gaussian_filter`)

#### `generate_motion_heatmap(self, file_name='motion_heatmap.jpg')`
**Purpose:** Creates and saves the color-overlaid heatmap image.

**Returns:** Boolean (True if save successful)

**Steps:**
1. **Select base image:**
   ```python
   output_image = self.average_image if use_average_image_overlay 
                  else cv2.imread(self.images[0])
   ```

2. **Compute mean heatmap value:**
   ```python
   mean_stdev = np.mean(self.heatmap)
   ```

3. **Apply color overlay per block:**
   ```python
   for each block (vertical_index, horizontal_index):
       offset = color_intensity_factor * (heatmap[v][h] - mean_stdev)
       for each pixel in block:
           output_image[row][col][2] += offset  # Red channel (BGR format)
           output_image[row][col][0] -= offset  # Blue channel
           clamp to [0, 255]
   ```
   - **Color scheme:** High motion → red tint, low motion → blue tint

4. **Write to disk:**
   ```python
   return cv2.imwrite(file_name, output_image)
   ```

#### `_clip_rgb(value)` (static)
**Purpose:** Clamps color values to valid range [0, 255]

```python
@staticmethod
def _clip_rgb(value):
    return int(max(min(value, 255), 0))
```

---

## Algorithm Explanation

### Motion Detection Theory

The algorithm detects motion through **temporal intensity variance analysis**:

1. **Assumption:** Static regions have stable pixel intensities over time
2. **Observation:** Moving objects cause intensity fluctuations at fixed spatial locations
3. **Metric:** Standard deviation of intensity time series measures motion magnitude

### Signal Processing Pipeline

#### 1. Sampling Strategy
- Divides image into grid of blocks
- Samples one random pixel per block across all frames
- **Rationale:** Reduces computation while capturing representative motion
- **Weakness:** Single-pixel sampling is noisy and non-deterministic

#### 2. High-Pass Filtering
```python
# Butterworth filter design
b, a = scipy.signal.butter(5, 0.2, 'high')
filtered = scipy.signal.filtfilt(b, a, signal, padlen=0)
```

**Purpose:**
- Removes low-frequency components (global lighting changes, slow camera drift)
- Preserves high-frequency components (motion events)
- Cutoff frequency 0.2 (normalized, where 1.0 = Nyquist frequency)

**Filter characteristics:**
- Type: Butterworth (maximally flat passband)
- Order: 5 (steeper rolloff)
- `filtfilt`: Zero-phase filtering (forward-backward pass)

#### 3. Motion Metric: Standard Deviation
```python
motion_intensity = np.std(filtered_signal)
```

**Interpretation:**
- High std → large intensity fluctuations → motion present
- Low std → stable intensity → static region
- **Alternative metrics:** Variance, peak-to-peak, energy

#### 4. Spatial Smoothing
```python
heatmap = gaussian_filter(raw_heatmap, sigma=1.5)
```

**Purpose:**
- Reduces block artifacts
- Creates visually smooth transitions
- sigma=1.5 controls smoothing radius (larger = more blur)

#### 5. Color Mapping
```python
offset = color_intensity_factor * (block_value - mean_value)
red_channel += offset
blue_channel -= offset
```

**Effect:**
- Above-average motion → increases red, decreases blue → red tint
- Below-average motion → decreases red, increases blue → blue tint
- `color_intensity_factor` controls saturation intensity

---

## API Reference

### Class: `MotionHeatmapGenerator`

#### Constructor

```python
generator = MotionHeatmapGenerator(
    num_vertical_divisions=4,
    num_horizontal_divisions=4,
    images=["frame001.jpg", "frame002.jpg", ...],
    use_average_image_overlay=True,
    sigma=1.5,
    color_intensity_factor=7,
    print_debug=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_vertical_divisions` | int | *required* | Number of rows in heatmap grid |
| `num_horizontal_divisions` | int | *required* | Number of columns in heatmap grid |
| `images` | List[str] | *required* | Ordered list of image file paths |
| `use_average_image_overlay` | bool | True | If True, overlay on averaged frame; if False, overlay on first frame |
| `sigma` | float | 1.5 | Gaussian smoothing standard deviation |
| `color_intensity_factor` | int | 7 | Multiplier for color overlay intensity |
| `print_debug` | bool | True | Print progress messages during processing |

**Raises:**
- No explicit exceptions (should be added)
- May raise `TypeError` if image read fails (returns None)

**Side Effects:**
- Reads all images from disk
- Prints warnings if dimensions not evenly divisible
- Prints progress if `print_debug=True`

#### Method: `generate_motion_heatmap`

```python
success = generator.generate_motion_heatmap(file_name='motion_heatmap.jpg')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_name` | str | 'motion_heatmap.jpg' | Output file path |

**Returns:**
- `bool`: True if save successful, False otherwise

**Side Effects:**
- Writes image file to disk
- Prints progress if `print_debug=True`

---

## Usage Examples

### Basic Usage

```python
from MotionHeatmapGenerator import MotionHeatmapGenerator

# Initialize with frame sequence
generator = MotionHeatmapGenerator(
    num_vertical_divisions=4,
    num_horizontal_divisions=4,
    images=["frame001.jpg", "frame002.jpg", "frame003.jpg"]
)

# Generate and save heatmap
generator.generate_motion_heatmap("output_heatmap.jpg")
```

### High-Resolution Heatmap

```python
# More divisions = finer spatial resolution
generator = MotionHeatmapGenerator(
    num_vertical_divisions=16,
    num_horizontal_divisions=16,
    images=image_list,
    sigma=2.0  # More smoothing for finer grid
)
generator.generate_motion_heatmap("high_res_heatmap.jpg")
```

### Overlay on First Frame

```python
# Use first frame as background instead of average
generator = MotionHeatmapGenerator(
    num_vertical_divisions=8,
    num_horizontal_divisions=8,
    images=image_list,
    use_average_image_overlay=False  # Use first frame
)
generator.generate_motion_heatmap("first_frame_overlay.jpg")
```

### Subtle Heatmap

```python
# Reduce color intensity for subtle overlay
generator = MotionHeatmapGenerator(
    num_vertical_divisions=6,
    num_horizontal_divisions=6,
    images=image_list,
    color_intensity_factor=3,  # Lower value = less intense colors
    sigma=3.0  # More smoothing
)
generator.generate_motion_heatmap("subtle_heatmap.jpg")
```

### Processing Video Frames

```python
import cv2
import glob

# Extract frames from video (external preprocessing)
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

---

## Dependencies

### Required Packages

```python
# From setup.py
install_requires=[
    "opencv-python",    # Image I/O and processing
    "numpy",            # Numerical arrays
    "scipy",            # Signal filtering and Gaussian smoothing
    "scikit-image",     # Listed but not actually used in code
]
```

### Actual Imports Used

```python
import collections      # defaultdict for block intensities
import itertools        # Grid iteration (product)
import math             # floor function
import random           # Pixel sampling
import cv2              # opencv-python
import scipy.ndimage    # Gaussian filter
import scipy.signal     # Butterworth filter
import numpy as np      # Array operations
```

**Note:** `scikit-image` is listed in dependencies but never imported. Can be removed.

### Version Compatibility

- **Python:** >= 3.6 (specified in setup.py)
- **OpenCV:** Any recent version (uses basic imread/imwrite)
- **NumPy:** Any recent version
- **SciPy:** >= 1.0 (uses scipy.signal.butter, scipy.ndimage)
- **Issue:** Code uses deprecated `scipy.ndimage.filters.gaussian_filter` (removed in SciPy 1.10+)

---

## Known Issues and Bugs

### Critical Issues

1. **Import Error in `__init__.py`**
   - **Location:** `MotionHeatmapGenerator/__init__.py`, line 7
   - **Current:** `from.motion_heatmap_generator import MotionHeatmapGenerator`
   - **Problem:** Missing space or malformed relative import
   - **Fix:** Change to `from .motion_heatmap_generator import MotionHeatmapGenerator`
   - **Impact:** Package import will fail

2. **Package Layout Mismatch**
   - **Problem:** `setup.py` expects `src/` layout but code is in `MotionHeatmapGenerator/`
   - **Impact:** `pip install` will fail
   - **Fix Option A:** Move code to `src/MotionHeatmapGenerator/`
   - **Fix Option B:** Update setup.py: `packages=find_packages(), package_dir={}`

3. **Test Import Path Wrong**
   - **Location:** `tests/test_motion_heatmap_generator.py`
   - **Current:** `from src.motion_heatmap_generator import ...`
   - **Problem:** No `src/` directory exists
   - **Impact:** Tests cannot run

### High-Priority Bugs

4. **No Input Validation**
   - **Problem:** No checks for empty `images` list, non-existent files, or None from imread
   - **Impact:** Cryptic errors or crashes
   - **Example:**
     ```python
     # This will crash with unhelpful error
     generator = MotionHeatmapGenerator(4, 4, [])
     ```

5. **Non-Deterministic Output**
   - **Problem:** `random.random()` used for pixel sampling without seed control
   - **Impact:** Cannot reproduce results
   - **Fix:** Add `random_seed` parameter:
     ```python
     if random_seed is not None:
         random.seed(random_seed)
     ```

6. **Unsafe Array Indexing**
   - **Location:** Constructor, lines ~30-31
   - **Current:** `self.height = len(sample_image)`
   - **Problem:** Assumes `cv2.imread()` succeeded (may return None)
   - **Fix:**
     ```python
     sample_image = cv2.imread(self.images[0])
     if sample_image is None:
         raise FileNotFoundError(f"Cannot read image: {self.images[0]}")
     self.height, self.width = sample_image.shape[:2]
     ```

7. **Deprecated SciPy API**
   - **Location:** Constructor, line ~75
   - **Current:** `scipy.ndimage.filters.gaussian_filter`
   - **Problem:** Removed in SciPy 1.10.0+
   - **Fix:** Change to `scipy.ndimage.gaussian_filter`

8. **Float Image Type Issue**
   - **Problem:** `self.average_image` accumulates as float but may not be properly converted to uint8
   - **Impact:** cv2.imwrite may produce incorrect output or fail
   - **Fix:**
     ```python
     output_image = np.clip(output_image, 0, 255).astype(np.uint8)
     ```

### Medium-Priority Issues

9. **filtfilt padlen=0 Risk**
   - **Problem:** Short sequences may cause ValueError
   - **Impact:** Crashes with < 10 frames (depends on filter order)
   - **Fix:** Check sequence length or use adaptive padlen

10. **No Multi-Resolution Support**
    - **Problem:** Assumes all images same size
    - **Impact:** Crashes or corrupts output if frames differ
    - **Fix:** Validate dimensions in loop or resize frames

11. **Performance: Nested Python Loops**
    - **Location:** `generate_motion_heatmap`, lines ~80-95
    - **Problem:** Iterates every pixel in Python (slow for large images)
    - **Impact:** Processing time grows quadratically with resolution
    - **Better approach:** Resize heatmap to image size and blend vectorized

### Low-Priority Issues

12. **Missing Docstrings**
    - No class or method documentation
    - No type hints

13. **Hardcoded Color Channels**
    - Only modifies red/blue channels
    - No option for different colormaps

14. **Duplicate README Content**
    - Same text repeated twice in README.md

---

## Limitations

### Design Limitations

1. **Single Pixel Sampling Per Block**
   - Only one randomly chosen pixel represents entire block
   - Noisy, non-robust to local artifacts
   - **Better:** Average all pixels in block or sample grid

2. **No Video File Support**
   - Requires pre-extracted frames
   - User must handle video decoding externally

3. **Fixed Colormap**
   - Hardcoded red/blue overlay
   - No support for standard colormaps (jet, viridis, etc.)

4. **Memory Inefficient**
   - Loads all frame data in memory for average image
   - Could use incremental averaging

5. **Limited Motion Metrics**
   - Only standard deviation supported
   - Could add: variance, peak-to-peak, frequency analysis

### Algorithmic Limitations

6. **High-Pass Filter Artifacts**
   - Butterworth filter may introduce ringing
   - Cutoff frequency (0.2) not tunable by user
   - May suppress genuine slow motion

7. **Blockiness**
   - If dimensions not divisible by division counts, blocks uneven
   - Gaussian smoothing helps but doesn't eliminate

8. **Intensity-Only Analysis**
   - Uses brightness only (mean of BGR)
   - Doesn't consider motion vectors, optical flow, or edge detection

9. **No Background Subtraction**
   - Cannot distinguish foreground motion from background
   - High-pass filter only removes very slow trends

### Practical Limitations

10. **No CLI**
    - Must use as library; no command-line tool

11. **No Configuration File Support**
    - All parameters must be hardcoded

12. **No Intermediate Output**
    - Cannot access raw heatmap array
    - Cannot export heatmap as data (only rendered image)

---

## Testing Status

### Current Test Suite
- **Location:** `tests/test_motion_heatmap_generator.py`
- **Status:** ❌ Broken/Incomplete
- **Issues:**
  - Wrong import path (`src.motion_heatmap_generator`)
  - References non-existent test images
  - Only one placeholder test (no assertions)

### Test Coverage
- **Unit Tests:** 0% (no working tests)
- **Integration Tests:** None
- **Manual Testing:** Unknown

### Tests That Should Exist

#### Unit Tests Needed

1. **Input Validation Tests**
   ```python
   def test_empty_images_list():
       # Should raise ValueError
   
   def test_nonexistent_image_file():
       # Should raise FileNotFoundError
   
   def test_mismatched_image_sizes():
       # Should raise ValueError or handle gracefully
   ```

2. **Determinism Tests**
   ```python
   def test_reproducible_with_seed():
       # Same seed → same output
   
   def test_different_without_seed():
       # No seed → different outputs
   ```

3. **Dimension Tests**
   ```python
   def test_uneven_divisions():
       # Should handle or warn appropriately
   
   def test_divisions_larger_than_image():
       # Should handle edge case
   ```

4. **Algorithm Tests**
   ```python
   def test_static_frames_produce_low_heatmap():
       # Identical frames → uniform low-intensity heatmap
   
   def test_moving_object_produces_hotspot():
       # Synthetic frames with moving bright spot → heatmap peak at motion location
   ```

5. **Output Tests**
   ```python
   def test_generate_motion_heatmap_returns_true():
       # Successful save returns True
   
   def test_output_file_exists():
       # File created on disk
   
   def test_output_image_dimensions():
       # Output matches input dimensions
   ```

---

## Improvement Suggestions

### Quick Wins (Low Effort, High Impact)

1. **Fix Import Bug**
   ```python
   # In __init__.py
   from .motion_heatmap_generator import MotionHeatmapGenerator
   ```

2. **Add Input Validation**
   ```python
   if not images:
       raise ValueError("images list cannot be empty")
   sample_image = cv2.imread(images[0])
   if sample_image is None:
       raise FileNotFoundError(f"Cannot read image: {images[0]}")
   ```

3. **Fix Deprecated API**
   ```python
   # Replace
   self.heatmap = scipy.ndimage.filters.gaussian_filter(...)
   # With
   self.heatmap = scipy.ndimage.gaussian_filter(...)
   ```

4. **Add Docstrings**
   ```python
   class MotionHeatmapGenerator:
       """
       Generate motion heatmaps from video frame sequences.
       
       This class analyzes temporal intensity variations across a sequence
       of images to identify regions of motion, producing a color-coded
       heatmap overlay.
       
       Parameters
       ----------
       num_vertical_divisions : int
           Number of vertical blocks in heatmap grid
       ...
       """
   ```

5. **Fix Package Structure**
   - Option A: Update setup.py to match current layout
   - Option B: Reorganize to src/ layout

### Medium-Term Improvements

6. **Add Deterministic Mode**
   ```python
   def __init__(self, ..., random_seed=None):
       if random_seed is not None:
           random.seed(random_seed)
   ```

7. **Vectorize Overlay Generation**
   ```python
   # Instead of nested loops, use:
   heatmap_resized = cv2.resize(self.heatmap, (self.width, self.height))
   # Apply colormap
   colored_heatmap = cv2.applyColorMap(
       (heatmap_resized * 255 / heatmap_resized.max()).astype(np.uint8),
       cv2.COLORMAP_JET
   )
   # Blend with alpha
   output = cv2.addWeighted(base_image, 0.7, colored_heatmap, 0.3, 0)
   ```

8. **Add Block Averaging Instead of Single Pixel**
   ```python
   # Sample entire block region instead of one pixel
   row_start = int(row * height / divisions)
   row_end = int((row + 1) * height / divisions)
   col_start = int(col * width / divisions)
   col_end = int((col + 1) * width / divisions)
   block_region = frame[row_start:row_end, col_start:col_end]
   intensity = np.mean(block_region)
   ```

9. **Add Progress Bar**
   ```python
   from tqdm import tqdm
   for index, file_name in enumerate(tqdm(self.images, desc="Processing frames")):
       ...
   ```

10. **Support Video Input Directly**
    ```python
    def __init__(self, ..., video_path=None, images=None):
        if video_path is not None:
            # Extract frames internally
            cap = cv2.VideoCapture(video_path)
            self.images = []  # Extract to temp dir
    ```

### Long-Term Enhancements

11. **Add Colormap Options**
    ```python
    def generate_motion_heatmap(self, file_name, colormap=cv2.COLORMAP_JET):
        ...
    ```

12. **Export Heatmap Data**
    ```python
    def get_heatmap_array(self):
        """Return raw heatmap as numpy array."""
        return self.heatmap.copy()
    ```

13. **Add CLI Interface**
    ```python
    # New file: cli.py
    import argparse
    def main():
        parser = argparse.ArgumentParser(description="Generate motion heatmaps")
        parser.add_argument("--input", nargs="+", required=True)
        parser.add_argument("--output", default="heatmap.jpg")
        parser.add_argument("--divisions", type=int, default=8)
        ...
    ```

14. **Support Multiple Motion Metrics**
    ```python
    def __init__(self, ..., motion_metric='std'):
        # Options: 'std', 'variance', 'range', 'energy'
    ```

15. **Add Configuration File Support**
    ```yaml
    # config.yaml
    num_vertical_divisions: 8
    num_horizontal_divisions: 8
    sigma: 2.0
    color_intensity_factor: 5
    ```

---

## Performance Analysis

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Frame reading | O(N × H × W) | N frames, H×W pixels |
| Pixel sampling | O(N × B) | B blocks (divisions²) |
| Filtering | O(N × B) | Per-block filtfilt |
| Std computation | O(N × B) | Per-block std |
| Gaussian smoothing | O(B²) | 2D convolution on heatmap grid |
| Overlay generation | O(H × W) | Per-pixel loop (slow!) |

### Bottlenecks

1. **Per-Pixel Python Loop** (Most Critical)
   - Current: Nested Python loops iterate every pixel
   - ~1000x slower than vectorized operations
   - **Solution:** Use cv2.resize + cv2.addWeighted

2. **Frame Reading**
   - Reads each frame multiple times from disk if not cached
   - **Solution:** Process streaming or use memory-mapped files

3. **Average Image Accumulation**
   - Stores full-resolution float array
   - Memory: ~12 MB for 1920×1080 image
   - **Solution:** Incremental averaging (update running mean)

### Scalability

**Small video (480p, 100 frames, 8×8 grid):**
- Processing time: ~5-10 seconds
- Memory: ~50 MB

**Medium video (1080p, 300 frames, 16×16 grid):**
- Processing time: ~60-120 seconds (estimate)
- Memory: ~200 MB

**Large video (4K, 1000 frames, 32×32 grid):**
- Processing time: 10+ minutes (estimate)
- Memory: ~1 GB

**Optimized version (vectorized overlay):**
- Expected 50-100× speedup on overlay step
- Overall ~10-20× faster

---

## Edge Cases

### Input Edge Cases

1. **Empty Image List**
   - `images=[]`
   - **Current:** Crashes (IndexError)
   - **Should:** Raise ValueError with clear message

2. **Single Frame**
   - `images=["frame.jpg"]`
   - **Current:** May crash in filtfilt (need 2+ points)
   - **Should:** Raise ValueError or return zero heatmap

3. **Two Frames**
   - `images=["f1.jpg", "f2.jpg"]`
   - **Current:** Insufficient for filtfilt
   - **Should:** Document minimum frame requirement (suggest 10+)

4. **Non-Existent Files**
   - `images=["missing.jpg"]`
   - **Current:** cv2.imread returns None → crash
   - **Should:** Raise FileNotFoundError

5. **Mismatched Dimensions**
   - Frames with different resolutions
   - **Current:** Crashes or corrupts output
   - **Should:** Validate and raise ValueError

### Parameter Edge Cases

6. **divisions = 0**
   - **Current:** Division by zero
   - **Should:** Raise ValueError

7. **divisions > image dimension**
   - E.g., 100×100 image, 200 divisions
   - **Current:** Invalid pixel sampling
   - **Should:** Raise ValueError or auto-limit

8. **sigma = 0**
   - No smoothing
   - **Current:** Works but produces blocky heatmap
   - **Should:** Document minimum recommended value

9. **color_intensity_factor = 0**
   - No color overlay
   - **Current:** Works (invisible heatmap)
   - **Should:** Document or warn

10. **Negative Parameters**
    - sigma < 0, divisions < 0, etc.
    - **Current:** Undefined behavior
    - **Should:** Validate and raise ValueError

### Algorithmic Edge Cases

11. **All Frames Identical**
    - No motion
    - **Expected:** Uniform low-intensity heatmap
    - **Risk:** Std=0 everywhere, divisions by zero?

12. **Single Moving Pixel**
    - Most pixels static, one pixel changes
    - **Expected:** Small hotspot
    - **Risk:** May be missed if not in sampled pixel set

13. **Camera Shake**
    - Entire frame shifts slightly each frame
    - **Current:** High-pass filter should help
    - **Risk:** May still show false motion everywhere

14. **Lighting Changes**
    - Gradual brightness change across sequence
    - **Current:** High-pass filter should remove
    - **Risk:** Sudden lighting changes may register as motion

---

## Additional Notes

### Why This Approach?

**Advantages:**
- Simple and interpretable
- No training required (not ML-based)
- Fast per-frame processing (sampling reduces computation)
- Works with any image sequence

**Disadvantages:**
- Less accurate than optical flow methods
- Cannot distinguish direction or velocity
- Sensitive to noise and lighting
- Single-pixel sampling is unreliable

### Alternative Approaches

1. **Optical Flow (e.g., Farneback, Lucas-Kanade)**
   - More accurate motion vectors
   - Can visualize direction and speed
   - More computationally expensive

2. **Background Subtraction**
   - Separates foreground motion from static background
   - Good for surveillance applications
   - Requires background model

3. **Frame Differencing**
   - Simple: abs(frame[t] - frame[t-1])
   - Fast but noisy
   - No temporal filtering

4. **Deep Learning (e.g., optical flow CNNs)**
   - State-of-the-art accuracy
   - Requires GPU and trained model
   - Overkill for simple visualization

### Use Cases

**Good for:**
- Quick visualization of motion patterns
- Surveillance footage analysis (where was activity?)
- Traffic flow studies (which lanes are busiest?)
- Sports analysis (player movement density)
- User testing (where do users look/interact?)

**Not good for:**
- Precise motion tracking
- Real-time processing (too slow)
- Directional motion analysis
- Small/fast object detection

---

## Summary for AI Consumption

**Key Takeaways:**

1. **Project Goal:** Visualize motion patterns in video by generating color heatmaps
2. **Core Algorithm:** Sample pixel intensities → high-pass filter → compute std → smooth → overlay colors
3. **Main Class:** `MotionHeatmapGenerator` (single class, procedural design)
4. **Status:** Alpha stage, functional but has bugs and performance issues
5. **Critical Bugs:** Import error in `__init__.py`, packaging mismatch, no input validation
6. **Performance:** Slow due to nested Python loops; needs vectorization
7. **Testing:** No working tests; needs comprehensive test suite
8. **Dependencies:** OpenCV, NumPy, SciPy (uses deprecated API)
9. **Improvements Needed:** Fix imports, validate inputs, vectorize overlay, add tests, update docs

**If You Need to Modify This Project:**
1. First, fix the import bug in `__init__.py`
2. Add input validation (empty lists, file existence, None checks)
3. Replace deprecated `scipy.ndimage.filters` with `scipy.ndimage`
4. Consider vectorizing the overlay generation (major speedup)
5. Add comprehensive unit tests with synthetic data
6. Update README (remove duplicates, fix formatting)

**If You Need to Use This Project:**
1. Ensure you have OpenCV, NumPy, SciPy installed
2. Extract video frames to separate images first
3. Create list of frame paths in order
4. Instantiate `MotionHeatmapGenerator` with desired grid resolution
5. Call `generate_motion_heatmap()` to save output
6. Experiment with `sigma` and `color_intensity_factor` for visual tuning

---

**Document Version:** 1.0  
**Last Updated:** November 4, 2025  
**Prepared for:** AI Analysis and Code Understanding
