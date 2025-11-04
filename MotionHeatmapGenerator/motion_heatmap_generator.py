import collections
import itertools
import math
import random
import cv2
import scipy.ndimage
import scipy.signal
import numpy as np

class MotionHeatmapGenerator:
    """
    Generate motion heatmaps from video frame sequences.
    
    This class analyzes temporal intensity variations across a sequence
    of images to identify regions of motion, producing a color-coded
    heatmap overlay.
    
    Parameters
    ----------
    num_vertical_divisions : int
        Number of vertical blocks in heatmap grid (must be > 0)
    num_horizontal_divisions : int
        Number of horizontal blocks in heatmap grid (must be > 0)
    images : list of str
        Ordered list of image file paths (must be non-empty)
    use_average_image_overlay : bool, optional
        If True, overlay on averaged frame; if False, overlay on first frame (default: True)
    sigma : float, optional
        Gaussian smoothing standard deviation (default: 1.5)
    color_intensity_factor : int, optional
        Multiplier for color overlay intensity (default: 7)
    print_debug : bool, optional
        Print progress messages during processing (default: True)
    random_seed : int, optional
        Random seed for deterministic pixel sampling (default: None)
        
    Raises
    ------
    ValueError
        If images list is empty, divisions are invalid, or parameters are negative
    FileNotFoundError
        If first image file cannot be read
    """
    def __init__(
        self,
        num_vertical_divisions,
        num_horizontal_divisions,
        images,
        use_average_image_overlay=True,
        sigma=1.5,
        color_intensity_factor=7,
        print_debug=True,
        random_seed=None,
    ):
        # Input validation
        if not images:
            raise ValueError("images list cannot be empty")
        if num_vertical_divisions <= 0:
            raise ValueError(f"num_vertical_divisions must be > 0, got {num_vertical_divisions}")
        if num_horizontal_divisions <= 0:
            raise ValueError(f"num_horizontal_divisions must be > 0, got {num_horizontal_divisions}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
            
        self.num_vertical_divisions = num_vertical_divisions
        self.num_horizontal_divisions = num_horizontal_divisions
        self.color_intensity_factor = color_intensity_factor
        self.use_average_image_overlay = use_average_image_overlay
        self.images = images
        self.print_debug = print_debug
        
        # Set random seed for deterministic output if provided
        if random_seed is not None:
            random.seed(random_seed)

        # Read and validate first image
        sample_image = cv2.imread(self.images[0])
        if sample_image is None:
            raise FileNotFoundError(f"Cannot read image: {self.images[0]}")
        
        # Use proper shape extraction
        self.height, self.width = sample_image.shape[:2]
        
        # Validate that divisions are reasonable
        if num_vertical_divisions > self.height:
            raise ValueError(f"num_vertical_divisions ({num_vertical_divisions}) cannot exceed image height ({self.height})")
        if num_horizontal_divisions > self.width:
            raise ValueError(f"num_horizontal_divisions ({num_horizontal_divisions}) cannot exceed image width ({self.width})")

        if self.height % self.num_vertical_divisions!= 0:
            print('Warning: number of vertical divisions {} isn\'t equally divisible by the image height {}; this will result in a blocky output image.'.format(
                self.num_vertical_divisions,
                self.height,
            ))
        if self.width % self.num_horizontal_divisions!= 0:
            print('Warning: number of horizontal divisions {} isn\'t equally divisible by the image width {}; this will result in a blocky output image.'.format(
                self.num_horizontal_divisions,
                self.width,
            ))

        self.pixel_locations = {}
        for row, col in itertools.product(range(self.num_vertical_divisions), range(self.num_horizontal_divisions)):
            self.pixel_locations[(row, col)] = (
                int(row * self.height / self.num_vertical_divisions + math.floor(random.random() * self.height / self.num_vertical_divisions)),
                int(col * self.width / self.num_horizontal_divisions + math.floor(random.random() * self.width / self.num_horizontal_divisions)),
            )

        self.block_intensities = collections.defaultdict(list)
        self.average_image = np.zeros((self.height, self.width, 3))
        for index, file_name in enumerate(self.images):
            if self.print_debug:
                print('Processing input frame {} of {}'.format(index + 1, len(self.images)))
            frame = cv2.imread(file_name, cv2.IMREAD_COLOR)
            
            # Validate frame was read successfully
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {file_name}")
            
            # Validate frame dimensions match
            if frame.shape[:2] != (self.height, self.width):
                raise ValueError(
                    f"Image dimension mismatch: {file_name} has shape {frame.shape[:2]}, "
                    f"expected ({self.height}, {self.width})"
                )
            
            if self.use_average_image_overlay:
                self.average_image += frame
            for row, col in itertools.product(range(self.num_vertical_divisions), range(self.num_horizontal_divisions)):
                pixel_row, pixel_col = self.pixel_locations[(row, col)]
                self.block_intensities[(row, col)].append(round(np.mean(frame[pixel_row][pixel_col])))
        
        # Validate sufficient frames for filtering
        min_frames_required = 10
        if len(self.images) < min_frames_required:
            if self.print_debug:
                print(f'Warning: Only {len(self.images)} frames provided. For best results, use at least {min_frames_required} frames.')
        
        # Apply high-pass filter to remove low-frequency trends
        b, a = scipy.signal.butter(5, 0.2, 'high')
        for block, intensity in self.block_intensities.items():
            # Use adaptive padlen to avoid errors with short sequences
            padlen = min(len(intensity) - 1, 3 * max(len(a), len(b)))
            if padlen > 0:
                self.block_intensities[block] = scipy.signal.filtfilt(b, a, intensity, padlen=padlen)
            else:
                # Skip filtering for very short sequences
                self.block_intensities[block] = intensity
                
        if self.use_average_image_overlay:
            self.average_image /= len(self.images)

        unfiltered_heatmap = np.zeros((self.num_vertical_divisions, self.num_horizontal_divisions))
        for row, col in itertools.product(range(self.num_vertical_divisions), range(self.num_horizontal_divisions)):
            unfiltered_heatmap[row][col] = np.std(self.block_intensities[(row, col)])
        
        # Use non-deprecated scipy API
        self.heatmap = scipy.ndimage.gaussian_filter(unfiltered_heatmap, sigma=sigma)

    def generate_motion_heatmap(self, file_name='motion_heatmap.jpg'):
        """
        Generate and save the motion heatmap image.
        
        Parameters
        ----------
        file_name : str, optional
            Output file path (default: 'motion_heatmap.jpg')
            
        Returns
        -------
        bool
            True if save successful, False otherwise
        """
        # Get base image and ensure proper dtype
        if self.use_average_image_overlay:
            output_image = self.average_image.copy()
        else:
            output_image = cv2.imread(self.images[0], cv2.IMREAD_COLOR)
            if output_image is None:
                raise FileNotFoundError(f"Cannot read image: {self.images[0]}")
            output_image = output_image.astype(np.float64)
        
        mean_stdev = np.mean(self.heatmap)

        for vertical_index, horizontal_index in itertools.product(range(self.num_vertical_divisions), range(self.num_horizontal_divisions)):
            if self.print_debug:
                print('Processing output block {} of {}'.format(
                    vertical_index * self.num_horizontal_divisions + horizontal_index + 1,
                    self.num_horizontal_divisions * self.num_vertical_divisions,
                ))
            offset = self.color_intensity_factor * (self.heatmap[vertical_index][horizontal_index] - mean_stdev)
            for i, j in itertools.product(range(self.height // self.num_vertical_divisions), range(self.width // self.num_horizontal_divisions)):
                row = vertical_index * self.height / self.num_vertical_divisions + i
                col = horizontal_index * self.width / self.num_horizontal_divisions + j

                row = int(row)
                col = int(col)
                output_image[row][col][2] = self._clip_rgb(output_image[row][col][2] + offset)
                output_image[row][col][0] = self._clip_rgb(output_image[row][col][0] - offset)

        # Convert to uint8 before saving
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return cv2.imwrite(file_name, output_image)

    @staticmethod
    def _clip_rgb(value):
        return int(max(min(value, 255), 0))
