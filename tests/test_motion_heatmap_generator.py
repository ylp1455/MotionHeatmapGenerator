import unittest
import os
import tempfile
import numpy as np
import cv2
from MotionHeatmapGenerator import MotionHeatmapGenerator


class TestMotionHeatmapGenerator(unittest.TestCase):
    """Test suite for MotionHeatmapGenerator class."""
    
    def setUp(self):
        """Set up test fixtures - create temporary test images."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_images = []
        
        # Create synthetic test images (100x100 pixels, grayscale converted to BGR)
        for i in range(15):
            image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            # Add a moving bright spot to simulate motion
            x = 20 + i * 3
            y = 50
            if x < 90:
                cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
            
            image_path = os.path.join(self.temp_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(image_path, image)
            self.test_images.append(image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        for image_path in self.test_images:
            if os.path.exists(image_path):
                os.remove(image_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_empty_images_list(self):
        """Test that empty images list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            MotionHeatmapGenerator(2, 2, [])
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_nonexistent_image_file(self):
        """Test that nonexistent image file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            MotionHeatmapGenerator(2, 2, ["nonexistent.jpg"])
    
    def test_invalid_divisions(self):
        """Test that invalid division values raise ValueError."""
        with self.assertRaises(ValueError):
            MotionHeatmapGenerator(0, 2, self.test_images)
        
        with self.assertRaises(ValueError):
            MotionHeatmapGenerator(2, -1, self.test_images)
    
    def test_negative_sigma(self):
        """Test that negative sigma raises ValueError."""
        with self.assertRaises(ValueError):
            MotionHeatmapGenerator(2, 2, self.test_images, sigma=-1.0)
    
    def test_basic_initialization(self):
        """Test that generator initializes successfully with valid inputs."""
        generator = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            print_debug=False
        )
        self.assertEqual(generator.height, 100)
        self.assertEqual(generator.width, 100)
        self.assertIsNotNone(generator.heatmap)
    
    def test_deterministic_output_with_seed(self):
        """Test that same seed produces same heatmap."""
        gen1 = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            random_seed=42,
            print_debug=False
        )
        
        gen2 = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            random_seed=42,
            print_debug=False
        )
        
        np.testing.assert_array_almost_equal(gen1.heatmap, gen2.heatmap)
    
    def test_generate_motion_heatmap_creates_file(self):
        """Test that generate_motion_heatmap creates output file."""
        generator = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            print_debug=False
        )
        
        output_path = os.path.join(self.temp_dir, "test_heatmap.jpg")
        result = generator.generate_motion_heatmap(output_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_output_image_dimensions(self):
        """Test that output image has correct dimensions."""
        generator = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            print_debug=False
        )
        
        output_path = os.path.join(self.temp_dir, "test_heatmap.jpg")
        generator.generate_motion_heatmap(output_path)
        
        output_image = cv2.imread(output_path)
        self.assertEqual(output_image.shape[:2], (100, 100))
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_moving_object_produces_motion(self):
        """Test that moving object produces non-zero motion values."""
        generator = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            print_debug=False
        )
        
        # Heatmap should have variation (not all zeros)
        self.assertGreater(np.max(generator.heatmap), 0)
        self.assertGreater(np.std(generator.heatmap), 0)
    
    def test_use_first_frame_overlay(self):
        """Test overlay on first frame instead of average."""
        generator = MotionHeatmapGenerator(
            num_vertical_divisions=4,
            num_horizontal_divisions=4,
            images=self.test_images,
            use_average_image_overlay=False,
            print_debug=False
        )
        
        output_path = os.path.join(self.temp_dir, "test_first_frame.jpg")
        result = generator.generate_motion_heatmap(output_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
