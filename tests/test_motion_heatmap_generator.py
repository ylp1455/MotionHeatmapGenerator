import unittest
from src.motion_heatmap_generator import MotionHeatmapGenerator

class TestMotionHeatmapGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = MotionHeatmapGenerator(num_vertical_divisions=2, num_horizontal_divisions=2, images=["test1.jpg", "test2.jpg"])

    def test_generate_motion_heatmap(self):
        # Add your test cases here
        pass

if __name__ == '__main__':
    unittest.main()
