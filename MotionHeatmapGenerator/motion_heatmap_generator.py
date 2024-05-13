import collections
import itertools
import math
import random
import cv2
import scipy.ndimage
import scipy.signal
import numpy as np

class MotionHeatmapGenerator:
    def __init__(
        self,
        num_vertical_divisions,
        num_horizontal_divisions,
        images,
        use_average_image_overlay=True,
        sigma=1.5,
        color_intensity_factor=7,
        print_debug=True,
    ):
        self.num_vertical_divisions = num_vertical_divisions
        self.num_horizontal_divisions = num_horizontal_divisions
        self.color_intensity_factor = color_intensity_factor
        self.use_average_image_overlay = use_average_image_overlay
        self.images = images
        self.print_debug = print_debug

        sample_image = cv2.imread(self.images[0])
        self.height = len(sample_image)
        self.width = len(sample_image[0])

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
            if self.use_average_image_overlay:
                self.average_image += frame
            for row, col in itertools.product(range(self.num_vertical_divisions), range(self.num_horizontal_divisions)):
                pixel_row, pixel_col = self.pixel_locations[(row, col)]
                self.block_intensities[(row, col)].append(round(np.mean(frame[pixel_row][pixel_col])))
        b, a = scipy.signal.butter(5, 0.2, 'high')
        for block, intensity in self.block_intensities.items():
            self.block_intensities[block] = scipy.signal.filtfilt(b, a, intensity, padlen=0)
        if self.use_average_image_overlay:
            self.average_image /= len(self.images)

        unfiltered_heatmap = np.zeros((self.num_vertical_divisions, self.num_horizontal_divisions))
        for row, col in itertools.product(range(self.num_vertical_divisions), range(self.num_horizontal_divisions)):
            unfiltered_heatmap[row][col] = np.std(self.block_intensities[(row, col)])
        self.heatmap = scipy.ndimage.filters.gaussian_filter(unfiltered_heatmap, sigma=sigma)

    def generate_motion_heatmap(self, file_name='motion_heatmap.jpg'):
        output_image = self.average_image if self.use_average_image_overlay else cv2.imread(self.images[0], cv2.IMREAD_COLOR)
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

        return cv2.imwrite(file_name, output_image)

    @staticmethod
    def _clip_rgb(value):
        return int(max(min(value, 255), 0))
