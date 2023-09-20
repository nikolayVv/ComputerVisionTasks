import numpy as np
import cv2
from ms_method import get_mean_shift_vector
from ex2_utils import Tracker, extract_histogram, create_epanechnik_kernel, create_gaussian_kernel, create_uniform_kernel, backproject_histogram, get_patch


class MeanShiftTracker(Tracker):

    def initialize(self, image, region, kernel_shape, n_bins):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(round(region[2]), round(region[3]))
        
        # Initialization
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = np.array([region[2], region[3]]).astype(float)

        # Make sure the template and the kernel are the same size
        if self.window % 2 == 0:
            self.window += 1

        patch, _ = get_patch(image, self.position, (self.window, self.window))
        self.template = patch
        self.n_bins = n_bins
        self.kernel_shape = kernel_shape
        if self.kernel_shape == 'epanechnikov':
            kernel = create_epanechnik_kernel(int(self.window), int(self.window), 1)
        elif self.kernel_shape == 'gaussian':
            kernel = create_gaussian_kernel(int(self.window), int(self.window), 1)

        # Calculate the PDF of the template
        self.q = extract_histogram(patch, self.n_bins, kernel)
        # Normalize
        self.q = self.q / np.sum(self.q)

    def track(self, image, min_shifting, max_iter, alpha, adaptive_scale, gamma):
        # Calculate size and windows for different scales
        scale = np.round(adaptive_scale * self.size)
        scale_2 = np.add(self.size, scale)
        scale_3 = np.subtract(self.size, scale)

        window_1 = self.window
        window_2 = round(self.window + (self.window * adaptive_scale))
        window_3 = round(self.window - (self.window * adaptive_scale))

        if window_2 % 2 == 0:
            window_2 += 1

        if window_3 % 2 == 0:
            window_3 += 1

        # Apply adaptive scaling
        left_1 = max(round(self.position[0] - float(window_1) / self.parameters.enlarge_factor), 0)
        top_1 = max(round(self.position[1] - float(window_1) / self.parameters.enlarge_factor), 0)

        right_1 = min(round(self.position[0] + float(window_1) / self.parameters.enlarge_factor), image.shape[1] - 1)
        bottom_1 = min(round(self.position[1] + float(window_1) / self.parameters.enlarge_factor), image.shape[0] - 1)

        if right_1 - left_1 < self.template.shape[1] or bottom_1 - top_1 < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]
        else:
            if self.kernel_shape == 'epanechnikov':
                kernel_1 = create_epanechnik_kernel(int(window_1), int(window_1), 1)
            elif self.kernel_shape == 'gaussian':
                kernel_1 = create_gaussian_kernel(int(window_1), int(window_1), 1)

            patch_1, _ = get_patch(image, self.position, (window_1, window_1))
            p_1 = extract_histogram(patch_1, self.n_bins, kernel_1)
            # Normalize
            p_1 = p_1 / np.sum(p_1)
            bhattacharyya_1 = np.sum(np.sqrt(p_1 * self.q))

        left_2 = max(round(self.position[0] - float(window_2) / self.parameters.enlarge_factor), 0)
        top_2 = max(round(self.position[1] - float(window_2) / self.parameters.enlarge_factor), 0)

        right_2 = min(round(self.position[0] + float(window_2) / self.parameters.enlarge_factor), image.shape[1] - 1)
        bottom_2 = min(round(self.position[1] + float(window_2) / self.parameters.enlarge_factor), image.shape[0] - 1)

        if right_2 - left_2 < self.template.shape[1] or bottom_2 - top_2 < self.template.shape[0]:
            bhattacharyya_2 = 0
        else:
            if self.kernel_shape == 'epanechnikov':
                kernel_2 = create_epanechnik_kernel(int(window_2), int(window_2), 1)
            elif self.kernel_shape == 'gaussian':
                kernel_2 = create_gaussian_kernel(int(window_2), int(window_2), 1)

            patch_2, _ = get_patch(image, self.position, (window_2, window_2))
            p_2 = extract_histogram(patch_2, self.n_bins, kernel_2)
            # Normalize
            p_2 = p_2 / np.sum(p_2)
            bhattacharyya_2 = np.sum(np.sqrt(p_2 * self.q))

        left_3 = max(round(self.position[0] - float(window_3) / self.parameters.enlarge_factor), 0)
        top_3 = max(round(self.position[1] - float(window_3) / self.parameters.enlarge_factor), 0)

        right_3 = min(round(self.position[0] + float(window_3) / self.parameters.enlarge_factor), image.shape[1] - 1)
        bottom_3 = min(round(self.position[1] + float(window_3) / self.parameters.enlarge_factor), image.shape[0] - 1)

        if right_3 - left_3 < self.template.shape[1] or bottom_3 - top_3 < self.template.shape[0]:
            bhattacharyya_3 = 0
        else:
            if self.kernel_shape == 'epanechnikov':
                kernel_3 = create_epanechnik_kernel(int(window_3), int(window_3), 1)
            elif self.kernel_shape == 'gaussian':
                kernel_3 = create_gaussian_kernel(int(window_3), int(window_3), 1)

            patch_3, _ = get_patch(image, self.position, (window_3, window_3))
            p_3 = extract_histogram(patch_3, self.n_bins, kernel_3)
            # Normalize
            p_3 = p_3 / np.sum(p_3)
            bhattacharyya_3 = np.sum(np.sqrt(p_3 * self.q))

        # Compare the bhattacharyya measures and adapt the scale
        if bhattacharyya_2 > bhattacharyya_3:
            if bhattacharyya_1 < bhattacharyya_2:
                self.size = (gamma * scale_2) + ((1 - gamma) * self.size)
                self.window = window_2
        else:
            if bhattacharyya_1 < bhattacharyya_3:
                self.size = (gamma * scale_3) + ((1 - gamma) * self.size)
                self.window = window_3
        
        if self.kernel_shape == 'epanechnikov':
            kernel = create_epanechnik_kernel(int(self.window), int(self.window), 1)
            df_kernel = create_uniform_kernel(int(self.window), int(self.window))
        elif self.kernel_shape == 'gaussian':
            kernel = create_gaussian_kernel(int(self.window), int(self.window), 1)
            df_kernel = kernel

        x_coord = np.tile(np.arange(-int(self.window//2), int(self.window//2) + 1), (int(self.window), 1))
        y_coord = np.tile(np.arange(-int(self.window//2), int(self.window//2) + 1), (int(self.window), 1)).T 
        shift = np.inf
        curr_iter = 0
        center = self.position
        # Target localization
        while shift > min_shifting and curr_iter < max_iter:
            patch, _ = get_patch(image, center, (self.window, self.window))
            # Calculate the PDF of the target
            p = extract_histogram(patch, self.n_bins, kernel)
            # Normalize
            p = p / np.sum(p)
            v = np.sqrt(self.q / (p + 1e-3))
            w = backproject_histogram(patch, v, self.n_bins)
            center, shift = get_mean_shift_vector(w, center, df_kernel, x_coord, y_coord)
            curr_iter += 1
        
        # Model update
        self.position = (center[0], center[1])
        new_patch, _ = get_patch(image, self.position, (self.window, self.window))
        self.template = new_patch
        new_q = extract_histogram(new_patch, self.n_bins, kernel)
        self.q = ((1 - alpha) * self.q) + (alpha * new_q)
        self.q = self.q / np.sum(self.q)

        return [center[0] - self.size[0] / 2, center[1] - self.size[1] / 2, self.size[0], self.size[1]]

class MSParams():
    def __init__(self):
        self.enlarge_factor = 2

