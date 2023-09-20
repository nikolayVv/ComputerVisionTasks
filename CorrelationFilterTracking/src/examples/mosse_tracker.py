import numpy as np
import cv2
from utils.ex3_utils import get_patch, create_cosine_window, create_gauss_peak
from utils.tracker import Tracker
from utils.correlation import generate_filter, mosse_filter, generate_corr_response
from numpy.fft import fft2


class MosseTracker(Tracker):

    def name(self):
        return 'mosse'
    
    def __init__(self, enlarge_factor, alpha, beta, gamma, sigma, adaptive_scale):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.enlarge_factor = enlarge_factor
        self.adaptive_scale = adaptive_scale

        self.filter = None
        self.size = None
        self.window = None
        self.template = None
        self.position = None
        self.cosine_window = None
        self.gaussian = None


    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = round(max(round(region[2]), round(region[3])) * self.enlarge_factor)

        # Make sure the template and the kernel are the same size
        if self.window % 2 == 0:
            self.window += 1
        
        # INITIALIZATION
        self.size = np.array([region[2], region[3]]).astype(np.float32)
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.cosine_window = create_cosine_window((self.window, self.window))
        self.gaussian = create_gauss_peak((self.window, self.window), self.sigma)
        self.gaussian = fft2(self.gaussian)

        
        # PREPROCESS PATCH
        # Extract the patch
        patch, _ = get_patch(image, self.position, (self.window, self.window))
        # Convert it to grayscale
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Transform pixel values using log function
        c = 255 / np.log(1 + np.max(patch))
        patch = np.log(c * patch.astype(np.float32) + 1)
        # Normalize pixel values
        patch =  np.divide(patch - np.min(patch), np.max(patch) - np.min(patch)).astype(np.float32)

        # Calculate the filter
        patch = np.multiply(patch, self.cosine_window)
        self.filter = generate_filter(patch, self.gamma, self.gaussian)

    def track(self, image):
        # ADAPTIVE SCALE
        # Compute the 3 different scales
        scale = np.round(self.adaptive_scale * self.size)
        scale_2 = np.add(self.size, scale)
        scale_3 = np.subtract(self.size, scale)

        # Compute the 3 different windows
        window_1 = self.window
        window_2 = round(self.window + (self.window * self.adaptive_scale))
        window_3 = round(self.window - (self.window * self.adaptive_scale))

        # Compute the correlation response for all 3 cases
        corr_res_1 = generate_corr_response(image, self.position, (window_1, window_1), self.cosine_window, self.filter)
        corr_res_2 = generate_corr_response(image, self.position, (window_2, window_2), self.cosine_window, self.filter)
        corr_res_3 = generate_corr_response(image, self.position, (window_3, window_3), self.cosine_window, self.filter)

        # Find the maximum correlation response and set its size and window as the tracker's new values
        max_1 = np.max(corr_res_1)
        max_2 = np.max(corr_res_2)
        max_3 = np.max(corr_res_3)
        if max_2 > max_3:
            if max_1 < max_2:
                self.size = (self.beta * scale_2) + ((1 - self.beta) * self.size)
                self.window = window_2
        else:
            if max_1 < max_3:
                self.size = (self.beta * scale_3) + ((1 - self.beta) * self.size)
                self.window = window_3

        # MOSSE TRACKING
        # Update the target position and the filter
        self.position, self.filter = mosse_filter(image, self.position, (self.window, self.window), self.alpha, self.gamma, self.filter, self.cosine_window, self.gaussian)
        return [self.position[0] - (self.size[0] / 2), self.position[1] - (self.size[1] / 2), self.size[0], self.size[1]]