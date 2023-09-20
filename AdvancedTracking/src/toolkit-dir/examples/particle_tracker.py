import numpy as np
import cv2
from utils.ex2_utils import create_epanechnik_kernel, extract_histogram, get_patch
from utils.ex4_utils import sample_gauss
from utils.tracker import Tracker
from utils.recursive_bayes import get_dynamic_model, particle_filter

class ParticleTracker(Tracker):

    def name(self):
        return 'particle'
    
    def __init__(self, enlarge_factor, num_particles, bins, hist_color, motion_model, q_factor, alpha, distance_sigma, kernel_sigma):
        super().__init__()

        self.alpha = alpha
        self.distance_sigma = distance_sigma
        self.kernel_sigma = kernel_sigma
        self.enlarge_factor = enlarge_factor
        self.num_particles = num_particles
        self.bins = bins
        self.hist_color = hist_color
        self.motion_model = motion_model
        self.q_factor = q_factor

        self.kernel = None
        self.size = None
        self.window = None
        self.template = None
        self.position = None
        self.histogram = None
        self.system_matrix = None
        self.system_covariance = None
        self.particles = None
        self.weights = None
        self.q = None

    def initialize(self, image, region):
        region = [int(pos) for pos in region]

        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1

        # Change the color of the image
        if self.hist_color == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.hist_color == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.hist_color == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.hist_color == 'YCRCB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.enlarge_factor
        
        # INITIALIZATION
        self.size = np.array([region[2], region[3]])
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.template, _ = get_patch(image, self.position, self.size)

        self.q = max(0, int((image.shape[0] * image.shape[1]) / (self.size[0] * self.size[1]) * self.q_factor))

        # Generate visual model
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.kernel_sigma)
        self.histogram = extract_histogram(self.template, self.bins, self.kernel)
        self.histogram = np.divide(self.histogram, np.sum(self.histogram))

        self.system_matrix, self.system_covariance = get_dynamic_model(self.motion_model, self.q)
        state = np.zeros((self.system_matrix.shape[0], 1), dtype=np.float32).flatten()
        state[0] = self.position[0]
        state[1] = self.position[1]

        # Generate n particles
        self.particles = sample_gauss(state, self.system_covariance, self.num_particles)
        self.weights = np.divide(np.ones(self.num_particles), self.num_particles)
        
    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]
        
        # Change the color of the image
        if self.hist_color == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.hist_color == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.hist_color == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.hist_color == 'YCRCB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        # RESAMPLING
        weights_cumsumed = np.cumsum(self.weights)
        rand_samples = np.random.rand(self.num_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        particles_new = self.particles[sampled_idxs.flatten(), :]

        # PREDICT
        self.position, self.weights, self.particles = particle_filter(image, self.system_matrix, self.system_covariance, self.weights, self.num_particles, 
                                                        particles_new, self.kernel, self.bins, self.histogram, self.distance_sigma, self.position, self.particles)
        # UPDATE
        self.template, _ = get_patch(image, self.position, self.kernel.shape)
        new_histogram = extract_histogram(self.template, self.bins, self.kernel)
        new_histogram = np.divide(new_histogram, np.sum(new_histogram))
        self.histogram = self.alpha * new_histogram + (1 - self.alpha) * self.histogram
        self.histogram = np.divide(self.histogram, np.sum(self.histogram))

        return [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]]