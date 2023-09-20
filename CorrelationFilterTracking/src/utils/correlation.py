import cv2
import numpy as np

from utils.ex3_utils import get_patch
from numpy.fft import fft2, ifft2


def mosse_filter(img, center, window_size, alpha, gamma, old_h, cosine_window, gaussian):
    # LOCALIZATION
    # Generate the correlation response
    corr_response = generate_corr_response(img, center, window_size, cosine_window, old_h)

    # Calculate the new position of the target
    y, x = np.unravel_index(corr_response.argmax(), corr_response.shape)
    if x > window_size[0] / 2:
        x = x - window_size[0]
    if y > window_size[1] / 2:
        y = y - window_size[1]
    new_center = (center[0] + x, center[1] + y)

    # UPDATE
    # Extract the patch
    new_patch, _ = get_patch(img, new_center, window_size)
    # Convert it into grayscale
    new_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
    # Transform pixel values using log function
    c = 255 / np.log(1 + np.max(new_patch))
    new_patch = np.log(c * new_patch.astype(np.float32) + 1)
    # Normalize pixel values
    new_patch = np.divide(new_patch - np.min(new_patch), np.max(new_patch) - np.min(new_patch)).astype(np.float32)

    # Calculate the new filter
    new_patch = np.multiply(new_patch, cosine_window)
    h = generate_filter(new_patch, gamma, gaussian)
    new_h = ((1 - alpha) * old_h) + (alpha * h)

    return new_center, new_h

def mosse_filter_improved(img, center, window_size, alpha, gamma, old_h, cosine_window, gaussian, learning_rate, old_numerator, old_denominator, psr_threshold):
    # LOCALIZATION
    # Generate the correlation response
    corr_response = generate_corr_response(img, center, window_size, cosine_window, old_h)

    max_value = np.max(corr_response)
    y, x = np.unravel_index(corr_response.argmax(), corr_response.shape)
    if x > window_size[0] / 2:
        x = x - window_size[0]
    if y > window_size[1] / 2:
        y = y - window_size[1]

    sidelobe = np.copy(corr_response)
    sidelobe[y, x] = 0
    sidelobe[(y-5):(y+6), (x-5):(x+6)] = 0

    mean_sidelobe = np.mean(sidelobe)
    std_sidelobe = np.std(sidelobe)
    psr = (max_value - mean_sidelobe) / std_sidelobe
    
    if psr < psr_threshold:
        new_center = center
        new_h = old_h
        numerator = old_numerator
        denominator = old_denominator
    else:
        new_center = (center[0] + x, center[1] + y)

        # UPDATE
        # Extract the patch
        new_patch, _ = get_patch(img, new_center, window_size)
        # Convert it into grayscale
        new_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
        # Transform pixel values using log function
        c = 255 / np.log(1 + np.max(new_patch))
        new_patch = np.log(c * new_patch.astype(np.float32) + 1)
        # Normalize pixel values
        new_patch = np.divide(new_patch - np.min(new_patch), np.max(new_patch) - np.min(new_patch)).astype(np.float32)

        # Calculate the new filter
        new_patch = np.multiply(new_patch, cosine_window)
        h, numerator, denominator = generate_filter_improved(new_patch, gamma, gaussian, learning_rate, old_numerator, old_denominator)
        new_h = ((1 - alpha) * old_h) + (alpha * h)

    return new_center, new_h, numerator, denominator

def generate_filter(patch, gamma, gaussian):
    f = fft2(patch)
    f_conj = np.conjugate(f)
    
    return np.divide(np.multiply(gaussian, f_conj), np.multiply(f, f_conj) + gamma)


def generate_filter_improved(patch, gamma, gaussian, learning_rate, old_numerator, old_denominator):
    f = fft2(patch)
    f_conj = np.conjugate(f)
    # Update the numerator and denominator
    numerator = (learning_rate * np.multiply(gaussian, f_conj)) + ((1 - learning_rate) * old_numerator)
    denominator = ((learning_rate * np.multiply(f, f_conj)) + gamma) + ((1 - learning_rate) * old_denominator)

    return np.divide(numerator, denominator), numerator, denominator 


def generate_corr_response(img, center, window_size, cosine_window, h):
    # PREPROCESSING
    # Extract the patch
    patch, _ = get_patch(img, center, window_size)
    # Convert it into grayscale
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Transform pixel values using log function
    c = 255 / np.log(1 + np.max(patch))
    patch = np.log(c * patch.astype(np.float32) + 1)
    # Normalize pixel values
    patch = np.divide(patch - np.min(patch), np.max(patch) - np.min(patch)).astype(np.float32)

    patch = np.multiply(patch, cosine_window)
    f = fft2(patch)

    return ifft2(np.multiply(h, f))
