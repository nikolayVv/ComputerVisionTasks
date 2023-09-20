import math
import numpy as np
import cv2
from numpy.fft import fft2, ifft2
from matplotlib.colors import hsv_to_rgb


# Computes first and second Gaussian derivatives of an image
def gauss_deriv(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    
    D = -2 * (x * np.exp(-x**2 / (2 * sigma**2))) / (np.sqrt(2 * math.pi) * sigma**3)
    D = D / (np.sum(np.abs(D)) / 2)
    
    Dx = cv2.sepFilter2D(img, -1, D, G)
    Dy = cv2.sepFilter2D(img, -1, G, D)

    return Dx, Dy

# Performs Gaussian smoothing of an image
def gauss_smooth(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)
    
def show_flow(U, V, ax, type='field', set_aspect=False):
    if type == 'field':
        scaling = 0.1
        u = cv2.resize(gauss_smooth(U, 1.5), (0, 0), fx=scaling, fy=scaling)
        v = cv2.resize(gauss_smooth(V, 1.5), (0, 0), fx=scaling, fy=scaling)
        
        x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
        y_ = -(np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
        x, y = np.meshgrid(x_, y_)
        
        ax.quiver(x, y, -u * 5, v * 5)
        if set_aspect:
            ax.set_aspect(1.)
    elif type == 'magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        ax.imshow(np.minimum(1, magnitude))
    elif type == 'angle':
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    elif type == 'angle_magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.minimum(1, magnitude), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    else:
        print('Error: unknown optical flow visualization type.')
        exit(-1)

def rotate_image(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated