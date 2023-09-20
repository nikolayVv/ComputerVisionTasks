import numpy as np
import cv2
import time
from scipy import signal, ndimage
from ex1_utils import gauss_deriv, gauss_smooth

# im1 − first image matrix (grayscale)
# im2 − second image matrix (grayscale)
# N − size of the neighborhood (N x N)
# show_log - shows log of the Lucas-Kanade's execution time
# threshold_determinant - threshold for the determinant to make sure it is not 0
# threshold_harris - threshold for the Harris response
# alpha - constant used to calculate the harris_response
def lucas_kanade(im1, im2, N = 3, threshold_determinant = 1e-3, threshold_harris = 1e-8, alpha = 0.02, show_log = True):
    start_time = time.perf_counter()
    kernel = np.ones((N, N))

    # For each pixel calculate I_x, I_y and I_t
    I_x, I_y = gauss_deriv((im1 + im2) / 2, 1)
    I_t = gauss_smooth(im2 - im1, 1)
    I_x = gauss_smooth(I_x, 1)
    I_y = gauss_smooth(I_y, 1)

    # Calculate the summations by using convolution with a kernel
    I_x2 =  signal.convolve2d(I_x * I_x, kernel, mode='same')
    I_y2 =  signal.convolve2d(I_y * I_y, kernel, mode='same')
    I_xI_y =  signal.convolve2d(I_x * I_y, kernel, mode='same')
    I_xI_t =  signal.convolve2d(I_x * I_t, kernel, mode='same')
    I_yI_t =  signal.convolve2d(I_y * I_t, kernel, mode='same')

    # Calculate the value of the determinant and
    D = (I_x2 * I_y2) - (I_xI_y ** 2)
    trace = I_x2 + I_y2

    # Calculate the Harris response
    # D = determinant = multiplication of both eigenvalues
    # trace = addition of both eigenvalues
    harris_response = D - (alpha * (trace ** 2))
    threshold_harris = threshold_harris * np.max(harris_response)
    # If value is maximum in its neighborhood
    # or bigger than the threshold
    #print(ndimage.filters.maximum_filter(harris_response, size=(N, N)))
    max_filtered = ndimage.filters.maximum_filter(harris_response, size=(N, N), mode='constant')
    mask = np.where((harris_response >= threshold_harris) | (harris_response == max_filtered), 1, 0)
    # make sure the determinant is not 0
    D[D < threshold_determinant] = threshold_determinant

    # Calculate u and v
    u = ((I_y2 * I_xI_t) - (I_xI_y * I_yI_t)) / D
    v = ((I_x2 * I_yI_t) - (I_xI_y * I_xI_t)) / D

    end_time = time.perf_counter()
    if show_log:
        print(f"Lucas-Kanade's execution time for N={N} is: {end_time - start_time}s.")

    # Return the u and v components with the applied mask from the Harris response
    return -(u * mask), -(v * mask)

# im1 − first image matrix (grayscale)
# im2 − second image matrix (grayscale)
# n_iters − number of iterations (try several hundred)
# lmbd − constant used to calculate the value of D
# show_log - shows log of the Lucas-Kanade's execution time
def horn_schunck(im1, im2, n_iters = 1000, lmbd = 0.5, threshold = 0.87e-3, improvement = False):
    start_time = time.perf_counter()
    L_d = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]]).astype(np.float32)
    if improvement == False:
        u = np.zeros(im1.shape).astype(np.float32)
        v = np.zeros(im1.shape).astype(np.float32)
    else:
        u, v = lucas_kanade(im1, im2, show_log=False)
        u = u.astype(np.float32)
        v = v.astype(np.float32)
    max_iters = n_iters
    # For each pixel calculate I_x, I_y and I_t
    # make sure it is not 0
    I_x, I_y = gauss_deriv((im1 + im2) / 2, 1)
    I_t = gauss_smooth(im2 - im1, 1)

    # Calculate the value of the determinant
    D = lmbd + (I_x ** 2) + (I_y ** 2)
    D[D < 1e-9] = np.inf

    while (n_iters > 0):
        # Calculate the iterative corrections to the displacement
        u_a = signal.convolve2d(u, L_d, mode='same')
        v_a = signal.convolve2d(v, L_d, mode='same')

        # Calculate P depending on the new corrections
        P = (I_x * u_a) + (I_y * v_a) + I_t

        # Calculate the new u and v components
        u = u_a - (I_x * (P / D))
        v = v_a - (I_y * (P / D))

        # Check for convergence
        if (np.mean(np.abs(u - u_a)) < threshold) and (np.mean(np.abs(v - v_a)) < threshold):
            print(np.mean(np.abs(u-u_a)))
            break

        n_iters = n_iters - 1
    end_time = time.perf_counter()
    print(f"Horn-Schunk's execution time with {n_iters} from {max_iters} iterations left is: {end_time - start_time}s.")
    
    # Return the u and v components
    return u, v

# im1 − first image matrix (grayscale)
# im2 − second image matrix (grayscale)
# N − size of the neighborhood (N x N)
# levels - levels of the pyramid
# show_log - shows log of the Lucas-Kanade's execution time
def pyramidal_lucas_kanade(im1, im2, N = 7, levels = 3, show_log = True):
    start_time = time.perf_counter()
    shape = im1.shape

    u = np.zeros(shape).astype(np.float32)
    v = np.zeros(shape).astype(np.float32)
    prev_u = np.zeros((int(shape[0] / (2 ** (levels - 1))), int(shape[1] / (2 ** (levels - 1))))).astype(np.float32)
    prev_v = np.zeros((int(shape[0] / (2 ** (levels - 1))), int(shape[1] / (2 ** (levels - 1))))).astype(np.float32)
    im1_lvl = gauss_smooth(im1, 1)
    im2_lvl = gauss_smooth(im2, 1)

    for lvl in range(levels-1, -1, -1):
        # Resize the image to the current level size
        im1_lvl = cv2.resize(im1, (int(shape[1] / (2 ** lvl)), int(shape[0] / (2 ** lvl))), interpolation=cv2.INTER_AREA)
        im2_lvl = cv2.resize(im2, (int(shape[1] / (2 ** lvl)), int(shape[0] / (2 ** lvl))), interpolation=cv2.INTER_AREA)

        # Warp the second image with the previous level u and v components
        if (lvl < levels - 1):
            flow = np.stack((prev_u, prev_v), axis=-1)
            h, w = flow.shape[:2]
            flow = -flow
            flow[:, :, 0] += np.arange(w)
            flow[:, :, 1] += np.arange(h)[:, np.newaxis]
            im2_lvl = cv2.remap(im2_lvl, flow, None, cv2.INTER_LINEAR)

        # Compute the optical flow for the current level
        u_lvl, v_lvl = lucas_kanade(im1_lvl, im2_lvl, N, show_log=False)
        u_lvl = u_lvl.astype(np.float32)
        v_lvl = v_lvl.astype(np.float32)

        # Resize the u and v components and add them to the end result
        u_lvl_resized = cv2.resize(u_lvl, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        v_lvl_resized = cv2.resize(v_lvl, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        u = u + (u_lvl_resized * (2 ** lvl))
        v = v + (v_lvl_resized * (2 ** lvl))

        if (lvl != 0):
            # Save the resized u and v components
            prev_u = cv2.resize(u_lvl * 2, (int(shape[1] / (2 ** (lvl - 1))), int(shape[0] / (2 ** (lvl - 1)))), interpolation=cv2.INTER_AREA)
            prev_v = cv2.resize(v_lvl * 2, (int(shape[1] / (2 ** (lvl - 1))), int(shape[0] / (2 ** (lvl - 1)))), interpolation=cv2.INTER_AREA)
    end_time = time.perf_counter()
    if show_log:
        print(f"Pyramidal Lucas-Kanade's execution time for N={N} and L={levels} is: {end_time - start_time}s.")

    return u, v