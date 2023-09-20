import numpy as np
import time
from ex2_utils import get_patch, create_gaussian_kernel, create_uniform_kernel


# Calculate the new center using the Mean-Shift algorithm -> return the new center
## f -> NpArray of the function landscape
## center -> the center position of the current region
## kernel_shape -> shape of the kernel
## window_size -> size of the region and the kernel
## max_iter -> maximum iterations if there is no converegence
## min_shiftinh -> minimum shifting for convergence
def mean_shift(f, center, kernel_shape='epanechnikov', window_size=(5, 5), max_iter=100, min_shifting = 1e-5):
    start_time = time.perf_counter()

    # Create the Kernel
    if kernel_shape == 'epanechnikov':
        kernel = create_uniform_kernel(window_size[0], window_size[1])
    elif kernel_shape == 'gaussian':
        kernel = create_gaussian_kernel(window_size[0], window_size[1], 1)
    
    shift = np.inf
    curr_iter = 0
    # Calculate the Coordinates within window for each direction
    x_coord = np.tile(np.arange(-(window_size[0]//2), window_size[0]//2 + 1), (window_size[0], 1))
    y_coord = np.tile(np.arange(-(window_size[1]//2), window_size[1]//2 + 1), (window_size[1], 1)).T 

    # Calculate the new center using Mean-Shift algorithm
    while shift > min_shifting and curr_iter < max_iter:
        # Get the region
        region, _ = get_patch(f, center, (window_size[1], window_size[0]))
        # Calculate the new center and the shift
        new_center, shift = get_mean_shift_vector(region, center, kernel, x_coord, y_coord)
        center = new_center
        curr_iter += 1

    # Print the performance time
    end_time = time.perf_counter()
    print(f"Mean-Shift's execution time with left iterations {max_iter-curr_iter}/{max_iter} is: {end_time - start_time}s.")

    return np.round(center).astype(int)


# Calculate the Mean shift vector -> return the new center and the shifting
## region -> NpArray of the current region
## center -> the center position of the current region
## kernel -> NpArray of the kernel, which is the same size as the region
## x_coors, y_coord -> coordinates within each window for each direction (X, Y)
def get_mean_shift_vector(region, center, kernel, x_coord, y_coord):
    # Calculate the weights and normalize them
    weights = region * kernel
    weights = weights / np.sum(weights)

    # Calculate the mean shift vector
    mean_shift_x = np.sum(weights * x_coord) / np.sum(weights)
    mean_shift_y = np.sum(weights * y_coord) / np.sum(weights)
    mean_shift_vector = np.array([mean_shift_x, mean_shift_y])

    # Calculate the coordinates of the new center and the shift
    new_center = center + mean_shift_vector
    shift = np.linalg.norm(new_center - center)

    return new_center, shift