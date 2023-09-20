import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from ex2_utils import generate_responses_1, create_epanechnik_kernel, create_gaussian_kernel, create_uniform_kernel
from ms_method import mean_shift

TYPE1 = 'FUNCTION1'
TYPE2 = 'FUNCTION2'

if __name__ == '__main__':
    # DEFAULT, TYPE1 or TYPE2
    f_type = 'DEFAULT'
    x_starting = 30
    y_starting = 50
    window_size = (7, 7)
    n_iter = 200
    min_shifting = 0.035
    # gaussian or epanechnikov
    kernel_shape = 'epanechnikov'

    x = np.arange(0, 100, 1)
    y = np.arange(0, 100, 1)
    x, y = np.meshgrid(x, y)
    f = generate_responses_1(f_type)
    starting_point_2D = np.array([y_starting, x_starting])
    starting_point_3D = np.array([y_starting, x_starting, f[y_starting, x_starting] + 0.00005])

    # Calculate
    end_point_2D = mean_shift(f, starting_point_2D, kernel_shape, window_size, n_iter, min_shifting)
    end_point_3D = np.array([end_point_2D[0], end_point_2D[1], f[end_point_2D[0], end_point_2D[1]] + 0.00005])
    

    fig3D = plt.figure()
    ax3D = fig3D.add_subplot(111, projection='3d')
    ax3D.plot_surface(x, y, f, cmap='viridis')
    ax3D.scatter(starting_point_3D[1], starting_point_3D[0], starting_point_3D[2], color='black')
    ax3D.scatter(end_point_3D[1], end_point_3D[0], end_point_3D[2], color='red')
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Response')
    ax3D.set_title(f'Mean-Shift mode seeking ({window_size[0]}x{window_size[1]} window and {kernel_shape.capitalize()} kernel)')
    ax3D.view_init(elev=30, azim=-45)


    fig2D = plt.figure()
    ax2D = fig2D.add_subplot(111)
    ax2D.imshow(f, cmap='viridis')
    ax2D.scatter(starting_point_2D[1], starting_point_2D[0], color='black')
    ax2D.scatter(end_point_2D[1], end_point_2D[0], color='red')
    ax2D.set_title(f'Mean-Shift mode seeking ({window_size[0]}x{window_size[1]} window and {kernel_shape.capitalize()} kernel)')
    
    plt.show()