import numpy as np
import matplotlib.pyplot as plt
import cv2
from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, pyramidal_lucas_kanade, horn_schunck

TYPE1 = "RANDOM IMAGE PAIR"
TYPE2 = "COLLISION"
TYPE3 = "DISPARITY"
TYPE4 = "LAB2"


if __name__ == "__main__":
    mode = TYPE1
    N = 7
    L = 3
    n_iter = 1000
    threshold_determinant = 1e-3
    threshold_harris = 1e-8
    alpha = 0.02
    lmbd = 0.5
    threshold = 0.87e-3

    if mode == TYPE1:
        # Generation of the random images
        im1 = np.random.rand(200, 200).astype(np.float32)
        im2 = im1.copy()
        im2 = rotate_image(im2, 1)
    elif mode == TYPE2:
        # Loading and normalizing collision images
        im1 = cv2.normalize(cv2.imread("collision/00000166.jpg", cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im2 = cv2.normalize(cv2.imread("collision/00000167.jpg", cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    elif mode == TYPE3:
        # Loading and normalizing collision images
        im1 = cv2.normalize(cv2.imread("disparity/office2_left.png", cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im2 = cv2.normalize(cv2.imread("disparity/office2_right.png", cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    elif mode == TYPE4:
        # Loading and normalizing collision images
        im1 = cv2.normalize(cv2.imread("lab2/020.jpg", cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im2 = cv2.normalize(cv2.imread("lab2/021.jpg", cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Calculate the Lucas-Kanade optical flow
    U_lk, V_lk = lucas_kanade(im1, im2, N, threshold_determinant, threshold_harris, alpha)
    # Calculate the pyramidal Lucas-Kanade optical flow
    U_plk, V_plk = pyramidal_lucas_kanade(im1, im2, N, L)
    # Calculate the Horn-Schunck optical flow
    U_hs, V_hs = horn_schunck(im1, im2, n_iter, lmbd, threshold, improvement=False)

    if mode == TYPE1:
        # Show the plot of the optical flows for random noise 
        fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
        ax1_11.imshow(im1)
        ax1_12.imshow(im2)
        show_flow(U_lk, V_lk, ax1_21, type='angle')
        show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
        fig1.suptitle(f'Lucas−Kanade Optical Flow (N = {N})')

        fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
        ax2_11.imshow(im1)
        ax2_12.imshow(im2)
        show_flow(U_plk, V_plk, ax2_21, type='angle')
        show_flow(U_plk ,V_plk, ax2_22, type='field', set_aspect=True)
        fig2.suptitle(f'Pyramidal Lucas−Kanade Optical Flow (N = {N}, L = {L})')

        fig3, ((ax3_11, ax3_12), (ax3_21, ax3_22)) = plt.subplots(2, 2)
        ax3_11.imshow(im1)
        ax3_12.imshow(im2)
        show_flow(U_hs, V_hs, ax3_21, type='angle')
        show_flow(U_hs, V_hs, ax3_22, type='field', set_aspect=True)
        fig3.suptitle(f'Horn-Schunck Optical Flow (N_Iter = {n_iter})')
    else:
        # Show the plot of the optical flows for image
        extent = (0, im1.shape[1], -im1.shape[0], 0)

        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        show_flow(U_lk, V_lk, ax1, type='field', set_aspect=True)
        ax1.imshow(im1, alpha=0.8, extent=extent)
        fig1.suptitle(f'Lucas−Kanade Optical Flow (N = {N})')
        fig1.tight_layout()
        ax1.axis('off')

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        show_flow(U_plk, V_plk, ax2, type='field', set_aspect=True)
        ax2.imshow(im1, alpha=0.8, extent=extent)
        fig2.suptitle(f'Pyramidal Lucas−Kanade Optical Flow (N = {N}, L = {L})')
        fig2.tight_layout()
        ax2.axis('off')

        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
        show_flow(U_hs, V_hs, ax3, type='field', set_aspect=True)
        ax3.imshow(im1, alpha=0.8, extent=extent)
        fig3.suptitle(f'Horn-Schunck Optical Flow (N_Iter = {n_iter})')
        fig3.tight_layout()
        ax3.axis('off')


    plt.show()