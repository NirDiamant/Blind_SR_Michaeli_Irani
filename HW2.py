import scipy.io
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
from scipy import fftpack, signal
from scipy.sparse import csgraph
import time

## shrinks matrix by ratio alpha
def downsample_shrink_matrix(mat, alpha):
    # print(mat.shape)
    (mat_shape_x, mat_shape_y) = mat.shape
    new_size =(int(mat_shape_x / alpha), int(mat_shape_y / alpha))
    downsampled = np.zeros(new_size)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            downsampled[i, j] = mat[alpha * i, alpha * j]
    return downsampled

## only fills zeros wherever a point is going to be gone when shrinking
def downsample_zeros_matrix(mat, alpha):
    (mat_shape_x, mat_shape_y) = mat.shape
    for i in range(mat_shape_x):
        if (i % alpha):
            mat[i, :] = 0
            mat[:, i] = 0
    return mat

## creates a gaussian matrix
def gaussian(window_size,range, mu, sigma):
    z = np.linspace(-range, range, window_size)
    x, y = np.meshgrid(z, z)
    d = np.sqrt(x*x+y*y)
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    return g

## creates a sinc matrix
def sinc(window_size,range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x,x)
    s = np.sinc(xx)
    return s

## creates a laplacian matrix
def laplacian(window_size,range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x, x)
    return csgraph.laplacian(xx, normed=False)

## calculates the distnace as the nominator of the iterative function in the paper
def dist_weight(q_i,r_j_alpha, sigma):
    return np.exp(-0.5 * (np.linalg.norm(q_i - r_j_alpha) ** 2)/(sigma ** 2))

## divides image into patches with given step size, and saves the result in an array
def create_patches_from_image(img, patch_size, step_size):
    array_for_patches = []
    for i in range(0,int(img.shape[0] - patch_size),step_size):
        for j in range(0,int(img.shape[1] - patch_size),step_size):
            curr_patch = img[i:i+patch_size,j:j+patch_size]
            array_for_patches.append(curr_patch)
    return array_for_patches


def main():

    ## define constants
    init_time = time.time()
    img = plt.imread("DIPSourceHW2.png")
    img = np.array(img)
    img = img[:, :, 0]
    alpha = 4
    filter_range = 5
    window_size = 6
    mu = 0
    sigma = 1
    patch_size = 60
    q_patch_size = int(patch_size / alpha)
    sigma_NN = 1
    img_rows, img_cols = img.shape

    ## generate filters
    g = gaussian(window_size, filter_range, mu, sigma)
    gaussian_img = signal.convolve2d(img, g)
    downsampled_low_res_gaussian = downsample_shrink_matrix(gaussian_img, alpha)

    s_filter = sinc(window_size,filter_range)
    sinc_img = signal.convolve2d(img, s_filter)
    downsampled_low_res_sinc = downsample_shrink_matrix(sinc_img, alpha)



    ## creates patches from l (bigger size, r), and patches for comparing from l(size divided by alpha, q)

    r_patches = create_patches_from_image(img, patch_size, patch_size)
    q_patches = create_patches_from_image(img, q_patch_size, q_patch_size)

    ## initial k to start iterative algorithm with
    delta = fftpack.fftshift(scipy.signal.unit_impulse((patch_size,patch_size)))
    curr_k = delta

    ## create patches that correspond to the r patches, but downsampled by alpha factor
    r_alpha_patches = []
    for patch in r_patches:
        curr_patch = downsample_shrink_matrix(signal.convolve2d(patch, curr_k, mode='same'), alpha)
        r_alpha_patches.append(curr_patch)
    print(f'size of r alpha patch : {r_alpha_patches[0].shape}')

    ## calcuate the weights
    neighbors_weights = np.zeros((len(q_patches), len(r_alpha_patches)))

    for i in range(neighbors_weights.shape[0]):
        for j in range(neighbors_weights.shape[1]):
            neighbors_weights[i,j] = dist_weight(q_patches[i],r_alpha_patches[j],sigma_NN)

    neighbors_weights_sum = np.sum(neighbors_weights, axis=0)
    neighbors_weights = np.divide(neighbors_weights , np.expand_dims(neighbors_weights_sum, axis=0)) ## normalize each column



    ## generate a laplacian matrix
    C = laplacian(patch_size+1,patch_size/2)

    ## generating Rj by calculating: the downsampled version of k_rj @ k_inv
    Rj = []
    epsilon = 1e-10  # a value to add to a matrix to make it invertible


    for patch in r_patches:
        k_rj = signal.convolve2d(curr_k, patch, mode='same')
        k_inv = np.linalg.inv(curr_k + np.eye(curr_k.shape[0]) * epsilon)
        downsample_zeros_k_rj = downsample_zeros_matrix(k_rj,alpha)
        curr_Rj = downsample_shrink_matrix((downsample_zeros_k_rj @ k_inv),alpha)
        Rj.append(curr_Rj)


    ## calculate k hat   ## TODO
    for i in range(neighbors_weights.shape[0]):
        for j in range(neighbors_weights.shape[1]):
            neighbors_weights[i,j]









    # plt.title('original')
    # plt.imshow(img, cmap='gray')
    # plt.show()
    #
    #
    # plt.title('gaussian + downsampled')
    # plt.imshow(downsampled_low_res_gaussian, cmap='gray')
    # plt.show()
    #
    # plt.title('gaussian')
    # plt.imshow(gaussian_img, cmap='gray')
    # plt.show()
    #
    # plt.title('sinc')
    # plt.imshow(sinc_img, cmap='gray')
    # plt.show()
    #
    # plt.title('sinc + downsampled')
    # plt.imshow(downsampled_low_res_sinc, cmap='gray')
    # plt.show()

    # print(f'total time was:{end_time-init_time}')


if __name__ =="__main__":
    main()