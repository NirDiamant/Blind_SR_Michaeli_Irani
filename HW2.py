import scipy.io
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from scipy import fftpack, signal
from scipy.sparse import csgraph
from scipy.linalg import circulant
import time
import sklearn
from sklearn.decomposition import PCA
import sklearn.neighbors
import cv2


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

def downsample_shrink_matrix_1d(mat, alpha):
    # print(mat.shape)
    (mat_shape_x, mat_shape_y) = mat.shape
    new_size =(int(mat_shape_x / alpha), int(mat_shape_y))
    downsampled = np.zeros(new_size)
    for i in range(new_size[0]):
        downsampled[i,:] = mat[alpha * i, :]
    return downsampled

## only fills zeros wherever a point is going to be gone when shrinking
def downsample_zeros_matrix(mat, alpha):
    (mat_shape_x, mat_shape_y) = mat.shape
    for i in range(mat_shape_x):
        if (i % alpha):
            mat[i, :] = 0
            mat[:, i] = 0
    return mat


def upsample_matrix(mat,alpha):
    (mat_shape_x, mat_shape_y) = mat.shape
    new_size = (int(mat_shape_y * alpha), int(mat_shape_x * alpha))
    #FILTER = PIL.Image.BILINEAR
    upsampled_filtered_image = cv2.resize(mat, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    #upsampled_filtered_image = mat.resize(new_size, resample=FILTER)
    return upsampled_filtered_image


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


def wiener_filter(image, psf, k):
    image_dft = fftpack.fft2(image)
    psf_dft = fftpack.fft2(fftpack.ifftshift(psf), shape=image_dft.shape)
    filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
    recovered_dft = image_dft * filter_dft
    return np.real(fftpack.ifft2(recovered_dft))


def main():

    ## define constants
    init_time = time.time()
    img = plt.imread("DIPSourceHW2.png")
    img = np.array(img)
    img = img[:, :, 0]
    img = img / img.sum()
    alpha = 3
    window_size = 15
    filter_range = window_size/2 #was window_size/2. smaller range ==> wider gaussian,
    mu = 0
    sigma = 1
    patch_size = 12
    q_patch_size = int(patch_size / alpha)
    sigma_NN = 0.01
    Wiener_Filter_Constant = 0.01
    num_neighbors = 11
    pca_num_components = 15
    T = 16
    img_rows, img_cols = img.shape
    patch_step_size_factor = 0.5  #step size for the patch partition, in units of patches
    file_name = "patch20_alpha4_half_step_size_nn11_pca15"

    ## generate filters
    g = gaussian(window_size, filter_range, mu, sigma)
    plt.imshow(g, cmap='gray')
    plt.title("Real gaussian PSF")
    plt.show()

    gaussian_img = signal.convolve2d(img, g,boundary='wrap')
    gaussian_restored_true = wiener_filter(gaussian_img, g,
                                           0)
    plt.imshow(gaussian_restored_true, cmap='gray')
    plt.title("restored gaussian with real gaussian kernel")
    plt.show()
    gaussian_img = downsample_shrink_matrix(gaussian_img, alpha)

    gaussian_img_high_res = upsample_matrix(gaussian_img,alpha)
    print(f'gaussian_img shape: {gaussian_img.shape}')
    print(f'gaussian_img_high_res shape: {gaussian_img_high_res.shape}')
    plt.imshow(gaussian_img, cmap='gray')
    plt.title("gaussian img")
    plt.show()

    plt.imshow(gaussian_img_high_res, cmap='gray')
    plt.title("gaussian img high res initial")
    plt.show()


    downsampled_low_res_gaussian = downsample_shrink_matrix(gaussian_img, alpha)



    s_filter = sinc(window_size,filter_range)
    sinc_img = signal.convolve2d(img, s_filter)
    downsampled_low_res_sinc = downsample_shrink_matrix(sinc_img, alpha)



    ## creates patches from l (bigger size, r), and patches for comparing from l(size divided by alpha, q)

    r_patches = create_patches_from_image(img, patch_size, int(patch_size*patch_step_size_factor))
    q_patches = create_patches_from_image(img, q_patch_size, int(q_patch_size*patch_step_size_factor))

    q_patches_vec = []
    for q_patch in q_patches:
        curr_q_vec = q_patch.reshape(q_patch.size)
        q_patches_vec.append(curr_q_vec)

    pca = PCA(n_components=pca_num_components, svd_solver='full')
    pca.fit(q_patches_vec)
    q_patches_pca = pca.transform(q_patches_vec)

    ## generating Rj by calculating: taking every big patch (r), making it a vector, then a circulant matrix, and then downsample by alpha**2 (only rows)
    Rj = []
    for r_patch in r_patches:
        r_vec = r_patch.reshape(r_patch.size)
        r_circulant = circulant(r_vec)
        curr_Rj = downsample_shrink_matrix_1d(r_circulant, alpha**2)
        Rj.append(curr_Rj)
    print(f'size of Rj : {Rj[0].shape}')

    ## initial k to start iterative algorithm with
    delta = fftpack.fftshift(scipy.signal.unit_impulse((patch_size,patch_size)))
    curr_k = delta.reshape(delta.size)

    curr_k_image = curr_k.reshape((patch_size, patch_size))
    plt.imshow(curr_k_image, cmap='gray')
    plt.title("initial k")
    plt.show()

    ## generate a laplacian matrix
    C = laplacian(patch_size, (patch_size) / 2) ## we may consider adding 1 to the patch size in order to make it symatrical
    # plt.imshow(C, cmap='gray')
    # plt.title("Laplacian")
    # plt.show()
    C = C.reshape(C.size)
    C = np.expand_dims(C, axis=0)
    C_squared = C.T @ C
    print(f'shape of C: {C.shape}')
    epsilon = 1e-10  # a value to add to a matrix to make it invertible

    pad_size = int((patch_size - q_patch_size) / 2)
    for t in range(T):
        # curr_k = np.pad(curr_k,(pad_size + 1, pad_size))
        # print(f'curr_k shape: {curr_k.shape}')

        ## create patches that correspond to the r patches, but downsampled by alpha factor
        r_alpha_patches_pca = []
        for j ,patch in enumerate(r_patches):
            curr_patch_alpha = Rj[j] @ curr_k
            curr_patch_alpha_pca = pca.transform(np.expand_dims(curr_patch_alpha, axis=0))
            r_alpha_patches_pca.append(curr_patch_alpha_pca)
        #print(f'size of r alpha patch : {r_alpha_patches[0].shape}')

        tree = sklearn.neighbors.BallTree(q_patches_pca, leaf_size=2)
        # output = np.zeros_like(image)
        neighbors_weights = np.zeros((len(q_patches_pca), len(r_alpha_patches_pca)))
        for j, r_alpha_patch_pca in enumerate(r_alpha_patches_pca):
            representative_patch = np.expand_dims(r_alpha_patch_pca, 0)
            # print(f'representative_patch dim: {representative_patch.ndim}')
            _, neighbor_indices = tree.query(r_alpha_patch_pca, k=num_neighbors)
            for index in neighbor_indices:
                neighbors_weights[index, j] = dist_weight(q_patches_pca[index], r_alpha_patch_pca, sigma_NN)

        ## calcuate the weights
        # neighbors_weights = np.zeros((len(q_patches_pca), len(r_alpha_patches_pca)))
        # for i in range(neighbors_weights.shape[0]):
        #     for j in range(neighbors_weights.shape[1]):
        #         neighbors_weights[i,j] = dist_weight(q_patches_pca[i],r_alpha_patches_pca[j],sigma_NN)

        neighbors_weights_sum = np.sum(neighbors_weights, axis=0)
        neighbors_weights = np.divide(neighbors_weights , np.expand_dims(neighbors_weights_sum, axis=0)) ## normalize each column





        ## calculate k hat
        sum_left = np.zeros((curr_k.shape[0],curr_k.shape[0]))
        sum_right = np.zeros_like(curr_k)

        for i in range(neighbors_weights.shape[0]):
            for j in range(neighbors_weights.shape[1]):
                if not neighbors_weights[i, j]:
                    continue
                R_squared = Rj[j].T @ Rj[j]
                # C_squared = C.T @ C
                # print(f'R_squared shape: {R_squared.shape} and C_squared shape: {C_squared.shape}')
                # print(f'neighbors_weights[i, j]: {neighbors_weights[i, j]}')
                sum_left += neighbors_weights[i, j] * (R_squared) + (C_squared)
                sum_right += neighbors_weights[i, j] * Rj[j].T @ q_patches_vec[i]

        curr_k = np.linalg.inv((1/sigma_NN) * sum_left) @ sum_right
        # print(f'curr_k shape: {curr_k.shape}')

        gaussian_restored = wiener_filter(gaussian_img_high_res, curr_k.reshape((patch_size,patch_size)), Wiener_Filter_Constant)

        plt.imshow(gaussian_restored, cmap='gray')
        plt.title(f'restoration after iteration number: {t}')
        plt.show()

        curr_k_image = curr_k.reshape((patch_size, patch_size))
        plt.imshow(curr_k_image, cmap='gray')
        plt.title(f'curr_k as an image:')
        plt.show()

    curr_k_image = curr_k.reshape((patch_size,patch_size))
    plt.imshow(curr_k_image,cmap='gray')
    plt.show()





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