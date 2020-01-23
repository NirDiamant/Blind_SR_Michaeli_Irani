import scipy.io
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
from scipy import fftpack, signal
from scipy.linalg import circulant
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

## upsample matrix by factor alpha, using cubic interpolation
def upsample_matrix(mat,alpha):
    (mat_shape_x, mat_shape_y) = mat.shape
    new_size = (int(mat_shape_y * alpha), int(mat_shape_x * alpha))
    upsampled_filtered_image = cv2.resize(mat, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    return upsampled_filtered_image


## creates a gaussian matrix
def gaussian(window_size, range, mu, sigma):
    z = np.linspace(-range, range, window_size)
    x, y = np.meshgrid(z, z)
    d = np.sqrt(x*x+y*y)
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    g = g / g.sum()
    return g


## creates a sinc matrix
def sinc(window_size, range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x, x)
    s = np.sinc(xx)
    s = s / s.sum()
    return s

## creates a laplacian operator matrix
def laplacian(mat_length): #the matrix in a square, and the parameter is the length of the side of the square
    a = -1
    b = 4
    lap_mat = np.zeros((mat_length ** 2, mat_length ** 2))

    main_diag_pattern = np.zeros((mat_length, mat_length))
    for i in range(mat_length):
        for j in range(mat_length):
            if i == j:
                main_diag_pattern[i, j] = b
            if abs(i-j) == 1:
                main_diag_pattern[i, j] = a

    second_diag_pattern = np.zeros((mat_length, mat_length))
    for i in range(mat_length):
        second_diag_pattern[i, i] = a


    ### copying main digonal pattern to lap matrix all along the main block diagonal
    start_index = 0
    end_index = mat_length

    for num_matrices in range(mat_length):
        lap_mat[start_index:end_index, start_index:end_index] = main_diag_pattern
        start_index += mat_length
        end_index += mat_length

    ### copying second diagonal pattern to lap matrix all along the main block diagonal

    start_index_x = 0
    end_index_x = mat_length
    start_index_y = mat_length
    end_index_y = 2 * mat_length

    for num_matrices in range(mat_length - 1):
        lap_mat[start_index_x:end_index_x, start_index_y:end_index_y] = second_diag_pattern
        lap_mat[start_index_y:end_index_y, start_index_x:end_index_x] = second_diag_pattern

        start_index_x += mat_length
        end_index_x += mat_length
        start_index_y += mat_length
        end_index_y += mat_length

    return lap_mat


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


## restors image with wiener filter, given the blurred image, the psf and wiener filter constant
def wiener_filter(img, kernel, K):
    if np.sum(kernel):
        kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fftpack.fft2(dummy)
    kernel = fftpack.fft2(kernel, shape=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(fftpack.ifft2(dummy))
    return dummy


## image processing course teaching assistant's wiener filter implementation
def wiener_filter2(image, psf, k):
    image_dft = fftpack.fft2(image)
    psf_dft = fftpack.fft2(fftpack.ifftshift(psf), shape=image_dft.shape)
    filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
    recovered_dft = image_dft * filter_dft
    return np.real(fftpack.ifft2(recovered_dft))


## saves img to folder with specific format
def save_as_img(img,  title, my_format):
    img = img / img.max()
    img_256 = img * 255
    im = Image.fromarray(img_256)
    im = im.convert('RGB')
    im.save(title + my_format)


## calculates PSNR of two matrices
def calculate_psnr(im1, im2):
    MSE = np.mean((im1 - im2) ** 2)
    max_value = im1.max()
    PSNR = 20 * np.log10(max_value / (np.sqrt(MSE)))
    return PSNR


## calculate restored kernel from blurred image in order to generate higher resolution image
def calculate_kernel(blurred_img, alpha, T):
    patch_size = 15
    q_patch_size = int(patch_size / alpha)
    sigma_NN = 0.06  # last value that worked 0.059
    num_neighbors = 5
    pca_num_components = 25
    epsilon = 1e-10  # a value to add to a matrix to make it invertible
    patch_step_size_factor = 1 / (patch_size / alpha)  # step size for the patch partition, in units of patches
    title_name = "patch_" + str(patch_size) + "_alpha_" + str(alpha) + "_step_" + str(patch_step_size_factor) + "_NN_" \
                 + str(num_neighbors) + "_PCA_" + str(pca_num_components) + "_sigmaNN_" + str(
        sigma_NN) + "_knn_swapped_new_C_matan_new_wiener"

    ## creates patches from l (bigger size, r), and patches for comparing from l(size divided by alpha, q)
    r_patches = create_patches_from_image(blurred_img, patch_size, int(patch_size * patch_step_size_factor))
    q_patches = create_patches_from_image(blurred_img, q_patch_size, int(q_patch_size * patch_step_size_factor))

    q_patches_vec = []
    for q_patch in q_patches:
        curr_q_vec = q_patch.reshape(q_patch.size)
        q_patches_vec.append(curr_q_vec)
    q_patches_vec = np.array(q_patches_vec)

    ## generating Rj by calculating: taking every big patch (r), making it a vector, then a circulant matrix,
    # and then downsample by alpha**2 (only rows)
    Rj = []
    for r_patch in r_patches:
        r_vec = r_patch.reshape(r_patch.size)
        r_circulant = circulant(r_vec)
        curr_Rj = downsample_shrink_matrix_1d(r_circulant, alpha ** 2)
        Rj.append(curr_Rj)


    ## initial k to start iterative algorithm with
    delta = fftpack.fftshift(scipy.signal.unit_impulse((patch_size, patch_size)))
    curr_k = delta.reshape(delta.size)

    C = laplacian(patch_size)
    C_squared = C.T @ C


    for t in range(T):
        print(f'iteration number: {t}')

        r_alpha_patches = []
        for j, patch in enumerate(r_patches):
            curr_patch_alpha = Rj[j] @ curr_k
            r_alpha_patches.append(curr_patch_alpha)

        r_alpha_patches = np.array(r_alpha_patches)
        tree = sklearn.neighbors.BallTree(r_alpha_patches, leaf_size=2)
        neighbors_weights = np.zeros((len(q_patches_vec), len(r_alpha_patches)))
        for i, q_patch_vec in enumerate(q_patches_vec):
            representative_patch = np.expand_dims(q_patch_vec, 0)
            _, neighbor_indices = tree.query(representative_patch, k=num_neighbors)
            for index in neighbor_indices:
                neighbors_weights[i, index] = dist_weight(q_patch_vec, r_alpha_patches[index], sigma_NN)

        neighbors_weights_sum = np.sum(neighbors_weights, axis=1)
        epsilon_mat = np.eye(curr_k.shape[0]) * epsilon

        for row in range(neighbors_weights.shape[0]):
            row_sum = neighbors_weights_sum[row]
            if row_sum:
                neighbors_weights[row] = neighbors_weights[row] / row_sum  ## normalize each column

        ## calculate k hat
        sum_left = np.zeros((curr_k.shape[0], curr_k.shape[0]))
        sum_right = np.zeros_like(curr_k)

        for i in range(neighbors_weights.shape[0]):
            for j in range(neighbors_weights.shape[1]):
                if not neighbors_weights[i, j]:
                    continue
                R_squared = Rj[j].T @ Rj[j]

                sum_left += neighbors_weights[i, j] * (R_squared) + (C_squared)
                sum_right += neighbors_weights[i, j] * Rj[j].T @ q_patches_vec[i]

        curr_k = np.linalg.inv((1 / (sigma_NN ** 2)) * sum_left + epsilon_mat) @ sum_right
        curr_k_reshaped = curr_k.reshape((patch_size, patch_size))
    return curr_k_reshaped  ## as matrix in shape patch_size ** 2


def main():

    img = plt.imread("DIPSourceHW2.png")
    img = np.array(img)
    img = img[:, :, 0]
    img = img / img.max()
    img_new_size = np.zeros((img.shape[0]+2, img.shape[1]+2))
    img_new_size[1:-1, 1:-1] = img
    img = img_new_size

    ## define constants
    alpha = 3
    window_size = 15
    filter_range_sinc = window_size/4  #was window_size/2. smaller range ==> wider gaussian,
    filter_range_gaussian = window_size / 16
    mu = 0
    sigma = 1
    T = 5

    ## generate filters
    s_filter = sinc(window_size, filter_range_sinc)
    g_filter = gaussian(window_size, filter_range_gaussian, mu, sigma)

    ## generate low resolution images
    sinc_img_big = signal.convolve2d(img, s_filter, mode='same', boundary='wrap')
    sinc_img = downsample_shrink_matrix(sinc_img_big, alpha)
    sinc_img_high_res = upsample_matrix(sinc_img, alpha)

    gaussian_img_big = signal.convolve2d(img, g_filter, mode='same', boundary='wrap')
    gaussian_img = downsample_shrink_matrix(gaussian_img_big, alpha)
    gaussian_img_high_res = upsample_matrix(gaussian_img, alpha)

    plt.imshow(sinc_img_high_res, cmap='gray')
    plt.title("sinc blurred img")
    plt.show()

    plt.imshow(gaussian_img_high_res, cmap='gray')
    plt.title("gaussian blurred img")
    plt.show()

    ## restore kernels using Irani-Michaeli's paper's algorithm
    gaussian_restored_k = calculate_kernel(gaussian_img, alpha,T) ## restored kernel after blurring with gaussian
    sinc_restored_k = calculate_kernel(sinc_img, alpha, T) ## restored kernel after blurring with sinc

    ## restore each blurred image with both kernels
    sinc_restored_img = wiener_filter(sinc_img_high_res, sinc_restored_k, 0.1)
    sinc_blurred_restored_from_gaussian_img = wiener_filter(sinc_img_high_res, gaussian_restored_k, 0.1)
    gaussian_restored_img= wiener_filter(gaussian_img_high_res, gaussian_restored_k, 0.1)
    gaussian_blurred_restored_from_sinc_img = wiener_filter(gaussian_img_high_res, sinc_restored_k, 0.1)

    ## plot results and PSNR with original high-res image
    plot_results(sinc_restored_img, "sinc-ker", "sinc-ker", img)
    plot_results(sinc_blurred_restored_from_gaussian_img, "sinc-ker", "gauss-ker", img)
    plot_results(gaussian_restored_img, "gauss-ker", "gauss-ker", img)
    plot_results(gaussian_blurred_restored_from_sinc_img, "gauss-ker", "sinc-ker", img)


def plot_results(img_restored, blurred_with, restored_with, orig_img):
    plt.imshow(img_restored, cmap='gray')
    PSNR = calculate_psnr(orig_img[5:-5, 5:-5], img_restored[5:-5, 5:-5])
    plt.title(f'image blurred with {blurred_with} and restored with {restored_with}. PSNR={PSNR:.2f}')
    plt.show()


## sainity check of the laplacian matrix operator

# dummy_shape = 15
# dummy = np.zeros((dummy_shape,dummy_shape))
# dummy[3:7,:] = 255
# dummy = (dummy / dummy.max()).astype(int)
#
# plt.imshow(dummy,cmap='gray')
# plt.title("dummy")
# plt.show()
#
# img_vec = dummy.reshape(dummy.size)
# img_vec_t = img_vec.T
# img_lap = laplacian(dummy.shape[0])
#
# res = img_lap @ img_vec_t
# res_img = res.reshape((dummy_shape,dummy_shape))
#
# plt.imshow(res_img, cmap='gray')
# plt.title("laplacian matmul img vec")
# plt.show()


if __name__ == "__main__":
    main()
