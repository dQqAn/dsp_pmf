import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import cv2
import pandas as pd

from skimage.io import imshow, imread
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import filters, color, exposure, transform
from skimage.util import img_as_ubyte, img_as_float
from skimage.exposure import histogram, cumulative_distribution


def calc_color_overcast(image):
    # Calculate color overcast for each channel
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Create a dataframe to store the results
    channel_stats = pd.DataFrame(columns=['Mean', 'Std', 'Min', 'Median',
                                          'P_80', 'P_90', 'P_99', 'Max'])

    # Compute and store the statistics for each color channel
    for channel, name in zip([red_channel, green_channel, blue_channel],
                             ['Red', 'Green', 'Blue']):
        mean = np.mean(channel)
        std = np.std(channel)
        minimum = np.min(channel)
        median = np.median(channel)
        p_80 = np.percentile(channel, 80)
        p_90 = np.percentile(channel, 90)
        p_99 = np.percentile(channel, 99)
        maximum = np.max(channel)

        channel_stats.loc[name] = [mean, std, minimum, median, p_80, p_90, p_99, maximum]

    return channel_stats


def plot_cdf(image):
    """
    Plot the cumulative distribution function of an image.

    Parameters:
    image (ndarray): Input image.
    """

    # Convert the image to grayscale if needed
    if len(image.shape) == 3:
        image = rgb2gray(image[:, :, :3])

    # Compute the cumulative distribution function
    intensity = np.round(image * 255).astype(np.uint8)
    freq, bins = cumulative_distribution(intensity)

    # Plot the actual and target CDFs
    target_bins = np.arange(256)
    target_freq = np.linspace(0, 1, len(target_bins))
    plt.step(bins, freq, c='b', label='Actual CDF')
    plt.plot(target_bins, target_freq, c='r', label='Target CDF')

    # Plot an example lookup
    example_intensity = 50
    example_target = np.interp(freq[example_intensity], target_freq, target_bins)
    plt.plot([example_intensity, example_intensity, target_bins[-11], target_bins[-11]],
             [0, freq[example_intensity], freq[example_intensity], 0],
             'k--',
             label=f'Example lookup ({example_intensity} -> {example_target:.0f})')

    # Customize the plot
    plt.legend()
    plt.xlim(0, 255)
    plt.ylim(0, 1)
    plt.xlabel('Intensity Values')
    plt.ylabel('Cumulative Fraction of Pixels')
    plt.title('Cumulative Distribution Function')

    return freq, bins, target_freq, target_bins


if __name__ == '__main__':
    # image = plt.imread('images/view_1.png', None)
    #
    # # temp = image.shape[0]
    # # array = []
    # # for i in range(temp):
    # #     array.append(sum(image[i]))
    # # print(array)
    # #
    # # img = cv2.imread('images/lena.png', 0)
    # # temp = img.shape[0]
    # # array2 = []
    # # for i in range(temp):
    # #     array2.append(sum(img[i]))
    # # print(array2)
    #
    # print(calc_color_overcast(image))
    #
    # # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # # dark_image_grey = img_as_ubyte(rgb2gray(image))
    # # imshow(dark_image_grey)
    #
    # plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='white')
    #
    # # temp_image = img_as_ubyte(rgb2gray(image))
    # # freq, bins = histogram(temp_image)
    #
    # # freq, bins, target_freq, target_bins = plot_cdf(image)
    # plot_cdf(image)
    #
    # # plt.step(x=bins, y=freq / freq.sum())
    #
    # # plt.xlabel('intensity value', fontsize=12)
    # # plt.ylabel('fraction of pixels', fontsize=12)
    #
    # plt.show()

    #
    # pdf
    # image = cv2.imread('images/view_1.png')
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Step 2: Compute histogram of the grayscale image
    # histogram, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
    #
    # # Step 3: Normalize histogram to obtain PDF
    # pdf = histogram / np.sum(histogram)
    #
    # # Step 4: Plot the PDF
    # # plt.plot(pdf, color='black')
    # plt.plot(pdf, color='black', marker='o', linestyle='-', markersize=3)
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Probability Density')
    # plt.title('Probability Density Function (PDF)')
    # plt.show()

    #
    #
    # image = cv2.imread("images/view_3.jpg", cv2.IMREAD_GRAYSCALE)
    # # Flatten the pixel values
    # pixels = image.flatten()
    #
    # histogram, bins = np.histogram(pixels, bins=256, range=(0, 256))
    # # pdf, bins = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    #
    # # Normalize the PMF
    # pmf = histogram / float(np.sum(histogram))
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    #
    # ax1.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))  # !!!
    # ax1.axis('off')
    # ax1.set_title('Grayscale Image', fontsize=12)
    #
    # # threshold = 127
    # # ax2.axhline(y=.002, color='r', linestyle='--', label='Threshold')
    # # ax2.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    # # ax2.legend()
    #
    # # Plot the PMF
    # ax2.plot(bins[:-1], pmf, color='b')
    # ax2.set_xlabel('Pixel Value', fontsize=16)
    # ax2.set_ylabel('Probability', fontsize=16)  # plt.ylabel('Probability Density')
    # ax2.set_title('Probability Mass Function (PMF)', fontsize=12)  # plt.title('Probability Density Function (PDF)')
    #
    # plt.tight_layout()
    # plt.show()

    image_paths = {
        "apples": "images/apples.jpg",
        "bird": "images/bird.jpg",
        "car": "images/car.jpg",
        "cat_1": "images/cat_1.jpg",
        "cat_2": "images/cat_2.jpg",
        "dog": "images/dog.jpg",
        "flower_1": "images/flower_1.png",
        "flower_2": "images/flower_2.png",
        "hacker": "images/hacker.jpg",
        "lena": "images/lena.png"
    }

    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths.values()]

    num_rows = 2
    num_cols = 5

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 8))

    for i, (image, (row, col)) in enumerate(zip(images, np.ndindex((num_rows, num_cols)))):
        histogram, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
        pmf = histogram / np.sum(histogram)
        axs[row, col].plot(bins[:-1], pmf, color='k')
        axs[row, col].set_title(list(image_paths.keys())[i])
        axs[row, col].set_xlabel('Pixel value')
        axs[row, col].set_ylabel('PMF')

    plt.tight_layout()
    plt.show()
