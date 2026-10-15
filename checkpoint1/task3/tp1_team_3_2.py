"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Program loads an image, converts it to grayscale, and smooths it with a
    Gaussian filter (sigma = 1), then displays and saves the blurred image.

Assignment Information:
    Assignment:     7.2.3 tp1 Team 3
    Team ID:        LC5 - 02
    Author:         Sushant Santhosh, ssantho@purdue.edu
    Date:           10/15/2026

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_img(file_path):  # copied from 17.2.2
    # open the image file and turn it into a numpy array
    img = np.array(Image.open(file_path))

    # remove the alpha channel if the image is RGBA
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # change range from 0-255 to 0-1 so all images use the same scale
    if np.issubdtype(img.dtype, np.integer) or img.max() > 1.0:
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)

    # remove the sRGB gamma encoding to get linear light values
    linear = np.where(img <= 0.04045,
                      img / 12.92,
                      ((img + 0.055) / 1.055) ** 2.4)
    return linear


def rgb_to_grayscale(linear_rgb):  # copied from 17.2.2
    # split the image into its red, green, and blue channels
    r = linear_rgb[:, :, 0]
    g = linear_rgb[:, :, 1]
    b = linear_rgb[:, :, 2]
    # weighted sum of the channels based on how bright each color looks
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def create_gaussian_kernel(sigma):
    # kernel is (2 * ceil(3 * sigma) + 1) on each side, so it is always odd
    # and covers about 3 standard deviations in every direction
    size = 2 * math.ceil(3 * sigma) + 1
    # index of the middle row/column of the kernel
    center = size // 2
    # start with an empty square kernel filled with zeros
    kernel = np.zeros((size, size), dtype=np.float64)

    # loop over every position in the kernel
    for i in range(size):
        for j in range(size):
            # x and y are the offsets from the center of the kernel
            x = j - center
            y = i - center
            # apply the 2D Gaussian formula at this offset
            kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * \
                math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # normalize so all values add up to 1 and the brightness stays the same
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_filter(image, sigma):
    # build the kernel that will be slid across the image
    kernel = create_gaussian_kernel(sigma)
    k_size = kernel.shape[0]
    # number of zero rows/columns needed on each side of the image
    pad_size = (k_size - 1) // 2

    # add zero padding around the image
    padded = np.pad(image.astype(np.float64), pad_size,
                    mode="constant", constant_values=0)

    # gets size of the original image
    rows, cols = image.shape
    # separate output array so the original image is never changed
    output = np.zeros((rows, cols), dtype=np.float64)

    # slide the kernel over the image with a stride of 1
    for i in range(rows):
        for j in range(cols):
            # grab the patch of the padded image under the kernel
            window = padded[i:i + k_size, j:j + k_size]
            # weighted sum of the patch gives the blurred pixel value
            output[i, j] = np.sum(window * kernel)

    # if the input was 8-bit, return 8-bit so the data type matches
    # keep values inside 0-255, then drop the decimal
    if image.dtype == np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def main():
    # ask the user which image in ref_images to blur
    file_name = input("Enter the path to the image file: ")
    # load the image from the ref_images folder
    image = load_img(os.path.join("ref_images", file_name))

    # convert color images to grayscale before filtering
    if image.ndim == 3:
        image = rgb_to_grayscale(image)

    # blur the grayscale image with sigma = 1
    blurred = gaussian_filter(image, 1)

    # save the blurred image next to the program as <name>_smooth.png
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    plt.imsave(f"{base_name}_smooth.png", blurred, cmap="gray", vmin=0, vmax=1)

    # display the blurred image in grayscale without axis ticks
    plt.imshow(blurred, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()