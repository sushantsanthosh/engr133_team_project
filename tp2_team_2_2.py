"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Program takes in an image metadata CSV, cleans up each image, extracts nine color and shape features from it, saves the feature dataset to a CSV file, and prints the features for an image based on the user's input.

Assignment Information:
    Assignment:     8.2.2 tp2 Team 2
    Team ID:        LC5 - 02
    Author:         Sushant Santhosh, ssantho@purdue.edu
    Date:           10/22/2026

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
import pandas as pd
import cv2
from PIL import Image, ImageOps

TARGET_SIZE = (100, 100)
FEATURE_NAMES = ["hue_mean", "hue_std", "saturation_mean", "saturation_std", "value_mean", "value_std", "number_lines", "has_circle", "has_triangle"]


def load_img(file_path): # from 17.2.2
    img = np.array(Image.open(file_path)) # opens image and turns it into a numpy array

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3] # removes the A channel

    # changes range from 0-255 to 0-1
    if np.issubdtype(img.dtype, np.integer) or img.max() > 1.0:
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)

    # undo the gamma encoding to get linear light values
    linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return linear


def rgb_to_grayscale(linear_rgb): # from 17.2.2
    # split up the red, green, and blue channels
    r = linear_rgb[:, :, 0]
    g = linear_rgb[:, :, 1]
    b = linear_rgb[:, :, 2]
    # weighted sum since our eyes see green as the brightest and blue as the darkest
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def resize_image(original_image_array, new_size): # from 17.2.3
    # new_size is (width, height), bilinear blends nearby pixels when resizing
    image = Image.fromarray(original_image_array)
    resized = image.resize(new_size, Image.BILINEAR)
    return np.array(resized)


def pad_image(resized_img_array, target_size): # from 17.2.3
    # fills in any leftover space so the image is exactly the target size
    image = Image.fromarray(resized_img_array)
    padded = ImageOps.pad(image, target_size)
    return np.array(padded)


def rgb_to_hsv(r, g, b): # from 7.2.1
    r_p, g_p, b_p = r / 255, g / 255, b / 255 # changes range from 0-255 to 0-1
    c_max = max(r_p, g_p, b_p)
    c_min = min(r_p, g_p, b_p)
    delta = c_max - c_min

    # sets the hue based on the largest channel
    if delta == 0:
        h_p = 0 # gray pixel so there is no hue
    elif c_max == r_p:
        h_p = (60 * ((g_p - b_p) / delta) + 360) % 360 # add 360 so negative angles wrap around
    elif c_max == g_p:
        h_p = 60 * ((b_p - r_p) / delta) + 120
    else:
        h_p = 60 * ((r_p - g_p) / delta) + 240

    # saturation is 0 for black pixels so we don't divide by 0
    if c_max == 0:
        s_p = 0
    else:
        s_p = delta / c_max
    v_p = c_max # value is just the brightest channel

    # changes range back to 0-255 (hue goes from 0-360 degrees to 0-255)
    return h_p * 255 / 360, s_p * 255, v_p * 255


def convert_to_hsv(rgb_image): # from 7.2.1
    # output has the same height and width as the input, stored as 8-bit values
    rows, cols = rgb_image.shape[0], rgb_image.shape[1]
    hsv_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            # convert one pixel at a time (int() keeps the uint8 math from overflowing)
            r, g, b = rgb_image[i, j, :3]
            hsv_image[i, j] = rgb_to_hsv(int(r), int(g), int(b))
    return hsv_image


def create_gaussian_kernel(sigma): # from 7.2.3
    # kernel side lengths are always odd
    size = 2 * math.ceil(3 * sigma) + 1
    center = size // 2 # middle index of the kernel
    kernel = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            # distance from the center of the kernel
            x = j - center
            y = i - center
            kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # normalize so the values add up to 1 and brightness stays the same
    return kernel / np.sum(kernel)


def gaussian_filter(image, sigma): # from 7.2.3
    kernel = create_gaussian_kernel(sigma)
    k_size = kernel.shape[0]
    pad_size = (k_size - 1) // 2

    # add zero padding around the image
    padded = np.pad(image.astype(np.float64), pad_size, mode="constant", constant_values=0)

    # use a stride of 1
    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            window = padded[i:i + k_size, j:j + k_size]
            output[i, j] = np.sum(window * kernel)

    # returns 8-bit values if the inputs are 8-bit values
    if image.dtype == np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def sobel_filter(data, threshold=50): # from 8.2.1
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    data_array = np.array(data, dtype=float) # use floats to avoid early rounding errors

    # zero pad the image and slide both kernels across
    padded = np.pad(data_array, 1, mode="constant", constant_values=0)
    rows, cols = data_array.shape
    gx = np.zeros((rows, cols), dtype=float)
    gy = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            window = padded[i:i + 3, j:j + 3]
            gx[i, j] = np.sum(window * sobel_x)
            gy[i, j] = np.sum(window * sobel_y)

    # combine the gradients and keep the max at 255
    magnitude = np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 255)

    edges = np.where(magnitude >= threshold, 255, 0)
    return edges.astype(np.uint8)


def detect_circle(gray_img):
    # blur first so noise doesn't get picked up as circles, cv2 needs 8-bit images
    blurred_img = gaussian_filter(gray_img, sigma=1.5)
    blurred_img = np.array(blurred_img).astype(np.uint8)

    circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=50)
    
    if circles is None:
        return 0
    return 1


def detect_triangle(gray_img):
    blurred = gaussian_filter(gray_img, sigma=1.0).astype(np.uint8)
    edges = sobel_filter(blurred)
    # find the outlines of shapes in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        # simplifies the outline
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3 and cv2.contourArea(approx) > 625:
            return 1
    return 0


def count_lines(gray_img):
    edges = sobel_filter(gray_img)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return 0
    return len(lines)


def extract_features(img_array):
    # mean and standard deviation of the hue, saturation, and value channels
    hsv_img = convert_to_hsv(img_array)
    features = []
    for channel in range(3):
        features.append(float(np.mean(hsv_img[:, :, channel])))
        features.append(float(np.std(hsv_img[:, :, channel])))

    # use the grayscale version of the image for the shape features
    gray_img = rgb_to_grayscale(img_array)
    features.append(int(count_lines(gray_img)))
    features.append(int(detect_circle(gray_img)))
    features.append(int(detect_triangle(gray_img)))
    return features


def clean_image(file_path):
    # load the image and change the range of the 8-bit values to 0-255
    img_array = (load_img(file_path) * 255).astype(np.uint8)
    # resize and pad so every image is 100x100

    resized = resize_image(img_array, TARGET_SIZE)
    return pad_image(resized, TARGET_SIZE)


def main():
    # ask the user for the folder, metadata file, and output file names
    dataset_folder = input("Enter the name of the dataset folder:\n")
    metadata_file = input("Enter the name of the metadata file:\n")
    output_file = input("Enter the name of output csv features file:\n")

    # load the metadata
    metadata_df = pd.read_csv(metadata_file)
    print("\nTraining Metadata DataFrame:")
    print(metadata_df.head())

    # goes through every image and extracts its 9 features along with its path and class
    rows = []
    for _, row in metadata_df.iterrows():
        image_path = row["Path"]
        img_array = clean_image(os.path.join(dataset_folder, image_path))
        features = extract_features(img_array)
        rows.append(features + [image_path, row["ClassId"]])

    # make the 11 column dataset and save it as a csv
    features_df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["Path", "ClassId"])
    shape_cols = ["number_lines", "has_circle", "has_triangle"]
    features_df[shape_cols] = features_df[shape_cols].astype(float)
    features_df.to_csv(output_file, index=False)
    print(f"\nFeatures have been extracted and saved to {output_file}")

    print("\nFeature Dataset Shape:")
    print(features_df.shape)
    print("\nFeature Dataset Head:")
    print(features_df.head())

    # print the features for the image based on the user's input
    image_path = input("\nEnter the Path of the image you want to display the features for:\n")
    print(f"\nFeatures for {image_path}:")
    print(features_df[features_df["Path"] == image_path])


if __name__ == "__main__":
    main()
