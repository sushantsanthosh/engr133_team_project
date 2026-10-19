"""
Course Number: ENGR 13300
Fall 2026

Description:
    This part of the project blurs a given image by padding the image, and then using a kernel to modify each of the grey-scale image's pixel's brightness'

Assignment Information:
    Assignment:     Tp1 Team 2
    Team ID:        LC5 02
    Author:         Ram, rvadhadi@purdue.edu
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

import numpy as np
from PIL import Image, ImageOps

#creates the kernel by returning a np array full of 1/size^2
def create_mean_kernel(kernel_size):
    return np.full((kernel_size, kernel_size), 1/(kernel_size ** 2))

#pads image
def pad_image(resized_img, target_size):
    #converts to uint8
    u8arr = np.clip(np.rint(resized_img), 0, 255).astype(np.uint8)
    #pads img with ImageOps
    img = Image.fromarray(u8arr)
    padded_img = ImageOps.pad(img, target_size, method=Image.BILINEAR, color="black")
    return np.asarray(padded_img)

def apply_kernel(image, kernel):
    img = np.array(image)
    #sets kernel height and width
    k_h = (kernel.shape[0]-1)//2
    k_w = (kernel.shape[1]-1)//2
    #sets image height and width
    h = int(img.shape[0])
    w = int(img.shape[1])
    #sets the padded height and width
    pad_h = int(h + 2*k_h)
    pad_w = int(w + 2*k_w)
    #pads the img
    img = pad_image(img, (pad_w, pad_h))

    output = np.zeros((h, w))
    #iterates through img with stride 1
    for i in range(h):
        for j in range(w):
            #sets each patch of size kernel.shape
            patch = img[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            #multiplies each pixel by the kernel weighting to produce output
            output[i,j] = np.sum(patch * kernel)

    return output

def main():
    #sets default output precision to 6 decimal places
    np.set_printoptions(precision=6)

    #inputs kernel size
    k_size = int(input("Enter kernel size: "))
    print(f"Kernel size: {k_size}x{k_size}")
    print("Kernel sum: 1.000000")

    #creates and outputs kernel
    kernel = create_mean_kernel(k_size)
    print("Kernel:")
    print(kernel)
    

    #initializes test image array
    image = np.array([
    [0, 0,   0, 0, 0],
    [0, 0,   0, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0,   0, 0, 0],
    [0, 0,   0, 0, 0]
    ], dtype=np.uint8)

    #applies the kernel and outputs the filtered image
    filtered_img = apply_kernel(image, kernel)
    print("Filtered image:")
    print(filtered_img)
    
if __name__ == "__main__":
    main()
