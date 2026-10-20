"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     tp1 team 1
    Team ID:        LC5 - 02 
    Author:         Kristian Iliev, kiliev@purdue.edu
    Date:           10/15/2026

Contributors:
    Name, login@purdue [repeat for each]
    Ram Vadhadia, rvadhadi@purdue.edu
    alex demirdjan, ademirdj@purdue.edu
    sushant santhosh, ssantho@purdue.edu


My contributor(s) helped me:
    [X] understand the assignment expectations without
            telling me how they will approach it.
    [X] understand different ways to think about a solution
            without helping me plan my solution.
    [X] think through the meaning of a specific error or
            bug present in my code without looking at my code.
        Note that if you helped somebody else with their code, you
        have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.

"""


from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def load_image(path): #takes path for image and makes sure it is in 8-bit format
    image = Image.open(path) #opens path of the image
    image_array = np.array(image) #turns it into array
    if (image_array.dtype != np.uint8): #checks if its proper data type and returns value error if not
        raise ValueError("The image datatype is not unsigned 8-bit integer")
    if (image_array.ndim == 3): #checks to see if its an RGB, in order to linearize it
        image_array = image_array[:, :, :3]
    normalize_array = image_array / 255.0 #normalizes it
    if normalize_array.ndim == 3:  #if rgb
            for row in range(normalize_array.shape[0]):
                for column in range (normalize_array.shape[1]):             
                    for channel in range(normalize_array.shape[2]):
                        value = normalize_array[row][column][channel]
    
                        if (value <= 0.04045):
                            value = value / 12.92
                        else:
                            value = ((value + 0.055) / 1.055) ** 2.4
                        normalize_array[row][column][channel] = value #linearizes the array with value equations taken from Task 1
    return normalize_array #returns array

def resize_image(original_image_array, new_size):
    #creates variables for new size
    neww, newh = new_size
    #initializes variables for height and with of orig
    h, w = original_image_array.shape[:2]

    #gets the minimum of the ratios so image doesn't get cut off
    if(neww/w < newh/h):
        size = (int(neww), int(h * neww/w))
    else:
        size = (int(w * newh/h), int(newh))

    #converts the type to uint8 if not already
    if np.issubdtype(original_image_array.dtype, np.floating):
        u8arr = (np.clip(original_image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        u8arr = original_image_array.astype(np.uint8)

    #forms the image array
    img = Image.fromarray(u8arr)
    img = img.resize(size, resample=Image.Resampling.BILINEAR)

    return np.asarray(img)

def pad_image(resized_img, target_size):

    #converts to type to uint 8 if not already
    if np.issubdtype(resized_img.dtype, np.floating):
        u8arr = (np.clip(resized_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        u8arr = resized_img.astype(np.uint8)

    #forms and pads the image with black space
    h, w = u8arr.shape[:2]
    neww, newh = target_size

    padded_img = np.pad(u8arr,(((newh - h) // 2, (newh - h + 1) // 2),((neww - w) // 2, (neww - w + 1) // 2),(0, 0)),mode="edge")
    return padded_img

def rgb_to_hsv(R, G, B):
    # normalize each channel    
    R_prime = R / 255.0
    G_prime = G / 255.0
    B_prime = B / 255.0

    #Calculate intermediate values
    cmax = max(R_prime, G_prime, B_prime)
    cmin = min(R_prime, G_prime, B_prime)
    delta = cmax - cmin

    #Calculate Hue B_primeased on which channel is maximum:
    if (delta == 0):
        h_prime = 0

    elif (cmax == R_prime):
        h_prime = (60 * ((G_prime - B_prime) / delta)) % 360

    elif (cmax == G_prime):
        h_prime = (60 * ((B_prime - R_prime) / delta) + 120) % 360

    else:
        h_prime = (60 * ((R_prime - G_prime) / delta) + 240) % 360

    #Calculate Saturation
    if (cmax == 0):
        s_prime = 0
    else:
        s_prime = (delta / cmax)

    #Calculate Value
    v_prime = cmax

    #Rescale value into HSV
    h = (h_prime / 360) * 255
    s = (s_prime) * 255
    v = (v_prime) * 255

    return h, s, v

def convert_to_hsv(rgb_image):
    #intialize empty numpy array for hsv_image
    hsv_image = np.zeros(rgb_image.shape, dtype=np.uint8)

    #iterate for every pixel in rgb_image
    for row in range(rgb_image.shape[0]):
        for col in range(rgb_image.shape[1]):
            # get each pixel RGB
            r = rgb_image[row, col, 0]
            g = rgb_image[row, col, 1]
            b = rgb_image[row, col, 2]

            # convert to hsv
            h, s, v = rgb_to_hsv(r, g, b)
            h = round(h, 10)
            s = round(s, 10)
            v = round(v, 10)    
            hsv_image[row, col, 0] = h
            hsv_image[row, col, 1] = s
            hsv_image[row, col, 2] = v

    return hsv_image


def main():
    path = input("Enter the path of the RGB image you want to convert to hsv: ") #gets user input for temp path and scene path
    image_array = load_image(path)
    image_array = (image_array * 255).astype(np.uint8)
    resized_image = resize_image(image_array, (100, 100))
    image_array = pad_image(resized_image, (100, 100))
    x, y = input("Enter the coordinates of the pixel you want inspect: ").split(",")
    x = int(x)
    y = int(y)
    r = image_array[x, y, 0]
    g = image_array[x, y, 1]
    b = image_array[x, y, 2]
    print(f"RGB values of the ({x}, {y}) pixel: R={r}, G={g}, B={b}")
    print(f"Converting {path} to HSV...")
    hsv_image = convert_to_hsv(image_array)
    h = hsv_image[x, y, 0]
    s = hsv_image[x, y, 1]
    v = hsv_image[x, y, 2] 
    print(f"HSV values of the ({x}, {y}) pixel: H={h}, S={s}, V={v}")
    plt.imshow(hsv_image)
    plt.show()

if __name__ == "__main__":
    main()
