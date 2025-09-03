import numpy as np
import cv2
from ps1 import *


def main():

    # Image file paths:
    img1_filename = "ps1-1-a-1.jpg"
    img2_filename = "ps1-1-a-2.jpg"

    # # 1a
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    assert 100 < img1.shape[0] <= 512, "Check your image 1 dimensions"
    assert 100 < img1.shape[1] <= 512, "Check your image 1 dimensions"
    assert 100 < img2.shape[0] <= 512, "Check your image 2 dimensions"
    assert 100 < img2.shape[1] <= 512, "Check your image 2 dimensions"
    assert img1.shape[1] > img1.shape[0], "Image 1 should be a wide image"
    assert img2.shape[0] > img2.shape[1], "Image 2 should be a tall image"

    # No need to create PNG copies - using JPG files directly

    # # 2 Color Planes

    # # 2a
    swapped_green_blue_img = swap_green_blue(img1)
    cv2.imwrite('ps1-2-a-1.png', swapped_green_blue_img)

    # # 2b
    img1_green = extract_green(img1)
    assert len(img1_green.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('ps1-2-b-1.png', img1_green)

    # # 2c
    img1_red = extract_red(img1)
    assert len(img1_red.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('ps1-2-c-1.png', img1_red)

    # # 3 Replacement of Pixels
    # Note: For this, use ps1-2-b-1.png from 2-b as your monochrome image

    # # 3a
    # Use the monochrome green channel (ps1-2-b-1.png) as source
    src_gray = img1_green  # This is the monochrome green channel from image 1
    dest_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Monochrome version of image 2

    # Extract center 100x100 patch from source
    size = 100
    h_src, w_src = src_gray.shape
    sy, sx = h_src//2 - size//2, w_src//2 - size//2
    patch = src_gray[sy:sy+size, sx:sx+size]

    # Paste into center of destination
    h_dst, w_dst = dest_gray.shape
    dy, dx = h_dst//2 - size//2, w_dst//2 - size//2
    dest_gray[dy:dy+size, dx:dx+size] = patch

    cv2.imwrite('ps1-3-a-1.png', dest_gray)

    # # 3b
    # Copy the same patch but apply circular mask
    dest_gray_circle = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).copy()
    
    r = 50
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - size//2)**2 + (xx - size//2)**2 <= r*r
    
    dest_slice = dest_gray_circle[dy:dy+size, dx:dx+size].copy()
    dest_slice[mask] = patch[mask]
    dest_gray_circle[dy:dy+size, dx:dx+size] = dest_slice

    cv2.imwrite('ps1-3-b-1.png', dest_gray_circle)

    # # 4 Arithmetic and Geometric operations

    # # 4a
    min_green, max_green, mean_green, stddev_green = image_stats(img1_green)

    print("The min pixel value of img1_green is", min_green)
    print("The max pixel value of img1_green is", max_green)
    print("The mean pixel value of img1_green is {:.2f}".format(mean_green))
    print("The std dev of img1_green is {:.2f}".format(stddev_green))

    # # 4b
    normalized_img = center_and_normalize(img1_green, 10)
    cv2.imwrite('ps1-4-b-1.png', normalized_img)

    # # 4c
    shift_green = shift_image_left(img1_green, 2)
    cv2.imwrite('ps1-4-c-1.png', shift_green)

    # # 4d
    diff_green = difference_image(img1_green, shift_green)
    cv2.imwrite('ps1-4-d-1.png', diff_green)

    # # 5 Noise

    # Choose a sigma value:
    sigma = 50  # Increased from 20 to make noise more visible

    # # 5a
    channel = 1
    noisy_green = add_noise(img1, channel, sigma)
    cv2.imwrite('ps1-5-a-1.png', noisy_green)

    # # 5b
    channel = 0
    noisy_blue = add_noise(img1, channel, sigma)
    cv2.imwrite('ps1-5-b-1.png', noisy_blue)

    # # 7 Hybrid Images
    dog_img = cv2.imread('dog.bmp')
    cat_img = cv2.imread('cat.bmp')

    cutoff_frequency = 7

    hybrid_image = build_hybrid_image(dog_img, cat_img, cutoff_frequency)
    
    # Note that some processing in the following lines may be relative to your implementation.
    # If you have/have not altered the channel conventions (BGR -> RGB) and the multiplying factor of 255, you may choose to modify
    # or remove the lines 'vis.copy()*255' and 'vis[:,:,::-1] below.

    
    vis = vis_hybrid_image(hybrid_image)
    vis = (vis.copy() * 255)
    vis = vis.astype(np.uint8)
    cv2.imwrite("ps1-7-a-1.png", vis[:, :, ::-1])  # Save with correct filename


if __name__ == "__main__":
    HONOUR_PLEDGE = "By submitting to gradescope, you accept that you have neither received or given aid in the assignment. Please refer to \"Honor_Code_Policy.pdf\" to know about the policy about plagiarism for this class"
    print(HONOUR_PLEDGE)

    LATE_SUBMISSION_POLICY = "I have read the late assignments policy for CS6476. I understand that only my last commit before the deadline will be accepted without penalty."
    print(LATE_SUBMISSION_POLICY)

    main()
