import cv2
import numpy as np
from matplotlib import pyplot as plt


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=30,
        param2=20,
        minRadius=min(radii_range),
        maxRadius=max(radii_range)
    )
    
    if circles is None:
        return (0, 0), 'red'
    
    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda x: x[1])
    
    if len(circles) < 3:
        center_x, center_y, _ = circles[0]
        return (center_x, center_y), 'red'
    
    top_circle = circles[0]
    middle_circle = circles[1] 
    bottom_circle = circles[2]
    center_x, center_y, _ = middle_circle
    
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([30, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    
    def check_circle_color(circle, hsv_img):
        x, y, r = circle
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        circle_pixels = hsv_img[mask > 0]
        
        if len(circle_pixels) == 0:
            return 'off'
        
        red_mask1 = cv2.inRange(circle_pixels.reshape(-1, 1, 3), red_lower1, red_upper1)
        red_mask2 = cv2.inRange(circle_pixels.reshape(-1, 1, 3), red_lower2, red_upper2)
        red_pixels = np.sum(red_mask1) + np.sum(red_mask2)
        
        yellow_mask = cv2.inRange(circle_pixels.reshape(-1, 1, 3), yellow_lower, yellow_upper)
        yellow_pixels = np.sum(yellow_mask)
        
        green_mask = cv2.inRange(circle_pixels.reshape(-1, 1, 3), green_lower, green_upper)
        green_pixels = np.sum(green_mask)
        
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        if max_pixels < 50:
            return 'off'
        elif red_pixels == max_pixels:
            return 'red'
        elif yellow_pixels == max_pixels:
            return 'yellow'
        elif green_pixels == max_pixels:
            return 'green'
        else:
            return 'off'
    
    top_color = check_circle_color(top_circle, hsv)
    middle_color = check_circle_color(middle_circle, hsv)
    bottom_color = check_circle_color(bottom_circle, hsv)
    
    if top_color == 'red':
        state = 'red'
    elif middle_color == 'yellow':
        state = 'yellow'
    elif bottom_color == 'green':
        state = 'green'
    else:
        def get_brightness(circle, gray_img):
            x, y, r = circle
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            return np.mean(gray_img[mask > 0])
        
        top_brightness = get_brightness(top_circle, gray)
        middle_brightness = get_brightness(middle_circle, gray)
        bottom_brightness = get_brightness(bottom_circle, gray)
        
        if top_brightness > middle_brightness and top_brightness > bottom_brightness:
            state = 'red'
        elif middle_brightness > top_brightness and middle_brightness > bottom_brightness:
            state = 'yellow'
        else:
            state = 'green'
    
    return (center_x, center_y), state


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine
       the degree to which the template matches the original image
       We are required to implement each technique using the
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        float,
    )
    top_left = []
    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
    # Sum of squared differences
    if method == "tm_ssd":
        """Your code goes here"""
        raise NotImplementedError

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        """Your code goes here"""
        raise NotImplementedError

    # Cross Correlation
    elif method == "tm_ccor":
        """Your code goes here"""
        raise NotImplementedError

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        """Your code goes here"""
        raise NotImplementedError

    else:
        """Your code goes here"""
        # Invalid technique
    raise NotImplementedError
    return top_left


'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex_)
    raise NotImplementedError


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    x = np.asarray(x, dtype=np.complex_)
    raise NotImplementedError


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    raise NotImplementedError


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    raise NotImplementedError


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image. (Make sure the data type of the np array is float64)
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain

    """
    raise NotImplementedError


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image. (Make sure the data type of the np array is float64)
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """
    raise NotImplementedError