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
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    if lines is None:
        return (0, 0)
    
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([25, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    color_mask = cv2.bitwise_or(orange_mask, yellow_mask)
    
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) > 1e-6:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                
                if 0 <= t <= 1 and 0 <= u <= 1:
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    
                    if 0 <= px < img_in.shape[1] and 0 <= py < img_in.shape[0]:
                        if color_mask[py, px] > 0:
                            intersections.append((px, py))
    
    if not intersections:
        return (0, 0)
    
    intersections = np.array(intersections)
    
    if len(intersections) < 4:
        return (int(np.mean(intersections[:, 0])), int(np.mean(intersections[:, 1])))
    
    def distance_to_center(point, center):
        return np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    
    center_x = np.mean(intersections[:, 0])
    center_y = np.mean(intersections[:, 1])
    
    distances = [distance_to_center(p, (center_x, center_y)) for p in intersections]
    threshold = np.percentile(distances, 75)
    
    filtered_points = [p for i, p in enumerate(intersections) if distances[i] <= threshold]
    
    if len(filtered_points) >= 4:
        filtered_points = np.array(filtered_points)
        center_x = np.mean(filtered_points[:, 0])
        center_y = np.mean(filtered_points[:, 1])
    
    return (int(center_x), int(center_y))


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
    # Convert to grayscale if needed
    if len(img_orig.shape) == 3:
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    if len(img_template.shape) == 3:
        img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    
    # Convert to float64 for calculations
    img_orig = img_orig.astype(np.float64)
    img_template = img_template.astype(np.float64)
    
    # Calculate result matrix dimensions
    h_orig, w_orig = img_orig.shape
    h_template, w_template = img_template.shape
    result_h = h_orig - h_template + 1
    result_w = w_orig - w_template + 1
    
    if method == "tm_ssd":
        # Sum of Squared Differences - minimize for best match
        result = np.zeros((result_h, result_w), dtype=np.float64)
        for i in range(result_h):
            for j in range(result_w):
                window = img_orig[i:i+h_template, j:j+w_template]
                result[i, j] = np.sum((window - img_template) ** 2)
        top_left = np.unravel_index(np.argmin(result), result.shape)
    
    elif method == "tm_nssd":
        # Normalized Sum of Squared Differences
        result = np.zeros((result_h, result_w), dtype=np.float64)
        template_norm = np.sqrt(np.sum(img_template ** 2))
        for i in range(result_h):
            for j in range(result_w):
                window = img_orig[i:i+h_template, j:j+w_template]
                window_norm = np.sqrt(np.sum(window ** 2))
                if window_norm > 0:
                    result[i, j] = np.sum((window - img_template) ** 2) / (template_norm * window_norm)
                else:
                    result[i, j] = float('inf')
        top_left = np.unravel_index(np.argmin(result), result.shape)
    
    elif method == "tm_ccor":
        # Cross Correlation - use filter2D for efficiency
        result = cv2.filter2D(img_orig, -1, img_template)
        top_left = np.unravel_index(np.argmax(result), result.shape)
    
    elif method == "tm_nccor":
        # Normalized Cross Correlation with mean centering
        template_mean = np.mean(img_template)
        template_centered = img_template - template_mean
        template_norm = np.sqrt(np.sum(template_centered ** 2))
        
        result = np.zeros((result_h, result_w), dtype=np.float64)
        for i in range(result_h):
            for j in range(result_w):
                window = img_orig[i:i+h_template, j:j+w_template]
                window_mean = np.mean(window)
                window_centered = window - window_mean
                window_norm = np.sqrt(np.sum(window_centered ** 2))
                
                if template_norm > 0 and window_norm > 0:
                    result[i, j] = np.sum(template_centered * window_centered) / (template_norm * window_norm)
                else:
                    result[i, j] = 0
        top_left = np.unravel_index(np.argmax(result), result.shape)
    
    else:
        raise ValueError("Invalid method")
    
    return (int(top_left[1]), int(top_left[0]))


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