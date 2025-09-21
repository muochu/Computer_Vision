import cv2
import numpy as np
# import matplotlib.pyplot as plt


def traffic_light_detection(img, radius_range):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=30, param2=20, minRadius=min(radius_range), maxRadius=max(radius_range))
    
    if circles is None:  # no circles found
        return (0, 0), 'red'
    
    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda x: x[1])
    
    if len(circles) < 3:  # not enough circles
        return (circles[0][0], circles[0][1]), 'red'
    
    top = circles[0]
    middle = circles[1] 
    bottom = circles[2]
    
    # get precise center using moments from circular mask
    x, y, r = middle
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    M = cv2.moments(mask)
    if M['m00'] != 0:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
    else:
        center_x, center_y = x, y
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # red has two ranges because of hue wrap-around
    red1_low = np.array([0, 50, 50])  # first red range
    red1_high = np.array([10, 255, 255])
    red2_low = np.array([170, 50, 50])  # second red range
    red2_high = np.array([180, 255, 255])
    
    yellow_low = np.array([20, 50, 50])
    yellow_high = np.array([30, 255, 255])
    
    green_low = np.array([40, 50, 50])
    green_high = np.array([80, 255, 255])
    
    def get_color(circle, hsv_img):
        x, y, r = circle
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        pixels = hsv_img[mask > 0]
        
        if len(pixels) == 0:
            return 'off'
        
        # check red (two ranges)
        red1_mask = (pixels[...,0] >= red1_low[0]) & (pixels[...,0] <= red1_high[0]) & (pixels[...,1] >= red1_low[1]) & (pixels[...,1] <= red1_high[1]) & (pixels[...,2] >= red1_low[2]) & (pixels[...,2] <= red1_high[2])
        red2_mask = (pixels[...,0] >= red2_low[0]) & (pixels[...,0] <= red2_high[0]) & (pixels[...,1] >= red2_low[1]) & (pixels[...,1] <= red2_high[1]) & (pixels[...,2] >= red2_low[2]) & (pixels[...,2] <= red2_high[2])
        red_count = np.sum(red1_mask) + np.sum(red2_mask)
        
        # check yellow
        yellow_mask = (pixels[...,0] >= yellow_low[0]) & (pixels[...,0] <= yellow_high[0]) & (pixels[...,1] >= yellow_low[1]) & (pixels[...,1] <= yellow_high[1]) & (pixels[...,2] >= yellow_low[2]) & (pixels[...,2] <= yellow_high[2])
        yellow_count = np.sum(yellow_mask)
        
        # check green
        green_mask = (pixels[...,0] >= green_low[0]) & (pixels[...,0] <= green_high[0]) & (pixels[...,1] >= green_low[1]) & (pixels[...,1] <= green_high[1]) & (pixels[...,2] >= green_low[2]) & (pixels[...,2] <= green_high[2])
        green_count = np.sum(green_mask)
        
        max_count = max(red_count, yellow_count, green_count)
        
        if max_count < 50:
            return 'off'
        elif red_count == max_count:
            return 'red'
        elif yellow_count == max_count:
            return 'yellow'
        elif green_count == max_count:
            return 'green'
        else:
            return 'off'
    
    top_color = get_color(top, hsv)
    middle_color = get_color(middle, hsv)
    bottom_color = get_color(bottom, hsv)
    
    if top_color == 'red':
        state = 'red'
    elif middle_color == 'yellow':
        state = 'yellow'
    elif bottom_color == 'green':
        state = 'green'
    else:
        # fallback to brightness if color detection fails
        def brightness(circle, gray_img):
            x, y, r = circle
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            return np.mean(gray_img[mask > 0])
        
        top_bright = brightness(top, gray)
        middle_bright = brightness(middle, gray)
        bottom_bright = brightness(bottom, gray)
        
        if top_bright > middle_bright and top_bright > bottom_bright:
            state = 'red'
        elif middle_bright > top_bright and middle_bright > bottom_bright:
            state = 'yellow'
        else:
            state = 'green'
    
    return (center_x, center_y), state


def construction_sign_detection(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # orange range for construction signs
    orange_min = np.array([10, 100, 100])
    orange_max = np.array([25, 255, 255])
    
    orange_mask = cv2.inRange(hsv, orange_min, orange_max)
    
    ys, xs = np.where(orange_mask > 0)
    
    if len(xs) == 0:
        return (0, 0)
    
    # compute centroid
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    
    return (cx, cy)


def template_match(img, template, method):
    # convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float64)
    template = template.astype(np.float64)
    
    h, w = img.shape
    th, tw = template.shape
    result_h = h - th + 1
    result_w = w - tw + 1
    
    if method == "tm_ssd":
        result = np.zeros((result_h, result_w), dtype=np.float64)
        for i in range(result_h):
            for j in range(result_w):
                window = img[i:i+th, j:j+tw]
                result[i, j] = np.sum((window - template) ** 2)
        top_left = np.unravel_index(np.argmin(result), result.shape)
    
    elif method == "tm_nssd":
        result = np.zeros((result_h, result_w), dtype=np.float64)
        template_norm = np.sqrt(np.sum(template ** 2))
        for i in range(result_h):
            for j in range(result_w):
                window = img[i:i+th, j:j+tw]
                window_norm = np.sqrt(np.sum(window ** 2))
                if window_norm > 0:
                    result[i, j] = np.sum((window - template) ** 2) / (template_norm * window_norm)
                else:
                    result[i, j] = float('inf')
        top_left = np.unravel_index(np.argmin(result), result.shape)
    
    elif method == "tm_ccor":
        result = cv2.filter2D(img, -1, template)
        top_left = np.unravel_index(np.argmax(result), result.shape)
    
    elif method == "tm_nccor":
        template_mean = np.mean(template)
        template_centered = template - template_mean
        template_norm = np.sqrt(np.sum(template_centered ** 2))
        
        result = np.zeros((result_h, result_w), dtype=np.float64)
        for i in range(result_h):
            for j in range(result_w):
                window = img[i:i+th, j:j+tw]
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


# helper code for debugging template matching
# cv2.rectangle(img, top_left, bottom_right, 255, 2)
# plt.imshow(result, cmap='gray')
# plt.show()


def dft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    dft_matrix = np.exp(-2j * np.pi * k * n / N)
    y = np.dot(dft_matrix, x)
    
    return y


def idft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    idft_matrix = np.exp(2j * np.pi * k * n / N)
    y = np.dot(idft_matrix, x) / N
    
    return y


def dft2(img):
    img = np.asarray(img, dtype=np.complex128)
    rows, cols = img.shape
    
    dft_rows = np.zeros_like(img, dtype=np.complex128)
    for i in range(rows):
        dft_rows[i, :] = dft(img[i, :])
    
    dft_result = np.zeros_like(img, dtype=np.complex128)
    for j in range(cols):
        dft_result[:, j] = dft(dft_rows[:, j])
    
    return dft_result


def idft2(img):
    img = np.asarray(img, dtype=np.complex128)
    rows, cols = img.shape
    
    idft_cols = np.zeros_like(img, dtype=np.complex128)
    for j in range(cols):
        idft_cols[:, j] = idft(img[:, j])
    
    idft_result = np.zeros_like(img, dtype=np.complex128)
    for i in range(rows):
        idft_result[i, :] = idft(idft_cols[i, :])
    
    return idft_result


def compress_image_fft(img_bgr, threshold_percentage):
    img_bgr = img_bgr.astype(np.float64)
    h, w, c = img_bgr.shape
    
    img_compressed = np.zeros_like(img_bgr, dtype=np.float64)
    compressed_frequency_img = np.zeros_like(img_bgr, dtype=np.complex128)
    
    for channel in range(c):
        F = np.fft.fft2(img_bgr[:, :, channel])
        
        magnitude = np.abs(F)
        magnitude_flat = magnitude.flatten()
        
        sorted_indices = np.argsort(magnitude_flat)[::-1]
        
        keep_count = int(np.floor(threshold_percentage * len(magnitude_flat)))
        top_indices = sorted_indices[:keep_count]
        
        mask = np.zeros_like(magnitude_flat, dtype=bool)
        mask[top_indices] = True
        mask = mask.reshape(magnitude.shape)
        
        masked_F = F * mask
        
        channel_compressed = np.real(np.fft.ifft2(masked_F))
        
        img_compressed[:, :, channel] = channel_compressed
        compressed_frequency_img[:, :, channel] = masked_F
    
    return img_compressed, compressed_frequency_img


def low_pass_filter(img_bgr, r):
    img_bgr = img_bgr.astype(np.float64)
    h, w, c = img_bgr.shape
    
    img_low_pass = np.zeros_like(img_bgr, dtype=np.float64)
    low_pass_frequency_img = np.zeros_like(img_bgr, dtype=np.complex128)
    
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2
    
    for channel in range(c):
        F = np.fft.fft2(img_bgr[:, :, channel])
        F_shift = np.fft.fftshift(F)
        
        masked_F_shift = F_shift * mask
        masked_F = np.fft.ifftshift(masked_F_shift)
        
        channel_filtered = np.real(np.fft.ifft2(masked_F))
        
        img_low_pass[:, :, channel] = channel_filtered
        low_pass_frequency_img[:, :, channel] = masked_F_shift
    
    return img_low_pass, low_pass_frequency_img