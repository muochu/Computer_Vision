"""Problem Set 4: Motion Detection"""

import cv2
import numpy as np


# Utility function
def read_video(video_file, show=False):
    """Reads a video file and outputs a list of consecuative frames
  Args:
      image (string): Video file path
      show (bool):    Visualize the input video. WARNING doesn't work in
                      notebooks
  Returns:
      list(numpy.ndarray): list of frames
  """
    frames = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Opens a new window and displays the input
        if show:
            cv2.imshow("input", frame)
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # The following frees up resources and
    # closes all windows
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return frames
    
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    # sobel x gradient with 1/8 scale and ksize 3
    grad_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1/8)
    return grad_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    # sobel y gradient with 1/8 scale and ksize 3
    grad_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1/8)
    return grad_y


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    # get gradients
    I_x = gradient_x(img_a)
    I_y = gradient_y(img_a)
    I_t = img_b - img_a
    
    # make kernel
    if k_type == 'uniform':
        kernel = np.ones((k_size, k_size), dtype=np.float64) / (k_size * k_size)
    elif k_type == 'gaussian':
        kernel_1d = cv2.getGaussianKernel(k_size, sigma)
        kernel = kernel_1d @ kernel_1d.T
    else:
        raise ValueError(f"Unknown kernel type: {k_type}")
    
    # compute products for equations
    I_xx = cv2.filter2D(I_x * I_x, -1, kernel)
    I_yy = cv2.filter2D(I_y * I_y, -1, kernel)
    I_xy = cv2.filter2D(I_x * I_y, -1, kernel)
    I_xt = cv2.filter2D(I_x * I_t, -1, kernel)
    I_yt = cv2.filter2D(I_y * I_t, -1, kernel)
    
    # solve for each pixel
    U = np.zeros_like(img_a)
    V = np.zeros_like(img_a)
    
    for y in range(img_a.shape[0]):
        for x in range(img_a.shape[1]):
            A = np.array([[I_xx[y, x], I_xy[y, x]], 
                         [I_xy[y, x], I_yy[y, x]]])
            b = -np.array([I_xt[y, x], I_yt[y, x]])
            
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) > 1e-10:
                flow = np.linalg.solve(A, b)
                U[y, x] = flow[0]
                V[y, x] = flow[1]
            else:
                U[y, x] = 0
                V[y, x] = 0
    
    return U, V


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    # 5-tap separable filter for reduce operation
    # [1, 4, 6, 4, 1] / 16
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    
    # apply separable convolution
    # first convolve horizontally
    temp = cv2.filter2D(image, -1, kernel_1d.reshape(1, -1))
    # then convolve vertically  
    blurred = cv2.filter2D(temp, -1, kernel_1d.reshape(-1, 1))
    
    # downsample by factor of 2 (take every other pixel)
    h, w = blurred.shape
    reduced = blurred[::2, ::2]
    
    return reduced


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in 
    [0.0, 1.0].

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramid = []
    current_image = image.copy()
    
    for i in range(levels):
        pyramid.append(current_image)
        if i < levels - 1:  # don't reduce the last level
            current_image = reduce_image(current_image)
    
    return pyramid


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    if not img_list:
        return np.array([])
    
    # normalize and scale each image
    normalized_images = []
    for img in img_list:
        normalized = normalize_and_scale(img)
        normalized_images.append(normalized)
    
    # get dimensions
    heights = [img.shape[0] for img in normalized_images]
    widths = [img.shape[1] for img in normalized_images]
    
    # total width is sum of all widths
    total_width = sum(widths)
    max_height = max(heights)
    
    # create combined image
    combined = np.zeros((max_height, total_width), dtype=np.uint8)
    
    # place images side by side
    x_offset = 0
    for img in normalized_images:
        h, w = img.shape
        combined[:h, x_offset:x_offset+w] = img
        x_offset += w
    
    return combined


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    h, w = image.shape
    
    # upsample by inserting zeros between pixels
    upsampled = np.zeros((h*2, w*2), dtype=image.dtype)
    upsampled[::2, ::2] = image * 4  # multiply by 4 to compensate for zero insertion
    
    # 5-tap separable filter for expand operation
    # [1, 4, 6, 4, 1] / 16 (same as reduce but used differently)
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    
    # apply separable convolution
    # first convolve horizontally
    temp = cv2.filter2D(upsampled, -1, kernel_1d.reshape(1, -1))
    # then convolve vertically
    expanded = cv2.filter2D(temp, -1, kernel_1d.reshape(-1, 1))
    
    return expanded


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = []
    
    # for each level except the last one
    for i in range(len(g_pyr) - 1):
        # expand the next level to match current level size
        expanded = expand_image(g_pyr[i + 1])
        
        # crop expanded image to match current level size
        current_h, current_w = g_pyr[i].shape
        expanded = expanded[:current_h, :current_w]
        
        # laplacian = gaussian - expanded_gaussian
        laplacian = g_pyr[i] - expanded
        l_pyr.append(laplacian)
    
    # last level is just the smallest gaussian
    l_pyr.append(g_pyr[-1])
    
    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    h, w = image.shape
    
    # create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # add displacement to coordinates
    map_x = x_coords + U
    map_y = y_coords + V
    
    # convert to float32 for cv2.remap
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    # warp image using cv2.remap
    warped = cv2.remap(image, map_x, map_y, interpolation, borderMode=border_mode)
    
    return warped


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    # build gaussian pyramids for both images
    pyr_a = gaussian_pyramid(img_a, levels)
    pyr_b = gaussian_pyramid(img_b, levels)
    
    # start with zero flow at the coarsest level
    u = np.zeros_like(pyr_a[levels - 1])
    v = np.zeros_like(pyr_a[levels - 1])
    
    # iterate from coarsest to finest level
    for level in range(levels - 1, -1, -1):
        # get images at current level
        img_a_level = pyr_a[level]
        img_b_level = pyr_b[level]
        
        # if not the coarsest level, scale up flow from previous level
        if level < levels - 1:
            # expand flow by factor of 2 and adjust for current level size
            u = expand_image(u) * 2
            v = expand_image(v) * 2
            
            # crop to match current level size
            h, w = img_a_level.shape
            if u.shape[0] > h or u.shape[1] > w:
                u = u[:h, :w]
            if v.shape[0] > h or v.shape[1] > w:
                v = v[:h, :w]
        
        # warp image b using current flow estimate
        img_b_warped = warp(img_b_level, u, v, interpolation, border_mode)
        
        # compute optical flow between warped image and reference
        du, dv = optic_flow_lk(img_a_level, img_b_warped, k_size, k_type, sigma)
        
        # update flow estimate
        u = u + du
        v = v + dv
    
    return u, v

def classify_video(images):
    """Classifies a set of frames as either
        - int(1) == "Running"
        - int(2) == "Walking"
        - int(3) == "Clapping"
    Args:
        images list(numpy.array): greyscale floating-point frames of a video
    Returns:
        int:  Class of video
    """

    raise NotImplementedError
