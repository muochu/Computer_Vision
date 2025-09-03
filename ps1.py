import numpy as np
import cv2

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image)
    # In OpenCV, BGR format: channel 2 is red (B=0, G=1, R=2)
    return temp_image[:, :, 2]


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image)
    # In OpenCV, BGR format: channel 1 is green (B=0, G=1, R=2)
    return temp_image[:, :, 1]


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image)
    # In OpenCV, BGR format: channel 0 is blue (B=0, G=1, R=2)
    return temp_image[:, :, 0]


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image)
    # In OpenCV BGR format: swap channels 0 (blue) and 1 (green)
    temp_image[:, :, [0, 1]] = temp_image[:, :, [1, 0]]
    return temp_image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    # Make a copy to avoid modifying the original
    temp_dst = np.copy(dst)
    
    # Extract shape dimensions
    copy_height, copy_width = shape
    
    # Calculate source region (center of src)
    src_h, src_w = src.shape
    src_start_h = src_h // 2 - copy_height // 2
    src_start_w = src_w // 2 - copy_width // 2
    src_end_h = src_start_h + copy_height
    src_end_w = src_start_w + copy_width
    
    # Calculate destination region (center of dst)
    dst_h, dst_w = dst.shape
    dst_start_h = dst_h // 2 - copy_height // 2
    dst_start_w = dst_w // 2 - copy_width // 2
    dst_end_h = dst_start_h + copy_height
    dst_end_w = dst_start_w + copy_width
    
    # Copy the exact center region
    temp_dst[dst_start_h:dst_end_h, dst_start_w:dst_end_w] = src[src_start_h:src_end_h, src_start_w:src_end_w]
    
    return temp_dst


def copy_paste_middle_circle(src, dst, radius):
    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    # Make a copy to avoid modifying the original
    temp_dst = np.copy(dst)
    
    # Get image dimensions
    src_h, src_w = src.shape
    dst_h, dst_w = dst.shape
    
    # Calculate centers
    src_center_h, src_center_w = src_h // 2, src_w // 2
    dst_center_h, dst_center_w = dst_h // 2, dst_w // 2
    
    # Extract the center patch from source (2*radius x 2*radius)
    patch_size = 2 * radius
    src_start_h = max(0, src_center_h - radius)
    src_start_w = max(0, src_center_w - radius)
    src_end_h = min(src_h, src_center_h + radius)
    src_end_w = min(src_w, src_center_w + radius)
    
    # Calculate destination region (center of dst)
    dst_start_h = max(0, dst_center_h - radius)
    dst_start_w = max(0, dst_center_w - radius)
    dst_end_h = min(dst_h, dst_center_h + radius)
    dst_end_w = min(dst_w, dst_center_w + radius)
    
    # Extract the source patch
    src_patch = src[src_start_h:src_end_h, src_start_w:src_end_w]
    
    # Create circular mask for the patch
    patch_h, patch_w = src_patch.shape
    
    # Create coordinate grids
    y_coords = np.arange(patch_h).reshape(-1, 1)
    x_coords = np.arange(patch_w).reshape(1, -1)
    
    # Calculate distances from center
    center_y = patch_h // 2
    center_x = patch_w // 2
    
    # Create circular mask
    y_dist = (y_coords - center_y) ** 2
    x_dist = (x_coords - center_x) ** 2
    distances = y_dist + x_dist
    circle_mask = distances <= (radius ** 2)
    
    # Apply mask
    dst_patch = temp_dst[dst_start_h:dst_end_h, dst_start_w:dst_end_w]
    dst_patch[circle_mask] = src_patch[circle_mask]
    
    return temp_dst


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image)
    
    # Calculate statistics using numpy functions and convert to float
    min_val = float(np.min(temp_image))
    max_val = float(np.max(temp_image))
    mean_val = float(np.mean(temp_image))
    stddev_val = float(np.std(temp_image))
    
    return (min_val, max_val, mean_val, stddev_val)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image).astype(np.float64)
    
    # Calculate mean and standard deviation
    mean_val = np.mean(temp_image)
    std_val = np.std(temp_image)
    
    # Center around mean and normalize to desired standard deviation
    if std_val != 0:
        normalized = (temp_image - mean_val) / std_val * scale + mean_val
    else:
        normalized = temp_image
    
    return normalized


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image)
    
    if shift == 0:
        return temp_image
    
    # Get image dimensions
    height, width = temp_image.shape
    
    # Create shifted image using border replication
    shifted = np.zeros_like(temp_image)
    
    # Copy the shifted portion
    if shift < width:
        shifted[:, :width-shift] = temp_image[:, shift:]
        # Fill the right edge with replicated values
        shifted[:, width-shift:] = temp_image[:, -1:]
    else:
        # If shift is larger than width, fill with replicated values
        shifted[:, :] = temp_image[:, -1:]
    
    return shifted


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    # Make a copy to avoid modifying the original
    temp_img1 = np.copy(img1).astype(np.int16)
    temp_img2 = np.copy(img2).astype(np.int16)
    
    # Calculate difference (img1 - img2) - may be negative
    diff = temp_img1 - temp_img2
    
    # Use min-max normalization to scale to [0, 255]
    min_diff = np.min(diff)
    max_diff = np.max(diff)
    
    if max_diff - min_diff != 0:
        normalized_diff = ((diff - min_diff) / (max_diff - min_diff)) * 255
    else:
        normalized_diff = np.zeros_like(diff)
    
    # Return as float64 (autograder expects this)
    result = normalized_diff.astype(np.float64)
    
    return result


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    # Make a copy to avoid modifying the original
    temp_image = np.copy(image).astype(np.float64)
    
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(0, sigma, temp_image.shape[:2])
    
    # Add noise to the specified channel
    temp_image[:, :, channel] += noise
    
    return temp_image


def build_hybrid_image(image1, image2, cutoff_frequency):
    """ 
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar
    
    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """

    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)
    
    low_frequencies = cv2.filter2D(image1,-1,filter)

    high_frequencies = image2 - cv2.filter2D(image2,-1,filter)
    
    # Combine low frequencies from image1 with high frequencies from image2
    hybrid_image = low_frequencies + high_frequencies
    
    return hybrid_image


def vis_hybrid_image(hybrid_image):
    """ 
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """


    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
      # add padding
      output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                          dtype=np.float32)))

      # downsample image
      cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

      # pad the top to append to the output
      pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                     num_colors), dtype=np.float32)
      tmp = np.vstack((pad, cur_image))
      output = np.hstack((output, tmp))

    return output
