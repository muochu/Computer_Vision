"""Motion History Image generation."""
import numpy as np
import cv2


def frame_difference(frame1, frame2, threshold=30):
    """Compute binary motion mask from frame difference."""
    diff = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
    return (diff >= threshold).astype(np.uint8) * 255


def morphological_erode(binary_image, kernel_size=3):
    """Apply morphological erosion (from scratch)."""
    height, width = binary_image.shape
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    k_half = kernel_size // 2
    eroded = np.zeros_like(binary_image)
    
    for y in range(k_half, height - k_half):
        for x in range(k_half, width - k_half):
            region = binary_image[y - k_half:y + k_half + 1, 
                                  x - k_half:x + k_half + 1]
            if np.all(region == 255):
                eroded[y, x] = 255
    
    return eroded


def morphological_dilate(binary_image, kernel_size=3):
    """Apply morphological dilation (from scratch)."""
    height, width = binary_image.shape
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    k_half = kernel_size // 2
    dilated = np.zeros_like(binary_image)
    
    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 255:
                y_min = max(0, y - k_half)
                y_max = min(height, y + k_half + 1)
                x_min = max(0, x - k_half)
                x_max = min(width, x + k_half + 1)
                dilated[y_min:y_max, x_min:x_max] = 255
    
    return dilated


def morphological_open(binary_image, kernel_size=3):
    """Apply morphological opening (erosion followed by dilation) from scratch."""
    eroded = morphological_erode(binary_image, kernel_size)
    return morphological_dilate(eroded, kernel_size)


def create_mhi(frames, tau=15, theta=30, use_morphology=True):
    """Generate Motion History Image from frame sequence."""
    if len(frames) < 2:
        return np.zeros_like(frames[0]) if frames else None
    
    height, width = frames[0].shape
    mhi = np.zeros((height, width), dtype=np.float32)
    
    for i in range(1, len(frames)):
        binary = frame_difference(frames[i-1], frames[i], theta)
        
        if use_morphology:
            binary = morphological_open(binary)
        
        mask = (binary > 0).astype(np.float32)
        mhi = mask * tau + (1 - mask) * np.maximum(mhi - 1, 0)
    
    return mhi.astype(np.uint8)

