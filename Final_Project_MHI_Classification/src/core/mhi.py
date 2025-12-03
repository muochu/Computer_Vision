"""Motion History Image generation."""
import numpy as np
import cv2


def frame_difference(frame1, frame2, threshold=30):
    """Compute binary motion mask from frame difference."""
    diff = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
    return (diff >= threshold).astype(np.uint8) * 255


def morphological_open(binary_image, kernel_size=3):
    """Apply morphological opening for noise removal."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


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

