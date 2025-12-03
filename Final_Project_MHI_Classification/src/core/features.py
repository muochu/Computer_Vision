"""Feature extraction using Hu moments (from scratch)."""
import numpy as np


def compute_regular_moments(image, i, j):
    """Compute regular moment M_ij using vectorized operations."""
    height, width = image.shape
    y_coords, x_coords = np.ogrid[:height, :width]
    
    x_power = np.power(x_coords, i)
    y_power = np.power(y_coords, j)
    
    moment = np.sum(x_power * y_power * image)
    return float(moment)


def compute_centroid(image):
    """Compute image centroid."""
    m00 = compute_regular_moments(image, 0, 0)
    if m00 == 0:
        return 0.0, 0.0
    
    m10 = compute_regular_moments(image, 1, 0)
    m01 = compute_regular_moments(image, 0, 1)
    
    return m10 / m00, m01 / m00


def compute_central_moments(image, p, q):
    """Compute central moment mu_pq using vectorized operations."""
    x_bar, y_bar = compute_centroid(image)
    height, width = image.shape
    
    y_coords, x_coords = np.ogrid[:height, :width]
    x_centered = x_coords - x_bar
    y_centered = y_coords - y_bar
    
    moment = np.sum(np.power(x_centered, p) * np.power(y_centered, q) * image)
    return float(moment)


def compute_scale_invariant_moments(image, p, q):
    """Compute scale-invariant moment nu_pq."""
    mu_pq = compute_central_moments(image, p, q)
    mu_00 = compute_central_moments(image, 0, 0)
    
    if mu_00 == 0:
        return 0.0
    
    gamma = 1.0 + (p + q) / 2.0
    return mu_pq / (mu_00 ** gamma)


def extract_features(mhi, use_scaled=True):
    """Extract 8-dimensional feature vector from MHI."""
    if mhi is None or mhi.sum() == 0:
        return np.zeros(8, dtype=np.float32)
    
    moment_pairs = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (2, 2)]
    
    if use_scaled:
        moments = [compute_scale_invariant_moments(mhi, p, q) for p, q in moment_pairs]
    else:
        moments = [compute_central_moments(mhi, p, q) for p, q in moment_pairs]
    
    return np.array(moments, dtype=np.float32)

