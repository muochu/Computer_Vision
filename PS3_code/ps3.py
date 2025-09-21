"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

import cv2
import numpy as np
from typing import Tuple


class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        # path = r'D:\GaTech\TA - CV\ps05\ps05\ps5-1-b-1.png'
        #path1 = r'1a_notredame.jpg'
        #path2 = r'1b_notredame.jpg'


        #path1 = self.path1
        #path2 = self.path2

        # path1 = r'crop1.jpg'
        # path2 = r'crop2.jpg'

        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 1)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """

    raise NotImplementedError
    return 0.0


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    
    # Return corners in order: top-left, bottom-left, top-right, bottom-right
    corners = [
        (0, 0),                    # top-left
        (0, height - 1),           # bottom-left
        (width - 1, 0),            # top-right
        (width - 1, height - 1)    # bottom-right
    ]
    
    return corners


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    
    # Try circle detection first for larger images (wall images)
    if template is not None and len(image.shape) == 3:
        h, w = image.shape[:2]
        if h > 500 and w > 1000:  # Large image, try circle detection first
            circle_markers = _find_markers_circles(image)
            if len(circle_markers) >= 4:
                markers = _sort_markers(circle_markers)
                return markers[:4]
    
    # Start with template matching approach since we have template
    if template is None:
        # Fallback to circle detection if no template
        return _find_markers_circles(image)
    
    # Convert to grayscale for template matching
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    
    # Apply slight blur to handle noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    template_gray = cv2.GaussianBlur(template_gray, (3, 3), 0)
    
    # Template matching with multiple scales and methods
    markers = []
    # For small templates, try larger scale ranges
    if template_gray.shape[0] < 50:  # Small template
        scales = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    else:  # Large template
        scales = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]  # Try different methods
    
    for method in methods:
        for scale in scales:
            if scale != 1.0:
                h, w = template_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 0 and new_w > 0:
                    scaled_template = cv2.resize(template_gray, (new_w, new_h))
                else:
                    continue
            else:
                scaled_template = template_gray
            
            # Skip if template is larger than image
            if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                continue
                
            # Perform template matching
            result = cv2.matchTemplate(gray, scaled_template, method)
            
            # Find peaks in the result - adjust threshold based on method
            if method == cv2.TM_CCOEFF_NORMED:
                threshold = 0.4
            else:  # TM_CCORR_NORMED
                threshold = 0.7
                
            # Find local maxima to avoid clustered matches
            h, w = result.shape
            local_maxima = np.zeros_like(result)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if (result[i, j] > result[i-1, j] and 
                        result[i, j] > result[i+1, j] and
                        result[i, j] > result[i, j-1] and
                        result[i, j] > result[i, j+1] and
                        result[i, j] >= threshold):
                        local_maxima[i, j] = result[i, j]
            
            locations = np.where(local_maxima > 0)
            
            # Sort by confidence and take only the best matches
            if len(locations[0]) > 0:
                confidences = local_maxima[locations]
                sorted_indices = np.argsort(confidences)[::-1]  # Sort descending
                
                # Take only the best matches
                for idx in sorted_indices[:10]:  # Take top 10 matches
                    pt = (locations[1][idx], locations[0][idx])
                    # Calculate center of marker - use integer coordinates
                    center_x = int(pt[0] + scaled_template.shape[1] // 2)
                    center_y = int(pt[1] + scaled_template.shape[0] // 2)
                    markers.append((center_x, center_y))
        
        # If we found markers with this method, don't try the next method
        if len(markers) > 0:
            break
    
    # Remove duplicates and filter by distance
    markers = _remove_duplicate_markers(markers)
    
    # If we don't have exactly 4 markers, try circle detection as backup
    if len(markers) != 4:
        circle_markers = _find_markers_circles(image)
        if len(circle_markers) >= 4:
            markers = circle_markers
    
    # Sort markers into the required order: [top-left, bottom-left, top-right, bottom-right]
    if len(markers) >= 4:
        markers = _sort_markers(markers)
        return markers[:4]  # Return only the first 4
    
    return markers

def _find_markers_circles(image):
    """Fallback method using circle detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use more restrictive parameters to avoid false positives
    markers = []
    
    # Try with very precise parameters
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
                              param1=90, param2=50, minRadius=10, maxRadius=60)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            markers.append((int(x), int(y)))
    
    # If we don't have enough markers, try slightly more sensitive parameters
    if len(markers) < 4:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 25,
                                  param1=60, param2=35, minRadius=8, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                markers.append((int(x), int(y)))
    
    # Remove duplicates and filter out edge circles
    markers = _remove_duplicate_markers(markers, min_distance=50)
    markers = _filter_edge_circles(markers, image.shape[:2])
    
    return markers

def _filter_edge_circles(markers, image_shape):
    """Filter out circles that are too close to image edges"""
    h, w = image_shape
    filtered = []
    
    for marker in markers:
        x, y = marker
        # Keep circles that are at least 20 pixels from edges
        if (x > 20 and x < w - 20 and y > 20 and y < h - 20):
            filtered.append(marker)
    
    return filtered

def _find_markers_hybrid(image, template):
    """Hybrid approach: use template matching to find regions, then circle detection to refine"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    
    # Apply slight blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    template_gray = cv2.GaussianBlur(template_gray, (3, 3), 0)
    
    # Use template matching to find approximate regions
    scales = [1.0, 2.0, 3.0, 4.0, 5.0]
    candidate_regions = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = template_gray.shape
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 0 and new_w > 0:
                scaled_template = cv2.resize(template_gray, (new_w, new_h))
            else:
                continue
        else:
            scaled_template = template_gray
        
        if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
            continue
            
        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        # Find local maxima
        h, w = result.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if (result[i, j] > result[i-1, j] and 
                    result[i, j] > result[i+1, j] and
                    result[i, j] > result[i, j-1] and
                    result[i, j] > result[i, j+1] and
                    result[i, j] >= 0.4):
                    # Found a candidate region
                    center_x = j + scaled_template.shape[1] // 2
                    center_y = i + scaled_template.shape[0] // 2
                    candidate_regions.append((center_x, center_y, scaled_template.shape[0]))
    
    # Now use circle detection to find exact centers near candidate regions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15,
                              param1=30, param2=20, minRadius=5, maxRadius=150)
    
    markers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Check if this circle is near any candidate region
            for cx, cy, size in candidate_regions:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < size:  # Circle is within the template region
                    markers.append((int(x), int(y)))
                    break  # Found a match, don't check other regions
    
    # If we don't have enough markers, just use circle detection
    if len(markers) < 4:
        return _find_markers_circles(image)
    
    return markers

def _remove_duplicate_markers(markers, min_distance=30):
    """Remove markers that are too close to each other"""
    if len(markers) <= 1:
        return markers
    
    # Sort by x coordinate first
    markers = sorted(markers, key=lambda x: x[0])
    filtered = [markers[0]]
    
    for marker in markers[1:]:
        is_duplicate = False
        for existing in filtered:
            dist = np.sqrt((marker[0] - existing[0])**2 + (marker[1] - existing[1])**2)
            if dist < min_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(marker)
    
    return filtered

def _sort_markers(markers):
    """Sort markers into [top-left, bottom-left, top-right, bottom-right] order"""
    if len(markers) < 4:
        return markers
    
    # Convert to numpy array for easier manipulation
    markers = np.array(markers)
    
    # Find center point
    center = np.mean(markers, axis=0)
    
    # Split into left and right based on x coordinate relative to center
    left_markers = markers[markers[:, 0] < center[0]]
    right_markers = markers[markers[:, 0] >= center[0]]
    
    # Sort left markers by y coordinate (top to bottom)
    left_sorted = left_markers[left_markers[:, 1].argsort()]
    # Sort right markers by y coordinate (top to bottom)  
    right_sorted = right_markers[right_markers[:, 1].argsort()]
    
    # Combine in required order: [top-left, bottom-left, top-right, bottom-right]
    result = []
    if len(left_sorted) >= 2:
        result.extend([(int(left_sorted[0][0]), int(left_sorted[0][1])), 
                      (int(left_sorted[1][0]), int(left_sorted[1][1]))])
    if len(right_sorted) >= 2:
        result.extend([(int(right_sorted[0][0]), int(right_sorted[0][1])), 
                      (int(right_sorted[1][0]), int(right_sorted[1][1]))])
    
    return result


def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    
    # Create a copy of the input image to avoid modifying the original
    out_image = image.copy()
    
    # Validate that we have exactly 4 markers
    if len(markers) != 4:
        return out_image
    
    # Convert markers to numpy array for easier handling
    markers = np.array(markers, dtype=np.int32)
    
    # Draw lines connecting the four markers to form a box
    # The markers should be in order top left, bottom left, top right, bottom right
    # Connect top left to top right to bottom right to bottom left to top left
    
    # Top edge from top left to top right
    cv2.line(out_image, tuple(markers[0]), tuple(markers[2]), (0, 0, 255), thickness)
    
    # Right edge from top right to bottom right  
    cv2.line(out_image, tuple(markers[2]), tuple(markers[3]), (0, 0, 255), thickness)
    
    # Bottom edge from bottom right to bottom left
    cv2.line(out_image, tuple(markers[3]), tuple(markers[1]), (0, 0, 255), thickness)
    
    # Left edge from bottom left to top left
    cv2.line(out_image, tuple(markers[1]), tuple(markers[0]), (0, 0, 255), thickness)
    
    return out_image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    You should have used your find_markers method to find the corners and then
    compute the homography matrix prior to using this function.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """

    out_image = imageB.copy()
    
    # Get dimensions of both images
    hB, wB = imageB.shape[:2]
    hA, wA = imageA.shape[:2]
    
    # Create coordinate grids for the destination image
    y_coords, x_coords = np.mgrid[0:hB, 0:wB]
    
    # Convert to homogeneous coordinates
    ones = np.ones((hB, wB))
    dest_coords = np.stack([x_coords, y_coords, ones], axis=2)
    
    # Apply inverse homography to get source coordinates
    # We need to solve for source coordinates: H * src = dest
    # So: src = H^(-1) * dest
    inv_homography = np.linalg.inv(homography)
    
    # Transform all destination coordinates to source coordinates
    source_coords = np.dot(dest_coords, inv_homography.T)
    
    # Normalize homogeneous coordinates
    x_src = source_coords[:, :, 0] / source_coords[:, :, 2]
    y_src = source_coords[:, :, 1] / source_coords[:, :, 2]
    
    # Create masks for valid coordinates (within source image bounds)
    valid_mask = ((x_src >= 0) & (x_src < wA) & 
                  (y_src >= 0) & (y_src < hA))
    
    # Use bilinear interpolation to sample from imageA
    # Create coordinate maps for cv2.remap
    map_x = x_src.astype(np.float32)
    map_y = y_src.astype(np.float32)
    
    # Remap imageA to the destination coordinates
    warped_A = cv2.remap(imageA, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Only copy pixels where the mask is valid
    out_image[valid_mask] = warped_A[valid_mask]
    
    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """
    
    # Convert to numpy arrays
    src = np.array(srcPoints, dtype=np.float64)
    dst = np.array(dstPoints, dtype=np.float64)
    
    # Build the coefficient matrix A for the system Ah = 0
    # Each point pair gives us 2 equations, so we need 8 equations for 8 unknowns
    A = []
    
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        
        # First equation for this point pair
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        # Second equation for this point pair  
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    
    A = np.array(A, dtype=np.float64)
    
    # Solve using SVD to find the null space
    _, _, V = np.linalg.svd(A)
    
    # The solution is the last column of V
    h = V[-1, :]
    
    # Reshape to 3x3 matrix and normalize so that H[2,2] = 1
    homography = h.reshape(3, 3)
    homography = homography / homography[2, 2]
    
    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)

    # Read frames one by one until there are no more frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame

    # Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)



    def gradients(self, image_bw):
        '''Use convolution with Sobel filters to calculate the image gradient at each
            pixel location
            Input -
            :param image_bw: A numpy array of shape (M,N) containing the grayscale image
            Output -
            :return Ix: Array of shape (M,N) representing partial derivatives of image
                    in x-direction
            :return Iy: Array of shape (M,N) representing partial derivative of image
                    in y-direction
        '''

        raise NotImplementedError

        return Ix, Iy



    def second_moments(self, image_bw, ksize=7, sigma=10):
        """ Compute second moments from image.
            Compute image gradients, Ix and Iy at each pixel, the mixed derivatives and then the
            second moments (sx2, sxsy, sy2) at each pixel,using convolution with a Gaussian filter. You may call the
            previously written function for obtaining the gradients here.
            Input -
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of Gaussian filter
            Output -
            :return sx2: np array of shape (M,N) containing the second moment in x direction
            :return sy2: np array of shape (M,N) containing the second moment in y direction
            :return sxsy: np array of shape (M,N) containing the second moment in the x then the
                    y direction
        """

        sx2, sy2, sxsy = None, None, None

        raise NotImplementedError

        return sx2, sy2, sxsy


    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)
            R = det(M) - alpha * (trace(M))^2
            where M = [S_xx S_xy;
                       S_xy  S_yy],
                  S_xx = Gk * I_xx
                  S_yy = Gk * I_yy
                  S_xy  = Gk * I_xy,
            and * is a convolutional operation over a Gaussian kernel of size (k, k).
            (You can verify that this is equivalent to taking a (Gaussian) weighted sum
            over the window of size (k, k), see how convolutional operation works here:
                http://cs231n.github.io/convolutional-networks/)
            Ix, Iy are simply image derivatives in x and y directions, respectively.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of gaussian filter
            :param alpha: scalar term in Harris response score
            Output-
            :return R: np array of shape (M,N), indicating the corner score of each pixel.
            """


        raise NotImplementedError

        return R


    def nms_maxpool(self, R, k, ksize):
        """ Get top k interest points that are local maxima over (ksize,ksize)
        neighborhood.
        One simple way to do non-maximum suppression is to simply pick a
        local maximum over some window size (u, v). Note that this would give us all local maxima even when they
        have a really low score compare to other local maxima. It might be useful
        to threshold out low value score before doing the pooling.
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum. Multiply this binary
        image, multiplied with the cornerness response values.
        Args:
            R: np array of shape (M,N) with score response map
            k: number of interest points (take top k by confidence)
            ksize: kernel size of max-pooling operator
        Returns:
            x: np array of shape (k,) containing x-coordinates of interest points
            y: np array of shape (k,) containing y-coordinates of interest points
        """



        raise NotImplementedError

        return x, y


    def harris_corner(self, image_bw, k=100):
        """
            Implement the Harris Corner detector. You can call harris_response_map(), nms_maxpool() functions here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param k: maximum number of interest points to retrieve
            Output-
            :return x: np array of shape (p,) containing x-coordinates of interest points
            :return y: np array of shape (p,) containing y-coordinates of interest points
            """

        raise NotImplementedError

        return x1, y1





class Image_Mosaic(object):

    def __init__(self):
        pass

    def image_warp_inv(self, im_src, im_dst, H):
        '''
        Input -
        :param im_src: Image 1
        :param im_dst: Image 2
        :param H: numpy ndarray - 3x3 homography matrix
        Output -
        :return: Inverse Warped Resulting Image
        '''


        raise NotImplementedError

        return warped_img


    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''


        raise NotImplementedError

        return im_mos_out




