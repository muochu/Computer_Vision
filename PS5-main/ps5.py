"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np

from ps5_utils import run_kalman_filter, run_particle_filter

# I/O directories
input_dir = "input"
output_dir = "output"



# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # Set up the state vector with position and velocity
        self.state = np.array([init_x, init_y, 0., 0.], dtype=np.float64)
        
        # Store the noise matrices
        self.Q = Q
        self.R = R
        
        # Start with high uncertainty in our state estimate
        self.P = np.eye(4) * 100.0
        
        # This matrix describes how the state evolves over time
        # For constant velocity model: position += velocity * dt
        dt = 1.0
        self.D = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # This matrix tells us what we can actually measure
        # We only get x,y positions from our sensor, not velocities
        self.M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

    def predict(self):
        """Predict the next state based on our motion model"""
        # Update state using our dynamics model
        self.state = self.D @ self.state
        
        # Update our uncertainty about the state
        self.P = self.D @ self.P @ self.D.T + self.Q

    def correct(self, meas_x, meas_y):
        """Update our estimate using the new measurement"""
        # Put the measurement into a vector
        z = np.array([meas_x, meas_y], dtype=np.float64)
        
        # What measurement do we expect based on our current state?
        predicted_measurement = self.M @ self.state
        
        # How different is the actual measurement from what we expected?
        innovation = z - predicted_measurement
        
        # Calculate how much we trust this measurement
        S = self.M @ self.P @ self.M.T + self.R
        
        # Compute the Kalman gain - this tells us how much to trust the measurement
        K = self.P @ self.M.T @ np.linalg.inv(S)
        
        # Update our state estimate
        self.state = self.state + K @ innovation
        
        # Update our uncertainty
        I = np.eye(4)
        self.P = (I - K @ self.M) @ self.P

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        
        # Initialize particles around the template location
        template_x = self.template_rect['x'] + self.template_rect['w'] // 2
        template_y = self.template_rect['y'] + self.template_rect['h'] // 2
        
        # Create particles with some initial spread around the template
        self.particles = np.zeros((self.num_particles, 2))
        self.particles[:, 0] = template_x + np.random.normal(0, 20, self.num_particles)
        self.particles[:, 1] = template_y + np.random.normal(0, 20, self.num_particles)
        
        # Make sure particles stay within image bounds
        h, w = frame.shape[:2]
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w-1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h-1)
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Calculate MSE between template and frame cutout"""
        # Make sure both images are the same size
        if template.shape != frame_cutout.shape:
            return float('inf')
        
        # Convert to grayscale if needed
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
            
        if len(frame_cutout.shape) == 3:
            frame_gray = cv2.cvtColor(frame_cutout, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_cutout
        
        # Calculate MSE
        mse = np.mean((template_gray.astype(np.float32) - frame_gray.astype(np.float32)) ** 2)
        
        # Convert to similarity using Gaussian
        similarity = np.exp(-mse / (2 * self.sigma_exp ** 2))
        
        return similarity

    def resample_particles(self):
        """Resample particles based on their weights"""
        # Normalize weights to make sure they sum to 1
        weights_normalized = self.weights / np.sum(self.weights)
        
        # Use multinomial sampling to select particles based on weights
        indices = np.random.multinomial(self.num_particles, weights_normalized)
        
        # Create new particle array
        new_particles = np.zeros_like(self.particles)
        particle_idx = 0
        
        for i, count in enumerate(indices):
            for _ in range(count):
                if particle_idx < self.num_particles:
                    new_particles[particle_idx] = self.particles[i]
                    particle_idx += 1
        
        return new_particles

    def process(self, frame):
        """Process a new frame and update particle filter state"""
        h, w = frame.shape[:2]
        template_h, template_w = self.template.shape[:2]
        
        # Add noise to particles (dynamics step)
        noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
        
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        
        # Keep particles within image bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], template_w//2, w - template_w//2)
        self.particles[:, 1] = np.clip(self.particles[:, 1], template_h//2, h - template_h//2)
        
        # Calculate weights for each particle
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            
            # Extract patch from frame
            x1 = max(0, x - template_w//2)
            y1 = max(0, y - template_h//2)
            x2 = min(w, x + template_w//2)
            y2 = min(h, y + template_h//2)
            
            # Handle edge cases where patch goes outside image
            if x2 - x1 != template_w or y2 - y1 != template_h:
                self.weights[i] = 0.0
                continue
                
            frame_patch = frame[y1:y2, x1:x2]
            
            # Calculate similarity
            similarity = self.get_error_metric(self.template, frame_patch)
            self.weights[i] = similarity
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resample particles
        self.particles = self.resample_particles()
        
        # Reset weights after resampling
        self.weights = np.ones(self.num_particles) / self.num_particles

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Draw particles as colored dots
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            cv2.circle(frame_in, (x, y), 2, (0, 255, 0), -1)
        
        # Draw tracking rectangle around weighted mean
        template_h, template_w = self.template.shape[:2]
        x1 = int(x_weighted_mean - template_w//2)
        y1 = int(y_weighted_mean - template_h//2)
        x2 = int(x_weighted_mean + template_w//2)
        y2 = int(y_weighted_mean + template_h//2)
        cv2.rectangle(frame_in, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Calculate weighted standard deviation for uncertainty circle
        distances = []
        for i in range(self.num_particles):
            dist = np.sqrt((self.particles[i, 0] - x_weighted_mean)**2 + 
                          (self.particles[i, 1] - y_weighted_mean)**2)
            distances.append(dist * self.weights[i])
        
        uncertainty_radius = int(np.sum(distances))
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), 
                  uncertainty_radius, (0, 0, 255), 1)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha', 0.05)  # IIR filter parameter for template updating

    def process(self, frame):
        """Process frame and update template using IIR filter"""
        # First do the normal particle filter processing
        super(AppearanceModelPF, self).process(frame)
        
        # Now update the template using the best estimate
        # Get the weighted mean position
        x_mean = np.sum(self.particles[:, 0] * self.weights)
        y_mean = np.sum(self.particles[:, 1] * self.weights)
        
        # Extract the best patch from current frame
        template_h, template_w = self.template.shape[:2]
        h, w = frame.shape[:2]
        
        x1 = max(0, int(x_mean - template_w//2))
        y1 = max(0, int(y_mean - template_h//2))
        x2 = min(w, int(x_mean + template_w//2))
        y2 = min(h, int(y_mean + template_h//2))
        
        # Make sure we have the right size patch
        if x2 - x1 == template_w and y2 - y1 == template_h:
            best_patch = frame[y1:y2, x1:x2]
            
            # Update template using IIR filter: Template(t) = α * Best(t) + (1-α) * Template(t-1)
            self.template = (self.alpha * best_patch.astype(np.float32) + 
                           (1 - self.alpha) * self.template.astype(np.float32)).astype(np.uint8)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        
        # Override particles to include scale dimension [x, y, scale]
        template_x = self.template_rect['x'] + self.template_rect['w'] // 2
        template_y = self.template_rect['y'] + self.template_rect['h'] // 2
        
        # Initialize particles with position and scale
        self.particles = np.zeros((self.num_particles, 3))  # [x, y, scale]
        self.particles[:, 0] = template_x + np.random.normal(0, 20, self.num_particles)
        self.particles[:, 1] = template_y + np.random.normal(0, 20, self.num_particles)
        self.particles[:, 2] = 1.0 + np.random.normal(0, 0.1, self.num_particles)  # scale around 1.0
        
        # Keep particles within bounds
        h, w = frame.shape[:2]
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w-1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h-1)
        self.particles[:, 2] = np.clip(self.particles[:, 2], 0.5, 2.0)  # scale between 0.5 and 2.0
    
    def resample_particles(self):
        """Resample particles for multi-dimensional state"""
        # Normalize weights to make sure they sum to 1
        weights_normalized = self.weights / np.sum(self.weights)
        
        # Use multinomial sampling to select particles based on weights
        indices = np.random.multinomial(self.num_particles, weights_normalized)
        
        # Create new particle array
        new_particles = np.zeros_like(self.particles)
        particle_idx = 0
        
        for i, count in enumerate(indices):
            for _ in range(count):
                if particle_idx < self.num_particles:
                    new_particles[particle_idx] = self.particles[i]
                    particle_idx += 1
        
        return new_particles

    def process(self, frame):
        """Process frame with multi-dimensional state including scale"""
        h, w = frame.shape[:2]
        template_h, template_w = self.template.shape[:2]
        
        # Add noise to particles (dynamics step) - now includes scale
        noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_scale = np.random.normal(0, 0.05, self.num_particles)  # scale noise
        
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        self.particles[:, 2] += noise_scale
        
        # Keep particles within bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w-1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h-1)
        self.particles[:, 2] = np.clip(self.particles[:, 2], 0.5, 2.0)
        
        # Calculate weights for each particle
        for i in range(self.num_particles):
            x, y, scale = self.particles[i, 0], self.particles[i, 1], self.particles[i, 2]
            
            # Resize template according to particle scale
            new_h = int(template_h * scale)
            new_w = int(template_w * scale)
            
            if new_h < 5 or new_w < 5 or new_h > h or new_w > w:
                self.weights[i] = 0.0
                continue
                
            # Resize template
            resized_template = cv2.resize(self.template, (new_w, new_h))
            
            # Extract patch from frame
            x1 = max(0, int(x - new_w//2))
            y1 = max(0, int(y - new_h//2))
            x2 = min(w, int(x + new_w//2))
            y2 = min(h, int(y + new_h//2))
            
            if x2 - x1 != new_w or y2 - y1 != new_h:
                self.weights[i] = 0.0
                continue
                
            frame_patch = frame[y1:y2, x1:x2]
            
            # Calculate similarity
            similarity = self.get_error_metric(resized_template, frame_patch)
            self.weights[i] = similarity
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resample particles
        self.particles = self.resample_particles()
        
        # Reset weights after resampling
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Update template using best estimate (inherited from AppearanceModelPF)
        x_mean = np.sum(self.particles[:, 0] * self.weights)
        y_mean = np.sum(self.particles[:, 1] * self.weights)
        scale_mean = np.sum(self.particles[:, 2] * self.weights)
        
        # Extract best patch with current scale
        new_h = int(template_h * scale_mean)
        new_w = int(template_w * scale_mean)
        
        if new_h > 0 and new_w > 0 and new_h < h and new_w < w:
            x1 = max(0, int(x_mean - new_w//2))
            y1 = max(0, int(y_mean - new_h//2))
            x2 = min(w, int(x_mean + new_w//2))
            y2 = min(h, int(y_mean + new_h//2))
            
            if x2 - x1 == new_w and y2 - y1 == new_h:
                best_patch = frame[y1:y2, x1:x2]
                # Resize back to original template size
                resized_patch = cv2.resize(best_patch, (template_w, template_h))
                
                # Update template using IIR filter
                self.template = (self.alpha * resized_patch.astype(np.float32) + 
                               (1 - self.alpha) * self.template.astype(np.float32)).astype(np.uint8)


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 100  # Number of particles for tracking
    sigma_mse = 10.0  # Sigma for measurement similarity
    sigma_dyn = 8.0  # Sigma for particle dynamics (movement noise)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # More particles for noisy video
    sigma_mse = 15.0  # Higher sigma for noisy measurements
    sigma_dyn = 12.0  # Higher dynamics for noisy video

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 300  # More particles for appearance changes
    sigma_mse = 12.0  # Sigma for measurement similarity
    sigma_dyn = 10.0  # Sigma for particle dynamics
    alpha = 0.05  # IIR filter parameter for template updating

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 500  # Many particles for complex occlusions
    sigma_md = 15.0  # Higher sigma for multi-dimensional tracking
    sigma_dyn = 15.0  # Higher dynamics for occlusions and scale changes

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out
