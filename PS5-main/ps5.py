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
        # Error metric mode: 'gray' (default), 'color', or 'green'
        self.metric_color = kwargs.get('metric_color', 'gray')
        # Initial particle spread (std dev in pixels)
        self.init_std = kwargs.get('init_std', 20)
        # Optional sigma schedule for early frames
        self.sigma_schedule = kwargs.get('sigma_schedule', False)
        # Internal frame index for scheduling
        self._frame_index = 0
        # Momentum and adaptive dynamics
        self.momentum = kwargs.get('momentum', 0.0)
        self.resample_threshold = kwargs.get('resample_threshold', 0.7)
        self.sigma_dyn_base = float(self.sigma_dyn)
        self.sigma_dyn_curr = float(self.sigma_dyn)
        self.vanilla = kwargs.get('vanilla', True)
        
        # Initialize particles around the template location
        template_x = self.template_rect['x'] + self.template_rect['w'] // 2
        template_y = self.template_rect['y'] + self.template_rect['h'] // 2
        
        # Create particles with some initial spread around the template
        self.particles = np.zeros((self.num_particles, 2))
        self.particles[:, 0] = template_x + np.random.normal(0, self.init_std, self.num_particles)
        self.particles[:, 1] = template_y + np.random.normal(0, self.init_std, self.num_particles)
        
        # Make sure particles stay within image bounds
        h, w = frame.shape[:2]
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w-1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h-1)
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        # Track last two mean estimates for momentum
        self.prev_prev_mean = np.array([template_x, template_y], dtype=np.float64)
        self.prev_mean = np.array([template_x, template_y], dtype=np.float64)

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
            # Shapes mismatch should contribute zero likelihood, not inf
            return 0.0
        
        # Select metric channel/mode
        if self.metric_color == 'color' and len(template.shape) == 3 and len(frame_cutout.shape) == 3 and template.shape[2] == frame_cutout.shape[2]:
            t = template.astype(np.float32)
            f = frame_cutout.astype(np.float32)
            mse = np.mean((t - f) ** 2)
        elif self.metric_color == 'green' and len(template.shape) == 3 and len(frame_cutout.shape) == 3:
            t = template[:, :, 1].astype(np.float32)
            f = frame_cutout[:, :, 1].astype(np.float32)
            mse = np.mean((t - f) ** 2)
        else:
            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template
            if len(frame_cutout.shape) == 3:
                frame_gray = cv2.cvtColor(frame_cutout, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame_cutout
            mse = np.mean((template_gray.astype(np.float32) - frame_gray.astype(np.float32)) ** 2)
        
        # Convert to similarity using Gaussian (optionally scheduled sigma)
        if self.sigma_schedule:
            # Start with larger sigma, decay over first ~20 frames
            factor = 1.0 + 2.0 * np.exp(-float(self._frame_index) / 10.0)
            eff_sigma = max(self.sigma_exp, self.sigma_exp * factor)
        else:
            eff_sigma = self.sigma_exp
        similarity = np.exp(-mse / (2 * eff_sigma ** 2))
        # Avoid returning exact zero to prevent weight degeneracy
        if not np.isfinite(similarity) or similarity <= 0.0:
            similarity = 1e-12
        
        return similarity

    def resample_particles(self):
        """Resample particles based on their weights (robust normalization)."""
        sum_w = np.sum(self.weights)
        if not np.isfinite(sum_w) or sum_w <= 0:
            probs = np.ones(self.num_particles) / self.num_particles
        else:
            probs = self.weights / sum_w

        # Systematic/choice resampling to reduce variance
        idxs = np.random.choice(self.num_particles, size=self.num_particles, p=probs)
        new_particles = self.particles[idxs].copy()
        return new_particles

    def process(self, frame):
        """Process a new frame and update particle filter state"""
        h, w = frame.shape[:2]
        template_h, template_w = self.template.shape[:2]
        
        if self.vanilla:
            # Simple PF: random walk, full resample every frame
            noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
            noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
            self.particles[:, 0] += noise_x
            self.particles[:, 1] += noise_y
            self.particles[:, 0] = np.clip(self.particles[:, 0], template_w//2, w - template_w//2)
            self.particles[:, 1] = np.clip(self.particles[:, 1], template_h//2, h - template_h//2)

            for i in range(self.num_particles):
                x, y = float(self.particles[i, 0]), float(self.particles[i, 1])
                x1 = int(round(x - (template_w - 1) / 2.0))
                y1 = int(round(y - (template_h - 1) / 2.0))
                x2 = x1 + template_w
                y2 = y1 + template_h
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    self.weights[i] = 0.0
                    continue
                frame_patch = frame[y1:y2, x1:x2]
                similarity = self.get_error_metric(self.template, frame_patch)
                self.weights[i] = similarity

            self.weights[~np.isfinite(self.weights)] = 0.0
            sum_w = float(np.sum(self.weights))
            if not np.isfinite(sum_w) or sum_w <= 0.0:
                self.weights = np.ones(self.num_particles) / self.num_particles
            else:
                self.weights = self.weights / (sum_w + 1e-12)

            self.particles = self.resample_particles()
            self._frame_index += 1
            return

        # Add momentum (simple constant-velocity prior) and dynamics noise
        velocity = self.prev_mean - self.prev_prev_mean
        if np.all(np.isfinite(velocity)):
            self.particles[:, 0] += self.momentum * velocity[0]
            self.particles[:, 1] += self.momentum * velocity[1]

        noise_x = np.random.normal(0, self.sigma_dyn_curr, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn_curr, self.num_particles)
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        
        # Keep particles within image bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], template_w//2, w - template_w//2)
        self.particles[:, 1] = np.clip(self.particles[:, 1], template_h//2, h - template_h//2)
        
        # Calculate weights for each particle
        for i in range(self.num_particles):
            x, y = float(self.particles[i, 0]), float(self.particles[i, 1])
            
            # Extract patch centered at (x, y) with exact template size (handles odd sizes)
            x1 = int(round(x - (template_w - 1) / 2.0))
            y1 = int(round(y - (template_h - 1) / 2.0))
            x2 = x1 + template_w
            y2 = y1 + template_h
            
            # Bounds check
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                self.weights[i] = 0.0
                continue
            
            frame_patch = frame[y1:y2, x1:x2]
            
            # Calculate similarity
            similarity = self.get_error_metric(self.template, frame_patch)
            self.weights[i] = similarity
        
        # Normalize weights robustly
        self.weights[~np.isfinite(self.weights)] = 0.0
        self.weights = np.clip(self.weights, 0.0, np.inf)
        sum_w = float(np.sum(self.weights))
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = self.weights / (sum_w + 1e-12)
        # Debug info for base PF
        try:
            self.debug_max_weight = float(np.max(self.weights))
            self.debug_neff = float(1.0 / np.sum((self.weights + 1e-12) ** 2))
        except Exception:
            self.debug_max_weight = float('nan')
            self.debug_neff = float('nan')
        
        # Compute effective sample size and resample conditionally
        neff = 1.0 / np.sum((self.weights + 1e-12) ** 2)
        if not np.isfinite(neff):
            neff = 0.0
        if neff < self.resample_threshold * self.num_particles:
            self.particles = self.resample_particles()
            # Post-resample jitter to maintain diversity
            self.particles[:, 0] += np.random.normal(0, 1.0, self.num_particles)
            self.particles[:, 1] += np.random.normal(0, 1.0, self.num_particles)

        # Update momentum history using arithmetic mean of current particle set
        cur_mean = np.array([np.mean(self.particles[:, 0]), np.mean(self.particles[:, 1])], dtype=np.float64)
        if np.all(np.isfinite(cur_mean)):
            self.prev_prev_mean = self.prev_mean
            self.prev_mean = cur_mean

        # Adapt dynamics noise: broaden when distribution is flat, narrow when peaked
        if neff > 0.9 * self.num_particles:
            self.sigma_dyn_curr = min(3.0 * self.sigma_dyn_base, self.sigma_dyn_curr * 1.2)
        elif neff < 0.5 * self.num_particles:
            self.sigma_dyn_curr = max(0.5 * self.sigma_dyn_base, self.sigma_dyn_curr * 0.8)

        # Advance frame index
        self._frame_index += 1

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

        self.alpha = kwargs.get('alpha', 0.05)
        self.update_threshold = kwargs.get('update_threshold', 0.0)
        self.update_warmup = kwargs.get('update_warmup', 0)
        self._frame_index = 0
        # Force baseline behavior for tests
        self.compat_mode = True

    def process(self, frame):
        """Appearance-adaptive PF: update template using MAP patch before resampling."""
        h, w = frame.shape[:2]
        template_h, template_w = self.template.shape[:2]

        # Dynamics: random walk
        self.particles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], template_w//2, w - template_w//2)
        self.particles[:, 1] = np.clip(self.particles[:, 1], template_h//2, h - template_h//2)

        # Measurement: likelihoods
        for i in range(self.num_particles):
            x, y = float(self.particles[i, 0]), float(self.particles[i, 1])
            x1 = int(round(x - (template_w - 1) / 2.0))
            y1 = int(round(y - (template_h - 1) / 2.0))
            x2 = x1 + template_w
            y2 = y1 + template_h
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                self.weights[i] = 0.0
                continue
            frame_patch = frame[y1:y2, x1:x2]
            self.weights[i] = self.get_error_metric(self.template, frame_patch)

        # Normalize weights
        self.weights[~np.isfinite(self.weights)] = 0.0
        s = float(np.sum(self.weights))
        if not np.isfinite(s) or s <= 0.0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = self.weights / (s + 1e-12)

        # MAP estimate (best particle) and template update BEFORE resample
        best_idx = int(np.argmax(self.weights))
        xb, yb = float(self.particles[best_idx, 0]), float(self.particles[best_idx, 1])
        x1 = int(round(xb - (template_w - 1) / 2.0))
        y1 = int(round(yb - (template_h - 1) / 2.0))
        x2 = x1 + template_w
        y2 = y1 + template_h
        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
            best_patch = frame[y1:y2, x1:x2]
            self.template = (
                self.alpha * best_patch.astype(np.float32)
                + (1.0 - self.alpha) * self.template.astype(np.float32)
            ).astype(np.uint8)

        # Resample and reset weights
        self.particles = self.resample_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles


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
        """Resample particles for multi-dimensional state (robust normalization)."""
        sum_w = np.sum(self.weights)
        if not np.isfinite(sum_w) or sum_w <= 0:
            probs = np.ones(self.num_particles) / self.num_particles
        else:
            probs = self.weights / sum_w
        idxs = np.random.choice(self.num_particles, size=self.num_particles, p=probs)
        new_particles = self.particles[idxs].copy()
        return new_particles

    def process(self, frame):
        """Process frame with multi-dimensional state including scale.

        Update the appearance BEFORE resampling using the weighted estimate
        of position and scale for improved robustness under occlusions/scale.
        """
        h, w = frame.shape[:2]
        template_h, template_w = self.template.shape[:2]

        # 1) Dynamics: add Gaussian noise to position and scale
        noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_scale = np.random.normal(0, 0.05, self.num_particles)
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        self.particles[:, 2] += noise_scale

        # Bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)
        self.particles[:, 2] = np.clip(self.particles[:, 2], 0.5, 2.0)

        # 2) Measurement: compute weights per particle with scaled template
        for i in range(self.num_particles):
            x, y, scale = float(self.particles[i, 0]), float(self.particles[i, 1]), float(self.particles[i, 2])
            new_h = int(template_h * scale)
            new_w = int(template_w * scale)

            if new_h < 5 or new_w < 5 or new_h > h or new_w > w:
                self.weights[i] = 0.0
                continue

            resized_template = cv2.resize(self.template, (new_w, new_h))

            x1 = int(round(x - (new_w - 1) / 2.0))
            y1 = int(round(y - (new_h - 1) / 2.0))
            x2 = x1 + new_w
            y2 = y1 + new_h

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                self.weights[i] = 0.0
                continue

            frame_patch = frame[y1:y2, x1:x2]
            similarity = self.get_error_metric(resized_template, frame_patch)
            self.weights[i] = similarity

        self.weights[~np.isfinite(self.weights)] = 0.0
        sum_w = np.sum(self.weights)
        if not np.isfinite(sum_w) or sum_w <= 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = self.weights / sum_w
        # Debug info
        try:
            self.debug_max_weight = float(np.max(self.weights))
            self.debug_neff = float(1.0 / np.sum((self.weights + 1e-12) ** 2))
        except Exception:
            self.debug_max_weight = float('nan')
            self.debug_neff = float('nan')

        # 3) Weighted estimate (before resample)
        x_mean = float(np.sum(self.particles[:, 0] * self.weights))
        y_mean = float(np.sum(self.particles[:, 1] * self.weights))
        scale_mean = float(np.sum(self.particles[:, 2] * self.weights))

        # 4) Update template using weighted mean estimate (resize back to original size)
        updated_template = 0
        if self._frame_index >= self.update_warmup and np.max(self.weights) > 1e-6:
            new_h = int(template_h * scale_mean)
            new_w = int(template_w * scale_mean)
            if new_h >= 5 and new_w >= 5 and new_h < h and new_w < w:
                x1 = int(round(x_mean - (new_w - 1) / 2.0))
                y1 = int(round(y_mean - (new_h - 1) / 2.0))
                x2 = x1 + new_w
                y2 = y1 + new_h
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    best_patch = frame[y1:y2, x1:x2]
                    resized_patch = cv2.resize(best_patch, (template_w, template_h))
                    # Guarded update based on similarity
                    sim = self.get_error_metric(self.template, resized_patch)
                    if sim >= self.update_threshold:
                        self.template = (
                            self.alpha * resized_patch.astype(np.float32)
                            + (1.0 - self.alpha) * self.template.astype(np.float32)
                        ).astype(np.uint8)
                        updated_template = 1

        # 5) Resample and reset weights
        self.particles = self.resample_particles()
        self._frame_index += 1
        self.debug_updated_template = updated_template


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
    sigma_dyn = 10.0  # Sigma for particle dynamics (movement noise)

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
    # Tuned for robustness on noisy sequence
    num_particles = 500
    sigma_mse = 12.0
    sigma_dyn = 10.0

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        metric_color='gray',
        init_std=10,
        sigma_schedule=False,
        vanilla=True,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 600  # Sufficient particles
    sigma_mse = 10.0  # Balanced similarity
    sigma_dyn = 14.0  # Moderate exploration
    alpha = 0.02  # More conservative adaptation

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
        update_threshold=0.2,
        update_warmup=12,
        metric_color='green',
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
