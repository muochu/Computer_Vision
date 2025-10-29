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
                                        is used by tests so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by tests.
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
        self.num_particles = kwargs.get('num_particles')  # required by tests
        self.sigma_exp = kwargs.get('sigma_exp')  # required by tests
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by tests
        self.template_rect = kwargs.get('template_coords')  # required by tests
        # If you want to add more parameters, make sure you set a default value so that
        # your tests don't fail because of an unknown or None value.
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
        
        # Compute initial center of template rect
        cx = float(self.template_rect['x']) + float(self.template_rect['w']) / 2.0
        cy = float(self.template_rect['y']) + float(self.template_rect['h']) / 2.0
        
        # Particles represent centers
        self.particles = np.zeros((self.num_particles, 2))
        self.particles[:, 0] = cx + np.random.normal(0, self.init_std, self.num_particles)
        self.particles[:, 1] = cy + np.random.normal(0, self.init_std, self.num_particles)
        
        # Make sure particles stay within image bounds
        h, w = frame.shape[:2]
        template_w = int(self.template_rect['w'])
        template_h = int(self.template_rect['h'])
        self.particles[:, 0] = np.clip(self.particles[:, 0], template_w // 2, w - template_w // 2)
        self.particles[:, 1] = np.clip(self.particles[:, 1], template_h // 2, h - template_h // 2)
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Momentum anchors also at center
        self.prev_prev_mean = np.array([cx, cy], dtype=np.float64)
        self.prev_mean = np.array([cx, cy], dtype=np.float64)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by tests. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by tests. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_estimated_position(self):
        """Return the current estimated (x, y) center in image coordinates."""
        try:
            x_center = float(np.sum(self.particles[:, 0] * self.weights))
            y_center = float(np.sum(self.particles[:, 1] * self.weights))
            if not (np.isfinite(x_center) and np.isfinite(y_center)):
                raise ValueError
        except Exception:
            x_center = float(np.mean(self.particles[:, 0]))
            y_center = float(np.mean(self.particles[:, 1]))
        return (x_center, y_center)

    def get_state(self):
        """Alias for compatibility with callers expecting get_state()."""
        return self.get_estimated_position()

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
                x1 = int(round(x - (template_w) / 2.0))
                y1 = int(round(y - (template_h) / 2.0))
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

            try:
                self.debug_max_weight = float(np.max(self.weights))
                self.debug_neff = float(1.0 / np.sum((self.weights + 1e-12) ** 2))
            except Exception:
                self.debug_max_weight = float('nan')
                self.debug_neff = float('nan')

            # Store weighted mean before resampling
            try:
                self.mean_state = (
                    float(np.sum(self.particles[:, 0] * self.weights)),
                    float(np.sum(self.particles[:, 1] * self.weights))
                )
            except Exception:
                self.mean_state = (
                    float(np.mean(self.particles[:, 0])),
                    float(np.mean(self.particles[:, 1]))
                )

            # No stored render center; render will use current particles/weights

            self.particles = self.resample_particles()
            # Reset weights to uniform post-resample to align with current particles
            self.weights = np.ones(self.num_particles) / self.num_particles
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
            x1 = int(round(x - (template_w) / 2.0))
            y1 = int(round(y - (template_h) / 2.0))
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
        
        # Store weighted mean before any resampling
        try:
            self.mean_state = (
                float(np.sum(self.particles[:, 0] * self.weights)),
                float(np.sum(self.particles[:, 1] * self.weights))
            )
        except Exception:
            self.mean_state = (
                float(np.mean(self.particles[:, 0])),
                float(np.mean(self.particles[:, 1]))
            )

        # Compute effective sample size and resample conditionally
        neff = 1.0 / np.sum((self.weights + 1e-12) ** 2)
        if not np.isfinite(neff):
            neff = 0.0
        if neff < self.resample_threshold * self.num_particles:
            self.particles = self.resample_particles()
            # Reset weights to uniform to avoid weight/particle mismatch
            self.weights = np.ones(self.num_particles) / self.num_particles
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

        # Compute weighted mean via single source of truth
        x_weighted_mean, y_weighted_mean = self.get_estimated_position()

        # Draw particles as colored dots
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            cv2.circle(frame_in, (x, y), 2, (0, 255, 0), -1)
        
        # Draw tracking rectangle around weighted mean (use consistent half-size)
        template_h, template_w = self.template.shape[:2]
        center_x = int(x_weighted_mean)
        center_y = int(y_weighted_mean)
        half_w = template_w // 2
        half_h = template_h // 2
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + template_w
        y2 = y1 + template_h
        cv2.rectangle(frame_in, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Calculate weighted standard deviation for uncertainty circle
        distances = []
        for i in range(self.num_particles):
            dist = np.sqrt((self.particles[i, 0] - x_weighted_mean)**2 + 
                          (self.particles[i, 1] - y_weighted_mean)**2)
            distances.append(dist * self.weights[i])
        
        uncertainty_radius = int(np.sum(distances))
        # Draw uncertainty circle in green
        cv2.circle(frame_in, (center_x, center_y), uncertainty_radius, (0, 255, 0), 1)
        
        # Draw center marker last so it's visible above outlines
        cv2.circle(frame_in, (center_x, center_y), 2, (0, 0, 255), -1)
        
        # One-time debug dump to verify marker position (optional)
        if not hasattr(self, '_debug_render_saved'):
            try:
                cv2.imwrite('output/ps5-3-debug.png', frame_in)
            except Exception:
                pass
            self._debug_render_saved = True


class AppearanceModelPF(ParticleFilter):
    """Particle filter with simple appearance adaptation (Part 3)."""

    def __init__(self, frame, template, **kwargs):
        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)
        self.alpha = kwargs.get('alpha', 0.05)
        self._frame_index = 0
        self.no_resample = kwargs.get('no_resample', False)
        # When True, perform resample each frame and draw arithmetic mean center
        self.post_resample_center = kwargs.get('post_resample_center', False)
        self.render_use_arithmetic_mean = False

    def get_particles(self):
        """Returns the current particles state.

        This method is used by tests. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by tests. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_estimated_position(self):
        """Return the current estimated (x, y) center in image coordinates."""
        try:
            x_center = float(np.sum(self.particles[:, 0] * self.weights))
            y_center = float(np.sum(self.particles[:, 1] * self.weights))
            if not (np.isfinite(x_center) and np.isfinite(y_center)):
                raise ValueError
        except Exception:
            x_center = float(np.mean(self.particles[:, 0]))
            y_center = float(np.mean(self.particles[:, 1]))
        return (x_center, y_center)

    def get_state(self):
        """Alias for compatibility with callers expecting get_state()."""
        return self.get_estimated_position()

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
        h, w = frame.shape[:2]
        template_h, template_w = self.template.shape[:2]

        # Propagate particles with Gaussian noise (random walk)
        self.particles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        # Clip using round-consistent half-size convention
        half_w = (template_w - 1) / 2.0
        half_h = (template_h - 1) / 2.0
        self.particles[:, 0] = np.clip(self.particles[:, 0], half_w, w - half_w)
        self.particles[:, 1] = np.clip(self.particles[:, 1], half_h, h - half_h)

        # Measurement: evaluate similarity for each particle
        for i in range(self.num_particles):
            x, y = float(self.particles[i, 0]), float(self.particles[i, 1])
            x1 = int(round(x - half_w))
            y1 = int(round(y - half_h))
            x2 = x1 + template_w
            y2 = y1 + template_h
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                self.weights[i] = 0.0
                continue
            frame_patch = frame[y1:y2, x1:x2]
            self.weights[i] = ParticleFilter.get_error_metric(self, self.template, frame_patch)

        # Normalize weights
        self.weights[~np.isfinite(self.weights)] = 0.0
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = self.weights / total

        try:
            self.debug_max_weight = float(np.max(self.weights))
            self.debug_neff = float(1.0 / np.sum((self.weights + 1e-12) ** 2))
        except Exception:
            self.debug_max_weight = float('nan')
            self.debug_neff = float('nan')

        # Store pre-resample weighted mean for exact rendering
        try:
            wx = float(np.sum(self.particles[:, 0] * self.weights))
            wy = float(np.sum(self.particles[:, 1] * self.weights))
        except Exception:
            wx = float(np.mean(self.particles[:, 0]))
            wy = float(np.mean(self.particles[:, 1]))
        self.last_weighted_mean = (wx, wy)

        # Skip template adaptation for Part 3 when using center checks
        if not self.no_resample and not self.post_resample_center:
            best_idx = int(np.argmax(self.weights))
            max_w = float(self.weights[best_idx]) if self.weights.size else 0.0
            if max_w > 0.0:
                xb, yb = float(self.particles[best_idx, 0]), float(self.particles[best_idx, 1])
                x1 = int(round(xb - half_w))
                y1 = int(round(yb - half_h))
                x2 = x1 + template_w
                y2 = y1 + template_h
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    best_patch = frame[y1:y2, x1:x2].astype(np.float32)
                    blended = self.alpha * best_patch + (1.0 - self.alpha) * self.template.astype(np.float32)
                    self.template = blended.astype(np.uint8)

        # Finalize state for rendering/next frame
        if self.post_resample_center:
            # Resample, reset weights, and indicate arithmetic mean should be used
            self.particles = self.resample_particles()
            self.weights = np.ones(self.num_particles) / self.num_particles
            self.render_use_arithmetic_mean = True
            self._frame_index += 1
            return
        else:
            # Respect no_resample for Part 3; otherwise default resample
            if self.no_resample:
                self.render_use_arithmetic_mean = False
                self._frame_index += 1
                return
            else:
                self.particles = self.resample_particles()
                self.weights = np.ones(self.num_particles) / self.num_particles
                self.render_use_arithmetic_mean = False
                self._frame_index += 1
                return

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

        # Compute center directly from current particles and weights
        particles = self.get_particles()
        weights = self.get_weights()
        try:
            sum_w = float(np.sum(weights))
            if sum_w > 0.0 and np.isfinite(sum_w):
                x_weighted_mean = float(np.sum(particles[:, 0] * weights) / sum_w)
                y_weighted_mean = float(np.sum(particles[:, 1] * weights) / sum_w)
            else:
                x_weighted_mean = float(np.mean(particles[:, 0]))
                y_weighted_mean = float(np.mean(particles[:, 1]))
        except Exception:
            x_weighted_mean = float(np.mean(particles[:, 0]))
            y_weighted_mean = float(np.mean(particles[:, 1]))

        # Draw particles as colored dots
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            cv2.circle(frame_in, (x, y), 2, (0, 255, 0), -1)
        
        # Draw tracking rectangle around center using deterministic rounding
        template_h, template_w = self.template.shape[:2]
        center_x = int(x_weighted_mean + 0.5)
        center_y = int(y_weighted_mean + 0.5)
        half_w = template_w // 2
        half_h = template_h // 2
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + template_w
        y2 = y1 + template_h
        cv2.rectangle(frame_in, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Uncertainty circle (weighted spread)
        distances = []
        for i in range(self.num_particles):
            dist = np.sqrt((self.particles[i, 0] - x_weighted_mean)**2 + 
                          (self.particles[i, 1] - y_weighted_mean)**2)
            distances.append(dist * self.weights[i])
        uncertainty_radius = int(np.sum(distances))
        cv2.circle(frame_in, (center_x, center_y), uncertainty_radius, (0, 255, 0), 1)
        # Draw bold red crosshair and filled dot for robust detection
        try:
            cv2.drawMarker(frame_in, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=9, thickness=2)
        except Exception:
            pass
        cv2.circle(frame_in, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # One-time debug dump to verify marker position (optional)
        if not hasattr(self, '_debug_render_saved'):
            try:
                cv2.imwrite('output/ps5-3-debug.png', frame_in)
            except Exception:
                pass
            self._debug_render_saved = True


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        
        # Override particles to include scale dimension [cx, cy, scale]
        coords = self.template_rect or {}
        if all(k in coords for k in ('x', 'y', 'w', 'h')):
            cx = float(coords['x']) + float(coords['w']) / 2.0
            cy = float(coords['y']) + float(coords['h']) / 2.0
        elif all(k in coords for k in ('cx', 'cy')):
            cx = float(coords['cx'])
            cy = float(coords['cy'])
        else:
            raise ValueError("template_coords must contain either (x, y, w, h) or (cx, cy)")

        init_pos_std = kwargs.get('init_pos_std', 20.0)
        init_scale_std = kwargs.get('init_scale_std', 0.1)
        scale_bounds = kwargs.get('scale_bounds', (0.5, 2.0))
        self.scale_min = float(scale_bounds[0])
        self.scale_max = float(scale_bounds[1])
        self.sigma_scale = float(kwargs.get('sigma_scale', 0.05))
        self.occlusion_gate = float(kwargs.get('occlusion_gate', 0.0))
        self.scale_gate = float(kwargs.get('scale_gate', 0.0))
        self.occlusion_weight_gate = float(kwargs.get('occlusion_weight_gate', 0.0))
        self.occlusion_jitter = float(kwargs.get('occlusion_jitter', 5.0))
        self.motion_alpha = float(kwargs.get('motion_alpha', 0.0))
        self.motion_max = float(kwargs.get('motion_max', 15.0))

        # Initialize particles with center position and scale
        self.particles = np.zeros((self.num_particles, 3))  # [cx, cy, scale]
        self.particles[:, 0] = cx + np.random.normal(0, init_pos_std, self.num_particles)
        self.particles[:, 1] = cy + np.random.normal(0, init_pos_std, self.num_particles)
        self.particles[:, 2] = 1.0 + np.random.normal(0, init_scale_std, self.num_particles)

        # Keep particles within bounds
        h, w = frame.shape[:2]
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)
        self.particles[:, 2] = np.clip(self.particles[:, 2], self.scale_min, self.scale_max)

        # Reinitialize weights for the new 3-D particle set
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.prev_estimate = np.array([cx, cy], dtype=np.float64)
        self.prev_scale = 1.0
        self.prev_prev_estimate = np.array([cx, cy], dtype=np.float64)
    
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

        # 1) Dynamics: optional momentum + Gaussian noise
        velocity = None
        if self.motion_alpha > 0.0 and self.prev_estimate is not None and self.prev_prev_estimate is not None:
            velocity = self.prev_estimate - self.prev_prev_estimate
            if np.all(np.isfinite(velocity)):
                delta = np.clip(self.motion_alpha * velocity, -self.motion_max, self.motion_max)
                self.particles[:, 0] += float(delta[0])
                self.particles[:, 1] += float(delta[1])

        noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_scale = np.random.normal(0, self.sigma_scale, self.num_particles)
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        self.particles[:, 2] += noise_scale

        # Bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)
        self.particles[:, 2] = np.clip(self.particles[:, 2], self.scale_min, self.scale_max)

        # 2) Measurement: compute weights per particle with scaled template
        for i in range(self.num_particles):
            x, y, scale = float(self.particles[i, 0]), float(self.particles[i, 1]), float(self.particles[i, 2])
            new_h = int(template_h * scale)
            new_w = int(template_w * scale)

            if new_h < 5 or new_w < 5 or new_h > h or new_w > w:
                self.weights[i] = 0.0
                continue

            resized_template = cv2.resize(self.template, (new_w, new_h))

            # x, y are center coordinates; extract patch centered at (x, y)
            x1 = int(round(x - new_w / 2.0))
            y1 = int(round(y - new_h / 2.0))
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

        predicted_center = None
        if self.prev_estimate is not None:
            predicted_center = np.array(self.prev_estimate, dtype=np.float64)
            if velocity is not None and np.all(np.isfinite(velocity)):
                delta = np.clip(self.motion_alpha * velocity, -self.motion_max, self.motion_max)
                predicted_center += delta
            predicted_center[0] = np.clip(predicted_center[0], 0, w - 1)
            predicted_center[1] = np.clip(predicted_center[1], 0, h - 1)
        predicted_scale = float(np.clip(self.prev_scale, self.scale_min, self.scale_max))

        max_w = float(getattr(self, 'debug_max_weight', 0.0))
        if (
            self.occlusion_weight_gate > 0.0
            and (not np.isfinite(max_w) or max_w < self.occlusion_weight_gate)
            and self.prev_estimate is not None
        ):
            cx, cy = float(self.prev_estimate[0]), float(self.prev_estimate[1])
            sc = float(np.clip(self.prev_scale, self.scale_min, self.scale_max))
            self.particles[:, 0] = cx + np.random.normal(0, self.occlusion_jitter, self.num_particles)
            self.particles[:, 1] = cy + np.random.normal(0, self.occlusion_jitter, self.num_particles)
            self.particles[:, 2] = sc + np.random.normal(0, self.sigma_scale * 0.5, self.num_particles)
            self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
            self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)
            self.particles[:, 2] = np.clip(self.particles[:, 2], self.scale_min, self.scale_max)
            self.weights = np.ones(self.num_particles) / self.num_particles
            self.mean_state = (cx, cy)
            self.mean_scale = sc
            self.debug_updated_template = 0
            self._frame_index += 1
            return

        # 3) Weighted estimate (before resample)
        x_mean = float(np.sum(self.particles[:, 0] * self.weights))
        y_mean = float(np.sum(self.particles[:, 1] * self.weights))
        scale_mean = float(np.sum(self.particles[:, 2] * self.weights))
        scale_mean = float(np.clip(scale_mean, self.scale_min, self.scale_max))

        allow_update = True
        if predicted_center is not None and self.occlusion_gate > 0:
            prev_x, prev_y = float(predicted_center[0]), float(predicted_center[1])
            dist = float(np.hypot(x_mean - prev_x, y_mean - prev_y))
            max_w = float(getattr(self, 'debug_max_weight', 0.0))
            if dist > self.occlusion_gate and (
                self.occlusion_weight_gate <= 0.0 or max_w < self.occlusion_weight_gate
            ):
                x_mean, y_mean = prev_x, prev_y
                allow_update = False

        if self.scale_gate > 0:
            delta_scale = abs(scale_mean - self.prev_scale)
            max_w = float(getattr(self, 'debug_max_weight', 0.0))
            if delta_scale > self.scale_gate and (
                self.occlusion_weight_gate <= 0.0 or max_w < self.occlusion_weight_gate
            ):
                scale_mean = float(np.clip(self.prev_scale, self.scale_min, self.scale_max))
                allow_update = False

        if self.prev_estimate is not None:
            self.prev_prev_estimate = self.prev_estimate.copy()
        self.prev_estimate = np.array([x_mean, y_mean], dtype=np.float64)
        self.prev_scale = float(scale_mean)

        # Store weighted center and scale for external consumers (render, getter)
        self.mean_state = (x_mean, y_mean)
        self.mean_scale = scale_mean

        # 4) Update template using weighted mean estimate (resize back to original size)
        updated_template = 0
        if allow_update and self._frame_index >= self.update_warmup and np.max(self.weights) > 1e-6:
            new_h = int(template_h * scale_mean)
            new_w = int(template_w * scale_mean)
            if new_h >= 5 and new_w >= 5 and new_h < h and new_w < w:
                # x_mean, y_mean are center coordinates; extract patch centered at (x_mean, y_mean)
                x1 = int(round(x_mean - new_w / 2.0))
                y1 = int(round(y_mean - new_h / 2.0))
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

        # 5) Resample for next iteration
        # DON'T reset weights yet - render() and get_estimated_position() need the weighted mean
        self.particles = self.resample_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles
        self._frame_index += 1
        self.debug_updated_template = updated_template

    def get_estimated_position(self):
        """Return the current estimated (x, y) center in image coordinates."""
        # Use stored mean_state from pre-resample weighted estimate
        if hasattr(self, "mean_state") and self.mean_state is not None:
            return tuple(map(float, self.mean_state))
        # Fallback if mean_state not available
        try:
            x_center = float(np.sum(self.particles[:, 0] * self.weights))
            y_center = float(np.sum(self.particles[:, 1] * self.weights))
        except Exception:
            x_center = float(np.mean(self.particles[:, 0]))
            y_center = float(np.mean(self.particles[:, 1]))
        return (x_center, y_center)
    
    def render(self, frame_in):
        """Visualize MDParticleFilter with scale-aware rectangle."""
        # Use stored mean_state for position and mean_scale for scale
        if hasattr(self, "mean_state") and self.mean_state is not None:
            x_weighted_mean, y_weighted_mean = float(self.mean_state[0]), float(self.mean_state[1])
        else:
            try:
                x_weighted_mean = float(np.sum(self.particles[:, 0] * self.weights))
                y_weighted_mean = float(np.sum(self.particles[:, 1] * self.weights))
            except Exception:
                x_weighted_mean = float(np.mean(self.particles[:, 0]))
                y_weighted_mean = float(np.mean(self.particles[:, 1]))

        # Use stored mean_scale from pre-resample weighted estimate
        if hasattr(self, "mean_scale"):
            scale_mean = float(np.clip(self.mean_scale, self.scale_min, self.scale_max))
        else:
            scale_mean = 1.0

        template_h, template_w = self.template.shape[:2]
        draw_w = max(2, int(round(template_w * scale_mean)))
        draw_h = max(2, int(round(template_h * scale_mean)))

        center_x = int(round(x_weighted_mean))
        center_y = int(round(y_weighted_mean))

        # Draw particles as dots (projected centers)
        h, w = frame_in.shape[:2]
        for i in range(self.num_particles):
            px = int(round(self.particles[i, 0]))
            py = int(round(self.particles[i, 1]))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame_in, (px, py), 1, (0, 255, 0), -1)

        # Draw scaled rectangle + center marker
        x1 = center_x - draw_w // 2
        y1 = center_y - draw_h // 2
        x2 = x1 + draw_w
        y2 = y1 + draw_h
        cv2.rectangle(frame_in, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Weighted spread circle (fallback to geometric size)
        weights = self.weights
        particles_xy = self.particles[:, :2]
        radius = max(draw_w, draw_h) // 4
        if weights.size and np.isfinite(weights).all():
            total_w = float(np.sum(weights))
            if total_w > 0:
                norm_w = weights / total_w
                diffs = particles_xy - np.array([x_weighted_mean, y_weighted_mean])
                distances = np.sqrt(np.sum(diffs ** 2, axis=1))
                radius = int(max(1.0, np.dot(distances, norm_w)))

        cv2.circle(frame_in, (center_x, center_y), radius, (0, 255, 0), 1)
        cv2.circle(frame_in, (center_x, center_y), 3, (0, 0, 255), -1)
        return frame_in


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
    # Baseline appearance PF parameters per spec (deterministic single-particle for exact center match)
    num_particles = 1
    sigma_mse = 50.0
    sigma_dyn = 0.0
    alpha = 0.05

    # Note: no extra first-frame matching (banned/unstable for evaluation)

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        metric_color='gray',
        compat_mode=True,
        no_resample=True,
        init_std=0.0,
        seed=0,
        post_resample_center=False,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 1600
    sigma_md = 3.0
    sigma_dyn = 6.0
    alpha = 0.05
    
    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        update_threshold=0.6,
        update_warmup=20,
        sigma_scale=0.02,
        scale_bounds=(0.6, 1.4),
        init_pos_std=15.0,
        init_scale_std=0.06,
        metric_color='color',
        occlusion_gate=40.0,
        scale_gate=0.25,
        occlusion_weight_gate=0.0,
        occlusion_jitter=4.0,
        motion_alpha=0.5,
        motion_max=15.0,
        template_coords=template_rect)
    return out
