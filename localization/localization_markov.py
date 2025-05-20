import numpy as np
import matplotlib.pyplot as plt
from localization_base import Localization

class MarkovLocalization(Localization):
    """
    Markov Localization that uses a discrete probability distribution
    over all possible robot poses (x, y, theta).
    """
    
    def __init__(self, env, resolution=0.2, angle_resolution=10):
        """
        Initialize Markov Localization.
        
        Args:
            env: The environment
            resolution: Grid cell size for x, y discretization
            angle_resolution: Number of discrete angle bins
        """
        super().__init__(env)
        self.resolution = resolution
        self.angle_resolution = angle_resolution
        
        # Calculate grid dimensions
        self.x_bins = int(np.ceil(env.width / resolution))
        self.y_bins = int(np.ceil(env.height / resolution))
        
        # Initialize belief as empty
        self.belief = None
        
        # Store grid coordinates
        self.x_centers = np.linspace(resolution/2, env.width - resolution/2, self.x_bins)
        self.y_centers = np.linspace(resolution/2, env.height - resolution/2, self.y_bins)
        self.theta_centers = np.linspace(0, 2*np.pi, angle_resolution, endpoint=False)
        
        # Precalculate expected measurements for all grid cells
        self._precompute_measurements()
    
    def _precompute_measurements(self):
        """
        Precompute expected measurements for all grid cells.
        This significantly speeds up the measurement update step.
        """
        landmarks = self.env.get_landmarks()
        num_landmarks = landmarks.shape[0]
        
        # Initialize arrays to store expected distances and bearings
        self.expected_distances = np.zeros((self.x_bins, self.y_bins, num_landmarks))
        self.expected_bearings = np.zeros((self.x_bins, self.y_bins, self.angle_resolution, num_landmarks))
        
        # Compute expected measurements for each grid cell
        for i, x in enumerate(self.x_centers):
            for j, y in enumerate(self.y_centers):
                # Compute distances to landmarks
                dx = landmarks[:, 0] - x
                dy = landmarks[:, 1] - y
                self.expected_distances[i, j, :] = np.sqrt(dx**2 + dy**2)
                
                # Compute bearings to landmarks for each orientation
                for k, theta in enumerate(self.theta_centers):
                    bearings = np.arctan2(dy, dx) - theta
                    # Normalize to [-pi, pi]
                    self.expected_bearings[i, j, k, :] = np.mod(bearings + np.pi, 2*np.pi) - np.pi
    
    def initialize(self, x_range, y_range, theta_range=None):
        """
        Initialize the belief state with a uniform distribution.
        
        Args:
            x_range: Range of possible x positions as (min, max)
            y_range: Range of possible y positions as (min, max)
            theta_range: Range of possible orientations as (min, max)
        """
        # Create discrete probability distribution
        self.belief = np.ones((self.x_bins, self.y_bins, self.angle_resolution))
        
        # If ranges are specified, limit the initial belief
        if x_range is not None:
            x_min, x_max = x_range
            x_indices = np.where((self.x_centers >= x_min) & (self.x_centers <= x_max))[0]
            mask = np.zeros(self.x_bins, dtype=bool)
            mask[x_indices] = True
            self.belief = self.belief * mask.reshape(-1, 1, 1)
        
        if y_range is not None:
            y_min, y_max = y_range
            y_indices = np.where((self.y_centers >= y_min) & (self.y_centers <= y_max))[0]
            mask = np.zeros(self.y_bins, dtype=bool)
            mask[y_indices] = True
            self.belief = self.belief * mask.reshape(1, -1, 1)
        
        if theta_range is not None:
            theta_min, theta_max = theta_range
            theta_indices = np.where(
                (self.theta_centers >= theta_min) & 
                (self.theta_centers <= theta_max)
            )[0]
            mask = np.zeros(self.angle_resolution, dtype=bool)
            mask[theta_indices] = True
            self.belief = self.belief * mask.reshape(1, 1, -1)
        
        # Normalize
        self._normalize_belief()
    
    def _normalize_belief(self):
        """
        Normalize the belief to ensure it's a valid probability distribution.
        """
        # Handle zero-sum case to avoid division by zero
        if np.sum(self.belief) < 1e-10:
            self.belief = np.ones_like(self.belief)
            
        # Normalize
        self.belief = self.belief / np.sum(self.belief)
    
    def update_motion(self, dx, dy, dtheta=0):
        """
        Update belief based on robot motion using a motion model.
        This implements the prediction step of Bayes filtering.
        
        Args:
            dx: Change in x position
            dy: Change in y position
            dtheta: Change in orientation (radians)
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Convert changes to grid cell offsets
        cell_dx = dx / self.resolution
        cell_dy = dy / self.resolution
        angle_bin_dtheta = int(round(dtheta / (2*np.pi / self.angle_resolution))) % self.angle_resolution
        
        # Calculate standard deviations for the motion model
        # These values control how much the belief "spreads out" during motion
        sigma_x = max(0.5, abs(cell_dx) * 0.3)  # Cells
        sigma_y = max(0.5, abs(cell_dy) * 0.3)  # Cells
        sigma_theta = max(1, abs(angle_bin_dtheta) * 0.3)  # Angle bins
        
        # Create new belief grid
        new_belief = np.zeros_like(self.belief)
        
        # For each cell in the current belief
        for i in range(self.x_bins):
            for j in range(self.y_bins):
                for k in range(self.angle_resolution):
                    if self.belief[i, j, k] < 1e-10:
                        continue  # Skip cells with very low probability
                    
                    # Apply motion model: Calculate new position
                    new_i = i + cell_dx
                    new_j = j + cell_dy
                    new_k = (k + angle_bin_dtheta) % self.angle_resolution
                    
                    # Apply motion uncertainty using a discretized Gaussian
                    # We'll update a region around the predicted new position
                    i_range = max(0, int(new_i - 3*sigma_x)), min(self.x_bins, int(new_i + 3*sigma_x) + 1)
                    j_range = max(0, int(new_j - 3*sigma_y)), min(self.y_bins, int(new_j + 3*sigma_y) + 1)
                    
                    for ni in range(*i_range):
                        for nj in range(*j_range):
                            # For theta, we consider wrapping
                            for dk in range(-int(3*sigma_theta), int(3*sigma_theta) + 1):
                                nk = (int(new_k) + dk) % self.angle_resolution
                                
                                # Skip if position is outside the map
                                if self.env.check_collision(
                                    self.x_centers[ni], 
                                    self.y_centers[nj], 
                                    0.2  # Robot radius
                                ):
                                    continue
                                
                                # Calculate Gaussian probability
                                dist_i = (ni - new_i) / sigma_x
                                dist_j = (nj - new_j) / sigma_y
                                dist_k = min(
                                    (dk % self.angle_resolution) / sigma_theta,
                                    (self.angle_resolution - (dk % self.angle_resolution)) / sigma_theta
                                )
                                
                                # Multivariate Gaussian (ignoring normalization)
                                prob = np.exp(-0.5 * (dist_i**2 + dist_j**2 + dist_k**2))
                                
                                # Update the new belief
                                new_belief[ni, nj, nk] += self.belief[i, j, k] * prob
        
        # Update the belief and normalize
        self.belief = new_belief
        self._normalize_belief()
    
    def update_measurement(self, distances, bearings):
        """
        Update belief based on measurements using a sensor model.
        This implements the correction step of Bayes filtering.
        
        Args:
            distances: Distances to landmarks
            bearings: Bearings to landmarks
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Measurement noise (standard deviation)
        distance_sigma = 0.5  # meters
        bearing_sigma = 0.1   # radians
        
        # Calculate the likelihood for each grid cell
        likelihood = np.ones((self.x_bins, self.y_bins, self.angle_resolution))
        
        # For each landmark, calculate measurement likelihood
        for lm_idx in range(len(distances)):
            # Extract current measurement
            measured_dist = distances[lm_idx]
            measured_bearing = bearings[lm_idx]
            
            # For each cell and angle, calculate likelihood of this measurement
            for i in range(self.x_bins):
                for j in range(self.y_bins):
                    if np.sum(self.belief[i, j, :]) < 1e-10:
                        continue  # Skip cells with very low probability
                    
                    # Compute expected distance to this landmark
                    expected_dist = self.expected_distances[i, j, lm_idx]
                    
                    # Distance likelihood (Gaussian model)
                    dist_diff = measured_dist - expected_dist
                    dist_likelihood = np.exp(-0.5 * (dist_diff / distance_sigma)**2)
                    
                    for k in range(self.angle_resolution):
                        if self.belief[i, j, k] < 1e-10:
                            continue  # Skip orientations with very low probability
                        
                        # Expected bearing given position and orientation
                        expected_bearing = self.expected_bearings[i, j, k, lm_idx]
                        
                        # Bearing likelihood (von Mises distribution approximation)
                        bearing_diff = measured_bearing - expected_bearing
                        # Normalize to [-pi, pi]
                        bearing_diff = np.mod(bearing_diff + np.pi, 2*np.pi) - np.pi
                        bearing_likelihood = np.exp(-0.5 * (bearing_diff / bearing_sigma)**2)
                        
                        # Combined likelihood
                        likelihood[i, j, k] *= dist_likelihood * bearing_likelihood
        
        # Update the belief (Bayes rule)
        self.belief = self.belief * likelihood
        
        # Normalize
        self._normalize_belief()
    
    def get_belief(self):
        """
        Get the current belief state.
        
        Returns:
            The 3D belief array
        """
        return self.belief
    
    def get_estimate(self):
        """
        Get the best estimate of the robot's pose.
        
        Returns:
            tuple of (x, y, theta) representing the estimated pose
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Find the maximum probability
        i, j, k = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        
        # Convert to x, y, theta
        x = self.x_centers[i]
        y = self.y_centers[j]
        theta = self.theta_centers[k]
        
        return x, y, theta
    
    def visualize_belief(self, ax=None):
        """
        Visualize the belief state.
        
        Args:
            ax: Matplotlib axis to draw on (creates a new one if None)
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Sum over all orientations to get the 2D belief map
        belief_2d = np.sum(self.belief, axis=2)
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        # Plot heatmap
        im = ax.pcolormesh(X, Y, belief_2d, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Plot landmarks
        landmarks = self.env.get_landmarks()
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='^', s=100, label='Landmarks')
        
        # Draw walls
        walls = self.env.get_walls()
        for wall in walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
        
        # Draw the most likely pose
        x, y, theta = self.get_estimate()
        ax.plot(x, y, 'mo', markersize=10, label='Estimated Pose')
        
        # Draw the orientation
        dx = 0.3 * np.cos(theta)
        dy = 0.3 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='m', ec='m')
        
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title('Markov Localization Belief')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        return ax