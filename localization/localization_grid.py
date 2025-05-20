import numpy as np
import matplotlib.pyplot as plt
from localization_base import Localization

class GridLocalization(Localization):
    """
    Grid Localization that represents the environment as a fixed grid and
    maintains a probability for each cell. This is a simplified version of
    Markov Localization that only considers position, not orientation.
    """
    
    def __init__(self, env, resolution=0.2):
        """
        Initialize Grid Localization.
        
        Args:
            env: The environment
            resolution: Grid cell size
        """
        super().__init__(env)
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.x_bins = int(np.ceil(env.width / resolution))
        self.y_bins = int(np.ceil(env.height / resolution))
        
        # Initialize belief as empty
        self.belief = None
        
        # Store grid coordinates
        self.x_centers = np.linspace(resolution/2, env.width - resolution/2, self.x_bins)
        self.y_centers = np.linspace(resolution/2, env.height - resolution/2, self.y_bins)
        
        # Precompute distance to landmarks for each grid cell
        self._precompute_landmark_distances()
    
    def _precompute_landmark_distances(self):
        """
        Precompute distances to landmarks for all grid cells.
        This significantly speeds up the measurement update step.
        """
        landmarks = self.env.get_landmarks()
        num_landmarks = landmarks.shape[0]
        
        # Initialize array to store distances
        self.landmark_distances = np.zeros((self.x_bins, self.y_bins, num_landmarks))
        
        # Compute distances for each grid cell
        for i, x in enumerate(self.x_centers):
            for j, y in enumerate(self.y_centers):
                # Compute distances to landmarks
                dx = landmarks[:, 0] - x
                dy = landmarks[:, 1] - y
                self.landmark_distances[i, j, :] = np.sqrt(dx**2 + dy**2)
    
    def initialize(self, x_range, y_range, theta_range=None):
        """
        Initialize the belief state with a uniform distribution.
        
        Args:
            x_range: Range of possible x positions as (min, max)
            y_range: Range of possible y positions as (min, max)
            theta_range: Not used in Grid Localization
        """
        # Create uniform probability distribution
        self.belief = np.ones((self.x_bins, self.y_bins))
        
        # If ranges are specified, limit the initial belief
        if x_range is not None:
            x_min, x_max = x_range
            x_indices = np.where((self.x_centers >= x_min) & (self.x_centers <= x_max))[0]
            mask_x = np.zeros(self.x_bins, dtype=bool)
            mask_x[x_indices] = True
            self.belief = self.belief * mask_x.reshape(-1, 1)
        
        if y_range is not None:
            y_min, y_max = y_range
            y_indices = np.where((self.y_centers >= y_min) & (self.y_centers <= y_max))[0]
            mask_y = np.zeros(self.y_bins, dtype=bool)
            mask_y[y_indices] = True
            self.belief = self.belief * mask_y.reshape(1, -1)
        
        # Set zero probability for cells that are in collision
        for i, x in enumerate(self.x_centers):
            for j, y in enumerate(self.y_centers):
                if self.env.check_collision(x, y, 0.2):  # radius = 0.2
                    self.belief[i, j] = 0
        
        # Normalize
        self._normalize_belief()
    
    def _normalize_belief(self):
        """
        Normalize the belief to ensure it's a valid probability distribution.
        """
        # Handle zero-sum case to avoid division by zero
        if np.sum(self.belief) < 1e-10:
            # Reinitialize with uniform distribution
            self.belief = np.ones((self.x_bins, self.y_bins))
            # Set zero probability for cells that are in collision
            for i, x in enumerate(self.x_centers):
                for j, y in enumerate(self.y_centers):
                    if self.env.check_collision(x, y, 0.2):  # radius = 0.2
                        self.belief[i, j] = 0
        
        # Normalize
        self.belief = self.belief / np.sum(self.belief)
    
    def update_motion(self, dx, dy, dtheta=0):
        """
        Update belief based on robot motion using a motion model.
        
        Args:
            dx: Change in x position
            dy: Change in y position
            dtheta: Not used in Grid Localization
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Convert motion to grid cell offsets
        cell_dx = dx / self.resolution
        cell_dy = dy / self.resolution
        
        # Motion uncertainty parameters
        sigma_x = max(0.5, abs(cell_dx) * 0.3)  # Cells
        sigma_y = max(0.5, abs(cell_dy) * 0.3)  # Cells
        
        # Create new belief grid
        new_belief = np.zeros_like(self.belief)
        
        # Apply motion model to each cell
        for i in range(self.x_bins):
            for j in range(self.y_bins):
                if self.belief[i, j] < 1e-10:
                    continue  # Skip cells with very low probability
                
                # Calculate new position
                new_i = i + cell_dx
                new_j = j + cell_dy
                
                # Apply motion uncertainty
                i_range = max(0, int(new_i - 3*sigma_x)), min(self.x_bins, int(new_i + 3*sigma_x) + 1)
                j_range = max(0, int(new_j - 3*sigma_y)), min(self.y_bins, int(new_j + 3*sigma_y) + 1)
                
                for ni in range(*i_range):
                    for nj in range(*j_range):
                        # Skip if position is in collision
                        if self.env.check_collision(self.x_centers[ni], self.y_centers[nj], 0.2):
                            continue
                        
                        # Calculate Gaussian probability
                        dist_i = (ni - new_i) / sigma_x
                        dist_j = (nj - new_j) / sigma_y
                        prob = np.exp(-0.5 * (dist_i**2 + dist_j**2))
                        
                        # Update the new belief
                        new_belief[ni, nj] += self.belief[i, j] * prob
        
        # Update belief and normalize
        self.belief = new_belief
        self._normalize_belief()
    
    def update_measurement(self, distances, bearings=None):
        """
        Update belief based on distance measurements to landmarks.
        
        Args:
            distances: Distances to landmarks
            bearings: Bearings to landmarks (not used in basic Grid Localization)
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Measurement noise (standard deviation)
        distance_sigma = 0.5  # meters
        
        # Calculate the likelihood for each grid cell
        likelihood = np.ones((self.x_bins, self.y_bins))
        
        # For each landmark, calculate measurement likelihood
        for lm_idx, measured_dist in enumerate(distances):
            # For each cell, calculate likelihood of this measurement
            for i in range(self.x_bins):
                for j in range(self.y_bins):
                    # Compute expected distance to this landmark
                    expected_dist = self.landmark_distances[i, j, lm_idx]
                    
                    # Distance likelihood (Gaussian model)
                    dist_diff = measured_dist - expected_dist
                    dist_likelihood = np.exp(-0.5 * (dist_diff / distance_sigma)**2)
                    
                    # Update likelihood
                    likelihood[i, j] *= dist_likelihood
        
        # Update belief (Bayes rule)
        self.belief = self.belief * likelihood
        
        # Normalize
        self._normalize_belief()
    
    def get_belief(self):
        """
        Get the current belief state.
        
        Returns:
            The 2D belief array
        """
        return self.belief
    
    def get_estimate(self):
        """
        Get the best estimate of the robot's position.
        
        Returns:
            tuple of (x, y, 0) representing the estimated pose
            (theta is always 0 in this implementation)
        """
        if self.belief is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Find the maximum probability
        i, j = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        
        # Convert to x, y
        x = self.x_centers[i]
        y = self.y_centers[j]
        
        return x, y, 0  # theta is not estimated in Grid Localization
    
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
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        # Plot heatmap
        im = ax.pcolormesh(X, Y, self.belief, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Plot landmarks
        landmarks = self.env.get_landmarks()
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='^', s=100, label='Landmarks')
        
        # Draw walls
        walls = self.env.get_walls()
        for wall in walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
        
        # Draw the most likely pose
        x, y, _ = self.get_estimate()
        ax.plot(x, y, 'mo', markersize=10, label='Estimated Position')
        
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title('Grid Localization Belief')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        return ax