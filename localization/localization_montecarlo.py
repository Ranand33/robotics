import numpy as np
import matplotlib.pyplot as plt
from localization_base import Localization
import matplotlib.patches as patches

class MonteCarloLocalization(Localization):
    """
    Monte Carlo Localization (Particle Filter) that represents the belief state
    with a set of particles (samples) from the probability distribution.
    """
    
    def __init__(self, env, num_particles=1000):
        """
        Initialize Monte Carlo Localization.
        
        Args:
            env: The environment
            num_particles: Number of particles to use
        """
        super().__init__(env)
        self.num_particles = num_particles
        
        # Initialize particles and weights
        self.particles = None  # Shape: (num_particles, 3) for (x, y, theta)
        self.weights = None    # Shape: (num_particles,)
        
        # Motion model noise parameters
        self.alpha1 = 0.1  # Error in rotation from rotation
        self.alpha2 = 0.1  # Error in rotation from translation
        self.alpha3 = 0.1  # Error in translation from translation
        self.alpha4 = 0.1  # Error in translation from rotation
        
        # Measurement model parameters
        self.sigma_distance = 0.5   # Standard deviation for distance measurements
        self.sigma_bearing = 0.1    # Standard deviation for bearing measurements
        
        # Thresholds for resampling
        self.resampling_threshold = 0.5
        
        # Store landmarks
        self.landmarks = env.get_landmarks()
    
    def initialize(self, x_range, y_range, theta_range=None):
        """
        Initialize particles with a uniform distribution over the specified ranges.
        
        Args:
            x_range: Range of possible x positions as (min, max)
            y_range: Range of possible y positions as (min, max)
            theta_range: Range of possible orientations as (min, max)
        """
        # Set default ranges if not provided
        if x_range is None:
            x_range = (0, self.env.width)
        if y_range is None:
            y_range = (0, self.env.height)
        if theta_range is None:
            theta_range = (0, 2 * np.pi)
        
        # Initialize particles with uniform distribution
        self.particles = np.zeros((self.num_particles, 3))
        
        # Keep generating particles until we have enough valid ones
        valid_count = 0
        while valid_count < self.num_particles:
            # Generate candidate particles
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            theta = np.random.uniform(*theta_range)
            
            # Check collision
            if not self.env.check_collision(x, y, 0.2):  # radius = 0.2
                self.particles[valid_count] = [x, y, theta]
                valid_count += 1
        
        # Initialize weights
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def update_motion(self, dx, dy, dtheta=0):
        """
        Update particles based on robot motion using a probabilistic motion model.
        
        Args:
            dx: Change in x position
            dy: Change in y position
            dtheta: Change in orientation (radians)
        """
        if self.particles is None:
            raise ValueError("Particles not initialized. Call initialize() first.")
        
        # For particle filters, we'll use a sample-based motion model
        # with additive noise that depends on the motion
        
        # Convert dx, dy to distance and bearing
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) if distance > 1e-6 else 0
        
        # For each particle
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]
            
            if distance > 1e-6:
                # First rotation to face the motion direction
                rot1 = self._normalize_angle(bearing - theta)
                
                # Then translation
                trans = distance
                
                # Last rotation to achieve the desired final orientation
                rot2 = self._normalize_angle(dtheta - rot1)
                
                # Add noise to the controls using the motion model
                # Noise in the first rotation
                rot1_noise = rot1 + np.random.normal(0, self.alpha1 * abs(rot1) + self.alpha2 * trans)
                
                # Noise in the translation
                trans_noise = trans + np.random.normal(0, self.alpha3 * trans + self.alpha4 * (abs(rot1) + abs(rot2)))
                
                # Noise in the second rotation
                rot2_noise = rot2 + np.random.normal(0, self.alpha1 * abs(rot2) + self.alpha2 * trans)
                
                # Apply noisy motion
                theta_new = theta + rot1_noise
                x_new = x + trans_noise * np.cos(theta_new)
                y_new = y + trans_noise * np.sin(theta_new)
                theta_new = self._normalize_angle(theta_new + rot2_noise)
            else:
                # Only rotation, no movement
                rot_noise = dtheta + np.random.normal(0, self.alpha1 * abs(dtheta))
                x_new = x
                y_new = y
                theta_new = self._normalize_angle(theta + rot_noise)
            
            # Check if the new position is valid
            if not self.env.check_collision(x_new, y_new, 0.2):  # radius = 0.2
                self.particles[i] = [x_new, y_new, theta_new]
            # If invalid, keep the old position
    
    def _normalize_angle(self, angle):
        """Normalize angle to be in [-pi, pi]"""
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    
    def update_measurement(self, distances, bearings):
        """
        Update particle weights based on measurements.
        
        Args:
            distances: Distances to landmarks
            bearings: Bearings to landmarks
        """
        if self.particles is None:
            raise ValueError("Particles not initialized. Call initialize() first.")
        
        # Initialize weights
        self.weights = np.ones(self.num_particles)
        
        # For each particle, calculate the likelihood of the measurements
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]
            
            # For each landmark, calculate the expected measurement
            for j, (landmark_x, landmark_y) in enumerate(self.landmarks):
                # Expected measurement
                dx = landmark_x - x
                dy = landmark_y - y
                expected_distance = np.sqrt(dx**2 + dy**2)
                expected_bearing = self._normalize_angle(np.arctan2(dy, dx) - theta)
                
                # Actual measurement
                measured_distance = distances[j]
                measured_bearing = bearings[j]
                
                # Likelihood calculation
                # Distance likelihood (Gaussian model)
                dist_diff = measured_distance - expected_distance
                dist_likelihood = np.exp(-0.5 * (dist_diff / self.sigma_distance)**2) / (np.sqrt(2 * np.pi) * self.sigma_distance)
                
                # Bearing likelihood (wrapped Gaussian model)
                bearing_diff = self._normalize_angle(measured_bearing - expected_bearing)
                bearing_likelihood = np.exp(-0.5 * (bearing_diff / self.sigma_bearing)**2) / (np.sqrt(2 * np.pi) * self.sigma_bearing)
                
                # Combined likelihood
                self.weights[i] *= dist_likelihood * bearing_likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Check if we need to resample
        effective_particles = 1.0 / np.sum(self.weights**2)
        if effective_particles < self.num_particles * self.resampling_threshold:
            self._resample()
    
    def _resample(self):
        """
        Resample particles based on their weights using low variance sampling.
        """
        # New particle array
        new_particles = np.zeros((self.num_particles, 3))
        
        # Low variance resampling
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = self.weights[0]
        i = 0
        
        for m in range(self.num_particles):
            u = r + m / self.num_particles
            while u > c:
                i = (i + 1) % self.num_particles
                c += self.weights[i]
            new_particles[m] = self.particles[i]
        
        # Update particles
        self.particles = new_particles
        
        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_belief(self):
        """
        Get the current belief state.
        
        Returns:
            tuple of (particles, weights)
        """
        return self.particles, self.weights
    
    def get_estimate(self):
        """
        Get the best estimate of the robot's pose as the weighted average of particles.
        
        Returns:
            tuple of (x, y, theta) representing the estimated pose
        """
        if self.particles is None:
            raise ValueError("Particles not initialized. Call initialize() first.")
        
        # Weighted average of positions
        x = np.sum(self.weights * self.particles[:, 0])
        y = np.sum(self.weights * self.particles[:, 1])
        
        # For theta, we need to handle the circular nature
        cos_theta = np.sum(self.weights * np.cos(self.particles[:, 2]))
        sin_theta = np.sum(self.weights * np.sin(self.particles[:, 2]))
        theta = np.arctan2(sin_theta, cos_theta)
        
        return x, y, theta
    
    def visualize_belief(self, ax=None):
        """
        Visualize the belief state by plotting particles.
        
        Args:
            ax: Matplotlib axis to draw on (creates a new one if None)
        """
        if self.particles is None:
            raise ValueError("Particles not initialized. Call initialize() first.")
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot particles
        # Scale point sizes by weights
        sizes = 50 * self.weights * self.num_particles
        colors = self.weights / np.max(self.weights) if np.max(self.weights) > 0 else np.ones(self.num_particles)
        sc = ax.scatter(
            self.particles[:, 0], 
            self.particles[:, 1], 
            c=colors, 
            s=sizes, 
            alpha=0.5, 
            cmap='viridis',
            label='Particles'
        )
        plt.colorbar(sc, ax=ax, label='Normalized Weight')
        
        # Draw particle orientations (for a subset of particles)
        if self.num_particles > 100:
            # Draw arrows for only a subset of particles
            indices = np.random.choice(self.num_particles, 50, replace=False, p=self.weights)
        else:
            indices = range(self.num_particles)
        
        for i in indices:
            x, y, theta = self.particles[i]
            dx = 0.2 * np.cos(theta)
            dy = 0.2 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='b', ec='b', alpha=0.5)
        
        # Plot landmarks
        ax.scatter(
            self.landmarks[:, 0],
            self.landmarks[:, 1],
            c='g',
            marker='^',
            s=100,
            label='Landmarks'
        )
        
        # Draw walls
        walls = self.env.get_walls()
        for wall in walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
        
        # Draw the estimated pose
        x, y, theta = self.get_estimate()
        ax.plot(x, y, 'ro', markersize=10, label='Estimated Pose')
        dx = 0.3 * np.cos(theta)
        dy = 0.3 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='r', ec='r')
        
        # Compute weighted kernel density estimate for visualization
        try:
            from scipy.stats import gaussian_kde
            if self.num_particles > 20:  # Only do KDE if we have enough particles
                # Extract x, y coordinates
                xy = np.vstack([self.particles[:, 0], self.particles[:, 1]])
                
                # Create weighted KDE
                kde = gaussian_kde(xy, weights=self.weights)
                
                # Create a grid for evaluation
                x_grid = np.linspace(0, self.env.width, 100)
                y_grid = np.linspace(0, self.env.height, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])
                
                # Evaluate the KDE
                Z = kde(positions)
                Z = Z.reshape(X.shape)
                
                # Plot the contours
                contour = ax.contour(X, Y, Z, cmap='viridis', alpha=0.3)
        except (ImportError, np.linalg.LinAlgError):
            pass  # Skip KDE if scipy is not available or errors
        
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title('Monte Carlo Localization Belief')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        return ax