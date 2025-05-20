import numpy as np
import matplotlib.pyplot as plt
from localization_base import Localization
import matplotlib.patches as patches
from scipy.stats import multivariate_normal

class GaussianLocalization(Localization):
    """
    Gaussian Localization using Extended Kalman Filter (EKF).
    Represents the belief state as a multivariate Gaussian distribution.
    """
    
    def __init__(self, env):
        """
        Initialize Gaussian Localization with EKF.
        
        Args:
            env: The environment
        """
        super().__init__(env)
        
        # State dimension (x, y, theta)
        self.state_dim = 3
        
        # Mean of the Gaussian belief
        self.mu = None
        
        # Covariance of the Gaussian belief
        self.sigma = None
        
        # Process and measurement noise
        self.R = np.diag([0.1, 0.1, 0.05])  # Process noise (motion)
        
        # Store landmarks
        self.landmarks = env.get_landmarks()
        self.num_landmarks = len(self.landmarks)
        
        # Measurement noise (distance, bearing)
        self.Q = np.diag([0.5, 0.1])  # Measurement noise per landmark
    
    def initialize(self, x_range, y_range, theta_range=None):
        """
        Initialize the belief state with a Gaussian centered at the midpoint of the ranges.
        
        Args:
            x_range: Range of possible x positions as (min, max)
            y_range: Range of possible y positions as (min, max)
            theta_range: Range of possible orientations as (min, max)
        """
        # Set initial mean
        if x_range is not None:
            x_min, x_max = x_range
            x = (x_min + x_max) / 2
        else:
            x = self.env.width / 2
        
        if y_range is not None:
            y_min, y_max = y_range
            y = (y_min + y_max) / 2
        else:
            y = self.env.height / 2
        
        if theta_range is not None:
            theta_min, theta_max = theta_range
            theta = (theta_min + theta_max) / 2
        else:
            theta = 0.0
        
        self.mu = np.array([x, y, theta])
        
        # Set initial covariance based on ranges
        var_x = ((x_max - x_min) / 2)**2 if x_range is not None else (self.env.width / 2)**2
        var_y = ((y_max - y_min) / 2)**2 if y_range is not None else (self.env.height / 2)**2
        var_theta = ((theta_max - theta_min) / 2)**2 if theta_range is not None else np.pi**2
        
        self.sigma = np.diag([var_x, var_y, var_theta])
    
    def update_motion(self, dx, dy, dtheta=0):
        """
        Update belief based on robot motion using the EKF prediction step.
        
        Args:
            dx: Change in x position
            dy: Change in y position
            dtheta: Change in orientation (radians)
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Prediction step of the EKF
        # 1. Update the mean using the motion model
        theta = self.mu[2]
        
        # Nonlinear motion model
        # We assume the robot first rotates to face the direction of movement,
        # then moves forward, then adjusts its final orientation
        if np.abs(dx) > 1e-6 or np.abs(dy) > 1e-6:
            # Calculate the direction of movement
            movement_dir = np.arctan2(dy, dx)
            
            # Calculate the rotation needed for movement
            rot1 = self._normalize_angle(movement_dir - theta)
            
            # Calculate the distance moved
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate the final rotation
            rot2 = self._normalize_angle(dtheta - rot1)
            
            # Apply motion model with these decomposed actions
            x_new = self.mu[0] + distance * np.cos(self.mu[2] + rot1)
            y_new = self.mu[1] + distance * np.sin(self.mu[2] + rot1)
            theta_new = self._normalize_angle(self.mu[2] + rot1 + rot2)
        else:
            # Only rotation, no movement
            x_new = self.mu[0]
            y_new = self.mu[1]
            theta_new = self._normalize_angle(self.mu[2] + dtheta)
        
        # Update mean
        self.mu = np.array([x_new, y_new, theta_new])
        
        # 2. Calculate the Jacobian of the motion model
        G = np.eye(3)
        if np.abs(dx) > 1e-6 or np.abs(dy) > 1e-6:
            G[0, 2] = -distance * np.sin(self.mu[2] + rot1)
            G[1, 2] = distance * np.cos(self.mu[2] + rot1)
        
        # 3. Update the covariance
        self.sigma = G @ self.sigma @ G.T + self.R
    
    def _normalize_angle(self, angle):
        """Normalize angle to be in [-pi, pi]"""
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    
    def update_measurement(self, distances, bearings):
        """
        Update belief based on landmark measurements using the EKF correction step.
        
        Args:
            distances: Distances to landmarks
            bearings: Bearings to landmarks
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Extract current state estimate
        x, y, theta = self.mu
        
        # For each landmark, perform a separate measurement update
        for i, (distance, bearing) in enumerate(zip(distances, bearings)):
            # Get landmark position
            lm_x, lm_y = self.landmarks[i]
            
            # 1. Calculate expected measurement given current state
            dx = lm_x - x
            dy = lm_y - y
            expected_distance = np.sqrt(dx**2 + dy**2)
            expected_bearing = self._normalize_angle(np.arctan2(dy, dx) - theta)
            
            # 2. Calculate measurement residual (innovation)
            z = np.array([distance, bearing])
            expected_z = np.array([expected_distance, expected_bearing])
            y_diff = z - expected_z
            
            # Normalize bearing difference
            y_diff[1] = self._normalize_angle(y_diff[1])
            
            # 3. Calculate Jacobian of measurement model
            H = np.zeros((2, 3))
            # Distance jacobian
            H[0, 0] = -dx / expected_distance
            H[0, 1] = -dy / expected_distance
            H[0, 2] = 0
            # Bearing jacobian
            H[1, 0] = dy / (dx**2 + dy**2)
            H[1, 1] = -dx / (dx**2 + dy**2)
            H[1, 2] = -1
            
            # 4. Calculate Kalman gain
            S = H @ self.sigma @ H.T + self.Q
            K = self.sigma @ H.T @ np.linalg.inv(S)
            
            # 5. Update state estimate
            self.mu = self.mu + K @ y_diff
            self.mu[2] = self._normalize_angle(self.mu[2])  # Normalize theta
            
            # 6. Update covariance
            self.sigma = (np.eye(3) - K @ H) @ self.sigma
    
    def get_belief(self):
        """
        Get the current belief state.
        
        Returns:
            tuple of (mu, sigma) representing the Gaussian belief
        """
        return self.mu, self.sigma
    
    def get_estimate(self):
        """
        Get the best estimate of the robot's pose.
        
        Returns:
            tuple of (x, y, theta) representing the estimated pose
        """
        if self.mu is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        return self.mu[0], self.mu[1], self.mu[2]
    
    def visualize_belief(self, ax=None):
        """
        Visualize the Gaussian belief state.
        
        Args:
            ax: Matplotlib axis to draw on (creates a new one if None)
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Belief state not initialized. Call initialize() first.")
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the 2D position distribution (marginalizing over theta)
        mu_pos = self.mu[:2]
        sigma_pos = self.sigma[:2, :2]
        
        # Create a grid for evaluation
        x_grid = np.linspace(0, self.env.width, 100)
        y_grid = np.linspace(0, self.env.height, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Reshape grid points to 2D points
        pos = np.dstack((X, Y))
        
        # Evaluate the multivariate Gaussian at each point
        Z = multivariate_normal.pdf(pos, mean=mu_pos, cov=sigma_pos)
        
        # Plot contours
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Probability Density')
        
        # Plot the mean position
        ax.plot(self.mu[0], self.mu[1], 'ro', markersize=10, label='Mean')
        
        # Draw the orientation
        theta = self.mu[2]
        dx = 0.3 * np.cos(theta)
        dy = 0.3 * np.sin(theta)
        ax.arrow(self.mu[0], self.mu[1], dx, dy, head_width=0.1, head_length=0.2, fc='r', ec='r')
        
        # Draw 95% confidence ellipse
        lambda_, v = np.linalg.eig(sigma_pos)
        lambda_ = np.sqrt(lambda_)
        
        # Calculate ellipse parameters for 95% confidence (chi-squared with 2 DOF)
        chisquare_val = 5.991  # 95% confidence for 2 DOF
        ell_radius_x = np.sqrt(chisquare_val) * lambda_[0]
        ell_radius_y = np.sqrt(chisquare_val) * lambda_[1]
        ell_angle = np.arctan2(v[1, 0], v[0, 0])
        
        # Create ellipse patch
        ellipse = patches.Ellipse(
            (self.mu[0], self.mu[1]),
            width=2*ell_radius_x,
            height=2*ell_radius_y,
            angle=np.degrees(ell_angle),
            edgecolor='k',
            fc='none',
            lw=2,
            label='95% Confidence'
        )
        ax.add_patch(ellipse)
        
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
        
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title('Gaussian Localization Belief')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        return ax