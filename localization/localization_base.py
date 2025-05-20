import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

class Environment:
    """
    A 2D environment with landmarks for robot localization.
    """
    
    def __init__(self, width=10.0, height=10.0, num_landmarks=5, seed=None):
        """
        Initialize the environment.
        
        Args:
            width: Width of the environment
            height: Height of the environment
            num_landmarks: Number of landmarks to generate
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.num_landmarks = num_landmarks
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        # Generate random landmarks
        self.landmarks = np.random.uniform(
            low=[0, 0], 
            high=[width, height], 
            size=(num_landmarks, 2)
        )
        
        # Walls represented as line segments [x1, y1, x2, y2]
        self.walls = np.array([
            [0, 0, width, 0],      # Bottom wall
            [0, 0, 0, height],     # Left wall
            [width, 0, width, height],   # Right wall
            [0, height, width, height],  # Top wall
            # Add internal walls if needed
            [width/2, 0, width/2, height/2],  # Partial wall in the middle
        ])
    
    def get_landmarks(self):
        """
        Get the landmarks in the environment.
        
        Returns:
            numpy array of landmark coordinates (x, y)
        """
        return self.landmarks
    
    def get_walls(self):
        """
        Get the walls in the environment.
        
        Returns:
            numpy array of wall line segments [x1, y1, x2, y2]
        """
        return self.walls
    
    def measure_landmark_distances(self, robot_x, robot_y, add_noise=True, noise_sigma=0.1):
        """
        Measure distances to all landmarks from a robot position.
        
        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            add_noise: Whether to add Gaussian noise to measurements
            noise_sigma: Standard deviation of measurement noise
            
        Returns:
            numpy array of distances to landmarks
        """
        # Calculate true distances
        dx = self.landmarks[:, 0] - robot_x
        dy = self.landmarks[:, 1] - robot_y
        distances = np.sqrt(dx**2 + dy**2)
        
        # Add noise if required
        if add_noise:
            distances += np.random.normal(0, noise_sigma, size=distances.shape)
            
        return distances
    
    def measure_landmark_bearings(self, robot_x, robot_y, robot_theta, add_noise=True, noise_sigma=0.05):
        """
        Measure bearings to all landmarks from a robot position.
        
        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            robot_theta: Robot orientation (radians)
            add_noise: Whether to add Gaussian noise to measurements
            noise_sigma: Standard deviation of measurement noise
            
        Returns:
            numpy array of bearings to landmarks (radians)
        """
        # Calculate true bearings
        dx = self.landmarks[:, 0] - robot_x
        dy = self.landmarks[:, 1] - robot_y
        bearings = np.arctan2(dy, dx) - robot_theta
        
        # Normalize to [-pi, pi]
        bearings = np.mod(bearings + np.pi, 2 * np.pi) - np.pi
        
        # Add noise if required
        if add_noise:
            bearings += np.random.normal(0, noise_sigma, size=bearings.shape)
            bearings = np.mod(bearings + np.pi, 2 * np.pi) - np.pi
            
        return bearings
    
    def check_collision(self, x, y, radius=0.2):
        """
        Check if a position would cause a collision with walls.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Radius of the robot
            
        Returns:
            True if collision, False otherwise
        """
        # Check boundary collisions first
        if (x < radius or x > self.width - radius or 
            y < radius or y > self.height - radius):
            return True
        
        # Check collisions with internal walls
        for wall in self.walls:
            # Skip boundary walls which we already checked
            if ((wall[0] == 0 and wall[2] == 0) or 
                (wall[0] == self.width and wall[2] == self.width) or
                (wall[1] == 0 and wall[3] == 0) or
                (wall[1] == self.height and wall[3] == self.height)):
                continue
                
            # Check distance to line segment
            x1, y1, x2, y2 = wall
            A = x - x1
            B = y - y1
            C = x2 - x1
            D = y2 - y1
            
            # Length squared of the wall
            dot = A * C + B * D
            len_sq = C * C + D * D
            
            # Compute projection and clamp to segment
            param = max(0, min(1, dot / len_sq))
            
            # Closest point
            xx = x1 + param * C
            yy = y1 + param * D
            
            # Distance to closest point
            dx = x - xx
            dy = y - yy
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist < radius:
                return True
        
        return False
    
    def ray_cast(self, x, y, theta, max_range=5.0, angle_count=8):
        """
        Cast rays in different directions and return distances to obstacles.
        
        Args:
            x: Robot x position
            y: Robot y position
            theta: Robot orientation (radians)
            max_range: Maximum sensor range
            angle_count: Number of rays to cast
            
        Returns:
            numpy array of distances to obstacles
        """
        distances = np.zeros(angle_count)
        angles = np.linspace(
            theta - np.pi/2, 
            theta + np.pi/2, 
            angle_count
        )
        
        for i, angle in enumerate(angles):
            # Cast ray in direction angle
            distances[i] = self._cast_single_ray(x, y, angle, max_range)
        
        return distances
    
    def _cast_single_ray(self, x, y, angle, max_range):
        """
        Cast a single ray and return the distance to the nearest obstacle.
        
        Args:
            x: Starting x position
            y: Starting y position
            angle: Direction angle (radians)
            max_range: Maximum sensor range
            
        Returns:
            distance to nearest obstacle or max_range if none
        """
        ray_end_x = x + max_range * np.cos(angle)
        ray_end_y = y + max_range * np.sin(angle)
        
        min_distance = max_range
        
        # Check intersection with each wall
        for wall in self.walls:
            x1, y1, x2, y2 = wall
            x3, y3, x4, y4 = x, y, ray_end_x, ray_end_y
            
            # Compute determinants
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            # If lines are parallel
            if np.abs(den) < 1e-8:
                continue
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            
            # If intersection is within segments
            if 0 <= t <= 1 and 0 <= u <= 1:
                # Calculate intersection point
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                
                # Calculate distance
                dist = np.sqrt((ix - x)**2 + (iy - y)**2)
                
                # Update minimum distance
                if dist < min_distance:
                    min_distance = dist
        
        return min_distance
    
    def visualize(self, robot_x=None, robot_y=None, robot_theta=None, show_rays=False):
        """
        Visualize the environment, optionally showing the robot.
        
        Args:
            robot_x: Robot x position (optional)
            robot_y: Robot y position (optional)
            robot_theta: Robot orientation in radians (optional)
            show_rays: Whether to show rays cast from the robot
        """
        plt.figure(figsize=(10, 10))
        
        # Draw environment boundaries
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        
        # Draw walls
        for wall in self.walls:
            plt.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
        
        # Draw landmarks
        plt.scatter(
            self.landmarks[:, 0],
            self.landmarks[:, 1],
            c='g',
            s=100,
            marker='^',
            label='Landmarks'
        )
        
        # Draw robot if provided
        if robot_x is not None and robot_y is not None:
            # Draw robot body
            robot_circle = plt.Circle((robot_x, robot_y), 0.2, color='b', fill=True)
            plt.gca().add_patch(robot_circle)
            
            # Draw orientation indicator
            if robot_theta is not None:
                direction_x = robot_x + 0.3 * np.cos(robot_theta)
                direction_y = robot_y + 0.3 * np.sin(robot_theta)
                plt.plot([robot_x, direction_x], [robot_y, direction_y], 'k-', linewidth=2)
            
            # Draw rays if requested
            if show_rays and robot_theta is not None:
                ray_distances = self.ray_cast(robot_x, robot_y, robot_theta)
                angles = np.linspace(robot_theta - np.pi/2, robot_theta + np.pi/2, len(ray_distances))
                
                for distance, angle in zip(ray_distances, angles):
                    end_x = robot_x + distance * np.cos(angle)
                    end_y = robot_y + distance * np.sin(angle)
                    plt.plot([robot_x, end_x], [robot_y, end_y], 'r-', linewidth=1, alpha=0.5)
        
        plt.grid(True)
        plt.title('Environment with Landmarks')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        
        if robot_x is not None:
            plt.legend(['Walls', 'Landmarks', 'Robot'])
        else:
            plt.legend(['Walls', 'Landmarks'])
            
        plt.show()


class Robot:
    """
    A simple 2D robot model that can move in the environment.
    """
    
    def __init__(self, env, x=None, y=None, theta=None, motion_noise=0.1, turn_noise=0.05):
        """
        Initialize the robot.
        
        Args:
            env: The environment the robot is in
            x: Initial x position (random if None)
            y: Initial y position (random if None)
            theta: Initial orientation in radians (random if None)
            motion_noise: Standard deviation of motion noise
            turn_noise: Standard deviation of turn noise
        """
        self.env = env
        self.motion_noise = motion_noise
        self.turn_noise = turn_noise
        
        # Initialize position randomly if not provided
        if x is None or y is None:
            # Keep trying until we get a valid position
            while True:
                self.x = np.random.uniform(0, env.width)
                self.y = np.random.uniform(0, env.height)
                if not env.check_collision(self.x, self.y):
                    break
        else:
            self.x = x
            self.y = y
        
        # Initialize orientation randomly if not provided
        if theta is None:
            self.theta = np.random.uniform(0, 2 * np.pi)
        else:
            self.theta = theta
    
    def move(self, distance, turn_angle):
        """
        Move the robot by a distance and then turn by an angle.
        
        Args:
            distance: Distance to move forward
            turn_angle: Angle to turn (radians)
            
        Returns:
            tuple of (actual_distance, actual_turn) with noise applied
        """
        # Apply noise to motion
        actual_distance = distance + np.random.normal(0, self.motion_noise)
        actual_turn = turn_angle + np.random.normal(0, self.turn_noise)
        
        # Compute new position
        new_x = self.x + actual_distance * np.cos(self.theta)
        new_y = self.y + actual_distance * np.sin(self.theta)
        
        # Check if the new position is valid
        if not self.env.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
        
        # Update orientation
        self.theta += actual_turn
        # Normalize theta
        self.theta = np.mod(self.theta, 2 * np.pi)
        
        return actual_distance, actual_turn
    
    def get_measurements(self):
        """
        Get sensor measurements.
        
        Returns:
            tuple of (distances, bearings) to landmarks
        """
        distances = self.env.measure_landmark_distances(self.x, self.y)
        bearings = self.env.measure_landmark_bearings(self.x, self.y, self.theta)
        
        return distances, bearings
    
    def get_laser_scan(self):
        """
        Get laser scan measurements.
        
        Returns:
            array of distances to obstacles in different directions
        """
        return self.env.ray_cast(self.x, self.y, self.theta)
    
    def get_pose(self):
        """
        Get the current robot pose.
        
        Returns:
            tuple of (x, y, theta)
        """
        return self.x, self.y, self.theta


class Localization(ABC):
    """
    Abstract base class for localization algorithms.
    """
    
    def __init__(self, env):
        """
        Initialize the localization algorithm.
        
        Args:
            env: The environment
        """
        self.env = env
    
    @abstractmethod
    def initialize(self, x_range, y_range, theta_range=None):
        """
        Initialize the belief state of the localization algorithm.
        
        Args:
            x_range: Range of possible x positions as (min, max)
            y_range: Range of possible y positions as (min, max)
            theta_range: Range of possible orientations as (min, max)
        """
        pass
    
    @abstractmethod
    def update_motion(self, dx, dy, dtheta=0):
        """
        Update belief based on robot motion.
        
        Args:
            dx: Change in x position
            dy: Change in y position
            dtheta: Change in orientation (radians)
        """
        pass
    
    @abstractmethod
    def update_measurement(self, distances, bearings):
        """
        Update belief based on measurements.
        
        Args:
            distances: Distances to landmarks
            bearings: Bearings to landmarks
        """
        pass
    
    @abstractmethod
    def get_belief(self):
        """
        Get the current belief state.
        
        Returns:
            representation of the belief state (depends on implementation)
        """
        pass
    
    @abstractmethod
    def get_estimate(self):
        """
        Get the best estimate of the robot's pose.
        
        Returns:
            tuple of (x, y, theta) representing the estimated pose
        """
        pass
    
    @abstractmethod
    def visualize_belief(self, ax=None):
        """
        Visualize the belief state.
        
        Args:
            ax: Matplotlib axis to draw on (creates a new one if None)
        """
        pass