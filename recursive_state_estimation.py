import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class RecursiveStateEstimator(ABC):
    """
    Abstract base class for recursive state estimation algorithms.
    """
    
    @abstractmethod
    def predict(self, control_input=None, dt=1.0):
        """
        Prediction step: Update state and covariance predictions based on the system model.
        
        Args:
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            predicted_state: State prediction after applying the system model
            predicted_covariance: Covariance prediction after applying the system model
        """
        pass
    
    @abstractmethod
    def update(self, measurement):
        """
        Update step: Update state estimate based on the received measurement.
        
        Args:
            measurement: Measurement vector from sensors
            
        Returns:
            updated_state: State estimate updated with the measurement
            updated_covariance: Covariance updated with the measurement
        """
        pass
    
    @abstractmethod
    def estimate(self, measurement, control_input=None, dt=1.0):
        """
        Perform a complete estimation step (prediction + update).
        
        Args:
            measurement: Measurement vector from sensors
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            estimated_state: Updated state estimate
            estimated_covariance: Updated covariance estimate
        """
        pass


class KalmanFilter(RecursiveStateEstimator):
    """
    Implementation of the standard Kalman Filter for linear systems.
    The Kalman Filter is optimal for linear systems with Gaussian noise.
    
    State equation: x_k = F*x_{k-1} + B*u_k + w_k, w_k ~ N(0, Q)
    Measurement equation: z_k = H*x_k + v_k, v_k ~ N(0, R)
    
    where:
        x_k is the state vector at time k
        u_k is the control input at time k
        z_k is the measurement at time k
        F is the state transition matrix
        B is the control input matrix
        H is the measurement matrix
        w_k is the process noise with covariance Q
        v_k is the measurement noise with covariance R
    """
    
    def __init__(self, state_dim, measurement_dim, control_dim=0):
        """
        Initialize the Kalman Filter.
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            control_dim: Dimension of the control input vector (default: 0)
        """
        # State dimensions
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.control_dim = control_dim
        
        # State and covariance estimates
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)
        
        # System matrices (to be set by the user)
        self.F = np.eye(state_dim)  # State transition matrix
        self.H = np.zeros((measurement_dim, state_dim))  # Measurement matrix
        self.Q = np.eye(state_dim)  # Process noise covariance
        self.R = np.eye(measurement_dim)  # Measurement noise covariance
        self.B = np.zeros((state_dim, control_dim)) if control_dim > 0 else None  # Control input matrix
    
    def predict(self, control_input=None, dt=1.0):
        """
        Prediction step of the Kalman Filter.
        
        Args:
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            predicted_state: State prediction after applying the motion model
            predicted_covariance: Covariance prediction after applying the motion model
        """
        # Apply control input if available
        if control_input is not None and self.B is not None:
            self.state = self.F @ self.state + self.B @ control_input
        else:
            self.state = self.F @ self.state
        
        # Update covariance
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        return self.state, self.covariance
    
    def update(self, measurement):
        """
        Update step of the Kalman Filter.
        
        Args:
            measurement: Measurement vector from sensors
            
        Returns:
            updated_state: State estimate updated with the measurement
            updated_covariance: Covariance updated with the measurement
        """
        # Reshape measurement to column vector if needed
        measurement = np.atleast_2d(measurement).T if measurement.ndim == 1 else measurement
        
        # Calculate innovation (measurement residual)
        y = measurement - self.H @ self.state
        
        # Calculate innovation covariance
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # Calculate Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.state = self.state + K @ y
        
        # Update covariance estimate using Joseph form (more numerically stable)
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ self.H) @ self.covariance @ (I - K @ self.H).T + K @ self.R @ K.T
        
        return self.state, self.covariance
    
    def estimate(self, measurement, control_input=None, dt=1.0):
        """
        Perform a complete Kalman Filter estimation step (prediction + update).
        
        Args:
            measurement: Measurement vector from sensors
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            estimated_state: Updated state estimate
            estimated_covariance: Updated covariance estimate
        """
        self.predict(control_input, dt)
        return self.update(measurement)


class ExtendedKalmanFilter(RecursiveStateEstimator):
    """
    Implementation of the Extended Kalman Filter (EKF) for nonlinear systems.
    The EKF linearizes the system and measurement models at each step.
    
    State equation: x_k = f(x_{k-1}, u_k) + w_k, w_k ~ N(0, Q)
    Measurement equation: z_k = h(x_k) + v_k, v_k ~ N(0, R)
    
    where:
        x_k is the state vector at time k
        u_k is the control input at time k
        z_k is the measurement at time k
        f() is the nonlinear state transition function
        h() is the nonlinear measurement function
        w_k is the process noise with covariance Q
        v_k is the measurement noise with covariance R
    """
    
    def __init__(self, state_dim, measurement_dim):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
        """
        # State dimensions
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State and covariance estimates
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)
        
        # System matrices
        self.Q = np.eye(state_dim)  # Process noise covariance
        self.R = np.eye(measurement_dim)  # Measurement noise covariance
    
    def state_transition_function(self, state, control_input=None, dt=1.0):
        """
        Nonlinear state transition function f(x, u, dt).
        This method should be overridden by the user to define the system dynamics.
        
        Args:
            state: Current state estimate
            control_input: Control input (optional)
            dt: Time step
            
        Returns:
            next_state: Predicted next state
        """
        # Default is identity (no change)
        return state.copy()
    
    def measurement_function(self, state):
        """
        Nonlinear measurement function h(x).
        This method should be overridden by the user to define the measurement model.
        
        Args:
            state: Current state estimate
            
        Returns:
            measurement: Predicted measurement
        """
        # Default returns zeros
        return np.zeros((self.measurement_dim, 1))
    
    def compute_jacobian_F(self, state, control_input=None, dt=1.0):
        """
        Compute the Jacobian of the state transition function with respect to the state.
        This method can be overridden by the user to provide an analytical Jacobian.
        
        Args:
            state: Current state estimate
            control_input: Control input (optional)
            dt: Time step
            
        Returns:
            F: Jacobian matrix of f() with respect to state
        """
        # Numerical Jacobian computation
        F = np.eye(self.state_dim)
        epsilon = 1e-5
        
        for i in range(self.state_dim):
            state_plus = state.copy()
            state_plus[i, 0] += epsilon
            
            f_plus = self.state_transition_function(state_plus, control_input, dt)
            f = self.state_transition_function(state, control_input, dt)
            
            F[:, i] = ((f_plus - f) / epsilon).flatten()
        
        return F
    
    def compute_jacobian_H(self, state):
        """
        Compute the Jacobian of the measurement function with respect to the state.
        This method can be overridden by the user to provide an analytical Jacobian.
        
        Args:
            state: Current state estimate
            
        Returns:
            H: Jacobian matrix of h() with respect to state
        """
        # Numerical Jacobian computation
        H = np.zeros((self.measurement_dim, self.state_dim))
        epsilon = 1e-5
        
        for i in range(self.state_dim):
            state_plus = state.copy()
            state_plus[i, 0] += epsilon
            
            h_plus = self.measurement_function(state_plus)
            h = self.measurement_function(state)
            
            H[:, i] = ((h_plus - h) / epsilon).flatten()
        
        return H
    
    def predict(self, control_input=None, dt=1.0):
        """
        Prediction step of the Extended Kalman Filter.
        
        Args:
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            predicted_state: State prediction after applying the nonlinear model
            predicted_covariance: Covariance prediction after applying the linearized model
        """
        # Apply nonlinear state transition function
        self.state = self.state_transition_function(self.state, control_input, dt)
        
        # Compute Jacobian of state transition function
        F = self.compute_jacobian_F(self.state, control_input, dt)
        
        # Update covariance using the linearized model
        self.covariance = F @ self.covariance @ F.T + self.Q
        
        return self.state, self.covariance
    
    def update(self, measurement):
        """
        Update step of the Extended Kalman Filter.
        
        Args:
            measurement: Measurement vector from sensors
            
        Returns:
            updated_state: State estimate updated with the measurement
            updated_covariance: Covariance updated with the measurement
        """
        # Reshape measurement to column vector if needed
        measurement = np.atleast_2d(measurement).T if measurement.ndim == 1 else measurement
        
        # Compute Jacobian of measurement function
        H = self.compute_jacobian_H(self.state)
        
        # Calculate innovation (measurement residual)
        expected_measurement = self.measurement_function(self.state)
        y = measurement - expected_measurement
        
        # Calculate innovation covariance
        S = H @ self.covariance @ H.T + self.R
        
        # Calculate Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.state = self.state + K @ y
        
        # Update covariance estimate
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance
        
        return self.state, self.covariance
    
    def estimate(self, measurement, control_input=None, dt=1.0):
        """
        Perform a complete EKF estimation step (prediction + update).
        
        Args:
            measurement: Measurement vector from sensors
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            estimated_state: Updated state estimate
            estimated_covariance: Updated covariance estimate
        """
        self.predict(control_input, dt)
        return self.update(measurement)


class ParticleFilter(RecursiveStateEstimator):
    """
    Implementation of a Particle Filter for nonlinear systems with non-Gaussian noise.
    Particle Filters approximate the state distribution using a set of weighted samples (particles).
    
    State equation: x_k = f(x_{k-1}, u_k) + w_k
    Measurement equation: z_k = h(x_k) + v_k
    
    where:
        x_k is the state vector at time k
        u_k is the control input at time k
        z_k is the measurement at time k
        f() is the nonlinear state transition function
        h() is the nonlinear measurement function
        w_k is the process noise (not necessarily Gaussian)
        v_k is the measurement noise (not necessarily Gaussian)
    """
    
    def __init__(self, state_dim, num_particles=100):
        """
        Initialize the Particle Filter.
        
        Args:
            state_dim: Dimension of the state vector
            num_particles: Number of particles to use
        """
        # State dimensions
        self.state_dim = state_dim
        self.num_particles = num_particles
        
        # Particles and weights
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        
        # State estimate (mean of particles)
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)
        
        # Resampling threshold
        self.resampling_threshold = num_particles / 2
    
    def initialize_particles(self, mean, covariance):
        """
        Initialize particles from a Gaussian distribution.
        
        Args:
            mean: Mean of the initial state distribution
            covariance: Covariance of the initial state distribution
        """
        self.particles = np.random.multivariate_normal(
            mean.flatten(), covariance, size=self.num_particles
        )
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.update_state_estimate()
    
    def state_transition_function(self, state, control_input=None, dt=1.0, noise=None):
        """
        Nonlinear state transition function f(x, u, dt) with added noise.
        This method should be overridden by the user to define the system dynamics.
        
        Args:
            state: Current state
            control_input: Control input (optional)
            dt: Time step
            noise: Process noise (optional)
            
        Returns:
            next_state: Predicted next state
        """
        # Default is identity (no change) plus noise
        if noise is not None:
            return state + noise
        return state
    
    def measurement_function(self, state, noise=None):
        """
        Nonlinear measurement function h(x) with added noise.
        This method should be overridden by the user to define the measurement model.
        
        Args:
            state: Current state
            noise: Measurement noise (optional)
            
        Returns:
            measurement: Predicted measurement
        """
        # Default returns the state itself (assuming full state observation) plus noise
        if noise is not None:
            return state + noise
        return state
    
    def measurement_likelihood(self, measurement, predicted_measurement):
        """
        Calculate the likelihood of a measurement given a predicted measurement.
        This method can be overridden by the user to provide a custom likelihood function.
        
        Args:
            measurement: Actual measurement
            predicted_measurement: Predicted measurement from a particle
            
        Returns:
            likelihood: Likelihood value (higher is better match)
        """
        # Default is a Gaussian likelihood
        error = measurement - predicted_measurement
        return np.exp(-0.5 * np.sum(error**2))
    
    def predict(self, control_input=None, dt=1.0):
        """
        Prediction step of the Particle Filter.
        Propagates each particle through the state transition function.
        
        Args:
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            predicted_state: State prediction (mean of particles)
            predicted_covariance: Covariance prediction
        """
        # Propagate each particle through state transition function
        for i in range(self.num_particles):
            # Generate process noise
            process_noise = np.random.randn(self.state_dim)
            
            # Apply state transition function with noise
            self.particles[i] = self.state_transition_function(
                self.particles[i], control_input, dt, process_noise
            )
        
        # Update state estimate
        self.update_state_estimate()
        
        return self.state, self.covariance
    
    def update(self, measurement):
        """
        Update step of the Particle Filter.
        Updates particle weights based on measurement likelihood.
        
        Args:
            measurement: Measurement vector from sensors
            
        Returns:
            updated_state: State estimate updated with the measurement
            updated_covariance: Covariance updated with the measurement
        """
        # Reshape measurement to 1D array if needed
        measurement = measurement.flatten() if measurement.ndim > 1 else measurement
        
        # Update weights based on measurement likelihood
        for i in range(self.num_particles):
            # Predict measurement for this particle
            predicted_measurement = self.measurement_function(self.particles[i])
            
            # Calculate likelihood and update weight
            likelihood = self.measurement_likelihood(measurement, predicted_measurement)
            self.weights[i] *= likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resample if effective number of particles is too low
        if 1.0 / np.sum(self.weights**2) < self.resampling_threshold:
            self.resample()
        
        # Update state estimate
        self.update_state_estimate()
        
        return self.state, self.covariance
    
    def estimate(self, measurement, control_input=None, dt=1.0):
        """
        Perform a complete Particle Filter estimation step (prediction + update).
        
        Args:
            measurement: Measurement vector from sensors
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            estimated_state: Updated state estimate
            estimated_covariance: Updated covariance estimate
        """
        self.predict(control_input, dt)
        return self.update(measurement)
    
    def resample(self):
        """
        Resample particles based on their weights.
        This implements the importance sampling step of the particle filter.
        """
        # Cumulative sum of weights
        cumsum_weights = np.cumsum(self.weights)
        
        # Draw starting point
        start_idx = np.random.uniform(0, 1.0 / self.num_particles)
        
        # Indices to draw
        indices = np.zeros(self.num_particles, dtype=int)
        
        # Systematic resampling algorithm
        for i in range(self.num_particles):
            idx = start_idx + i / self.num_particles
            while idx > cumsum_weights[indices[i]]:
                indices[i] += 1
        
        # Resample particles
        self.particles = self.particles[indices, :]
        
        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def update_state_estimate(self):
        """
        Update state estimate and covariance based on particles and weights.
        """
        # Calculate weighted mean
        self.state = np.sum(self.particles * self.weights[:, np.newaxis], axis=0).reshape(-1, 1)
        
        # Calculate weighted covariance
        zero_mean_particles = self.particles - self.state.T
        self.covariance = np.zeros((self.state_dim, self.state_dim))
        
        for i in range(self.num_particles):
            particle_diff = zero_mean_particles[i].reshape(-1, 1)
            self.covariance += self.weights[i] * (particle_diff @ particle_diff.T)


# Example: Tracking a moving object in 1D
def kalman_filter_example():
    """
    Example of using Kalman Filter to track a 1D moving object.
    The state is [position, velocity] and we observe only noisy positions.
    """
    # Create a Kalman Filter
    kf = KalmanFilter(state_dim=2, measurement_dim=1)
    
    # Set up the system matrices
    dt = 0.1  # Time step
    kf.F = np.array([[1, dt], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement matrix (observe only position)
    kf.Q = np.array([[0.01, 0], [0, 0.01]])  # Process noise
    kf.R = np.array([[0.1]])  # Measurement noise
    
    # Initialize the state
    kf.state = np.array([[0], [1]])  # Start at position 0 with velocity 1
    
    # Generate true trajectory
    time_steps = 100
    true_positions = np.zeros(time_steps)
    true_velocities = np.zeros(time_steps)
    true_positions[0] = 0
    true_velocities[0] = 1
    
    for t in range(1, time_steps):
        # True dynamics: position += velocity * dt, velocity changes slightly
        true_velocities[t] = true_velocities[t-1] + 0.01 * np.random.randn()
        true_positions[t] = true_positions[t-1] + true_velocities[t] * dt
    
    # Generate noisy measurements
    measurements = true_positions + 0.1 * np.random.randn(time_steps)
    
    # Run Kalman Filter
    estimated_positions = np.zeros(time_steps)
    estimated_velocities = np.zeros(time_steps)
    position_std = np.zeros(time_steps)
    velocity_std = np.zeros(time_steps)
    
    for t in range(time_steps):
        # Update filter with measurement
        kf.estimate(measurements[t])
        
        # Store estimates
        estimated_positions[t] = kf.state[0, 0]
        estimated_velocities[t] = kf.state[1, 0]
        position_std[t] = np.sqrt(kf.covariance[0, 0])
        velocity_std[t] = np.sqrt(kf.covariance[1, 1])
    
    # Plot results
    time = np.arange(time_steps) * dt
    plt.figure(figsize=(15, 10))
    
    # Position plot
    plt.subplot(2, 1, 1)
    plt.plot(time, true_positions, 'b-', label='True Position')
    plt.plot(time, measurements, 'r.', label='Noisy Measurements')
    plt.plot(time, estimated_positions, 'g-', label='Estimated Position')
    plt.fill_between(time, 
                     estimated_positions - 2*position_std,
                     estimated_positions + 2*position_std,
                     color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('Kalman Filter: Position Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.grid(True)
    
    # Velocity plot
    plt.subplot(2, 1, 2)
    plt.plot(time, true_velocities, 'b-', label='True Velocity')
    plt.plot(time, estimated_velocities, 'g-', label='Estimated Velocity')
    plt.fill_between(time, 
                     estimated_velocities - 2*velocity_std,
                     estimated_velocities + 2*velocity_std,
                     color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('Kalman Filter: Velocity Estimation')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Example: Tracking a nonlinear system with EKF
class NonlinearSystem(ExtendedKalmanFilter):
    """
    Example of a nonlinear system for tracking a vehicle with EKF.
    The state is [x, y, theta, v] where (x,y) is position, theta is heading, v is velocity.
    The measurement is [range, bearing] from a fixed sensor at the origin.
    """
    
    def __init__(self):
        super().__init__(state_dim=4, measurement_dim=2)
        
        # Process and measurement noise
        self.Q = np.diag([0.1, 0.1, 0.01, 0.1])  # Process noise
        self.R = np.diag([0.1, 0.01])  # Measurement noise
    
    def state_transition_function(self, state, control_input=None, dt=1.0):
        """
        Nonlinear state transition: Constant velocity model with constant heading
        """
        x, y, theta, v = state.flatten()
        
        # Calculate next state
        next_x = x + v * np.cos(theta) * dt
        next_y = y + v * np.sin(theta) * dt
        next_theta = theta  # Constant heading
        next_v = v  # Constant velocity
        
        return np.array([[next_x], [next_y], [next_theta], [next_v]])
    
    def measurement_function(self, state):
        """
        Nonlinear measurement: Range and bearing from origin
        """
        x, y, _, _ = state.flatten()
        
        # Calculate range and bearing
        range_val = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(y, x)
        
        return np.array([[range_val], [bearing]])
    
    def compute_jacobian_F(self, state, control_input=None, dt=1.0):
        """
        Analytical Jacobian of the state transition function
        """
        _, _, theta, v = state.flatten()
        
        # Jacobian of state transition
        F = np.eye(4)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        
        return F
    
    def compute_jacobian_H(self, state):
        """
        Analytical Jacobian of the measurement function
        """
        x, y, _, _ = state.flatten()
        r = np.sqrt(x**2 + y**2)
        
        # Jacobian of measurement function
        H = np.zeros((2, 4))
        H[0, 0] = x / r
        H[0, 1] = y / r
        H[1, 0] = -y / (r**2)
        H[1, 1] = x / (r**2)
        
        return H


def extended_kalman_filter_example():
    """
    Example of using Extended Kalman Filter to track a vehicle with nonlinear dynamics.
    """
    # Create an EKF for the nonlinear system
    ekf = NonlinearSystem()
    
    # Initialize the state
    ekf.state = np.array([[10], [0], [np.pi/4], [1]])  # Position, heading, velocity
    
    # Time parameters
    dt = 0.1
    time_steps = 100
    
    # Generate true trajectory
    true_states = np.zeros((time_steps, 4))
    true_states[0] = ekf.state.flatten()
    
    for t in range(1, time_steps):
        # Update true state with some noise
        process_noise = np.sqrt(np.diag(ekf.Q)) * np.random.randn(4)
        true_states[t] = ekf.state_transition_function(true_states[t-1].reshape(-1, 1), dt=dt).flatten() + process_noise
    
    # Generate measurements
    measurements = np.zeros((time_steps, 2))
    for t in range(time_steps):
        # Get measurement with noise
        measurement_noise = np.sqrt(np.diag(ekf.R)) * np.random.randn(2)
        measurements[t] = ekf.measurement_function(true_states[t].reshape(-1, 1)).flatten() + measurement_noise
    
    # Run EKF
    estimated_states = np.zeros((time_steps, 4))
    state_std = np.zeros((time_steps, 4))
    
    for t in range(time_steps):
        # Update filter with measurement
        ekf.estimate(measurements[t], dt=dt)
        
        # Store estimates
        estimated_states[t] = ekf.state.flatten()
        state_std[t] = np.sqrt(np.diag(ekf.covariance))
    
    # Plot results
    time = np.arange(time_steps) * dt
    plt.figure(figsize=(15, 10))
    
    # XY trajectory plot
    plt.subplot(2, 2, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True Trajectory')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'g-', label='Estimated Trajectory')
    plt.scatter(0, 0, c='r', marker='*', s=100, label='Sensor Location')
    plt.legend()
    plt.title('EKF: Vehicle Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True)
    
    # X position over time
    plt.subplot(2, 2, 2)
    plt.plot(time, true_states[:, 0], 'b-', label='True X')
    plt.plot(time, estimated_states[:, 0], 'g-', label='Estimated X')
    plt.fill_between(time, 
                     estimated_states[:, 0] - 2*state_std[:, 0],
                     estimated_states[:, 0] + 2*state_std[:, 0],
                     color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('EKF: X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    plt.grid(True)
    
    # Y position over time
    plt.subplot(2, 2, 3)
    plt.plot(time, true_states[:, 1], 'b-', label='True Y')
    plt.plot(time, estimated_states[:, 1], 'g-', label='Estimated Y')
    plt.fill_between(time, 
                     estimated_states[:, 1] - 2*state_std[:, 1],
                     estimated_states[:, 1] + 2*state_std[:, 1],
                     color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('EKF: Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    # Heading over time
    plt.subplot(2, 2, 4)
    plt.plot(time, true_states[:, 2], 'b-', label='True Heading')
    plt.plot(time, estimated_states[:, 2], 'g-', label='Estimated Heading')
    plt.fill_between(time, 
                     estimated_states[:, 2] - 2*state_std[:, 2],
                     estimated_states[:, 2] + 2*state_std[:, 2],
                     color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('EKF: Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (rad)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Example: Particle Filter for robot localization
class RobotLocalizationPF(ParticleFilter):
    """
    Particle Filter for robot localization with landmarks.
    The state is [x, y, theta] and measurements are ranges and bearings to landmarks.
    """
    
    def __init__(self, num_particles=100, landmarks=None):
        super().__init__(state_dim=3, num_particles=num_particles)
        self.landmarks = landmarks if landmarks is not None else []
    
    def state_transition_function(self, state, control_input=None, dt=1.0, noise=None):
        """
        Robot motion model: [x, y, theta] with velocity and angular velocity control
        """
        x, y, theta = state
        
        if control_input is not None:
            v, omega = control_input  # Linear and angular velocity
        else:
            v, omega = 0, 0
        
        # Add control noise if provided
        if noise is not None:
            v += noise[0]
            omega += noise[1]
        
        # Motion model
        next_theta = theta + omega * dt
        next_x = x + v * np.cos(theta) * dt
        next_y = y + v * np.sin(theta) * dt
        
        return np.array([next_x, next_y, next_theta])
    
    def measurement_function(self, state, noise=None):
        """
        Measurement model: ranges and bearings to landmarks
        """
        x, y, theta = state
        
        # Calculate expected measurements to all landmarks
        measurements = []
        for lm_x, lm_y in self.landmarks:
            # Range
            dx = lm_x - x
            dy = lm_y - y
            range_val = np.sqrt(dx**2 + dy**2)
            
            # Bearing (relative to robot's heading)
            bearing = np.arctan2(dy, dx) - theta
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            
            measurements.extend([range_val, bearing])
        
        # Add measurement noise if provided
        if noise is not None:
            measurements = np.array(measurements) + noise
        
        return np.array(measurements)
    
    def measurement_likelihood(self, measurement, predicted_measurement):
        """
        Calculate likelihood of measurement given predicted measurement
        """
        # Calculate difference for each measure (ranges and bearings)
        diff = measurement - predicted_measurement
        
        # Normalize bearings (every second element starting from index 1)
        for i in range(1, len(diff), 2):
            diff[i] = (diff[i] + np.pi) % (2 * np.pi) - np.pi
        
        # Weighted likelihood: range errors and bearing errors
        range_sigma = 0.5  # Expected range standard deviation
        bearing_sigma = 0.1  # Expected bearing standard deviation
        
        likelihood = 1.0
        for i in range(0, len(diff), 2):
            # Range likelihood
            range_likelihood = np.exp(-0.5 * (diff[i]/range_sigma)**2)
            
            # Bearing likelihood
            bearing_likelihood = np.exp(-0.5 * (diff[i+1]/bearing_sigma)**2)
            
            # Combined likelihood
            likelihood *= range_likelihood * bearing_likelihood
        
        return likelihood


def particle_filter_example():
    """
    Example of using Particle Filter for robot localization with landmarks.
    """
    # Define landmarks (x, y)
    landmarks = np.array([[5, 0], [0, 5], [5, 5]])
    
    # Create Particle Filter
    pf = RobotLocalizationPF(num_particles=500, landmarks=landmarks)
    
    # Initialize particles (uniform distribution in state space)
    initial_mean = np.array([2, 2, 0])  # Starting location
    initial_cov = np.diag([0.5, 0.5, 0.1])  # Uncertainty in initial position
    pf.initialize_particles(initial_mean, initial_cov)
    
    # Time parameters
    dt = 0.1
    time_steps = 100
    
    # True trajectory
    true_states = np.zeros((time_steps, 3))
    true_states[0] = initial_mean
    
    # Control inputs (linear and angular velocity)
    controls = np.zeros((time_steps, 2))
    for t in range(time_steps):
        # Circular trajectory
        controls[t, 0] = 0.5  # Linear velocity
        controls[t, 1] = 0.1  # Angular velocity
    
    # Generate true trajectory
    for t in range(1, time_steps):
        # Apply control with some noise
        control_noise = np.array([0.01, 0.01]) * np.random.randn(2)
        true_states[t] = pf.state_transition_function(true_states[t-1], controls[t-1], dt, control_noise)
    
    # Generate measurements
    measurements = np.zeros((time_steps, 2 * len(landmarks)))
    for t in range(time_steps):
        # Generate noisy measurements
        measurement_noise = np.array([0.1, 0.05] * len(landmarks)) * np.random.randn(2 * len(landmarks))
        measurements[t] = pf.measurement_function(true_states[t]) + measurement_noise
    
    # Run Particle Filter
    estimated_states = np.zeros((time_steps, 3))
    state_std = np.zeros((time_steps, 3))
    
    # Store all particle history for visualization
    all_particles = np.zeros((time_steps, pf.num_particles, 3))
    
    for t in range(time_steps):
        # Apply control and update with measurement
        pf.estimate(measurements[t], controls[t-1] if t > 0 else controls[0], dt)
        
        # Store estimates
        estimated_states[t] = pf.state.flatten()
        state_std[t] = np.sqrt(np.diag(pf.covariance))
        
        # Store particles
        all_particles[t] = pf.particles
    
    # Plot results
    time = np.arange(time_steps) * dt
    plt.figure(figsize=(15, 10))
    
    # Trajectory plot
    plt.subplot(2, 2, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True Trajectory')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'g-', label='Estimated Trajectory')
    
    # Plot landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='^', s=100, label='Landmarks')
    
    # Plot particles at selected time steps
    for t in [0, 25, 50, 75, 99]:
        plt.scatter(all_particles[t, :, 0], all_particles[t, :, 1], s=1, alpha=0.3, label=f'Particles t={t}' if t == 0 else "")
    
    plt.legend()
    plt.title('Particle Filter: Robot Localization')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True)
    
    # X position over time
    plt.subplot(2, 2, 2)
    plt.plot(time, true_states[:, 0], 'b-', label='True X')
    plt.plot(time, estimated_states[:, 0], 'g-', label='Estimated X')
    plt.fill_between(time, 
                    estimated_states[:, 0] - 2*state_std[:, 0],
                    estimated_states[:, 0] + 2*state_std[:, 0],
                    color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('Particle Filter: X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    plt.grid(True)
    
    # Y position over time
    plt.subplot(2, 2, 3)
    plt.plot(time, true_states[:, 1], 'b-', label='True Y')
    plt.plot(time, estimated_states[:, 1], 'g-', label='Estimated Y')
    plt.fill_between(time, 
                    estimated_states[:, 1] - 2*state_std[:, 1],
                    estimated_states[:, 1] + 2*state_std[:, 1],
                    color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('Particle Filter: Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    # Heading over time
    plt.subplot(2, 2, 4)
    plt.plot(time, true_states[:, 2], 'b-', label='True Heading')
    plt.plot(time, estimated_states[:, 2], 'g-', label='Estimated Heading')
    plt.fill_between(time, 
                    estimated_states[:, 2] - 2*state_std[:, 2],
                    estimated_states[:, 2] + 2*state_std[:, 2],
                    color='g', alpha=0.2, label='2σ Confidence')
    plt.legend()
    plt.title('Particle Filter: Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (rad)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Main function to run all examples
if __name__ == "__main__":
    print("Running Kalman Filter example...")
    kalman_filter_example()
    
    print("Running Extended Kalman Filter example...")
    extended_kalman_filter_example()
    
    print("Running Particle Filter example...")
    particle_filter_example()
