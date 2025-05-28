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
        """
        pass
    
    @abstractmethod
    def update(self, measurement):
        """
        Update step: Update state estimate based on the received measurement.
        """
        pass
    
    @abstractmethod
    def estimate(self, measurement, control_input=None, dt=1.0):
        """
        Perform a complete estimation step (prediction + update).
        """
        pass


class UnscentedKalmanFilter(RecursiveStateEstimator):
    """
    Implementation of the Unscented Kalman Filter (UKF) for nonlinear systems.
    
    The UKF uses the unscented transform to deal with nonlinearities by propagating
    a set of carefully chosen sigma points through the nonlinear functions and then
    reconstructing the Gaussian distribution.
    
    State equation: x_k = f(x_{k-1}, u_k) + w_k, w_k ~ N(0, Q)
    Measurement equation: z_k = h(x_k) + v_k, v_k ~ N(0, R)
    """
    
    def __init__(self, state_dim, measurement_dim, alpha=0.1, beta=2.0, kappa=0.0):
        """
        Initialize the Unscented Kalman Filter.
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            alpha: Controls spread of sigma points (usually small positive value, e.g. 1e-3)
            beta: Prior knowledge about distribution (2 is optimal for Gaussian)
            kappa: Secondary parameter for spread (usually 0 or 3-state_dim)
        """
        # State dimensions
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Derived UKF parameters
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.n_sigma_points = 2 * self.state_dim + 1
        
        # Calculate weights
        self._calculate_weights()
        
        # State and covariance estimates
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)
        
        # Process and measurement noise covariances
        self.Q = np.eye(state_dim)           # Process noise covariance
        self.R = np.eye(measurement_dim)     # Measurement noise covariance
    
    def _calculate_weights(self):
        """
        Calculate weights for mean and covariance.
        """
        # Weights for mean
        self.weights_mean = np.zeros(self.n_sigma_points)
        self.weights_mean[0] = self.lambda_ / (self.state_dim + self.lambda_)
        
        # Weights for covariance
        self.weights_cov = np.zeros(self.n_sigma_points)
        self.weights_cov[0] = self.weights_mean[0] + (1 - self.alpha**2 + self.beta)
        
        # Weights for remaining sigma points
        for i in range(1, self.n_sigma_points):
            self.weights_mean[i] = 1.0 / (2 * (self.state_dim + self.lambda_))
            self.weights_cov[i] = self.weights_mean[i]
    
    def _generate_sigma_points(self):
        """
        Generate sigma points around the current state estimate.
        
        Returns:
            sigma_points: Array of sigma points [n_sigma_points x state_dim]
        """
        # Initialize sigma points matrix
        sigma_points = np.zeros((self.n_sigma_points, self.state_dim))
        
        # First sigma point is the current state
        sigma_points[0, :] = self.state.flatten()
        
        # Calculate square root of covariance matrix
        # We use the Cholesky decomposition for numerical stability
        sqrt_cov = np.linalg.cholesky((self.state_dim + self.lambda_) * self.covariance)
        
        # Generate the remaining sigma points
        for i in range(self.state_dim):
            # Positive direction
            sigma_points[i + 1, :] = self.state.flatten() + sqrt_cov[:, i]
            # Negative direction
            sigma_points[i + 1 + self.state_dim, :] = self.state.flatten() - sqrt_cov[:, i]
        
        return sigma_points
    
    def state_transition_function(self, state, control_input=None, dt=1.0):
        """
        Nonlinear state transition function f(x, u, dt).
        This method should be overridden by the user to define the system dynamics.
        
        Args:
            state: Current state (1D array)
            control_input: Control input (optional)
            dt: Time step
            
        Returns:
            next_state: Predicted next state (1D array)
        """
        # Default is identity (no change)
        return state.copy()
    
    def measurement_function(self, state):
        """
        Nonlinear measurement function h(x).
        This method should be overridden by the user to define the measurement model.
        
        Args:
            state: Current state (1D array)
            
        Returns:
            measurement: Predicted measurement (1D array)
        """
        # Default returns zeros
        return np.zeros(self.measurement_dim)
    
    def predict(self, control_input=None, dt=1.0):
        """
        Prediction step of the Unscented Kalman Filter.
        
        Args:
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            predicted_state: State prediction
            predicted_covariance: Covariance prediction
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Propagate sigma points through state transition function
        transformed_sigma_points = np.zeros_like(sigma_points)
        for i in range(self.n_sigma_points):
            transformed_sigma_points[i, :] = self.state_transition_function(
                sigma_points[i, :], control_input, dt
            )
        
        # Reconstruct mean and covariance
        # Predicted state mean
        self.state = np.zeros((self.state_dim, 1))
        for i in range(self.n_sigma_points):
            self.state += self.weights_mean[i] * transformed_sigma_points[i, :].reshape(-1, 1)
        
        # Predicted state covariance
        self.covariance = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.n_sigma_points):
            diff = (transformed_sigma_points[i, :] - self.state.flatten()).reshape(-1, 1)
            self.covariance += self.weights_cov[i] * diff @ diff.T
        
        # Add process noise
        self.covariance += self.Q
        
        return self.state, self.covariance
    
    def update(self, measurement):
        """
        Update step of the Unscented Kalman Filter.
        
        Args:
            measurement: Measurement vector from sensors
            
        Returns:
            updated_state: State estimate updated with the measurement
            updated_covariance: Covariance updated with the measurement
        """
        # Reshape measurement to column vector if needed
        measurement = np.atleast_2d(measurement).T if measurement.ndim == 1 else measurement
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Propagate sigma points through measurement function
        transformed_sigma_points = np.zeros((self.n_sigma_points, self.measurement_dim))
        for i in range(self.n_sigma_points):
            transformed_sigma_points[i, :] = self.measurement_function(sigma_points[i, :])
        
        # Predicted measurement mean
        predicted_measurement = np.zeros((self.measurement_dim, 1))
        for i in range(self.n_sigma_points):
            predicted_measurement += self.weights_mean[i] * transformed_sigma_points[i, :].reshape(-1, 1)
        
        # Innovation (measurement residual) covariance
        S = np.zeros((self.measurement_dim, self.measurement_dim))
        for i in range(self.n_sigma_points):
            diff = (transformed_sigma_points[i, :] - predicted_measurement.flatten()).reshape(-1, 1)
            S += self.weights_cov[i] * diff @ diff.T
        
        # Add measurement noise
        S += self.R
        
        # Cross correlation matrix
        cross_correlation = np.zeros((self.state_dim, self.measurement_dim))
        for i in range(self.n_sigma_points):
            diff_state = (sigma_points[i, :] - self.state.flatten()).reshape(-1, 1)
            diff_meas = (transformed_sigma_points[i, :] - predicted_measurement.flatten()).reshape(-1, 1)
            cross_correlation += self.weights_cov[i] * diff_state @ diff_meas.T
        
        # Kalman gain
        K = cross_correlation @ np.linalg.inv(S)
        
        # Update state
        self.state += K @ (measurement - predicted_measurement)
        
        # Update covariance
        self.covariance -= K @ S @ K.T
        
        return self.state, self.covariance
    
    def estimate(self, measurement, control_input=None, dt=1.0):
        """
        Perform a complete UKF estimation step (prediction + update).
        
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


class GaussianFilter(RecursiveStateEstimator):
    """
    A general Gaussian filter implementation that provides flexibility in
    how the Gaussian approximation is computed.
    
    This class allows you to choose between different methods to handle
    nonlinearities including:
    - Linear approximation (Kalman Filter)
    - Linearization (Extended Kalman Filter)
    - Sigma point approximation (Unscented Kalman Filter)
    - Quadrature approximation (Cubature Kalman Filter)
    
    All of these maintain a Gaussian belief of the state.
    """
    
    # Type definitions for approximation methods
    LINEAR = 'linear'  # Kalman Filter
    LINEARIZED = 'linearized'  # Extended Kalman Filter
    SIGMA_POINT = 'sigma_point'  # Unscented Kalman Filter
    CUBATURE = 'cubature'  # Cubature Kalman Filter
    
    def __init__(self, state_dim, measurement_dim, method=SIGMA_POINT):
        """
        Initialize the Gaussian Filter.
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            method: Approximation method to use (default: sigma_point)
        """
        # Validate method
        valid_methods = [self.LINEAR, self.LINEARIZED, self.SIGMA_POINT, self.CUBATURE]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # State dimensions
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.method = method
        
        # State and covariance estimates
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)
        
        # System matrices for linear case
        self.F = np.eye(state_dim)  # State transition matrix
        self.H = np.zeros((measurement_dim, state_dim))  # Measurement matrix
        
        # Noise covariances
        self.Q = np.eye(state_dim)  # Process noise covariance
        self.R = np.eye(measurement_dim)  # Measurement noise covariance
        
        # For UKF parameters
        self.alpha = 0.1  # Controls spread of sigma points
        self.beta = 2.0   # Optimal for Gaussian distributions
        self.kappa = 0.0  # Secondary sigma point parameter
        
        # Create underlying filter based on method
        if method == self.LINEAR:
            self._init_linear()
        elif method == self.LINEARIZED:
            self._init_linearized()
        elif method == self.SIGMA_POINT:
            self._init_sigma_point()
        elif method == self.CUBATURE:
            self._init_cubature()
    
    def _init_linear(self):
        """Initialize a linear Kalman Filter."""
        self.filter = None  # We'll implement directly
    
    def _init_linearized(self):
        """Initialize an Extended Kalman Filter."""
        self.filter = None  # We'll implement directly
    
    def _init_sigma_point(self):
        """Initialize an Unscented Kalman Filter."""
        self.filter = UnscentedKalmanFilter(
            self.state_dim, 
            self.measurement_dim,
            self.alpha,
            self.beta,
            self.kappa
        )
        self.filter.Q = self.Q
        self.filter.R = self.R
    
    def _init_cubature(self):
        """Initialize a Cubature Kalman Filter."""
        # This is a special case of UKF with specific parameters
        self.alpha = 1.0
        self.beta = 0.0
        self.kappa = 0.0
        self.filter = UnscentedKalmanFilter(
            self.state_dim, 
            self.measurement_dim,
            self.alpha,
            self.beta,
            self.kappa
        )
        self.filter.Q = self.Q
        self.filter.R = self.R
    
    def state_transition_function(self, state, control_input=None, dt=1.0):
        """
        Nonlinear state transition function f(x, u, dt).
        This method should be overridden by the user for nonlinear methods.
        
        Args:
            state: Current state
            control_input: Control input (optional)
            dt: Time step
            
        Returns:
            next_state: Predicted next state
        """
        if state.ndim == 1:
            state = state.reshape(-1, 1)
        
        # Default linear case
        if control_input is not None:
            return np.dot(self.F, state) + control_input
        else:
            return np.dot(self.F, state)
    
    def measurement_function(self, state):
        """
        Measurement function h(x).
        This method should be overridden by the user for nonlinear methods.
        
        Args:
            state: Current state
            
        Returns:
            measurement: Predicted measurement
        """
        if state.ndim == 1:
            state = state.reshape(-1, 1)
        
        # Default linear case
        return np.dot(self.H, state)
    
    def compute_jacobian_F(self, state, control_input=None, dt=1.0):
        """
        Compute the Jacobian of the state transition function.
        This method is used for the linearized (EKF) method.
        
        Args:
            state: Current state
            control_input: Control input (optional)
            dt: Time step
            
        Returns:
            F: Jacobian matrix of f() with respect to state
        """
        # Default returns the linear F matrix
        return self.F
    
    def compute_jacobian_H(self, state):
        """
        Compute the Jacobian of the measurement function.
        This method is used for the linearized (EKF) method.
        
        Args:
            state: Current state
            
        Returns:
            H: Jacobian matrix of h() with respect to state
        """
        # Default returns the linear H matrix
        return self.H
    
    def set_functions(self, f=None, h=None, jacobian_f=None, jacobian_h=None):
        """
        Set the state transition and measurement functions.
        
        Args:
            f: State transition function f(x, u, dt)
            h: Measurement function h(x)
            jacobian_f: Jacobian of f with respect to state
            jacobian_h: Jacobian of h with respect to state
        """
        if f is not None:
            self.state_transition_function = f
            if self.method == self.SIGMA_POINT or self.method == self.CUBATURE:
                self.filter.state_transition_function = f
        
        if h is not None:
            self.measurement_function = h
            if self.method == self.SIGMA_POINT or self.method == self.CUBATURE:
                self.filter.measurement_function = h
        
        if jacobian_f is not None:
            self.compute_jacobian_F = jacobian_f
        
        if jacobian_h is not None:
            self.compute_jacobian_H = jacobian_h
    
    def predict_linear(self, control_input=None, dt=1.0):
        """
        Prediction step for linear Kalman Filter.
        
        Args:
            control_input: Control input vector
            dt: Time step
            
        Returns:
            predicted_state: State prediction
            predicted_covariance: Covariance prediction
        """
        # Apply control input if available
        if control_input is not None:
            self.state = self.F @ self.state + control_input
        else:
            self.state = self.F @ self.state
        
        # Update covariance
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        return self.state, self.covariance
    
    def update_linear(self, measurement):
        """
        Update step for linear Kalman Filter.
        
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
        
        # Update covariance estimate (Joseph form)
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ self.H) @ self.covariance @ (I - K @ self.H).T + K @ self.R @ K.T
        
        return self.state, self.covariance
    
    def predict_linearized(self, control_input=None, dt=1.0):
        """
        Prediction step for Extended Kalman Filter.
        
        Args:
            control_input: Control input vector
            dt: Time step
            
        Returns:
            predicted_state: State prediction
            predicted_covariance: Covariance prediction
        """
        # Apply nonlinear state transition function
        self.state = np.atleast_2d(
            self.state_transition_function(self.state, control_input, dt)
        ).T
        
        # Compute Jacobian of state transition function
        F = self.compute_jacobian_F(self.state, control_input, dt)
        
        # Update covariance using the linearized model
        self.covariance = F @ self.covariance @ F.T + self.Q
        
        return self.state, self.covariance
    
    def update_linearized(self, measurement):
        """
        Update step for Extended Kalman Filter.
        
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
        expected_measurement = np.atleast_2d(self.measurement_function(self.state)).T
        y = measurement - expected_measurement
        
        # Calculate innovation covariance
        S = H @ self.covariance @ H.T + self.R
        
        # Calculate Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.state = self.state + K @ y
        
        # Update covariance estimate (Joseph form)
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ self.R @ K.T
        
        return self.state, self.covariance
    
    def predict(self, control_input=None, dt=1.0):
        """
        Prediction step of the filter.
        
        Args:
            control_input: Control input vector (can be None if no control)
            dt: Time step
            
        Returns:
            predicted_state: State prediction
            predicted_covariance: Covariance prediction
        """
        if self.method == self.LINEAR:
            return self.predict_linear(control_input, dt)
        elif self.method == self.LINEARIZED:
            return self.predict_linearized(control_input, dt)
        else:  # SIGMA_POINT or CUBATURE
            self.filter.state = self.state
            self.filter.covariance = self.covariance
            self.state, self.covariance = self.filter.predict(control_input, dt)
            return self.state, self.covariance
    
    def update(self, measurement):
        """
        Update step of the filter.
        
        Args:
            measurement: Measurement vector from sensors
            
        Returns:
            updated_state: State estimate updated with the measurement
            updated_covariance: Covariance updated with the measurement
        """
        if self.method == self.LINEAR:
            return self.update_linear(measurement)
        elif self.method == self.LINEARIZED:
            return self.update_linearized(measurement)
        else:  # SIGMA_POINT or CUBATURE
            self.filter.state = self.state
            self.filter.covariance = self.covariance
            self.state, self.covariance = self.filter.update(measurement)
            return self.state, self.covariance
    
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
        self.predict(control_input, dt)
        return self.update(measurement)


# Example: Target tracking with bearing and range measurements
class TargetTrackingUKF(UnscentedKalmanFilter):
    """
    UKF implementation for tracking a target using range and bearing measurements.
    The state is [x, y, vx, vy] where (x,y) is position and (vx,vy) is velocity.
    The measurement is [range, bearing] from a fixed sensor.
    """
    
    def __init__(self):
        # Initialize with state dimension 4 (x, y, vx, vy) and measurement dimension 2 (range, bearing)
        super().__init__(state_dim=4, measurement_dim=2)
        
        # Set initial state and covariance
        self.state = np.zeros((4, 1))
        self.covariance = np.eye(4)
        
        # Set noise covariances
        self.Q = np.diag([0.1, 0.1, 0.01, 0.01])  # Process noise
        self.R = np.diag([0.1, 0.01])  # Measurement noise (range, bearing)
    
    def state_transition_function(self, state, control_input=None, dt=1.0):
        """
        Nonlinear state transition: Constant velocity model.
        
        Args:
            state: Current state [x, y, vx, vy]
            control_input: Not used in this model
            dt: Time step
            
        Returns:
            next_state: Predicted next state
        """
        # Unpack state
        x, y, vx, vy = state if state.ndim == 1 else state.flatten()
        
        # Apply constant velocity model
        next_x = x + vx * dt
        next_y = y + vy * dt
        next_vx = vx  # Constant velocity
        next_vy = vy  # Constant velocity
        
        return np.array([next_x, next_y, next_vx, next_vy])
    
    def measurement_function(self, state):
        """
        Nonlinear measurement: Range and bearing from origin.
        
        Args:
            state: Current state [x, y, vx, vy]
            
        Returns:
            measurement: [range, bearing]
        """
        # Unpack state (only need position)
        x, y = state[0:2] if state.ndim == 1 else state.flatten()[0:2]
        
        # Calculate range and bearing
        range_val = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(y, x)
        
        return np.array([range_val, bearing])


def gaussian_filter_example():
    """
    Example of using the Gaussian Filter for target tracking with range and bearing measurements.
    Compares the performance of EKF, UKF, and CKF approaches.
    """
    # Time parameters
    dt = 0.1  # Time step
    time_steps = 100
    
    # Generate true trajectory (circular path)
    true_states = np.zeros((time_steps, 4))
    
    # Initialize at position (10, 0) with velocity (0, 2)
    true_states[0, :] = [10.0, 0.0, 0.0, 2.0]
    
    # Generate trajectory
    for t in range(1, time_steps):
        # Add slight random acceleration
        accel_x = 0.1 * np.random.randn()
        accel_y = 0.1 * np.random.randn()
        
        # Update velocity
        true_states[t, 2] = true_states[t-1, 2] + accel_x * dt
        true_states[t, 3] = true_states[t-1, 3] + accel_y * dt
        
        # Update position
        true_states[t, 0] = true_states[t-1, 0] + true_states[t, 2] * dt
        true_states[t, 1] = true_states[t-1, 1] + true_states[t, 3] * dt
    
    # Generate noisy measurements (range and bearing)
    measurements = np.zeros((time_steps, 2))
    for t in range(time_steps):
        # Get position
        x = true_states[t, 0]
        y = true_states[t, 1]
        
        # Calculate true range and bearing
        true_range = np.sqrt(x**2 + y**2)
        true_bearing = np.arctan2(y, x)
        
        # Add noise
        range_noise = 0.3 * np.random.randn()
        bearing_noise = 0.01 * np.random.randn()
        
        measurements[t, 0] = true_range + range_noise
        measurements[t, 1] = true_bearing + bearing_noise
    
    # Create filters for comparison
    # 1. Extended Kalman Filter
    ekf = GaussianFilter(state_dim=4, measurement_dim=2, method=GaussianFilter.LINEARIZED)
    
    # 2. Unscented Kalman Filter
    ukf = GaussianFilter(state_dim=4, measurement_dim=2, method=GaussianFilter.SIGMA_POINT)
    
    # 3. Cubature Kalman Filter
    ckf = GaussianFilter(state_dim=4, measurement_dim=2, method=GaussianFilter.CUBATURE)
    
    # Define the target tracking functions
    def f(state, control_input=None, dt=0.1):
        # State transition: constant velocity model
        x, y, vx, vy = state.flatten()
        next_x = x + vx * dt
        next_y = y + vy * dt
        next_vx = vx
        next_vy = vy
        return np.array([next_x, next_y, next_vx, next_vy]).reshape(-1, 1)
    
    def h(state):
        # Measurement: range and bearing
        x, y = state.flatten()[0:2]
        range_val = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(y, x)
        return np.array([range_val, bearing]).reshape(-1, 1)
    
    def jacobian_f(state, control_input=None, dt=0.1):
        # Jacobian of f with respect to state
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F
    
    def jacobian_h(state):
        # Jacobian of h with respect to state
        x, y = state.flatten()[0:2]
        r = np.sqrt(x**2 + y**2)
        
        H = np.zeros((2, 4))
        # Range derivatives
        H[0, 0] = x / r
        H[0, 1] = y / r
        # Bearing derivatives
        H[1, 0] = -y / (r**2)
        H[1, 1] = x / (r**2)
        
        return H
    
    # Set noise covariances
    Q = np.diag([0.01, 0.01, 0.1, 0.1])  # Process noise
    R = np.diag([0.1, 0.01])  # Measurement noise (range, bearing)
    
    for filter_ in [ekf, ukf, ckf]:
        filter_.Q = Q
        filter_.R = R
        
        # Set functions
        filter_.set_functions(f, h, jacobian_f, jacobian_h)
        
        # Initialize state with first measurement
        first_range = measurements[0, 0]
        first_bearing = measurements[0, 1]
        
        # Initialize position from first measurement
        init_x = first_range * np.cos(first_bearing)
        init_y = first_range * np.sin(first_bearing)
        
        # Initialize state with position and zero velocity
        filter_.state = np.array([[init_x], [init_y], [0.0], [0.0]])
        
        # Initial uncertainty
        filter_.covariance = np.diag([1.0, 1.0, 1.0, 1.0])
    
    # Run filters
    ekf_states = np.zeros((time_steps, 4))
    ukf_states = np.zeros((time_steps, 4))
    ckf_states = np.zeros((time_steps, 4))
    
    for t in range(time_steps):
        # Update all filters with current measurement
        ekf.estimate(measurements[t])
        ukf.estimate(measurements[t])
        ckf.estimate(measurements[t])
        
        # Store state estimates
        ekf_states[t] = ekf.state.flatten()
        ukf_states[t] = ukf.state.flatten()
        ckf_states[t] = ckf.state.flatten()
    
    # Calculate RMSE for position
    def calculate_rmse(estimated_states, true_states):
        # Calculate position errors
        errors = np.sqrt((estimated_states[:, 0] - true_states[:, 0])**2 + 
                         (estimated_states[:, 1] - true_states[:, 1])**2)
        return np.mean(errors)
    
    ekf_rmse = calculate_rmse(ekf_states, true_states)
    ukf_rmse = calculate_rmse(ukf_states, true_states)
    ckf_rmse = calculate_rmse(ckf_states, true_states)
    
    print(f"Position RMSE - EKF: {ekf_rmse:.4f}, UKF: {ukf_rmse:.4f}, CKF: {ckf_rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Trajectory plot
    plt.subplot(2, 1, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True')
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], 'b-', label=f'EKF (RMSE: {ekf_rmse:.4f})')
    plt.plot(ukf_states[:, 0], ukf_states[:, 1], 'g-', label=f'UKF (RMSE: {ukf_rmse:.4f})')
    plt.plot(ckf_states[:, 0], ckf_states[:, 1], 'r-', label=f'CKF (RMSE: {ckf_rmse:.4f})')
    plt.scatter(0, 0, c='m', marker='*', s=100, label='Sensor')
    
    # Plot range circles for context
    ranges = [5, 10, 15]
    for r in ranges:
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.legend()
    plt.title('Target Tracking: Comparison of Different Gaussian Filters')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True)
    
    # Position error plot
    plt.subplot(2, 1, 2)
    time = np.arange(time_steps) * dt
    
    # Calculate position errors over time
    ekf_error = np.sqrt((ekf_states[:, 0] - true_states[:, 0])**2 + 
                        (ekf_states[:, 1] - true_states[:, 1])**2)
    ukf_error = np.sqrt((ukf_states[:, 0] - true_states[:, 0])**2 + 
                        (ukf_states[:, 1] - true_states[:, 1])**2)
    ckf_error = np.sqrt((ckf_states[:, 0] - true_states[:, 0])**2 + 
                        (ckf_states[:, 1] - true_states[:, 1])**2)
    
    plt.plot(time, ekf_error, 'b-', label='EKF Error')
    plt.plot(time, ukf_error, 'g-', label='UKF Error')
    plt.plot(time, ckf_error, 'r-', label='CKF Error')
    plt.legend()
    plt.title('Position Estimation Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Run the example
if __name__ == "__main__":
    print("Running Gaussian Filter example...")
    gaussian_filter_example()