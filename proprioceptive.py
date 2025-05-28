import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz
import time
from abc import ABC, abstractmethod

class ProprioceptiveSensor(ABC):
    """
    Abstract base class for proprioceptive sensors, which measure
    the robot's own state (position, velocity, force, etc.)
    """
    
    def __init__(self, name, sampling_rate=100, noise_level=0.01):
        """
        Initialize the sensor with basic parameters.
        
        Args:
            name: Name identifier for the sensor
            sampling_rate: Data acquisition rate in Hz
            noise_level: Standard deviation of Gaussian noise as a fraction of range
        """
        self.name = name
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.last_reading_time = None
        self.data_buffer = []
        self.time_buffer = []
        self.buffer_size = int(5 * sampling_rate)  # Store 5 seconds of data
        
    @abstractmethod
    def read(self):
        """
        Read the current sensor value. Must be implemented by subclasses.
        
        Returns:
            Current sensor reading (implementation-dependent format)
        """
        pass
    
    def add_noise(self, value, sensor_range):
        """
        Add Gaussian noise to the sensor reading.
        
        Args:
            value: The clean sensor value
            sensor_range: The range of the sensor output for noise scaling
            
        Returns:
            The value with added noise
        """
        noise = np.random.normal(0, self.noise_level * sensor_range)
        return value + noise
    
    def update_buffer(self, value):
        """
        Update the sensor's internal data buffer with a new reading.
        
        Args:
            value: New sensor reading to add to the buffer
        """
        current_time = time.time()
        
        if self.last_reading_time is None:
            self.last_reading_time = current_time
            
        self.data_buffer.append(value)
        self.time_buffer.append(current_time)
        
        # Keep buffer at designated size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            self.time_buffer.pop(0)
            
        self.last_reading_time = current_time
        
    def get_buffer_data(self):
        """
        Retrieve the data from the sensor's buffer.
        
        Returns:
            Tuple of (timestamps, data) arrays
        """
        return np.array(self.time_buffer), np.array(self.data_buffer)
    
    def apply_lowpass_filter(self, cutoff_freq=10):
        """
        Apply a low-pass filter to the buffered data.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
            
        Returns:
            Filtered data array
        """
        if len(self.data_buffer) < 4:  # Need minimum data for filtering
            return np.array(self.data_buffer)
        
        # Design Butterworth filter
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        
        # Apply filter
        return signal.filtfilt(b, a, self.data_buffer)
    
    def apply_kalman_filter(self, process_variance=1e-4, measurement_variance=1e-2):
        """
        Apply a simple Kalman filter to the buffered data.
        
        Args:
            process_variance: Process noise variance
            measurement_variance: Measurement noise variance
            
        Returns:
            Filtered data array
        """
        if len(self.data_buffer) < 2:
            return np.array(self.data_buffer)
            
        # Initialize Kalman filter
        x_hat = self.data_buffer[0]  # Initial state estimate
        P = 1.0  # Initial estimate uncertainty
        
        # Storage for filtered data
        filtered_data = np.zeros(len(self.data_buffer))
        filtered_data[0] = x_hat
        
        # Process the data
        for i in range(1, len(self.data_buffer)):
            # Prediction step
            x_hat_minus = x_hat
            P_minus = P + process_variance
            
            # Update step
            K = P_minus / (P_minus + measurement_variance)
            x_hat = x_hat_minus + K * (self.data_buffer[i] - x_hat_minus)
            P = (1 - K) * P_minus
            
            filtered_data[i] = x_hat
            
        return filtered_data
    
    def detect_drift(self, threshold=0.1):
        """
        Detect if sensor is experiencing drift.
        
        Args:
            threshold: Drift threshold as fraction of range
            
        Returns:
            True if drift detected, False otherwise
        """
        if len(self.data_buffer) < 10:
            return False
            
        # Simple drift detection - check if there's a consistent trend
        # by comparing the average of the first and last quarters of the buffer
        quarter = len(self.data_buffer) // 4
        first_quarter_avg = np.mean(self.data_buffer[:quarter])
        last_quarter_avg = np.mean(self.data_buffer[-quarter:])
        
        drift = abs(last_quarter_avg - first_quarter_avg)
        return drift > threshold
    
    def plot_data(self, filtered=False, filter_type='lowpass'):
        """
        Plot the sensor data from the buffer.
        
        Args:
            filtered: Whether to show filtered data
            filter_type: Type of filter to apply ('lowpass' or 'kalman')
        """
        times, data = self.get_buffer_data()
        
        # Convert times to relative seconds from start
        if len(times) > 0:
            relative_times = times - times[0]
        else:
            relative_times = times
            
        plt.figure(figsize=(10, 6))
        plt.plot(relative_times, data, 'b-', label='Raw Data')
        
        if filtered and len(data) > 0:
            if filter_type == 'lowpass':
                filtered_data = self.apply_lowpass_filter()
                plt.plot(relative_times, filtered_data, 'r-', label='Lowpass Filtered')
            elif filter_type == 'kalman':
                filtered_data = self.apply_kalman_filter()
                plt.plot(relative_times, filtered_data, 'g-', label='Kalman Filtered')
                
        plt.title(f'{self.name} Sensor Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.legend()
        plt.grid(True)
        plt.show()


class JointPositionSensor(ProprioceptiveSensor):
    """
    Sensor for measuring joint position or angle.
    """
    
    def __init__(self, name, joint_index, joint_type='revolute', 
                 min_value=0, max_value=2*np.pi, sampling_rate=100, 
                 noise_level=0.01, resolution=0.01):
        """
        Initialize joint position sensor.
        
        Args:
            name: Sensor name
            joint_index: Index of the joint being measured
            joint_type: 'revolute' for angular or 'prismatic' for linear
            min_value: Minimum measurable value (radians or meters)
            max_value: Maximum measurable value (radians or meters)
            sampling_rate: Data acquisition rate in Hz
            noise_level: Noise level as fraction of range
            resolution: Smallest detectable change in position
        """
        super().__init__(name, sampling_rate, noise_level)
        self.joint_index = joint_index
        self.joint_type = joint_type
        self.min_value = min_value
        self.max_value = max_value
        self.range = max_value - min_value
        self.resolution = resolution
        self.current_position = 0.0
        
    def read(self, actual_position=None):
        """
        Read the current joint position.
        
        Args:
            actual_position: Real joint position (for simulation)
            
        Returns:
            Measured joint position (with noise and resolution effects)
        """
        # If actual position provided (simulation), use it
        if actual_position is not None:
            self.current_position = actual_position
            
        # Add noise to the reading
        noisy_position = self.add_noise(self.current_position, self.range)
        
        # Quantize to resolution
        quantized_position = np.round(noisy_position / self.resolution) * self.resolution
        
        # Clamp to sensor range
        clamped_position = np.clip(quantized_position, self.min_value, self.max_value)
        
        # Update buffer
        self.update_buffer(clamped_position)
        
        return clamped_position
    
    def calibrate(self, known_positions):
        """
        Calibrate the position sensor using known reference positions.
        
        Args:
            known_positions: List of (measured, actual) position pairs
            
        Returns:
            Calibration parameters (scale, offset)
        """
        measured = np.array([p[0] for p in known_positions])
        actual = np.array([p[1] for p in known_positions])
        
        # Linear regression to find scale and offset
        A = np.vstack([measured, np.ones(len(measured))]).T
        scale, offset = np.linalg.lstsq(A, actual, rcond=None)[0]
        
        print(f"Calibration results: scale={scale}, offset={offset}")
        return scale, offset
    
    def apply_calibration(self, reading, scale, offset):
        """
        Apply calibration to a sensor reading.
        
        Args:
            reading: Raw sensor reading
            scale: Calibration scale factor
            offset: Calibration offset
            
        Returns:
            Calibrated reading
        """
        return scale * reading + offset


class JointVelocitySensor(ProprioceptiveSensor):
    """
    Sensor for measuring joint velocity.
    """
    
    def __init__(self, name, joint_index, joint_type='revolute', 
                 max_velocity=10.0, sampling_rate=100, 
                 noise_level=0.02):
        """
        Initialize joint velocity sensor.
        
        Args:
            name: Sensor name
            joint_index: Index of the joint being measured
            joint_type: 'revolute' for angular or 'prismatic' for linear
            max_velocity: Maximum measurable velocity (rad/s or m/s)
            sampling_rate: Data acquisition rate in Hz
            noise_level: Noise level as fraction of range
        """
        super().__init__(name, sampling_rate, noise_level)
        self.joint_index = joint_index
        self.joint_type = joint_type
        self.max_velocity = max_velocity
        self.range = 2 * max_velocity  # Range is -max_velocity to +max_velocity
        self.current_velocity = 0.0
        self.position_buffer = []
        self.dt = 1.0 / sampling_rate
        
    def read(self, actual_velocity=None, position=None):
        """
        Read the current joint velocity.
        
        Args:
            actual_velocity: Real joint velocity (for direct simulation)
            position: Current joint position (for derivative-based simulation)
            
        Returns:
            Measured joint velocity
        """
        # Case 1: If actual velocity is provided directly
        if actual_velocity is not None:
            self.current_velocity = actual_velocity
        
        # Case 2: If position is provided, compute velocity by differentiation
        elif position is not None:
            self.position_buffer.append(position)
            
            # Keep buffer at reasonable size
            if len(self.position_buffer) > 10:
                self.position_buffer.pop(0)
                
            # Compute velocity by differentiation (simple 2-point if few samples)
            if len(self.position_buffer) >= 2:
                if len(self.position_buffer) == 2:
                    # Simple 2-point differentiation
                    self.current_velocity = (self.position_buffer[1] - self.position_buffer[0]) / self.dt
                else:
                    # Use central difference for better accuracy
                    self.current_velocity = (self.position_buffer[-1] - self.position_buffer[-3]) / (2 * self.dt)
        
        # Add noise to the reading
        noisy_velocity = self.add_noise(self.current_velocity, self.range)
        
        # Clamp to sensor range
        clamped_velocity = np.clip(noisy_velocity, -self.max_velocity, self.max_velocity)
        
        # Update buffer
        self.update_buffer(clamped_velocity)
        
        return clamped_velocity


class JointTorqueSensor(ProprioceptiveSensor):
    """
    Sensor for measuring joint torque or force.
    """
    
    def __init__(self, name, joint_index, joint_type='revolute', 
                 max_torque=100.0, sampling_rate=1000, 
                 noise_level=0.02, temperature_sensitivity=0.001):
        """
        Initialize joint torque/force sensor.
        
        Args:
            name: Sensor name
            joint_index: Index of the joint being measured
            joint_type: 'revolute' for torque or 'prismatic' for force
            max_torque: Maximum measurable torque/force (Nm or N)
            sampling_rate: Data acquisition rate in Hz
            noise_level: Noise level as fraction of range
            temperature_sensitivity: Sensitivity to temperature (fraction/°C)
        """
        super().__init__(name, sampling_rate, noise_level)
        self.joint_index = joint_index
        self.joint_type = joint_type
        self.max_torque = max_torque
        self.range = 2 * max_torque  # Range is -max_torque to +max_torque
        self.current_torque = 0.0
        self.temperature_sensitivity = temperature_sensitivity
        self.reference_temperature = 25.0  # °C
        self.current_temperature = self.reference_temperature
        
    def read(self, actual_torque=None, temperature=None):
        """
        Read the current joint torque/force.
        
        Args:
            actual_torque: Real joint torque/force (Nm or N)
            temperature: Current temperature (°C)
            
        Returns:
            Measured joint torque/force
        """
        # Update temperature if provided
        if temperature is not None:
            self.current_temperature = temperature
            
        # If actual torque provided (simulation), use it
        if actual_torque is not None:
            self.current_torque = actual_torque
            
        # Apply temperature effect
        temp_diff = self.current_temperature - self.reference_temperature
        temperature_effect = self.current_torque * self.temperature_sensitivity * temp_diff
        temperature_adjusted_torque = self.current_torque + temperature_effect
        
        # Add noise to the reading
        noisy_torque = self.add_noise(temperature_adjusted_torque, self.range)
        
        # Clamp to sensor range
        clamped_torque = np.clip(noisy_torque, -self.max_torque, self.max_torque)
        
        # Update buffer
        self.update_buffer(clamped_torque)
        
        return clamped_torque
    
    def estimate_external_torque(self, commanded_torque, joint_velocity, 
                                friction_coeff=0.1, inertia=0.01):
        """
        Estimate external torque by subtracting expected dynamic torques.
        
        Args:
            commanded_torque: Motor command torque
            joint_velocity: Current joint velocity
            friction_coeff: Viscous friction coefficient
            inertia: Joint inertia
            
        Returns:
            Estimated external torque
        """
        # Simple dynamic model: τ_measured = τ_commanded - τ_friction - τ_inertia
        friction_torque = friction_coeff * joint_velocity
        
        # Estimate acceleration from velocity changes
        if len(self.data_buffer) > 2:
            dt = self.time_buffer[-1] - self.time_buffer[-2]
            if dt > 0:
                joint_accel = (self.data_buffer[-1] - self.data_buffer[-2]) / dt
                inertial_torque = inertia * joint_accel
            else:
                inertial_torque = 0
        else:
            inertial_torque = 0
            
        # Calculate external torque
        expected_dynamic_torque = commanded_torque - friction_torque - inertial_torque
        external_torque = self.current_torque - expected_dynamic_torque
        
        return external_torque


class IMUSensor(ProprioceptiveSensor):
    """
    Inertial Measurement Unit (IMU) for proprioceptive sensing.
    """
    
    def __init__(self, name, sampling_rate=100, noise_level=0.01, 
                 gyro_drift=0.01, accel_bias=0.05):
        """
        Initialize the IMU sensor.
        
        Args:
            name: Sensor name
            sampling_rate: Data acquisition rate in Hz
            noise_level: Noise level as fraction of range
            gyro_drift: Gyroscope drift in deg/s
            accel_bias: Accelerometer bias in m/s²
        """
        super().__init__(name, sampling_rate, noise_level)
        self.gyro_drift = gyro_drift
        self.accel_bias = accel_bias
        
        # Sensor ranges
        self.accel_range = 16.0  # ±16g
        self.gyro_range = 2000.0  # ±2000 deg/s
        
        # Current values
        self.acceleration = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        
        # For integration
        self.last_gyro_time = None
        
        # Separate buffers for different measurements
        self.accel_buffer = []
        self.gyro_buffer = []
        self.orient_buffer = []
        
    def read(self, actual_accel=None, actual_gyro=None, actual_orient=None, dt=None):
        """
        Read the current IMU data.
        
        Args:
            actual_accel: Real acceleration vector [x,y,z] in m/s²
            actual_gyro: Real angular velocity vector [x,y,z] in deg/s
            actual_orient: Real orientation vector [roll,pitch,yaw] in degrees
            dt: Time step for integration (if None, computed from sampling rate)
            
        Returns:
            Dict with accelerometer, gyroscope, and orientation data
        """
        current_time = time.time()
        
        if dt is None:
            dt = 1.0 / self.sampling_rate
            
        # Update acceleration if provided
        if actual_accel is not None:
            self.acceleration = np.array(actual_accel)
            
            # Add bias and noise
            for i in range(3):
                bias = self.accel_bias * np.random.uniform(-1, 1)
                self.acceleration[i] += bias
                self.acceleration[i] = self.add_noise(self.acceleration[i], self.accel_range)
                
            self.accel_buffer.append(self.acceleration.copy())
            if len(self.accel_buffer) > self.buffer_size:
                self.accel_buffer.pop(0)
                
        # Update angular velocity if provided
        if actual_gyro is not None:
            self.angular_velocity = np.array(actual_gyro)
            
            # Add drift and noise
            for i in range(3):
                drift = self.gyro_drift * np.random.uniform(-1, 1)
                self.angular_velocity[i] += drift
                self.angular_velocity[i] = self.add_noise(self.angular_velocity[i], self.gyro_range)
                
            self.gyro_buffer.append(self.angular_velocity.copy())
            if len(self.gyro_buffer) > self.buffer_size:
                self.gyro_buffer.pop(0)
                
            # Integrate gyro to get orientation if we have time data
            if self.last_gyro_time is not None and actual_orient is None:
                gyro_dt = current_time - self.last_gyro_time
                self.orientation += self.angular_velocity * gyro_dt
                
                # Normalize yaw to [-180, 180]
                self.orientation[2] = ((self.orientation[2] + 180) % 360) - 180
                
            self.last_gyro_time = current_time
            
        # Update orientation if provided directly
        if actual_orient is not None:
            self.orientation = np.array(actual_orient)
            
            # Add noise
            for i in range(3):
                self.orientation[i] = self.add_noise(self.orientation[i], 360)
                
        self.orient_buffer.append(self.orientation.copy())
        if len(self.orient_buffer) > self.buffer_size:
            self.orient_buffer.pop(0)
            
        # Update main buffer with a combined reading
        combined_reading = np.concatenate((self.acceleration, self.angular_velocity, self.orientation))
        self.update_buffer(combined_reading)
        
        return {
            'acceleration': self.acceleration.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'orientation': self.orientation.copy()
        }
    
    def get_orientation_quaternion(self):
        """
        Convert Euler angles to quaternion orientation.
        
        Returns:
            Quaternion [w, x, y, z]
        """
        # Convert to radians
        roll, pitch, yaw = np.radians(self.orientation)
        
        # Compute quaternion
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def integrate_position(self, initial_velocity=None, initial_position=None):
        """
        Integrate acceleration twice to estimate position.
        Note: This is prone to drift and should only be used for short durations.
        
        Args:
            initial_velocity: Initial velocity vector [vx, vy, vz]
            initial_position: Initial position vector [x, y, z]
            
        Returns:
            Tuple of (velocity, position) arrays
        """
        if len(self.accel_buffer) < 2:
            return (np.zeros((1, 3)), np.zeros((1, 3)))
            
        # Convert list of vectors to array
        accel_array = np.array(self.accel_buffer)
        
        # Create time array
        dt = 1.0 / self.sampling_rate
        time_array = np.arange(len(accel_array)) * dt
        
        # First integration: acceleration -> velocity
        velocity = np.zeros((len(time_array), 3))
        for i in range(3):  # x, y, z components
            vel = cumtrapz(accel_array[:, i], time_array, initial=0)
            velocity[:, i] = vel
            
            # Apply initial velocity if provided
            if initial_velocity is not None:
                velocity[:, i] += initial_velocity[i]
        
        # Second integration: velocity -> position
        position = np.zeros((len(time_array), 3))
        for i in range(3):  # x, y, z components
            pos = cumtrapz(velocity[:, i], time_array, initial=0)
            position[:, i] = pos
            
            # Apply initial position if provided
            if initial_position is not None:
                position[:, i] += initial_position[i]
                
        return velocity, position
    
    def complementary_filter(self, alpha=0.98):
        """
        Apply complementary filter to combine gyroscope and accelerometer data
        for more stable orientation estimates.
        
        Args:
            alpha: Weight for gyroscope data (0-1)
            
        Returns:
            Filtered orientation [roll, pitch, yaw]
        """
        if len(self.accel_buffer) < 2 or len(self.gyro_buffer) < 2:
            return self.orientation
            
        # Get latest accelerometer reading
        accel = self.acceleration
        
        # Calculate roll and pitch from accelerometer (gravity vector)
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0:
            accel_roll = np.arctan2(accel[1], accel[2])
            accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        else:
            accel_roll = 0
            accel_pitch = 0
            
        # Convert to degrees
        accel_roll = np.degrees(accel_roll)
        accel_pitch = np.degrees(accel_pitch)
        
        # Combine with gyroscope data using complementary filter
        filtered_roll = alpha * self.orientation[0] + (1 - alpha) * accel_roll
        filtered_pitch = alpha * self.orientation[1] + (1 - alpha) * accel_pitch
        
        # Yaw can't be determined from accelerometer, so we keep the gyro-integrated value
        filtered_yaw = self.orientation[2]
        
        return np.array([filtered_roll, filtered_pitch, filtered_yaw])


class StrainGaugeSensor(ProprioceptiveSensor):
    """
    Strain gauge sensor for measuring deformation/strain in structural elements.
    """
    
    def __init__(self, name, location, orientation='axial', 
                 gauge_factor=2.0, sampling_rate=1000, noise_level=0.005,
                 temperature_sensitivity=0.0001):
        """
        Initialize strain gauge sensor.
        
        Args:
            name: Sensor name
            location: Placement location identifier
            orientation: 'axial', 'torsional', or 'biaxial'
            gauge_factor: Sensitivity of the strain gauge
            sampling_rate: Data acquisition rate in Hz
            noise_level: Noise level as fraction of range
            temperature_sensitivity: Sensitivity to temperature (fraction/°C)
        """
        super().__init__(name, sampling_rate, noise_level)
        self.location = location
        self.orientation = orientation
        self.gauge_factor = gauge_factor
        self.temperature_sensitivity = temperature_sensitivity
        self.reference_temperature = 25.0  # °C
        self.current_temperature = self.reference_temperature
        
        # Strain measurement range
        self.max_strain = 0.01  # 1% strain (10,000 microstrain)
        self.range = 2 * self.max_strain  # Range is -max_strain to +max_strain
        
        # Current values
        if orientation == 'biaxial':
            self.strain = np.zeros(2)  # [axial, transverse]
        else:
            self.strain = 0.0
        
    def read(self, actual_strain=None, temperature=None):
        """
        Read the current strain.
        
        Args:
            actual_strain: Real strain value(s)
            temperature: Current temperature (°C)
            
        Returns:
            Measured strain value(s)
        """
        # Update temperature if provided
        if temperature is not None:
            self.current_temperature = temperature
            
        # If actual strain provided (simulation), use it
        if actual_strain is not None:
            self.strain = actual_strain
            
        # Apply temperature effect
        temp_diff = self.current_temperature - self.reference_temperature
        
        if isinstance(self.strain, np.ndarray):
            # Biaxial strain
            temperature_effect = self.strain * self.temperature_sensitivity * temp_diff
            temperature_adjusted_strain = self.strain + temperature_effect
            
            # Add noise to each component
            noisy_strain = np.zeros_like(self.strain)
            for i in range(len(self.strain)):
                noisy_strain[i] = self.add_noise(temperature_adjusted_strain[i], self.range)
                
            # Clamp to sensor range
            clamped_strain = np.clip(noisy_strain, -self.max_strain, self.max_strain)
        else:
            # Single strain value
            temperature_effect = self.strain * self.temperature_sensitivity * temp_diff
            temperature_adjusted_strain = self.strain + temperature_effect
            
            # Add noise
            noisy_strain = self.add_noise(temperature_adjusted_strain, self.range)
            
            # Clamp to sensor range
            clamped_strain = np.clip(noisy_strain, -self.max_strain, self.max_strain)
            
        # Update buffer
        self.update_buffer(clamped_strain)
        
        return clamped_strain
    
    def calculate_stress(self, youngs_modulus=200e9, poissons_ratio=0.3):
        """
        Calculate stress from strain (Hooke's Law).
        
        Args:
            youngs_modulus: Young's modulus of the material (Pa)
            poissons_ratio: Poisson's ratio of the material
            
        Returns:
            Stress value(s) in Pa
        """
        if isinstance(self.strain, np.ndarray) and len(self.strain) == 2:
            # Biaxial strain -> plane stress
            strain_x, strain_y = self.strain
            stress_x = (youngs_modulus / (1 - poissons_ratio**2)) * (strain_x + poissons_ratio * strain_y)
            stress_y = (youngs_modulus / (1 - poissons_ratio**2)) * (strain_y + poissons_ratio * strain_x)
            return np.array([stress_x, stress_y])
        else:
            # Uniaxial strain -> uniaxial stress
            return youngs_modulus * self.strain
    
    def calculate_force(self, cross_sectional_area=0.001, youngs_modulus=200e9):
        """
        Calculate force from strain and cross-sectional area.
        
        Args:
            cross_sectional_area: Cross-sectional area in m²
            youngs_modulus: Young's modulus of the material (Pa)
            
        Returns:
            Force in N
        """
        if self.orientation == 'axial':
            stress = youngs_modulus * self.strain
            return stress * cross_sectional_area
        else:
            return None  # Not applicable for other orientations


class ProprioceptiveSensorFusion:
    """
    Sensor fusion for combining data from multiple proprioceptive sensors.
    """
    
    def __init__(self, sensors=None):
        """
        Initialize sensor fusion system.
        
        Args:
            sensors: List of ProprioceptiveSensor instances
        """
        self.sensors = sensors if sensors is not None else []
        self.joint_states = {}  # Dict to store joint state estimates
        
    def add_sensor(self, sensor):
        """
        Add a sensor to the fusion system.
        
        Args:
            sensor: ProprioceptiveSensor instance
        """
        self.sensors.append(sensor)
        
    def update_joint_states(self):
        """
        Update joint state estimates using all available sensor data.
        
        Returns:
            Dictionary of joint states
        """
        # Process each sensor
        for sensor in self.sensors:
            # Extract joint information from position, velocity, and torque sensors
            if isinstance(sensor, JointPositionSensor):
                joint_id = sensor.joint_index
                if joint_id not in self.joint_states:
                    self.joint_states[joint_id] = {'position': None, 'velocity': None, 'torque': None}
                    
                # Update position estimate
                self.joint_states[joint_id]['position'] = sensor.data_buffer[-1] if sensor.data_buffer else None
                
            elif isinstance(sensor, JointVelocitySensor):
                joint_id = sensor.joint_index
                if joint_id not in self.joint_states:
                    self.joint_states[joint_id] = {'position': None, 'velocity': None, 'torque': None}
                    
                # Update velocity estimate
                self.joint_states[joint_id]['velocity'] = sensor.data_buffer[-1] if sensor.data_buffer else None
                
            elif isinstance(sensor, JointTorqueSensor):
                joint_id = sensor.joint_index
                if joint_id not in self.joint_states:
                    self.joint_states[joint_id] = {'position': None, 'velocity': None, 'torque': None}
                    
                # Update torque estimate
                self.joint_states[joint_id]['torque'] = sensor.data_buffer[-1] if sensor.data_buffer else None
                
        return self.joint_states
    
    def kalman_fusion(self, joint_id):
        """
        Apply Kalman fusion to joint position and velocity estimates.
        
        Args:
            joint_id: Joint identifier
            
        Returns:
            Fused state [position, velocity]
        """
        if joint_id not in self.joint_states:
            return None
            
        # Simple 2-state Kalman filter for position and velocity
        joint_state = self.joint_states[joint_id]
        
        # Need both position and velocity measurements
        if joint_state['position'] is None or joint_state['velocity'] is None:
            return None
            
        # State transition matrix for constant velocity model
        dt = 0.01  # Assumed time step
        F = np.array([[1, dt], [0, 1]])
        
        # Observation matrix
        H = np.eye(2)
        
        # Process noise covariance
        Q = np.array([[0.01, 0], [0, 0.1]])
        
        # Measurement noise covariance
        R = np.array([[0.1, 0], [0, 0.2]])
        
        # Initial state is current measurements
        x = np.array([joint_state['position'], joint_state['velocity']])
        
        # Initial covariance
        P = np.eye(2)
        
        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Measurement
        z = np.array([joint_state['position'], joint_state['velocity']])
        
        # Update step
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x_updated = x_pred + K @ y
        P_updated = (np.eye(2) - K @ H) @ P_pred
        
        # Store updated state as the fused estimate
        fused_position, fused_velocity = x_updated
        
        # Update joint state
        self.joint_states[joint_id]['position'] = fused_position
        self.joint_states[joint_id]['velocity'] = fused_velocity
        
        return x_updated
    
    def detect_sensor_faults(self, threshold=3.0):
        """
        Detect sensor faults using data consistency checks.
        
        Args:
            threshold: Fault detection threshold (standard deviations)
            
        Returns:
            Dictionary of detected faults
        """
        faults = {}
        
        for sensor in self.sensors:
            if len(sensor.data_buffer) < 10:
                continue
                
            # Get sensor data
            data = np.array(sensor.data_buffer)
            
            # Check for stuck sensor
            if np.std(data) < 1e-6:
                faults[sensor.name] = 'stuck'
                continue
                
            # Check for outliers
            mean_val = np.mean(data)
            std_val = np.std(data)
            z_scores = np.abs((data - mean_val) / std_val)
            
            if np.any(z_scores > threshold):
                faults[sensor.name] = 'outlier'
                continue
                
            # Check for drift
            if sensor.detect_drift():
                faults[sensor.name] = 'drift'
                continue
                
        return faults


# Example usage

def simulate_robot_joint():
    """
    Simulate a robot joint with various proprioceptive sensors.
    """
    print("Simulating robot joint with proprioceptive sensors...")
    
    # Create sensors
    position_sensor = JointPositionSensor("Joint 1 Encoder", joint_index=0, 
                                         noise_level=0.02, resolution=0.001)
    
    velocity_sensor = JointVelocitySensor("Joint 1 Tachometer", joint_index=0, 
                                         max_velocity=5.0, noise_level=0.03)
    
    torque_sensor = JointTorqueSensor("Joint 1 Torque", joint_index=0, 
                                     max_torque=50.0, noise_level=0.02)
    
    imu_sensor = IMUSensor("Link 1 IMU", noise_level=0.01, gyro_drift=0.02)
    
    strain_sensor = StrainGaugeSensor("Link 1 Strain", location="shoulder_link", 
                                     noise_level=0.01)
    
    # Create sensor fusion system
    fusion = ProprioceptiveSensorFusion([position_sensor, velocity_sensor, 
                                         torque_sensor, imu_sensor, strain_sensor])
    
    # Simulate joint motion (simple sine wave)
    dt = 0.01
    simulation_time = 5.0  # seconds
    time_points = np.arange(0, simulation_time, dt)
    
    # Simulated joint angle trajectory
    frequency = 0.5
    amplitude = np.pi/4
    position_trajectory = amplitude * np.sin(2 * np.pi * frequency * time_points)
    
    # Velocity is the derivative of position
    velocity_trajectory = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * time_points)
    
    # Torque depends on acceleration and load
    acceleration_trajectory = -4 * np.pi**2 * frequency**2 * amplitude * np.sin(2 * np.pi * frequency * time_points)
    inertia = 0.5
    load_torque = 5.0 * np.sin(2 * np.pi * 0.25 * time_points)  # External load
    torque_trajectory = inertia * acceleration_trajectory + load_torque
    
    # Strain depends on torque
    strain_coefficient = 1e-6  # Simplified relationship
    strain_trajectory = strain_coefficient * torque_trajectory
    
    # IMU measurements (simplistic model)
    gravity = 9.81
    accel_trajectory = np.zeros((len(time_points), 3))
    gyro_trajectory = np.zeros((len(time_points), 3))
    orient_trajectory = np.zeros((len(time_points), 3))
    
    for i, t in enumerate(time_points):
        # Accelerometer measures gravity and joint acceleration
        accel_trajectory[i, 0] = -gravity * np.sin(position_trajectory[i])
        accel_trajectory[i, 1] = gravity * np.cos(position_trajectory[i])
        accel_trajectory[i, 2] = acceleration_trajectory[i]
        
        # Gyroscope measures angular velocity
        gyro_trajectory[i, 0] = 0
        gyro_trajectory[i, 1] = 0
        gyro_trajectory[i, 2] = np.degrees(velocity_trajectory[i])
        
        # Orientation is the integral of angular velocity
        orient_trajectory[i, 0] = 0
        orient_trajectory[i, 1] = 0
        orient_trajectory[i, 2] = np.degrees(position_trajectory[i])
    
    # Data collection
    sensor_readings = []
    
    for i, t in enumerate(time_points):
        # Read from each sensor
        position = position_sensor.read(position_trajectory[i])
        velocity = velocity_sensor.read(velocity_trajectory[i])
        torque = torque_sensor.read(torque_trajectory[i])
        
        # IMU readings
        imu_data = imu_sensor.read(
            actual_accel=accel_trajectory[i],
            actual_gyro=gyro_trajectory[i],
            actual_orient=orient_trajectory[i],
            dt=dt
        )
        
        # Strain readings
        strain = strain_sensor.read(strain_trajectory[i])
        
        # Update fusion
        fusion.update_joint_states()
        
        # Store sensor readings
        sensor_readings.append({
            't': t,
            'position': position,
            'velocity': velocity,
            'torque': torque,
            'imu': imu_data,
            'strain': strain
        })
    
    # Apply filtering to the position sensor data
    filtered_position = position_sensor.apply_lowpass_filter(cutoff_freq=5)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Position
    plt.subplot(3, 2, 1)
    plt.plot(time_points, position_trajectory, 'b-', label='True Position')
    plt.plot([reading['t'] for reading in sensor_readings], 
             [reading['position'] for reading in sensor_readings], 
             'r.', alpha=0.5, label='Measured Position')
    plt.plot(time_points[:len(filtered_position)], filtered_position, 'g-', 
             label='Filtered Position')
    plt.legend()
    plt.title('Joint Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.grid(True)
    
    # Velocity
    plt.subplot(3, 2, 2)
    plt.plot(time_points, velocity_trajectory, 'b-', label='True Velocity')
    plt.plot([reading['t'] for reading in sensor_readings], 
             [reading['velocity'] for reading in sensor_readings], 
             'r.', alpha=0.5, label='Measured Velocity')
    plt.legend()
    plt.title('Joint Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.grid(True)
    
    # Torque
    plt.subplot(3, 2, 3)
    plt.plot(time_points, torque_trajectory, 'b-', label='True Torque')
    plt.plot([reading['t'] for reading in sensor_readings], 
             [reading['torque'] for reading in sensor_readings], 
             'r.', alpha=0.5, label='Measured Torque')
    plt.legend()
    plt.title('Joint Torque')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.grid(True)
    
    # IMU Angular Velocity
    plt.subplot(3, 2, 4)
    plt.plot(time_points, gyro_trajectory[:, 2], 'b-', label='True Angular Velocity')
    plt.plot([reading['t'] for reading in sensor_readings], 
             [reading['imu']['angular_velocity'][2] for reading in sensor_readings], 
             'r.', alpha=0.5, label='Measured Angular Velocity')
    plt.legend()
    plt.title('IMU Angular Velocity (Z-axis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.grid(True)
    
    # Strain
    plt.subplot(3, 2, 5)
    plt.plot(time_points, strain_trajectory, 'b-', label='True Strain')
    plt.plot([reading['t'] for reading in sensor_readings], 
             [reading['strain'] for reading in sensor_readings], 
             'r.', alpha=0.5, label='Measured Strain')
    plt.legend()
    plt.title('Link Strain')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.grid(True)
    
    # Fused state
    plt.subplot(3, 2, 6)
    
    # Apply Kalman fusion at each time step
    fused_positions = []
    fused_velocities = []
    
    for i, t in enumerate(time_points):
        if i % 10 == 0:  # Apply fusion every 10 steps for clarity
            fused_state = fusion.kalman_fusion(0)
            if fused_state is not None:
                fused_positions.append(fused_state[0])
                fused_velocities.append(fused_state[1])
                
    plt.plot(time_points[:len(fused_positions)], fused_positions, 'g-', label='Fused Position')
    plt.plot(time_points[:len(fused_velocities)], fused_velocities, 'm-', label='Fused Velocity')
    plt.legend()
    plt.title('Kalman Fusion Results')
    plt.xlabel('Time (s)')
    plt.ylabel('State Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Detect any sensor faults
    faults = fusion.detect_sensor_faults()
    if faults:
        print("Detected sensor faults:")
        for sensor_name, fault_type in faults.items():
            print(f"  {sensor_name}: {fault_type}")
    else:
        print("No sensor faults detected.")
    
    return fusion

if __name__ == "__main__":
    fusion = simulate_robot_joint()