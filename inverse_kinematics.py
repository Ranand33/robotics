import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class RobotLink:
    """
    A class representing a robot link, which connects two consecutive joints
    """
    def __init__(self, length, theta=0, alpha=0, d=0, a=0, offset=0, joint_type='revolute'):
        """
        Initialize a robot link using Denavit-Hartenberg parameters.
        
        Args:
            length: Length of the link
            theta: Joint angle (for revolute joints) or 0 (for prismatic joints)
            alpha: Twist angle
            d: Link offset
            a: Link length (same as length for standard DH)
            offset: Joint angle offset
            joint_type: 'revolute' or 'prismatic'
        """
        self.length = length
        self.theta = theta  # Joint angle
        self.alpha = alpha  # Twist angle
        self.d = d          # Link offset
        self.a = a          # Link length
        self.offset = offset  # Joint angle offset
        self.joint_type = joint_type  # 'revolute' or 'prismatic'
        
    def transform(self, q):
        """
        Calculate the transformation matrix for this link.
        
        Args:
            q: Joint value (angle for revolute joints, displacement for prismatic)
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        if self.joint_type == 'revolute':
            theta = q + self.offset
            d = self.d
        else:  # prismatic
            theta = self.theta + self.offset
            d = q
            
        # Denavit-Hartenberg transformation matrix
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, self.a*ct],
            [st, ct*ca, -ct*sa, self.a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])


class RobotArm:
    """
    A class representing a general robot arm with multiple links
    """
    def __init__(self, links, base_position=np.array([0, 0, 0])):
        """
        Initialize a robot arm.
        
        Args:
            links: List of RobotLink objects
            base_position: Position of the robot base
        """
        self.links = links
        self.num_joints = len(links)
        self.base_position = base_position
        
    def forward_kinematics(self, joint_values):
        """
        Calculate the forward kinematics for the robot.
        
        Args:
            joint_values: List of joint values (angles or displacements)
            
        Returns:
            Tuple containing:
                - List of 4x4 transformation matrices for each joint
                - List of 3D positions for each joint
        """
        T = np.eye(4)  # Identity matrix for the base
        transforms = [T]  # Start with base transform
        positions = [self.base_position]  # Start with base position
        
        for i, (link, q) in enumerate(zip(self.links, joint_values)):
            T = T @ link.transform(q)  # Apply joint transformation
            transforms.append(T)
            positions.append(self.base_position + T[:3, 3])  # Extract position from transform
            
        return transforms, positions
    
    def jacobian(self, joint_values):
        """
        Calculate the Jacobian matrix at the given joint values.
        
        Args:
            joint_values: List of joint values (angles or displacements)
            
        Returns:
            6xn Jacobian matrix where n is the number of joints
        """
        _, positions = self.forward_kinematics(joint_values)
        end_effector = positions[-1]
        jacobian = np.zeros((6, self.num_joints))
        
        # Calculate transformation matrices up to each joint
        transforms, _ = self.forward_kinematics(joint_values)
        
        for i in range(self.num_joints):
            if self.links[i].joint_type == 'revolute':
                # For revolute joints, the Jacobian consists of:
                # - Linear velocity component: z_{i-1} × (o_n - o_{i-1})
                # - Angular velocity component: z_{i-1}
                z_axis = transforms[i][:3, 2]  # z-axis of the ith joint frame
                joint_pos = positions[i]
                
                # Linear velocity component (cross product)
                jacobian[:3, i] = np.cross(z_axis, end_effector - joint_pos)
                # Angular velocity component
                jacobian[3:, i] = z_axis
            else:  # prismatic
                # For prismatic joints, the Jacobian consists of:
                # - Linear velocity component: z_{i-1}
                # - Angular velocity component: 0
                z_axis = transforms[i][:3, 2]  # z-axis of the ith joint frame
                
                # Linear velocity component
                jacobian[:3, i] = z_axis
                # Angular velocity component is zero
                jacobian[3:, i] = 0
                
        return jacobian
    
    def inverse_kinematics_analytical_2r(self, target_position):
        """
        Analytical inverse kinematics solution for a 2R planar robot.
        
        Args:
            target_position: 2D or 3D target position [x, y, (z)]
            
        Returns:
            Two possible solutions for joint values
        """
        if self.num_joints != 2:
            raise ValueError("Analytical IK for 2R robot requires exactly 2 joints")
        
        # Extract target x, y coordinates
        x = target_position[0]
        y = target_position[1]
        
        # Extract link lengths
        l1 = self.links[0].length
        l2 = self.links[1].length
        
        # Calculate target distance from base
        r = np.sqrt(x**2 + y**2)
        
        # Check if the target is reachable
        if r > l1 + l2 or r < abs(l1 - l2):
            raise ValueError(f"Target position ({x}, {y}) is not reachable by the 2R robot")
        
        # Calculate the second joint angle using the law of cosines
        cos_q2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if cos_q2 > 1 or cos_q2 < -1:
            raise ValueError("Target position is not reachable")
            
        # Two possible solutions for q2 (elbow up and elbow down)
        q2_1 = np.arccos(cos_q2)
        q2_2 = -q2_1
        
        # Calculate the first joint angle for both solutions
        q1_1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2_1), l1 + l2 * np.cos(q2_1))
        q1_2 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2_2), l1 + l2 * np.cos(q2_2))
        
        # Return both solutions
        return [[q1_1, q2_1], [q1_2, q2_2]]
    
    def inverse_kinematics_jacobian(self, target_position, target_orientation=None,
                                   max_iter=1000, tolerance=1e-3, alpha=0.5):
        """
        Jacobian-based inverse kinematics solver.
        
        Args:
            target_position: 3D target position [x, y, z]
            target_orientation: 3D target orientation as euler angles [roll, pitch, yaw] (optional)
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            alpha: Step size (learning rate)
            
        Returns:
            Joint values that achieve the target position (and orientation if specified)
        """
        # Initial joint values (current configuration)
        q = np.zeros(self.num_joints)
        
        # Target vector
        target = np.array(target_position)
        if target_orientation is not None:
            target = np.concatenate([target, target_orientation])
            use_orientation = True
            target_size = 6  # Position and orientation
        else:
            use_orientation = False
            target_size = 3  # Position only
        
        # Iteration loop
        for i in range(max_iter):
            # Calculate forward kinematics
            transforms, positions = self.forward_kinematics(q)
            current_position = positions[-1]
            
            # Calculate current end effector state
            current = current_position
            if use_orientation:
                # Extract orientation (roll, pitch, yaw) from transformation matrix
                T = transforms[-1]
                # This is a simplified extraction and might not work for all robot configurations
                # In a real implementation, you would use proper Euler angle extraction
                roll = np.arctan2(T[2, 1], T[2, 2])
                pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))
                yaw = np.arctan2(T[1, 0], T[0, 0])
                current = np.concatenate([current, np.array([roll, pitch, yaw])])
            
            # Calculate error
            error = target - current[:target_size]
            if np.linalg.norm(error) < tolerance:
                print(f"Converged after {i} iterations")
                break
                
            # Calculate Jacobian
            J = self.jacobian(q)
            
            # If we're not using orientation, we only need the position part of the Jacobian
            if not use_orientation:
                J = J[:3, :]
                
            # Use pseudoinverse to find joint velocities
            J_pinv = np.linalg.pinv(J)
            dq = alpha * J_pinv @ error
            
            # Update joint values
            q = q + dq
            
            # Optional: Normalize angles to [-pi, pi]
            for j in range(self.num_joints):
                if self.links[j].joint_type == 'revolute':
                    q[j] = ((q[j] + np.pi) % (2 * np.pi)) - np.pi
                    
        if i == max_iter - 1:
            print(f"Did not converge after {max_iter} iterations, final error: {np.linalg.norm(error)}")
                    
        return q
    
    def inverse_kinematics_ccd(self, target_position, max_iter=100, tolerance=1e-3):
        """
        Cyclic Coordinate Descent IK solver.
        
        Args:
            target_position: 3D target position [x, y, z]
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Joint values that achieve the target position
        """
        # Initial joint values (current configuration)
        q = np.zeros(self.num_joints)
        
        # Convert target to numpy array
        target = np.array(target_position)
        
        # Iteration loop
        for i in range(max_iter):
            # Check if we've reached the target
            _, positions = self.forward_kinematics(q)
            end_effector = positions[-1]
            error = np.linalg.norm(target - end_effector)
            
            if error < tolerance:
                print(f"CCD converged after {i} iterations")
                break
                
            # CCD: Adjust joints one at a time, starting from the end-effector
            for j in range(self.num_joints - 1, -1, -1):
                # Skip prismatic joints for simplicity (could be added later)
                if self.links[j].joint_type != 'revolute':
                    continue
                    
                # Forward kinematics to get the current positions
                _, positions = self.forward_kinematics(q)
                joint_pos = positions[j]
                end_effector = positions[-1]
                
                # Vector from current joint to end effector
                joint_to_end = end_effector - joint_pos
                
                # Vector from current joint to target
                joint_to_target = target - joint_pos
                
                # Normalize vectors
                joint_to_end = joint_to_end / np.linalg.norm(joint_to_end)
                joint_to_target = joint_to_target / np.linalg.norm(joint_to_target)
                
                # Calculate the angle between the vectors
                dot_product = np.dot(joint_to_end, joint_to_target)
                # Clamp dot product to [-1, 1] to avoid numerical issues
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Determine the rotation direction using cross product
                cross_product = np.cross(joint_to_end, joint_to_target)
                # Determine sign based on the z-component of the cross product
                # This is a simplified approach and might not work for all configurations
                if cross_product[2] < 0:
                    angle = -angle
                    
                # Update the joint angle
                q[j] += angle
                
                # Optional: Normalize angle to [-pi, pi]
                q[j] = ((q[j] + np.pi) % (2 * np.pi)) - np.pi
                
        if i == max_iter - 1:
            print(f"CCD did not converge after {max_iter} iterations, final error: {error}")
                
        return q
    
    def inverse_kinematics_analytical_3r(self, target_position, target_orientation=None):
        """
        Analytical inverse kinematics solution for a 3R robot arm.
        This assumes a specific configuration with 3 revolute joints and all links in a plane.
        
        Args:
            target_position: 2D or 3D target position [x, y, (z)]
            target_orientation: Target orientation of the end effector (if needed)
            
        Returns:
            Joint angles [q1, q2, q3]
        """
        if self.num_joints != 3:
            raise ValueError("Analytical IK for 3R robot requires exactly 3 joints")
            
        # Extract target coordinates
        x = target_position[0]
        y = target_position[1]
        
        # Extract link lengths
        l1 = self.links[0].length
        l2 = self.links[1].length
        l3 = self.links[2].length
        
        # If target orientation is specified, we can directly solve for q3
        if target_orientation is not None:
            # Assuming target_orientation is the global angle of the end-effector
            phi = target_orientation  # Target orientation of the end effector
            
            # Calculate the position of the wrist (before the last link)
            wrist_x = x - l3 * np.cos(phi)
            wrist_y = y - l3 * np.sin(phi)
            
            # Now solve the 2R problem for the first two joints
            # Similar to the 2R case with the target at the wrist
            r = np.sqrt(wrist_x**2 + wrist_y**2)
            
            # Check if the wrist position is reachable
            if r > l1 + l2 or r < abs(l1 - l2):
                raise ValueError(f"Wrist position ({wrist_x}, {wrist_y}) is not reachable by the first two links")
                
            # Calculate q2 using the law of cosines
            cos_q2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
            q2 = np.arccos(cos_q2)
            
            # Calculate q1
            q1 = np.arctan2(wrist_y, wrist_x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
            
            # Calculate q3
            q3 = phi - q1 - q2
            
            # Normalize q3 to [-pi, pi]
            q3 = ((q3 + np.pi) % (2 * np.pi)) - np.pi
            
            return [q1, q2, q3]
        else:
            # Without target orientation, we have an extra degree of freedom
            # We can choose q3 arbitrarily (e.g., q3 = 0) or use other criteria
            q3 = 0
            
            # Calculate the position of the wrist
            wrist_x = x - l3 * np.cos(q1 + q2 + q3)
            wrist_y = y - l3 * np.sin(q1 + q2 + q3)
            
            # Now solve the 2R problem for the first two joints
            # This becomes an iterative problem because the wrist position depends on all joint angles
            # For simplicity, we can use a numerical approach
            
            # This is a simplified approach that might not work in all cases
            return self.inverse_kinematics_jacobian(target_position)
    
    def inverse_kinematics_dls(self, target_position, target_orientation=None,
                              max_iter=1000, tolerance=1e-3, lambda_val=0.5):
        """
        Damped Least Squares (DLS) / Levenberg-Marquardt inverse kinematics solver.
        
        Args:
            target_position: 3D target position [x, y, z]
            target_orientation: 3D target orientation as euler angles [roll, pitch, yaw] (optional)
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            lambda_val: Damping factor
            
        Returns:
            Joint values that achieve the target position (and orientation if specified)
        """
        # Initial joint values (current configuration)
        q = np.zeros(self.num_joints)
        
        # Target vector
        target = np.array(target_position)
        if target_orientation is not None:
            target = np.concatenate([target, target_orientation])
            use_orientation = True
            target_size = 6  # Position and orientation
        else:
            use_orientation = False
            target_size = 3  # Position only
        
        # Iteration loop
        for i in range(max_iter):
            # Calculate forward kinematics
            transforms, positions = self.forward_kinematics(q)
            current_position = positions[-1]
            
            # Calculate current end effector state
            current = current_position
            if use_orientation:
                # Extract orientation (roll, pitch, yaw) from transformation matrix
                T = transforms[-1]
                # Simplified extraction
                roll = np.arctan2(T[2, 1], T[2, 2])
                pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))
                yaw = np.arctan2(T[1, 0], T[0, 0])
                current = np.concatenate([current, np.array([roll, pitch, yaw])])
            
            # Calculate error
            error = target - current[:target_size]
            if np.linalg.norm(error) < tolerance:
                print(f"DLS converged after {i} iterations")
                break
                
            # Calculate Jacobian
            J = self.jacobian(q)
            
            # If we're not using orientation, we only need the position part of the Jacobian
            if not use_orientation:
                J = J[:3, :]
                
            # Damped Least Squares: (J^T * J + λ^2 * I)^-1 * J^T * e
            J_T = J.T
            lambda_sq = lambda_val ** 2
            I = np.eye(self.num_joints)
            
            # Calculate inverse
            tmp = J_T @ J + lambda_sq * I
            tmp_inv = np.linalg.inv(tmp)
            
            # Calculate joint velocity
            dq = tmp_inv @ J_T @ error
            
            # Update joint values
            q = q + dq
            
            # Optional: Normalize angles to [-pi, pi]
            for j in range(self.num_joints):
                if self.links[j].joint_type == 'revolute':
                    q[j] = ((q[j] + np.pi) % (2 * np.pi)) - np.pi
                    
        if i == max_iter - 1:
            print(f"DLS did not converge after {max_iter} iterations, final error: {np.linalg.norm(error)}")
                    
        return q
    
    def plot_robot(self, joint_values, ax=None, show=True, target_position=None, show_path=False, prev_positions=None):
        """
        Plot the robot in its current configuration.
        
        Args:
            joint_values: List of joint values
            ax: Matplotlib axis to plot on (or None to create a new one)
            show: Whether to show the plot immediately
            target_position: Optional target position to show
            show_path: Whether to show the path of the end-effector
            prev_positions: List of previous end-effector positions
            
        Returns:
            The matplotlib axis
        """
        # Create a new 3D axis if none provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        # Compute forward kinematics
        _, positions = self.forward_kinematics(joint_values)
        
        # Extract x, y, z coordinates
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        
        # Plot the robot links
        ax.plot(xs, ys, zs, 'o-', linewidth=3, markersize=10)
        
        # Plot the base
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color='blue', marker='o', s=100, label='Base')
        
        # Plot the end-effector
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='red', marker='o', s=100, label='End-effector')
        
        # Plot the target if provided
        if target_position is not None:
            ax.scatter([target_position[0]], [target_position[1]], 
                      [target_position[2] if len(target_position) > 2 else 0],
                      color='green', marker='x', s=100, label='Target')
            
        # Plot the path if requested
        if show_path and prev_positions is not None:
            path_xs = [p[0] for p in prev_positions]
            path_ys = [p[1] for p in prev_positions]
            path_zs = [p[2] for p in prev_positions]
            ax.plot(path_xs, path_ys, path_zs, 'g--', alpha=0.5, label='Path')
            
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set axis limits - could be improved to be more dynamic
        ax_range = sum([link.length for link in self.links])
        ax.set_xlim([-ax_range, ax_range])
        ax.set_ylim([-ax_range, ax_range])
        ax.set_zlim([0, ax_range])
        
        # Add a grid and legend
        ax.grid(True)
        ax.legend()
        
        if show:
            plt.tight_layout()
            plt.show()
            
        return ax
    
    def animate_to_target(self, start_q, target_position, method='jacobian', 
                         num_steps=50, interval=100):
        """
        Animate the robot's movement from start position to target position.
        
        Args:
            start_q: Starting joint configuration
            target_position: Target end-effector position
            method: IK method to use ('jacobian', 'ccd', 'dls')
            num_steps: Number of steps in the animation
            interval: Time between frames in milliseconds
            
        Returns:
            Animation object
        """
        # Solve IK to get the final joint configuration
        if method == 'jacobian':
            end_q = self.inverse_kinematics_jacobian(target_position)
        elif method == 'ccd':
            end_q = self.inverse_kinematics_ccd(target_position)
        elif method == 'dls':
            end_q = self.inverse_kinematics_dls(target_position)
        elif method == '2r_analytical' and self.num_joints == 2:
            # Use the first solution from the analytical IK
            end_q = self.inverse_kinematics_analytical_2r(target_position)[0]
        elif method == '3r_analytical' and self.num_joints == 3:
            end_q = self.inverse_kinematics_analytical_3r(target_position)
        else:
            raise ValueError(f"Method {method} not supported or not applicable to this robot")
            
        # Generate interpolated joint configurations
        q_traj = []
        for i in range(num_steps):
            t = i / (num_steps - 1)  # Normalized time from 0 to 1
            q = start_q * (1 - t) + end_q * t  # Linear interpolation
            q_traj.append(q)
            
        # Set up the animation figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Track the end-effector path
        ee_positions = []
        
        # Initialize the plot
        self.plot_robot(start_q, ax=ax, show=False, target_position=target_position)
        
        # Animation update function
        def update(frame):
            ax.clear()
            _, positions = self.forward_kinematics(q_traj[frame])
            ee_positions.append(positions[-1])
            return self.plot_robot(q_traj[frame], ax=ax, show=False, 
                                 target_position=target_position, 
                                 show_path=True, prev_positions=ee_positions)
            
        # Create and return the animation
        anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=False)
        plt.close()  # Prevent duplicate display
        
        return anim


# Example usage for testing the code

def create_2r_planar_robot():
    """Create a simple 2R planar robot"""
    link1 = RobotLink(length=1.0)
    link2 = RobotLink(length=0.8)
    return RobotArm([link1, link2])

def create_3r_planar_robot():
    """Create a 3R planar robot"""
    link1 = RobotLink(length=1.0)
    link2 = RobotLink(length=0.8)
    link3 = RobotLink(length=0.5)
    return RobotArm([link1, link2, link3])

def create_puma560_robot():
    """
    Create a simplified PUMA 560 robot with 6 DOF.
    """
    # PUMA 560 DH parameters (simplified)
    links = [
        RobotLink(length=0, theta=0, alpha=-np.pi/2, d=0, a=0),
        RobotLink(length=0.4318, theta=0, alpha=0, d=0.14909, a=0.4318),
        RobotLink(length=0.4331, theta=0, alpha=-np.pi/2, d=0, a=0.0203),
        RobotLink(length=0, theta=0, alpha=np.pi/2, d=0.4331, a=0),
        RobotLink(length=0, theta=0, alpha=-np.pi/2, d=0, a=0),
        RobotLink(length=0, theta=0, alpha=0, d=0.0562, a=0)
    ]
    return RobotArm(links)


def test_2r_robot_analytical_ik():
    """Test the analytical IK solution for a 2R robot"""
    robot = create_2r_planar_robot()
    
    # Define a target position
    target = [1.2, 0.8, 0]
    
    # Solve IK analytically
    solutions = robot.inverse_kinematics_analytical_2r(target)
    print("Analytical IK solutions for 2R robot:")
    for i, q in enumerate(solutions):
        print(f"Solution {i+1}: {q}")
        
        # Verify the solution using forward kinematics
        _, positions = robot.forward_kinematics(q)
        ee_pos = positions[-1]
        print(f"  End-effector position: {ee_pos}")
        print(f"  Error: {np.linalg.norm(np.array(target) - ee_pos)}")
        
        # Plot the robot with this solution
        robot.plot_robot(q, target_position=target)

def test_2r_robot_jacobian_ik():
    """Test the Jacobian IK method for a 2R robot"""
    robot = create_2r_planar_robot()
    
    # Define a target position
    target = [1.2, 0.8, 0]
    
    # Solve IK using Jacobian method
    q = robot.inverse_kinematics_jacobian(target)
    print(f"Jacobian IK solution for 2R robot: {q}")
    
    # Verify the solution
    _, positions = robot.forward_kinematics(q)
    ee_pos = positions[-1]
    print(f"End-effector position: {ee_pos}")
    print(f"Error: {np.linalg.norm(np.array(target) - ee_pos)}")
    
    # Plot the robot
    robot.plot_robot(q, target_position=target)

def test_2r_robot_ccd_ik():
    """Test the CCD IK method for a 2R robot"""
    robot = create_2r_planar_robot()
    
    # Define a target position
    target = [1.2, 0.8, 0]
    
    # Solve IK using CCD
    q = robot.inverse_kinematics_ccd(target)
    print(f"CCD IK solution for 2R robot: {q}")
    
    # Verify the solution
    _, positions = robot.forward_kinematics(q)
    ee_pos = positions[-1]
    print(f"End-effector position: {ee_pos}")
    print(f"Error: {np.linalg.norm(np.array(target) - ee_pos)}")
    
    # Plot the robot
    robot.plot_robot(q, target_position=target)

def test_2r_robot_dls_ik():
    """Test the DLS IK method for a 2R robot"""
    robot = create_2r_planar_robot()
    
    # Define a target position
    target = [1.2, 0.8, 0]
    
    # Solve IK using DLS
    q = robot.inverse_kinematics_dls(target)
    print(f"DLS IK solution for 2R robot: {q}")
    
    # Verify the solution
    _, positions = robot.forward_kinematics(q)
    ee_pos = positions[-1]
    print(f"End-effector position: {ee_pos}")
    print(f"Error: {np.linalg.norm(np.array(target) - ee_pos)}")
    
    # Plot the robot
    robot.plot_robot(q, target_position=target)

def test_3r_robot_ik():
    """Test different IK methods for a 3R robot"""
    robot = create_3r_planar_robot()
    
    # Define a target position and orientation
    target_pos = [1.5, 0.5, 0]
    target_orient = np.pi/4  # 45 degrees
    
    # Solve using analytical method (when possible)
    try:
        q_analytical = robot.inverse_kinematics_analytical_3r(target_pos, target_orient)
        print(f"Analytical IK for 3R robot: {q_analytical}")
        robot.plot_robot(q_analytical, target_position=target_pos)
    except Exception as e:
        print(f"Analytical IK failed: {e}")
    
    # Solve using Jacobian method
    q_jacobian = robot.inverse_kinematics_jacobian(target_pos)
    print(f"Jacobian IK for 3R robot: {q_jacobian}")
    robot.plot_robot(q_jacobian, target_position=target_pos)
    
    # Solve using DLS (often most reliable)
    q_dls = robot.inverse_kinematics_dls(target_pos)
    print(f"DLS IK for 3R robot: {q_dls}")
    robot.plot_robot(q_dls, target_position=target_pos)

def test_animation():
    """Test the animation feature"""
    robot = create_2r_planar_robot()
    
    # Starting and target positions
    start_q = [0, 0]
    target = [1.2, 0.8, 0]
    
    # Create animation
    anim = robot.animate_to_target(start_q, target, method='2r_analytical', num_steps=50)
    
    # Display the animation
    from IPython.display import HTML
    return HTML(anim.to_jshtml())

def run_all_tests():
    # Run all the tests
    print("\nTesting 2R Robot Analytical IK")
    print("------------------------------")
    test_2r_robot_analytical_ik()
    
    print("\nTesting 2R Robot Jacobian IK")
    print("----------------------------")
    test_2r_robot_jacobian_ik()
    
    print("\nTesting 2R Robot CCD IK")
    print("----------------------")
    test_2r_robot_ccd_ik()
    
    print("\nTesting 2R Robot DLS IK")
    print("----------------------")
    test_2r_robot_dls_ik()
    
    print("\nTesting 3R Robot IK Methods")
    print("-------------------------")
    test_3r_robot_ik()

if __name__ == "__main__":
    run_all_tests()
    
    # Uncomment to run animation test (only works in Jupyter/IPython)
    # animation = test_animation()
    # display(animation)