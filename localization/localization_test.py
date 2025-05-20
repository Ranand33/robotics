import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from localization_base import Environment, Robot
from localization_markov import MarkovLocalization
from localization_gaussian import GaussianLocalization
from localization_grid import GridLocalization
from localization_montecarlo import MonteCarloLocalization
import time

def test_localization(algorithm, num_steps=30, seed=42):
    """
    Test a localization algorithm with a simulated robot.
    
    Args:
        algorithm: The localization algorithm to test
        num_steps: Number of simulation steps
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create environment
    env = Environment(width=10.0, height=10.0, num_landmarks=5, seed=seed)
    
    # Create robot initially at a known position
    robot = Robot(env, x=2.0, y=2.0, theta=0.0)
    
    # Initialize localization algorithm
    algorithm.initialize(
        x_range=(0, env.width),
        y_range=(0, env.height),
        theta_range=(0, 2*np.pi)
    )
    
    # Lists to store data for plotting
    true_poses = []
    estimated_poses = []
    estimation_errors = []
    
    # Run simulation
    for step in range(num_steps):
        # Record true position
        true_x, true_y, true_theta = robot.get_pose()
        true_poses.append([true_x, true_y, true_theta])
        
        # Get measurements
        distances, bearings = robot.get_measurements()
        
        # Update localization with measurements
        algorithm.update_measurement(distances, bearings)
        
        # Get estimated position
        est_x, est_y, est_theta = algorithm.get_estimate()
        estimated_poses.append([est_x, est_y, est_theta])
        
        # Calculate position error
        position_error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
        
        # Calculate angle error (handle circular difference)
        angle_diff = true_theta - est_theta
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        angle_error = np.abs(angle_diff)
        
        estimation_errors.append([position_error, angle_error])
        
        # Move the robot randomly
        if step < num_steps - 1:  # Don't move on the last step
            distance = np.random.uniform(0.5, 1.0)
            turn_angle = np.random.uniform(-np.pi/4, np.pi/4)
            
            # Calculate dx, dy from distance and current orientation
            dx = distance * np.cos(true_theta)
            dy = distance * np.sin(true_theta)
            
            # Move robot
            actual_distance, actual_turn = robot.move(distance, turn_angle)
            
            # Update localization with the motion
            algorithm.update_motion(dx, dy, actual_turn)
    
    # Convert to numpy arrays
    true_poses = np.array(true_poses)
    estimated_poses = np.array(estimated_poses)
    estimation_errors = np.array(estimation_errors)
    
    # Create figure for final visualization
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot trajectory
    ax1 = axs[0, 0]
    ax1.plot(true_poses[:, 0], true_poses[:, 1], 'b-', label='True Trajectory')
    ax1.plot(estimated_poses[:, 0], estimated_poses[:, 1], 'r--', label='Estimated Trajectory')
    ax1.scatter(true_poses[0, 0], true_poses[0, 1], c='g', s=100, marker='o', label='Start')
    ax1.scatter(true_poses[-1, 0], true_poses[-1, 1], c='r', s=100, marker='x', label='End')
    
    # Draw landmarks
    landmarks = env.get_landmarks()
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='k', marker='^', s=100, label='Landmarks')
    
    # Draw walls
    walls = env.get_walls()
    for wall in walls:
        ax1.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
    
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.set_aspect('equal')
    ax1.set_title('Robot Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot position error
    ax2 = axs[0, 1]
    ax2.plot(range(num_steps), estimation_errors[:, 0], 'b-')
    ax2.set_title('Position Estimation Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (m)')
    ax2.grid(True)
    
    # Plot orientation error
    ax3 = axs[1, 0]
    ax3.plot(range(num_steps), np.rad2deg(estimation_errors[:, 1]), 'r-')
    ax3.set_title('Orientation Estimation Error')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Orientation Error (degrees)')
    ax3.grid(True)
    
    # Plot final belief state
    ax4 = axs[1, 1]
    algorithm.visualize_belief(ax4)
    
    # Add overall title
    plt.suptitle(f'Localization Results: {algorithm.__class__.__name__}', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    avg_pos_error = np.mean(estimation_errors[:, 0])
    avg_ang_error = np.mean(np.rad2deg(estimation_errors[:, 1]))
    print(f"Algorithm: {algorithm.__class__.__name__}")
    print(f"Average position error: {avg_pos_error:.2f} m")
    print(f"Average orientation error: {avg_ang_error:.2f} degrees")
    
    return true_poses, estimated_poses, estimation_errors


def animate_localization(algorithm, num_steps=30, seed=42):
    """
    Create an animation of the localization process.
    
    Args:
        algorithm: The localization algorithm to test
        num_steps: Number of simulation steps
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create environment
    env = Environment(width=10.0, height=10.0, num_landmarks=5, seed=seed)
    
    # Create robot initially at a known position
    robot = Robot(env, x=2.0, y=2.0, theta=0.0)
    
    # Initialize localization algorithm
    algorithm.initialize(
        x_range=(0, env.width),
        y_range=(0, env.height),
        theta_range=(0, 2*np.pi)
    )
    
    # Store data for animation
    true_traj_x = []
    true_traj_y = []
    est_traj_x = []
    est_traj_y = []
    
    # Create figure for animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Setup first plot for trajectory
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.set_aspect('equal')
    ax1.set_title('Robot Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    
    # Draw landmarks and walls
    landmarks = env.get_landmarks()
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='k', marker='^', s=100, label='Landmarks')
    
    walls = env.get_walls()
    for wall in walls:
        ax1.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
    
    # True trajectory line
    true_traj_line, = ax1.plot([], [], 'b-', label='True Trajectory')
    
    # Estimated trajectory line
    est_traj_line, = ax1.plot([], [], 'r--', label='Estimated Trajectory')
    
    # Robot position
    robot_pos = ax1.scatter([], [], c='g', s=150, marker='o', label='Robot')
    
    # Robot orientation
    robot_orient, = ax1.plot([], [], 'g-', linewidth=2)
    
    # Estimated position
    est_pos = ax1.scatter([], [], c='r', s=100, marker='x', label='Estimate')
    
    # Estimated orientation
    est_orient, = ax1.plot([], [], 'r-', linewidth=2)
    
    ax1.legend()
    
    # Create animation function
    def animate(i):
        if i == 0:
            # Clear stored trajectories at the start
            true_traj_x.clear()
            true_traj_y.clear()
            est_traj_x.clear()
            est_traj_y.clear()
            
            # Clear the belief visualization
            ax2.clear()
            ax2.set_title(f'{algorithm.__class__.__name__} Belief State')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
        
        # Get robot pose
        true_x, true_y, true_theta = robot.get_pose()
        true_traj_x.append(true_x)
        true_traj_y.append(true_y)
        
        # Get measurements
        distances, bearings = robot.get_measurements()
        
        # Update localization with measurements
        algorithm.update_measurement(distances, bearings)
        
        # Get estimated position
        est_x, est_y, est_theta = algorithm.get_estimate()
        est_traj_x.append(est_x)
        est_traj_y.append(est_y)
        
        # Update trajectory lines
        true_traj_line.set_data(true_traj_x, true_traj_y)
        est_traj_line.set_data(est_traj_x, est_traj_y)
        
        # Update robot position
        robot_pos.set_offsets([true_x, true_y])
        
        # Update robot orientation
        head_len = 0.3
        robot_orient.set_data(
            [true_x, true_x + head_len * np.cos(true_theta)],
            [true_y, true_y + head_len * np.sin(true_theta)]
        )
        
        # Update estimated position
        est_pos.set_offsets([est_x, est_y])
        
        # Update estimated orientation
        est_orient.set_data(
            [est_x, est_x + head_len * np.cos(est_theta)],
            [est_y, est_y + head_len * np.sin(est_theta)]
        )
        
        # Visualize belief state
        ax2.clear()
        algorithm.visualize_belief(ax2)
        
        # Move the robot on the next frame
        if i < num_steps - 1:
            distance = np.random.uniform(0.5, 1.0)
            turn_angle = np.random.uniform(-np.pi/4, np.pi/4)
            
            # Calculate dx, dy from distance and current orientation
            dx = distance * np.cos(true_theta)
            dy = distance * np.sin(true_theta)
            
            # Move robot
            actual_distance, actual_turn = robot.move(distance, turn_angle)
            
            # Update localization with the motion
            algorithm.update_motion(dx, dy, actual_turn)
        
        return (true_traj_line, est_traj_line, robot_pos, 
                robot_orient, est_pos, est_orient)
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=num_steps, 
        interval=500, blit=False, repeat=False
    )
    
    plt.suptitle(f'Localization Animation: {algorithm.__class__.__name__}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return anim


def compare_all_algorithms(num_steps=30, seed=42):
    """
    Compare all localization algorithms on the same scenario.
    
    Args:
        num_steps: Number of simulation steps
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create environment
    env = Environment(width=10.0, height=10.0, num_landmarks=5, seed=seed)
    
    # Create robot initially at a known position
    robot = Robot(env, x=2.0, y=2.0, theta=0.0)
    
    # Create localization algorithms
    markov = MarkovLocalization(env, resolution=0.2, angle_resolution=12)
    gaussian = GaussianLocalization(env)
    grid = GridLocalization(env, resolution=0.2)
    monte_carlo = MonteCarloLocalization(env, num_particles=1000)
    
    algorithms = [markov, gaussian, grid, monte_carlo]
    
    # Initialize all algorithms
    for alg in algorithms:
        alg.initialize(
            x_range=(0, env.width),
            y_range=(0, env.height),
            theta_range=(0, 2*np.pi)
        )
    
    # Lists to store data for plotting
    true_poses = []
    estimated_poses = {
        'Markov': [],
        'Gaussian': [],
        'Grid': [],
        'Monte Carlo': []
    }
    
    # Run simulation
    for step in range(num_steps):
        print(f"Simulation step {step+1}/{num_steps}")
        
        # Record true position
        true_x, true_y, true_theta = robot.get_pose()
        true_poses.append([true_x, true_y, true_theta])
        
        # Get measurements
        distances, bearings = robot.get_measurements()
        
        # Update all localizations with measurements
        for alg in algorithms:
            alg.update_measurement(distances, bearings)
        
        # Get estimated positions
        estimated_poses['Markov'].append(markov.get_estimate())
        estimated_poses['Gaussian'].append(gaussian.get_estimate())
        estimated_poses['Grid'].append(grid.get_estimate())
        estimated_poses['Monte Carlo'].append(monte_carlo.get_estimate())
        
        # Move the robot randomly
        if step < num_steps - 1:  # Don't move on the last step
            distance = np.random.uniform(0.5, 1.0)
            turn_angle = np.random.uniform(-np.pi/4, np.pi/4)
            
            # Calculate dx, dy from distance and current orientation
            dx = distance * np.cos(true_theta)
            dy = distance * np.sin(true_theta)
            
            # Move robot
            actual_distance, actual_turn = robot.move(distance, turn_angle)
            
            # Update all localizations with the motion
            for alg in algorithms:
                alg.update_motion(dx, dy, actual_turn)
    
    # Convert to numpy arrays
    true_poses = np.array(true_poses)
    for key in estimated_poses:
        estimated_poses[key] = np.array(estimated_poses[key])
    
    # Calculate errors
    position_errors = {}
    orientation_errors = {}
    
    for key in estimated_poses:
        est = estimated_poses[key]
        
        # Position errors
        dx = est[:, 0] - true_poses[:, 0]
        dy = est[:, 1] - true_poses[:, 1]
        position_errors[key] = np.sqrt(dx**2 + dy**2)
        
        # Orientation errors
        angle_diff = est[:, 2] - true_poses[:, 2]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        orientation_errors[key] = np.abs(angle_diff)
    
    # Create figure for comparison
    fig = plt.figure(figsize=(15, 12))
    
    # Define colors for each algorithm
    colors = {
        'Markov': 'blue',
        'Gaussian': 'red',
        'Grid': 'green',
        'Monte Carlo': 'purple'
    }
    
    # Plot trajectory
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(true_poses[:, 0], true_poses[:, 1], 'k-', linewidth=2, label='True Trajectory')
    
    for key in estimated_poses:
        est = estimated_poses[key]
        ax1.plot(est[:, 0], est[:, 1], '--', color=colors[key], label=f'{key}')
    
    ax1.scatter(true_poses[0, 0], true_poses[0, 1], c='g', s=100, marker='o', label='Start')
    ax1.scatter(true_poses[-1, 0], true_poses[-1, 1], c='r', s=100, marker='x', label='End')
    
    # Draw landmarks
    landmarks = env.get_landmarks()
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='k', marker='^', s=100, label='Landmarks')
    
    # Draw walls
    walls = env.get_walls()
    for wall in walls:
        ax1.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
    
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.set_aspect('equal')
    ax1.set_title('Robot Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot position errors
    ax2 = fig.add_subplot(2, 2, 2)
    
    for key in position_errors:
        err = position_errors[key]
        ax2.plot(range(num_steps), err, '-', color=colors[key], label=f'{key}')
    
    ax2.set_title('Position Estimation Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (m)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot orientation errors
    ax3 = fig.add_subplot(2, 2, 3)
    
    for key in orientation_errors:
        err = np.rad2deg(orientation_errors[key])
        ax3.plot(range(num_steps), err, '-', color=colors[key], label=f'{key}')
    
    ax3.set_title('Orientation Estimation Error')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Orientation Error (degrees)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot average errors as bar chart
    ax4 = fig.add_subplot(2, 2, 4)
    
    avg_pos_errors = []
    avg_orient_errors = []
    alg_names = []
    
    for key in position_errors:
        avg_pos_errors.append(np.mean(position_errors[key]))
        avg_orient_errors.append(np.rad2deg(np.mean(orientation_errors[key])))
        alg_names.append(key)
    
    x = np.arange(len(alg_names))
    width = 0.35
    
    ax4.bar(x - width/2, avg_pos_errors, width, label='Position Error (m)')
    ax4.bar(x + width/2, avg_orient_errors, width, label='Orientation Error (deg)')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(alg_names)
    ax4.set_title('Average Errors')
    ax4.set_ylabel('Error')
    ax4.legend()
    ax4.grid(True)
    
    # Add overall title
    plt.suptitle('Comparison of Localization Algorithms', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    print(f"{'Algorithm':<15} {'Avg Position Error (m)':<25} {'Avg Orientation Error (deg)':<25}")
    print("-" * 60)
    
    for key in position_errors:
        avg_pos = np.mean(position_errors[key])
        avg_orient = np.rad2deg(np.mean(orientation_errors[key]))
        print(f"{key:<15} {avg_pos:<25.2f} {avg_orient:<25.2f}")
    
    # Calculate execution time for a single update
    print("\nExecution Time for a Single Update (ms):")
    print("-" * 60)
    print(f"{'Algorithm':<15} {'Motion Update':<20} {'Measurement Update':<20}")
    print("-" * 60)
    
    # Create a new environment for timing tests
    test_env = Environment(width=10.0, height=10.0, num_landmarks=5, seed=seed)
    
    # Create algorithms
    test_markov = MarkovLocalization(test_env, resolution=0.2, angle_resolution=12)
    test_gaussian = GaussianLocalization(test_env)
    test_grid = GridLocalization(test_env, resolution=0.2)
    test_monte_carlo = MonteCarloLocalization(test_env, num_particles=1000)
    
    test_algorithms = [
        ("Markov", test_markov),
        ("Gaussian", test_gaussian),
        ("Grid", test_grid),
        ("Monte Carlo", test_monte_carlo)
    ]
    
    # Initialize all algorithms
    for _, alg in test_algorithms:
        alg.initialize(
            x_range=(0, test_env.width),
            y_range=(0, test_env.height),
            theta_range=(0, 2*np.pi)
        )
    
    # Generate test data
    test_distances = np.random.uniform(1, 5, 5)
    test_bearings = np.random.uniform(-np.pi, np.pi, 5)
    test_dx, test_dy, test_dtheta = 0.5, 0.3, 0.1
    
    # Perform timing tests
    for name, alg in test_algorithms:
        # Time motion update
        start_time = time.time()
        for _ in range(100):  # Repeat to get stable timing
            alg.update_motion(test_dx, test_dy, test_dtheta)
        motion_time = (time.time() - start_time) * 10  # ms per call
        
        # Time measurement update
        start_time = time.time()
        for _ in range(100):  # Repeat to get stable timing
            alg.update_measurement(test_distances, test_bearings)
        measurement_time = (time.time() - start_time) * 10  # ms per call
        
        print(f"{name:<15} {motion_time:<20.2f} {measurement_time:<20.2f}")
    
    return true_poses, estimated_poses, position_errors, orientation_errors


# Main script
if __name__ == "__main__":
    # Create environment
    env = Environment(width=10.0, height=10.0, num_landmarks=5, seed=42)
    
    # Visualize environment
    env.visualize()
    
    # Choose which tests to run
    RUN_INDIVIDUAL_TESTS = True
    RUN_ANIMATION = True
    RUN_COMPARISON = True
    
    if RUN_INDIVIDUAL_TESTS:
        print("\nTesting Markov Localization")
        markov = MarkovLocalization(env, resolution=0.2, angle_resolution=12)
        test_localization(markov, num_steps=20, seed=42)
        
        print("\nTesting Gaussian Localization")
        gaussian = GaussianLocalization(env)
        test_localization(gaussian, num_steps=20, seed=42)
        
        print("\nTesting Grid Localization")
        grid = GridLocalization(env, resolution=0.2)
        test_localization(grid, num_steps=20, seed=42)
        
        print("\nTesting Monte Carlo Localization")
        monte_carlo = MonteCarloLocalization(env, num_particles=1000)
        test_localization(monte_carlo, num_steps=20, seed=42)
    
    if RUN_ANIMATION:
        print("\nAnimating Monte Carlo Localization")
        monte_carlo = MonteCarloLocalization(env, num_particles=1000)
        anim = animate_localization(monte_carlo, num_steps=20, seed=42)
    
    if RUN_COMPARISON:
        print("\nComparing all localization algorithms")
        compare_all_algorithms(num_steps=20, seed=42)