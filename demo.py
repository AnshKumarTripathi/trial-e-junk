#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import numpy as np

class EbotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_navigator')
        
        # Initialize publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Waypoints in order [x, y, yaw]
        # Original waypoints
        self.original_waypoints = [
            [-1.53, -1.95, 1.57],   # P1
            [0.13, 1.24, 0.00],     # P2
            [0.38, -3.32, -1.57]    # P3
        ]
        
        # Calculate intermediate waypoints for safer navigation
        self.waypoints = self.calculate_intermediate_waypoints()
        
        # Print waypoint plan
        self.print_waypoint_plan()
        
        # Starting position
        self.start_pos = [-1.5339, -6.6156, 1.57]
        
        # Tolerances
        self.POSITION_TOLERANCE = 0.3  # meters
        self.ORIENTATION_TOLERANCE = 10.0  # degrees
        
        # Robot geometry
        self.ROBOT_FRONT_OFFSET = 0.25  # Distance from center to front of robot (meters)
        self.CLEARANCE_BUFFER = 0.4  # Extra clearance for intermediate waypoints (meters)
        
        # Control parameters
        self.KP_LINEAR = 0.8
        self.KP_ANGULAR = 1.2
        self.MAX_LINEAR_VEL = 0.5
        self.MAX_ANGULAR_VEL = 1.0
        self.MIN_DISTANCE = 0.5  # Minimum distance to obstacles
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_waypoint = 0
        self.lidar_data = None
        self.obstacle_detected = False
        
        # Navigation states
        self.state = "WAYPOINT_NAV"  # WALL_FOLLOWING, WAYPOINT_NAV, COLLISION_RECOVERY, STOPPED
        self.target_reached = False
        
        # Wall-following parameters
        self.FRONT_SECTOR_START = 160
        self.FRONT_SECTOR_END = 200
        self.RIGHT_SECTOR_START = 200
        self.RIGHT_SECTOR_END = 280
        self.LEFT_SECTOR_START = 80
        self.LEFT_SECTOR_END = 160
        
        # Waypoint progress tracking
        self.completed_waypoints = set()
        self.current_target_waypoint = 0
        self.collision_recovery_steps = 0
        self.max_recovery_steps = 10
        
        # Greedy recovery when stuck
        self.stuck_counter = 0
        self.max_stuck_counter = 20
        self.last_position = [0.0, 0.0]
        self.greedy_mode = False
        
        # Create timer for main control loop
        self.timer = self.create_timer(0.1, self.navigation_loop)
        
        self.get_logger().info("Ebot Navigator initialized")
        self.get_logger().info(f"Starting navigation to {len(self.waypoints)} waypoints")

    def lidar_callback(self, msg):
        """Process LiDAR data for obstacle detection"""
        self.lidar_data = msg.ranges
        
        # Check for obstacles in front of robot
        if self.lidar_data:
            # Get ranges in front of robot (assuming 360-degree scan)
            front_ranges = []
            if len(self.lidar_data) >= 360:
                # Check front 45 degrees for better obstacle detection
                front_start = 157  # 180 - 23 degrees
                front_end = 203    # 180 + 23 degrees
                front_ranges = self.lidar_data[front_start:front_end]
            else:
                # For different LiDAR configurations
                mid_point = len(self.lidar_data) // 2
                check_range = len(self.lidar_data) // 8  # 45 degrees
                front_ranges = self.lidar_data[mid_point - check_range:mid_point + check_range]
            
        # Check for obstacles with more conservative threshold
        valid_ranges = [r for r in front_ranges if not math.isnan(r) and r > 0]
        if valid_ranges:
            min_distance = min(valid_ranges)
            # Use a more conservative distance threshold
            self.obstacle_detected = min_distance < (self.MIN_DISTANCE + 0.2)
            
            # Check for collision (very close obstacle)
            if min_distance < (self.MIN_DISTANCE * 0.5):
                self.get_logger().error("COLLISION DETECTED! Initiating recovery...")
                self.state = "COLLISION_RECOVERY"
                self.collision_recovery_steps = 0
        else:
            self.obstacle_detected = False

    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        self.current_yaw = yaw

    def calculate_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calculate_intermediate_waypoints(self):
        """
        Calculate intermediate waypoints for axis-by-axis navigation
        For each waypoint transition, we align one axis first, then the other
        """
        intermediate_waypoints = []
        
        for i in range(len(self.original_waypoints) - 1):
            current = self.original_waypoints[i]
            next_wp = self.original_waypoints[i + 1]
            
            # Add current waypoint
            intermediate_waypoints.append(current)
            
            # Calculate intermediate waypoint
            # Strategy: Align one axis at a time (Y first, then X)
            # This ensures robot moves straight along one axis before turning
            intermediate_x = current[0]  # Keep current X
            intermediate_y = next_wp[1]  # Move to next Y
            intermediate_yaw = current[2]  # Keep current orientation
            
            # Only add intermediate if there's actual movement
            dx = abs(next_wp[0] - current[0])
            dy = abs(next_wp[1] - current[1])
            
            if dx > 0.1 and dy > 0.1:  # Both axes change significantly
                intermediate_waypoint = [intermediate_x, intermediate_y, intermediate_yaw]
                intermediate_waypoints.append(intermediate_waypoint)
        
        # Add the final waypoint
        intermediate_waypoints.append(self.original_waypoints[-1])
        
        return intermediate_waypoints

    def print_waypoint_plan(self):
        """Print the waypoint navigation plan"""
        self.get_logger().info("=== Navigation Plan ===")
        for i, waypoint in enumerate(self.waypoints):
            wp_type = "Main" if i % 2 == 0 else "Intermediate"
            self.get_logger().info(f"Waypoint {i+1} ({wp_type}): [{waypoint[0]:.2f}, {waypoint[1]:.2f}, {math.degrees(waypoint[2]):.1f}°]")
        self.get_logger().info("======================")

    def should_use_intermediate_waypoint(self, current_wp, next_wp):
        """Determine if we should use intermediate waypoint for safer navigation"""
        if current_wp >= len(self.original_waypoints) - 1:
            return False
            
        current = self.original_waypoints[current_wp]
        next_wp_orig = self.original_waypoints[current_wp + 1]
        
        # Calculate if there's a significant diagonal movement
        dx = abs(next_wp_orig[0] - current[0])
        dy = abs(next_wp_orig[1] - current[1])
        
        # If both x and y changes are significant, use intermediate waypoint
        return dx > 0.5 and dy > 0.5

    def check_waypoint_reached(self):
        """Check if current waypoint is reached within tolerance"""
        if self.current_waypoint >= len(self.waypoints):
            return True
            
        target = self.waypoints[self.current_waypoint]
        target_x, target_y, target_yaw = target
        
        # Check position tolerance
        distance = self.calculate_distance(
            self.current_x, self.current_y, target_x, target_y
        )
        
        # Check orientation tolerance
        angle_diff = abs(self.normalize_angle(self.current_yaw - target_yaw))
        angle_diff_deg = math.degrees(angle_diff)
        
        position_ok = distance <= self.POSITION_TOLERANCE
        orientation_ok = angle_diff_deg <= self.ORIENTATION_TOLERANCE
        
        return position_ok and orientation_ok

    def check_obstacles_in_sector(self, sector_start, sector_end):
        """Check for obstacles in a specific LiDAR sector"""
        if not self.lidar_data or len(self.lidar_data) < 360:
            return False, float('inf')
            
        sector_ranges = self.lidar_data[sector_start:sector_end]
        valid_ranges = [r for r in sector_ranges if not math.isnan(r) and r > 0]
        
        if not valid_ranges:
            return False, float('inf')
            
        min_distance = min(valid_ranges)
        obstacle_present = min_distance < self.MIN_DISTANCE
        
        return obstacle_present, min_distance

    def check_obstacle_surroundings(self):
        """Check obstacles in front, left, and right sectors"""
        front_obstacle, front_dist = self.check_obstacles_in_sector(
            self.FRONT_SECTOR_START, self.FRONT_SECTOR_END
        )
        right_obstacle, right_dist = self.check_obstacles_in_sector(
            self.RIGHT_SECTOR_START, self.RIGHT_SECTOR_END
        )
        left_obstacle, left_dist = self.check_obstacles_in_sector(
            self.LEFT_SECTOR_START, self.LEFT_SECTOR_END
        )
        
        return {
            'front': front_obstacle,
            'right': right_obstacle,
            'left': left_obstacle,
            'front_dist': front_dist,
            'right_dist': right_dist,
            'left_dist': left_dist
        }

    def get_obstacle_avoidance_direction(self):
        """Calculate direction to avoid obstacles using wall-following"""
        if not self.lidar_data:
            return 0.0
            
        # Analyze left and right sectors for better obstacle avoidance
        num_ranges = len(self.lidar_data)
        mid_point = num_ranges // 2
        
        # Check left sector (90-180 degrees)
        left_start = mid_point - num_ranges // 4
        left_end = mid_point
        left_ranges = self.lidar_data[left_start:left_end]
        left_avg_distance = np.mean([r for r in left_ranges if not math.isnan(r) and r > 0])
        
        # Check right sector (180-270 degrees)
        right_start = mid_point
        right_end = mid_point + num_ranges // 4
        right_ranges = self.lidar_data[right_start:right_end]
        right_avg_distance = np.mean([r for r in right_ranges if not math.isnan(r) and r > 0])
        
        # Choose direction with more clearance
        if left_avg_distance > right_avg_distance:
            # Turn left
            return 0.5  # Positive angular velocity
        else:
            # Turn right
            return -0.5  # Negative angular velocity

    def is_waypoint_completed(self, waypoint_index):
        """Check if a waypoint has been completed"""
        return waypoint_index in self.completed_waypoints

    def mark_waypoint_completed(self, waypoint_index):
        """Mark a waypoint as completed"""
        self.completed_waypoints.add(waypoint_index)
        self.get_logger().info(f"Waypoint {waypoint_index + 1} marked as completed")

    def get_next_target_waypoint(self):
        """Get the next uncompleted waypoint"""
        for i in range(len(self.waypoints)):
            if not self.is_waypoint_completed(i):
                return i
        return len(self.waypoints)  # All waypoints completed

    def collision_recovery(self):
        """Handle collision recovery by moving back and realigning"""
        if self.collision_recovery_steps < self.max_recovery_steps:
            # Move backward
            cmd = Twist()
            cmd.linear.x = -0.2  # Move backward
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.collision_recovery_steps += 1
            self.get_logger().warn(f"Collision recovery step {self.collision_recovery_steps}")
            return True
        else:
            # Recovery complete, realign to current target
            self.state = "WAYPOINT_NAV"
            self.collision_recovery_steps = 0
            self.get_logger().info("Collision recovery complete, resuming waypoint navigation")
            return False

    def wall_following_navigation(self, obstacles):
        """Implement wall-following navigation algorithm"""
        cmd = Twist()
        
        if not obstacles['front'] and not obstacles['right']:
            # Clear path ahead and to the right - move forward
            cmd.linear.x = self.MAX_LINEAR_VEL * 0.8
            cmd.angular.z = 0.0
            self.get_logger().info("Clear path - moving forward")
            
        elif obstacles['front'] and not obstacles['right']:
            # Obstacle in front, clear to the right - turn right
            cmd.linear.x = 0.1  # Slow forward movement
            cmd.angular.z = -self.MAX_ANGULAR_VEL * 0.8  # Turn right
            self.get_logger().info("Obstacle in front, clear right - turning right")
            
        elif obstacles['front'] and obstacles['right'] and not obstacles['left']:
            # Obstacle in front and right, clear to the left - turn left
            cmd.linear.x = 0.1  # Slow forward movement
            cmd.angular.z = self.MAX_ANGULAR_VEL * 0.8  # Turn left
            self.get_logger().info("Obstacle in front and right, clear left - turning left")
            
        elif obstacles['front'] and obstacles['right'] and obstacles['left']:
            # All directions blocked - don't stop if waypoints not completed
            # Return None to trigger greedy mode
            return None
            
        else:
            # Default case - move forward slowly
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
            
        return cmd

    def greedy_navigation_to_waypoint(self, target_x, target_y, obstacles):
        """
        Greedy approach: Aggressively navigate towards waypoint
        Used when stuck but waypoints not completed
        """
        cmd = Twist()
        
        # Calculate angle to target
        angle_to_target = math.atan2(target_y - self.current_y, target_x - self.current_x)
        angle_error = self.normalize_angle(angle_to_target - self.current_yaw)
        
        self.get_logger().warn("GREEDY MODE: Attempting to reach waypoint aggressively")
        
        # If completely blocked, try turning towards the most open direction
        if obstacles['front'] and obstacles['right'] and obstacles['left']:
            # Turn towards the direction with more clearance
            if obstacles['left_dist'] > obstacles['right_dist']:
                cmd.angular.z = self.MAX_ANGULAR_VEL  # Turn left
                cmd.linear.x = 0.0
                self.get_logger().warn("All blocked, turning LEFT (more clearance)")
            else:
                cmd.angular.z = -self.MAX_ANGULAR_VEL  # Turn right
                cmd.linear.x = 0.0
                self.get_logger().warn("All blocked, turning RIGHT (more clearance)")
        
        # If front blocked but sides open
        elif obstacles['front']:
            if not obstacles['left']:
                cmd.angular.z = self.MAX_ANGULAR_VEL * 0.8  # Turn left
                cmd.linear.x = 0.1
                self.get_logger().warn("Front blocked, turning LEFT")
            elif not obstacles['right']:
                cmd.angular.z = -self.MAX_ANGULAR_VEL * 0.8  # Turn right
                cmd.linear.x = 0.1
                self.get_logger().warn("Front blocked, turning RIGHT")
            else:
                # Both sides blocked, turn towards target
                cmd.angular.z = self.KP_ANGULAR * angle_error
                cmd.linear.x = 0.0
        
        # If front is clear, move towards target
        else:
            # Align towards target
            if abs(angle_error) > 0.2:
                cmd.angular.z = self.KP_ANGULAR * angle_error
                cmd.linear.x = self.MAX_LINEAR_VEL * 0.3
            else:
                cmd.linear.x = self.MAX_LINEAR_VEL * 0.5
                cmd.angular.z = self.KP_ANGULAR * angle_error * 0.5
        
        return cmd
    
    def check_if_stuck(self):
        """Check if robot is stuck (not moving)"""
        distance_moved = self.calculate_distance(
            self.last_position[0], self.last_position[1],
            self.current_x, self.current_y
        )
        
        if distance_moved < 0.05:  # Moved less than 5cm
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Update last position
        self.last_position = [self.current_x, self.current_y]
        
        # Check if stuck for too long
        if self.stuck_counter >= self.max_stuck_counter:
            return True
        
        return False

    def navigation_loop(self):
        """Main navigation control loop with hybrid wall-following and waypoint navigation"""
        # Check if all waypoints are completed
        if len(self.completed_waypoints) >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("All waypoints reached! Navigation complete.")
            return
            
        # Handle collision recovery
        if self.state == "COLLISION_RECOVERY":
            if self.collision_recovery():
                return  # Still in recovery
            # Recovery complete, continue with navigation
            
        # Get obstacle information
        obstacles = self.check_obstacle_surroundings()
        
        # Get current target waypoint
        self.current_target_waypoint = self.get_next_target_waypoint()
        
        if self.current_target_waypoint >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("All waypoints completed!")
            return
            
        # Check if current target waypoint is reached
        if self.check_waypoint_reached_for_index(self.current_target_waypoint):
            self.mark_waypoint_completed(self.current_target_waypoint)
            self.get_logger().info(f"Waypoint {self.current_target_waypoint + 1} completed!")
            # Reset stuck counter when waypoint reached
            self.stuck_counter = 0
            self.greedy_mode = False
            return
            
        # Determine navigation strategy
        target = self.waypoints[self.current_target_waypoint]
        target_x, target_y, target_yaw = target
        
        # Calculate distance to target
        distance_to_target = self.calculate_distance(
            self.current_x, self.current_y, target_x, target_y
        )
        
        # Calculate angle to target
        angle_to_target = math.atan2(target_y - self.current_y, target_x - self.current_x)
        angle_error = self.normalize_angle(angle_to_target - self.current_yaw)
        
        # Check if robot is stuck
        is_stuck = self.check_if_stuck()
        
        # If stuck and waypoints not completed, enable greedy mode
        if is_stuck and len(self.completed_waypoints) < len(self.waypoints):
            self.greedy_mode = True
            self.get_logger().warn("Robot appears stuck! Enabling GREEDY MODE")
        
        # Decision logic for navigation strategy
        cmd = Twist()
        
        # GREEDY MODE: Use when stuck but waypoints not completed
        if self.greedy_mode:
            cmd = self.greedy_navigation_to_waypoint(target_x, target_y, obstacles)
            self.state = "GREEDY_NAV"
            
            # Disable greedy mode after some movement
            if self.stuck_counter < 5:
                self.greedy_mode = False
                self.get_logger().info("Robot moving again, disabling greedy mode")
        
        # If obstacles are detected, use wall-following
        elif obstacles['front'] or obstacles['right']:
            self.state = "WALL_FOLLOWING"
            cmd = self.wall_following_navigation(obstacles)
            
            # If wall-following returns None (completely blocked), use greedy mode
            if cmd is None:
                if len(self.completed_waypoints) < len(self.waypoints):
                    self.get_logger().warn("All directions blocked but waypoints incomplete! Using greedy mode")
                    cmd = self.greedy_navigation_to_waypoint(target_x, target_y, obstacles)
                    self.state = "GREEDY_NAV"
                else:
                    self.stop_robot()
                    self.get_logger().error("All directions blocked - stopping")
                    return
            else:
                self.get_logger().info("Using wall-following navigation")
            
        else:
            # Clear path - use waypoint navigation
            self.state = "WAYPOINT_NAV"
            
            # Check if we need to turn towards target
            if abs(angle_error) > 0.1:  # 0.1 rad ≈ 6 degrees
                # Turn towards target
                cmd.angular.z = self.KP_ANGULAR * angle_error
                cmd.linear.x = 0.0
                self.get_logger().info("Turning towards target")
                
            elif distance_to_target > self.POSITION_TOLERANCE:
                # Move towards target
                linear_vel = min(self.KP_LINEAR * distance_to_target, self.MAX_LINEAR_VEL)
                cmd.linear.x = linear_vel
                
                # Fine-tune orientation while moving
                cmd.angular.z = self.KP_ANGULAR * angle_error * 0.3
                self.get_logger().info("Moving towards target")
                
            else:
                # Close to target, orient to final angle
                final_angle_error = self.normalize_angle(target_yaw - self.current_yaw)
                if abs(final_angle_error) > 0.1:
                    cmd.angular.z = self.KP_ANGULAR * final_angle_error
                    cmd.linear.x = 0.0
                    self.get_logger().info("Orienting to final angle")
                else:
                    # Waypoint reached
                    self.mark_waypoint_completed(self.current_target_waypoint)
                    return
        
        # Limit velocities
        cmd.linear.x = max(-self.MAX_LINEAR_VEL, min(self.MAX_LINEAR_VEL, cmd.linear.x))
        cmd.angular.z = max(-self.MAX_ANGULAR_VEL, min(self.MAX_ANGULAR_VEL, cmd.angular.z))
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
        # Log current status
        self.get_logger().info(
            f"Target: {self.current_target_waypoint + 1}, "
            f"Pos: ({self.current_x:.2f}, {self.current_y:.2f}) -> "
            f"({target_x:.2f}, {target_y:.2f}), "
            f"Dist: {distance_to_target:.2f}m, "
            f"State: {self.state}, "
            f"Stuck: {self.stuck_counter}, "
            f"Completed: {len(self.completed_waypoints)}/{len(self.waypoints)}"
        )

    def check_waypoint_reached_for_index(self, waypoint_index):
        """Check if a specific waypoint is reached within tolerance"""
        if waypoint_index >= len(self.waypoints):
            return True
            
        target = self.waypoints[waypoint_index]
        target_x, target_y, target_yaw = target
        
        # Check position tolerance
        distance = self.calculate_distance(
            self.current_x, self.current_y, target_x, target_y
        )
        
        # Check orientation tolerance
        angle_diff = abs(self.normalize_angle(self.current_yaw - target_yaw))
        angle_diff_deg = math.degrees(angle_diff)
        
        # Use stricter tolerance for intermediate waypoints to ensure full clearance
        # Intermediate waypoints are at odd indices (1, 3, 5...)
        is_intermediate = waypoint_index % 2 == 1
        
        if is_intermediate:
            # For intermediate waypoints, use tighter position tolerance
            # This makes the robot travel closer/past the point before turning
            # ensuring the entire robot body clears the corner
            position_tolerance = 0.1  # Tighter tolerance (meters)
            # More relaxed on final orientation for intermediate points
            orientation_tolerance = self.ORIENTATION_TOLERANCE * 2
        else:
            # For main waypoints, use standard tolerances
            position_tolerance = self.POSITION_TOLERANCE
            orientation_tolerance = self.ORIENTATION_TOLERANCE
        
        position_ok = distance <= position_tolerance
        orientation_ok = angle_diff_deg <= orientation_tolerance
        
        return position_ok and orientation_ok

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    
    navigator = EbotNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("Navigation interrupted by user")
    finally:
        navigator.stop_robot()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
