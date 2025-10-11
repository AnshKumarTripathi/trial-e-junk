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
        self.state = "ROTATING"  # ROTATING, MOVING, AVOIDING
        self.target_reached = False
        
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
        """Calculate intermediate waypoints for safer navigation"""
        intermediate_waypoints = []
        
        for i in range(len(self.original_waypoints) - 1):
            current = self.original_waypoints[i]
            next_wp = self.original_waypoints[i + 1]
            
            # Add current waypoint
            intermediate_waypoints.append(current)
            
            # Calculate intermediate waypoint
            # Strategy: Move horizontally first, then vertically
            intermediate_x = next_wp[0]  # Same x as next waypoint
            intermediate_y = current[1]  # Same y as current waypoint
            intermediate_yaw = current[2]  # Keep current orientation
            
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

    def get_obstacle_avoidance_direction(self):
        """Calculate direction to avoid obstacles"""
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

    def navigation_loop(self):
        """Main navigation control loop"""
        if self.current_waypoint >= len(self.waypoints):
            # All waypoints reached
            self.stop_robot()
            self.get_logger().info("All waypoints reached! Navigation complete.")
            return
            
        # Check if current waypoint is reached
        if self.check_waypoint_reached():
            self.current_waypoint += 1
            if self.current_waypoint < len(self.waypoints):
                # Determine if this is an intermediate waypoint
                is_intermediate = self.current_waypoint % 2 == 0  # Even indices are intermediate
                waypoint_type = "Intermediate" if is_intermediate else "Main"
                self.get_logger().info(f"{waypoint_type} waypoint {self.current_waypoint} reached! Moving to waypoint {self.current_waypoint + 1}")
                self.state = "ROTATING"
            return
            
        target = self.waypoints[self.current_waypoint]
        target_x, target_y, target_yaw = target
        
        # Calculate errors
        distance_error = self.calculate_distance(
            self.current_x, self.current_y, target_x, target_y
        )
        
        # Calculate angle to target position
        angle_to_target = math.atan2(target_y - self.current_y, target_x - self.current_x)
        angle_error = self.normalize_angle(angle_to_target - self.current_yaw)
        
        # Calculate final orientation error
        final_angle_error = self.normalize_angle(target_yaw - self.current_yaw)
        
        # Create velocity command
        cmd = Twist()
        
        # Handle obstacle avoidance
        if self.obstacle_detected:
            self.state = "AVOIDING"
            self.get_logger().warn("Obstacle detected! Avoiding...")
            
            # Stop forward movement and turn away from obstacle
            cmd.linear.x = 0.0
            avoidance_angle = self.get_obstacle_avoidance_direction()
            cmd.angular.z = self.KP_ANGULAR * avoidance_angle
            
        else:
            # Normal navigation
            if self.state == "AVOIDING":
                self.state = "ROTATING"
                
            if self.state == "ROTATING":
                # First, orient towards target position
                if abs(angle_error) > 0.1:  # 0.1 rad ≈ 6 degrees
                    cmd.angular.z = self.KP_ANGULAR * angle_error
                    cmd.linear.x = 0.0
                else:
                    self.state = "MOVING"
                    
            elif self.state == "MOVING":
                # Move towards target
                if distance_error > self.POSITION_TOLERANCE:
                    # Move forward
                    linear_vel = min(self.KP_LINEAR * distance_error, self.MAX_LINEAR_VEL)
                    cmd.linear.x = linear_vel
                    
                    # Fine-tune orientation while moving
                    cmd.angular.z = self.KP_ANGULAR * angle_error * 0.5
                else:
                    # Close to target, now orient to final angle
                    self.state = "ROTATING"
                    if abs(final_angle_error) > 0.1:
                        cmd.angular.z = self.KP_ANGULAR * final_angle_error
                        cmd.linear.x = 0.0
                    else:
                        # Waypoint reached
                        self.stop_robot()
                        return
        
        # Limit velocities
        cmd.linear.x = max(-self.MAX_LINEAR_VEL, min(self.MAX_LINEAR_VEL, cmd.linear.x))
        cmd.angular.z = max(-self.MAX_ANGULAR_VEL, min(self.MAX_ANGULAR_VEL, cmd.angular.z))
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
        # Log current status
        if self.current_waypoint < len(self.waypoints):
            self.get_logger().info(
                f"Waypoint {self.current_waypoint + 1}: "
                f"Pos: ({self.current_x:.2f}, {self.current_y:.2f}) -> "
                f"({target_x:.2f}, {target_y:.2f}), "
                f"Dist: {distance_error:.2f}m, "
                f"State: {self.state}"
            )

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
