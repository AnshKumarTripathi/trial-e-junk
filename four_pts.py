#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math

class EbotWallFollower(Node):
    def __init__(self):
        super().__init__('ebot_wall_follower')
        
        # Initialize publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Target waypoints (for reference and goal detection)
        self.waypoints = [
            [-1.53, -1.95, 1.57],   # P1
            [0.13, 1.24, 0.00],     # P2
            [0.38, -3.32, -1.57]    # P3
        ]
        self.current_waypoint_index = 0
        self.WAYPOINT_REACHED_TOLERANCE = 0.5  # meters
        
        # Control parameters
        self.MAX_LINEAR_VEL = 0.3  # Conservative speed for wall following
        self.MAX_ANGULAR_VEL = 0.8
        self.TURN_ANGULAR_VEL = 0.6  # Speed for turning
        
        # Obstacle detection thresholds
        self.WALL_DISTANCE = 0.6  # Desired distance from right wall
        self.OBSTACLE_THRESHOLD = 0.5  # Distance to consider as obstacle
        
        # LiDAR sectors (for 360-degree scan)
        # Assuming 360 readings, index 180 is front (0 degrees)
        self.FRONT_SECTOR_START = 160
        self.FRONT_SECTOR_END = 200
        self.RIGHT_SECTOR_START = 220  # 220-280 (right side)
        self.RIGHT_SECTOR_END = 280
        self.LEFT_SECTOR_START = 80    # 80-140 (left side)
        self.LEFT_SECTOR_END = 140
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.lidar_data = None
        
        # Navigation state
        self.state = "RUNNING"  # RUNNING, STOPPED, GOAL_REACHED
        self.obstacle_front = False
        self.obstacle_right = False
        self.obstacle_left = False
        
        # Create timer for main control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Ebot Wall Follower initialized")
        self.get_logger().info("Algorithm: Right-wall following")

    def lidar_callback(self, msg):
        """Process LiDAR data for obstacle detection"""
        self.lidar_data = msg.ranges
        
        if not self.lidar_data or len(self.lidar_data) < 360:
            return
        
        # Check front sector
        front_ranges = self.lidar_data[self.FRONT_SECTOR_START:self.FRONT_SECTOR_END]
        valid_front = [r for r in front_ranges if not math.isnan(r) and r > 0]
        self.obstacle_front = min(valid_front) < self.OBSTACLE_THRESHOLD if valid_front else False
        
        # Check right sector
        right_ranges = self.lidar_data[self.RIGHT_SECTOR_START:self.RIGHT_SECTOR_END]
        valid_right = [r for r in right_ranges if not math.isnan(r) and r > 0]
        self.obstacle_right = min(valid_right) < self.WALL_DISTANCE if valid_right else False
        
        # Check left sector
        left_ranges = self.lidar_data[self.LEFT_SECTOR_START:self.LEFT_SECTOR_END]
        valid_left = [r for r in left_ranges if not math.isnan(r) and r > 0]
        self.obstacle_left = min(valid_left) < self.OBSTACLE_THRESHOLD if valid_left else False

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

    def check_goal_reached(self):
        """Check if final goal (P3) is reached"""
        if self.current_waypoint_index >= len(self.waypoints):
            return True
        
        target = self.waypoints[-1]  # Check if we're near P3 (final goal)
        distance = self.calculate_distance(
            self.current_x, self.current_y, target[0], target[1]
        )
        
        if distance < self.WAYPOINT_REACHED_TOLERANCE:
            self.get_logger().info(f"Goal P3 reached! Distance: {distance:.2f}m")
            return True
        
        return False

    def right_wall_following_algorithm(self):
        """
        Pure right-wall following algorithm:
        1. If obstacle on right AND no obstacle in front → move forward
        2. If obstacle in front AND no obstacle on right → turn right
        3. If obstacles on all sides → stop
        """
        cmd = Twist()
        
        # Log current obstacle status
        self.get_logger().info(
            f"Obstacles - Front: {self.obstacle_front}, "
            f"Right: {self.obstacle_right}, "
            f"Left: {self.obstacle_left}"
        )
        
        # Rule 3: All directions blocked - STOP
        if self.obstacle_front and self.obstacle_right and self.obstacle_left:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.state = "STOPPED"
            self.get_logger().error("All directions blocked - STOPPING")
            
        # Rule 2: Obstacle in front, no obstacle on right - TURN RIGHT
        elif self.obstacle_front and not self.obstacle_right:
            cmd.linear.x = 0.0
            cmd.angular.z = -self.TURN_ANGULAR_VEL  # Turn right (negative)
            self.get_logger().info("Turning RIGHT (obstacle ahead, clear on right)")
            
        # Rule 1: Obstacle on right, no obstacle in front - MOVE FORWARD
        elif self.obstacle_right and not self.obstacle_front:
            cmd.linear.x = self.MAX_LINEAR_VEL
            cmd.angular.z = 0.0
            self.get_logger().info("Moving FORWARD (following right wall)")
            
        # Extra case: No obstacle on right, no obstacle in front
        # This means we've lost the right wall - keep moving forward to find it
        elif not self.obstacle_right and not self.obstacle_front:
            cmd.linear.x = self.MAX_LINEAR_VEL * 0.7
            cmd.angular.z = -0.2  # Slight right bias to find wall again
            self.get_logger().info("Searching for right wall")
            
        # Extra case: Obstacle on left but not front
        elif self.obstacle_left and not self.obstacle_front:
            cmd.linear.x = self.MAX_LINEAR_VEL * 0.8
            cmd.angular.z = -0.1  # Slight right turn
            self.get_logger().info("Moving forward with slight right adjustment")
            
        # Default: Move forward slowly
        else:
            cmd.linear.x = self.MAX_LINEAR_VEL * 0.5
            cmd.angular.z = 0.0
            self.get_logger().info("Default: Moving forward slowly")
        
        return cmd

    def control_loop(self):
        """Main control loop"""
        # Check if goal is reached
        if self.check_goal_reached():
            self.stop_robot()
            self.state = "GOAL_REACHED"
            self.get_logger().info("GOAL REACHED! Navigation complete.")
            return
        
        # Check if stopped
        if self.state == "STOPPED":
            self.stop_robot()
            return
        
        # Wait for LiDAR data
        if self.lidar_data is None:
            self.get_logger().warn("Waiting for LiDAR data...")
            return
        
        # Execute wall-following algorithm
        cmd = self.right_wall_following_algorithm()
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
        # Log position
        self.get_logger().info(
            f"Position: ({self.current_x:.2f}, {self.current_y:.2f}, "
            f"{math.degrees(self.current_yaw):.1f}°)"
        )

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    
    wall_follower = EbotWallFollower()
    
    try:
        rclpy.spin(wall_follower)
    except KeyboardInterrupt:
        wall_follower.get_logger().info("Wall following interrupted by user")
    finally:
        wall_follower.stop_robot()
        wall_follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

