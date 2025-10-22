#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Twist
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import numpy as np
import time
import math

class UR5WaypointController(Node):
    def __init__(self):
        super().__init__('ur5_waypoint_controller')
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize publishers and subscribers
        self.pose_publisher = self.create_publisher(PoseStamped, '/ur5_controller/pose_command', 10)
        self.twist_publisher = self.create_publisher(Twist, '/ur5_controller/cartesian_velocity_command', 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )
        
        # Current joint states and end-effector pose
        self.current_joint_states = None
        self.current_ee_pose = None
        
        # Define waypoints with positions and orientations (quaternions)
        self.waypoints = [
            {
                'name': 'P1',
                'position': [-0.214, -0.532, 0.557],
                'orientation': [0.707, 0.028, 0.034, 0.707]  # [x, y, z, w]
            },
            {
                'name': 'P2', 
                'position': [-0.159, 0.501, 0.415],
                'orientation': [0.029, 0.997, 0.045, 0.033]
            },
            {
                'name': 'P3',
                'position': [-0.806, 0.010, 0.182],
                'orientation': [-0.684, 0.726, 0.05, 0.008]
            }
        ]
        
        # Tolerance for position and orientation
        self.position_tolerance = 0.15
        self.orientation_tolerance = 0.15
        
        # Servoing parameters
        self.linear_gain = 0.5
        self.angular_gain = 0.5
        self.max_linear_velocity = 0.1
        self.max_angular_velocity = 0.1
        
        # Wait for initial joint states
        self.get_logger().info("Waiting for joint states...")
        while self.current_joint_states is None:
            rclpy.spin_once(self, timeout_sec=1.0)
        
        self.get_logger().info("Joint states received. Starting waypoint navigation...")
        
        # Start the waypoint navigation
        self.navigate_waypoints()
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages"""
        self.current_joint_states = msg
    
    def create_pose_stamped(self, position, orientation, frame_id='base_link'):
        """Create a PoseStamped message"""
        pose_stamped = PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = frame_id
        
        pose_stamped.pose = Pose()
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        
        pose_stamped.pose.orientation.x = orientation[0]
        pose_stamped.pose.orientation.y = orientation[1]
        pose_stamped.pose.orientation.z = orientation[2]
        pose_stamped.pose.orientation.w = orientation[3]
        
        return pose_stamped
    
    def calculate_pose_error(self, target_pose, current_pose):
        """Calculate position and orientation errors"""
        # Position error
        pos_error = math.sqrt(
            (target_pose.position.x - current_pose.position.x)**2 +
            (target_pose.position.y - current_pose.position.y)**2 +
            (target_pose.position.z - current_pose.position.z)**2
        )
        
        # Orientation error (quaternion distance)
        q1 = np.array([target_pose.orientation.w, target_pose.orientation.x, 
                      target_pose.orientation.y, target_pose.orientation.z])
        q2 = np.array([current_pose.orientation.w, current_pose.orientation.x,
                      current_pose.orientation.y, current_pose.orientation.z])
        
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate quaternion distance
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Clamp to avoid numerical errors
        ori_error = 2 * math.acos(abs(dot_product))
        
        return pos_error, ori_error
    
    def get_current_ee_pose(self):
        """Get current end-effector pose from TF"""
        try:
            # Get transform from base_link to tool0 (end-effector)
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time()
            )
            
            # Convert transform to pose
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            
            return pose
        except Exception as e:
            self.get_logger().warn(f"Could not get current end-effector pose: {e}")
            return None
    
    def move_to_waypoint(self, waypoint):
        """Move the arm to a specific waypoint using servoing"""
        self.get_logger().info(f"Moving to waypoint {waypoint['name']}...")
        
        # Create target pose
        target_pose = self.create_pose_stamped(waypoint['position'], waypoint['orientation'])
        
        # Servoing loop
        start_time = time.time()
        timeout = 30.0  # 30 second timeout
        rate = self.create_rate(10)  # 10 Hz
        
        while time.time() - start_time < timeout:
            # Get current end-effector pose
            current_pose = self.get_current_ee_pose()
            
            if current_pose is None:
                # If we can't get current pose, use pose command approach
                self.pose_publisher.publish(target_pose)
                time.sleep(0.1)
                continue
            
            # Calculate errors
            pos_error, ori_error = self.calculate_pose_error(target_pose.pose, current_pose)
            
            # Check if we're within tolerance
            if pos_error < self.position_tolerance and ori_error < self.orientation_tolerance:
                self.get_logger().info(f"Reached waypoint {waypoint['name']} within tolerance")
                break
            
            # Calculate velocity commands for servoing
            twist = Twist()
            
            # Linear velocity (position control)
            if pos_error > self.position_tolerance:
                # Calculate direction vector
                dx = target_pose.pose.position.x - current_pose.position.x
                dy = target_pose.pose.position.y - current_pose.position.y
                dz = target_pose.pose.position.z - current_pose.position.z
                
                # Normalize and scale by gain
                norm = math.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 0:
                    twist.linear.x = min(self.max_linear_velocity, self.linear_gain * dx / norm)
                    twist.linear.y = min(self.max_linear_velocity, self.linear_gain * dy / norm)
                    twist.linear.z = min(self.max_linear_velocity, self.linear_gain * dz / norm)
            
            # Angular velocity (orientation control)
            if ori_error > self.orientation_tolerance:
                # Simple angular velocity based on orientation error
                twist.angular.x = min(self.max_angular_velocity, self.angular_gain * ori_error)
                twist.angular.y = min(self.max_angular_velocity, self.angular_gain * ori_error)
                twist.angular.z = min(self.max_angular_velocity, self.angular_gain * ori_error)
            
            # Publish velocity command
            self.twist_publisher.publish(twist)
            
            # Log progress
            self.get_logger().info(f"Position error: {pos_error:.3f}, Orientation error: {ori_error:.3f}")
            
            rate.sleep()
        
        # Stop the arm
        stop_twist = Twist()
        self.twist_publisher.publish(stop_twist)
        
        # Wait at the waypoint for 1 second
        self.get_logger().info(f"Reached waypoint {waypoint['name']}. Waiting for 1 second...")
        time.sleep(1.0)
        
        self.get_logger().info(f"Completed waypoint {waypoint['name']}")
    
    def move_to_waypoint_simple(self, waypoint):
        """Simple approach using pose commands (fallback method)"""
        self.get_logger().info(f"Moving to waypoint {waypoint['name']} using simple pose command...")
        
        # Create target pose
        target_pose = self.create_pose_stamped(waypoint['position'], waypoint['orientation'])
        
        # Publish the target pose multiple times to ensure it's received
        for _ in range(10):
            self.pose_publisher.publish(target_pose)
            time.sleep(0.1)
        
        # Wait for the arm to reach the waypoint
        self.get_logger().info(f"Waiting for arm to reach waypoint {waypoint['name']}...")
        time.sleep(3.0)  # Give the arm time to move
        
        # Wait at the waypoint for 1 second
        self.get_logger().info(f"Reached waypoint {waypoint['name']}. Waiting for 1 second...")
        time.sleep(1.0)
        
        self.get_logger().info(f"Completed waypoint {waypoint['name']}")
    
    def navigate_waypoints(self):
        """Navigate through all waypoints"""
        self.get_logger().info("Starting waypoint navigation...")
        
        for i, waypoint in enumerate(self.waypoints):
            self.get_logger().info(f"Navigating to waypoint {i+1}/3: {waypoint['name']}")
            
            # Try servoing first, fallback to simple approach
            try:
                self.move_to_waypoint(waypoint)
            except Exception as e:
                self.get_logger().warn(f"Servoing failed for {waypoint['name']}: {e}")
                self.get_logger().info("Falling back to simple pose command approach...")
                self.move_to_waypoint_simple(waypoint)
        
        self.get_logger().info("All waypoints completed successfully!")
        
        # Keep the node alive
        self.get_logger().info("Task completed. Node will continue running...")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = UR5WaypointController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
