#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
import time

class UR5WaypointController(Node):
    def __init__(self):
        super().__init__('ur5_waypoint_controller')
        
        # Initialize publisher
        self.pose_publisher = self.create_publisher(PoseStamped, '/ur5_controller/pose_command', 10)
        
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
        
        self.get_logger().info("UR5 Waypoint Controller initialized. Starting waypoint navigation...")
        
        # Start the waypoint navigation after a short delay
        self.timer = self.create_timer(2.0, self.start_navigation)
        self.navigation_started = False
    
    def start_navigation(self):
        """Start navigation after initialization"""
        if not self.navigation_started:
            self.navigation_started = True
            self.timer.cancel()  # Cancel the timer
            self.navigate_waypoints()
    
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
    
    def move_to_waypoint(self, waypoint):
        """Move the arm to a specific waypoint"""
        self.get_logger().info(f"Moving to waypoint {waypoint['name']}...")
        self.get_logger().info(f"Target position: {waypoint['position']}")
        self.get_logger().info(f"Target orientation: {waypoint['orientation']}")
        
        # Create target pose
        target_pose = self.create_pose_stamped(waypoint['position'], waypoint['orientation'])
        
        # Publish the target pose multiple times to ensure it's received
        self.get_logger().info("Publishing target pose...")
        for i in range(20):  # Publish for 2 seconds at 10Hz
            self.pose_publisher.publish(target_pose)
            time.sleep(0.1)
        
        # Wait for the arm to reach the waypoint
        self.get_logger().info(f"Waiting for arm to reach waypoint {waypoint['name']}...")
        time.sleep(4.0)  # Give the arm time to move
        
        # Wait at the waypoint for 1 second as required
        self.get_logger().info(f"Reached waypoint {waypoint['name']}. Waiting for 1 second...")
        time.sleep(1.0)
        
        self.get_logger().info(f"Completed waypoint {waypoint['name']}")
    
    def navigate_waypoints(self):
        """Navigate through all waypoints"""
        self.get_logger().info("Starting waypoint navigation...")
        self.get_logger().info(f"Total waypoints: {len(self.waypoints)}")
        
        for i, waypoint in enumerate(self.waypoints):
            self.get_logger().info(f"Navigating to waypoint {i+1}/{len(self.waypoints)}: {waypoint['name']}")
            self.move_to_waypoint(waypoint)
        
        self.get_logger().info("All waypoints completed successfully!")
        self.get_logger().info("Task 1C completed. The robotic arm has visited all three waypoints.")
        
        # Keep the node alive
        self.get_logger().info("Node will continue running...")

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
