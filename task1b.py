#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 5 ]
# Author List:		[ Your Name ]
# Filename:		    task1b_boiler_plate.py
# Functions:		[ depthimagecb, colorimagecb, bad_fruit_detection, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/color/image_raw, /camera/depth/image_rect_raw ]

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np
import math
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros import Buffer, TransformListener

# runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

class FruitsTF(Node):
    """
    ROS2 Boilerplate for fruit detection and TF publishing.
    Students should implement detection logic inside the TODO sections.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        
        # Team ID - change this to your team ID
        self.team_id = 5
        
        # Fruit detection parameters
        self.min_fruit_area = 500      # Minimum area for fruit detection
        self.max_fruit_area = 10000    # Maximum area for fruit detection
        self.min_circularity = 0.3     # Minimum circularity for fruit detection
        
        # Color range for greyish-white fruits (bad fruits)
        self.lower_grey = np.array([0, 0, 50])
        self.upper_grey = np.array([180, 30, 200])
        
        # Morphological kernel
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # Timer for periodic processing
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF boilerplate node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 
                        Use this function to receive image depth data and convert to CV2 image.

        Args:
            data (Image): Input depth image frame received from aligned depth camera topic

        Returns:
            None
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 
        #   -> Use `data` variable to convert ROS Image message to CV2 Image type
        #   -> HINT: You may use CvBridge to do the same
        #   -> Store the converted image into `self.depth_image`

        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image.

        Args:
            data (Image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            None
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :
        #   -> Use `data` variable to convert ROS Image message to CV2 Image type
        #   -> HINT: You may use CvBridge to do the same
        #   -> Store the converted image into `self.cv_image`
        #   -> Check if you need any rotation or flipping of the image 
        #      (as input data may be oriented differently than expected).
        #      You may use cv2 functions such as `cv2.flip` or `cv2.rotate`.

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')

        ############################################


    def bad_fruit_detection(self, rgb_image):
        '''
        Description:    Function to detect bad fruits in the image frame.
                        Use this function to detect bad fruits and return their center coordinates, distance from camera, angle, width and ids list.

        Args:
            rgb_image (cv2 image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            list: A list of detected bad fruit information, where each entry is a dictionary containing:
                - 'center': (x, y) coordinates of the fruit center
                - 'distance': distance from the camera in meters
                - 'angle': angle of the fruit in degrees
                - 'width': width of the fruit in pixels
                - 'id': unique identifier for the fruit
        '''
        ############ ADD YOUR CODE HERE ############
        # INSTRUCTIONS & HELP :
        #   ->  Implement bad fruit detection logic using image processing techniques
        #   ->  You may use techniques such as color filtering, contour detection, etc.
        #   ->  For each detected bad fruit, create a dictionary with its information and append
        #       to the bad_fruits list
        #   ->  Return the bad_fruits list at the end of the function
        # Step 1: Convert RGB image to HSV color space
        #   - Use cv2.cvtColor to convert the input image to HSV for better color segmentation

        # Step 2: Define lower and upper HSV bounds for "bad fruit" color
        #   - Choose HSV ranges that correspond to the color of bad fruits (e.g., brown/black spots)

        # Step 3: Create a binary mask using cv2.inRange
        #   - This mask highlights pixels within the specified HSV range

        # Step 4: Find contours in the mask
        #   - Use cv2.findContours to detect continuous regions (potential bad fruits)

        # Step 5: Loop through each contour
        #   - Filter out small contours by area threshold to remove noise
        #   - For each valid contour:
        #       a. Compute bounding rectangle (cv2.boundingRect)
        #       b. Calculate center coordinates (cX, cY)
        #       c. (Optional) Calculate distance and angle if depth data is available
        #       d. Store fruit info (center, distance, angle, width, id) in a dictionary
        #       e. Append dictionary to bad_fruits list

        # Step 6: Return the bad_fruits list
        bad_fruits = []

        if rgb_image is None:
            return bad_fruits

        # Step 1: Convert RGB image to HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create mask for greyish-white fruits (bad fruits)
        mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)
        
        # Step 3: Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Loop through each contour
        fruit_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out small contours by area threshold
            if self.min_fruit_area < area < self.max_fruit_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                    if circularity > self.min_circularity:
                        # Compute bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate center coordinates
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX = x + w // 2
                            cY = y + h // 2
                        
                        # Get distance from depth image if available
                        distance = 0.0
                        if self.depth_image is not None:
                            try:
                                depth_value = self.depth_image[cY, cX]
                                if not (np.isnan(depth_value) or np.isinf(depth_value) or depth_value <= 0):
                                    distance = float(depth_value) / 1000.0  # Convert mm to meters
                            except:
                                distance = 0.0
                        
                        # Calculate angle (simplified - you can enhance this)
                        angle = 0.0  # You can calculate actual angle if needed
                        
                        # Store fruit info
                        fruit_info = {
                            'center': (cX, cY),
                            'distance': distance,
                            'angle': angle,
                            'width': w,
                            'id': fruit_id
                        }
                        
                        bad_fruits.append(fruit_info)
                        fruit_id += 1

        return bad_fruits


    def process_image(self):
        '''
        Description:    Timer-driven loop for periodic image processing.

        Returns:
            None
        '''
        ############ Function VARIABLES ############

        # These are the variables defined from camera info topic such as image pixel size, focalX, focalY, etc.
        # Make sure you verify these variable values once. As it may affect your result.
        # You can find more on these variables here -> http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312
            

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #   ->  Get fruit center, distance from rgb, angle, width and ids list from 'detect_fruit_center' defined above

        #   ->  Loop over detected box ids received to calculate position and orientation transform to publish TF 

        #   
        #   ->  Use center_fruit_list to get realsense depth and log them down.

        #   ->  Use this formula to rectify x, y, z based on focal length, center value and size of image
        #       x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
        #       y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
        #       z = distance_from_rgb
        #       where, 
        #               cX, and cY from 'center_fruit_list'
        #               distance_from_rgb is depth of object calculated in previous step
        #               sizeCamX, sizeCamY, centerCamX, centerCamY, focalX and focalY are defined above

        #   ->  Now, mark the center points on image frame using cX and cY variables with help of 'cv2.circle' function 

        #   ->  Here, till now you receive coordinates from camera_link to fruit center position. 
        #       So, publish this transform w.r.t. camera_link using Geometry Message - TransformStamped 
        #       so that we will collect its position w.r.t base_link in next step.
        #       Use the following frame_id-
        #           frame_id = 'camera_link'
        #           child_frame_id = 'cam_<fruit_id>'          Ex: cam_20, where 20 is fruit ID

        #   ->  Then finally lookup transform between base_link and obj frame to publish the TF
        #       You may use 'lookup_transform' function to pose of obj frame w.r.t base_link 

        #   ->  And now publish TF between object frame and base_link
        #       Use the following frame_id-
        #           frame_id = 'base_link'
        #           child_frame_id = f'{teamid}_bad_fruit_{fruit_id}'    Ex: 5_bad_fruit_1, where 5 is team ID and 1 is fruit ID

        #   ->  At last show cv2 image window having detected markers drawn and center points located using 'cv2.imshow' function.
        #       Refer MD book on portal for sample image -> https://portal.e-yantra.org/

        if self.cv_image is None or self.depth_image is None:
            return

        # Get fruit detection results
        bad_fruits = self.bad_fruit_detection(self.cv_image)
        
        # Create visualization image
        vis_image = self.cv_image.copy()
        
        # Process each detected fruit
        for fruit in bad_fruits:
            cX, cY = fruit['center']
            distance = fruit['distance']
            fruit_id = fruit['id']
            
            # Calculate 3D position using the provided formula
            if distance > 0:
                x = distance * (sizeCamX - cX - centerCamX) / focalX
                y = distance * (sizeCamY - cY - centerCamY) / focalY
                z = distance
            else:
                x = y = z = 0.0
            
            # Mark center points on image
            cv2.circle(vis_image, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(vis_image, f"bad_fruit_{fruit_id}", (cX - 30, cY - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish transform from camera_link to fruit
            cam_transform = TransformStamped()
            cam_transform.header.stamp = self.get_clock().now().to_msg()
            cam_transform.header.frame_id = 'camera_link'
            cam_transform.child_frame_id = f'cam_{fruit_id}'
            cam_transform.transform.translation.x = x
            cam_transform.transform.translation.y = y
            cam_transform.transform.translation.z = z
            cam_transform.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(cam_transform)
            
            # Lookup transform from base_link to camera_link
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_link', rclpy.time.Time()
                )
                
                # Calculate final position w.r.t base_link
                final_x = transform.transform.translation.x + x
                final_y = transform.transform.translation.y + y
                final_z = transform.transform.translation.z + z
                
                # Publish final transform from base_link to fruit
                final_transform = TransformStamped()
                final_transform.header.stamp = self.get_clock().now().to_msg()
                final_transform.header.frame_id = 'base_link'
                final_transform.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
                final_transform.transform.translation.x = final_x
                final_transform.transform.translation.y = final_y
                final_transform.transform.translation.z = final_z
                final_transform.transform.rotation.w = 1.0
                
                self.tf_broadcaster.sendTransform(final_transform)
                
                self.get_logger().info(f'Published transform for {self.team_id}_bad_fruit_{fruit_id} at position: ({final_x:.3f}, {final_y:.3f}, {final_z:.3f})')
                
            except Exception as e:
                self.get_logger().warn(f'Could not lookup transform: {e}')
        
        # Show visualization
        if SHOW_IMAGE:
            cv2.imshow('fruits_tf_view', vis_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
