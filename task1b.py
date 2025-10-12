#!/usr/bin/env python3

import cv2
import numpy as np
import math


class FruitDetector:
    def __init__(self):
        # Fruit detection parameters
        self.min_fruit_area = 500      # Minimum area for fruit detection
        self.max_fruit_area = 10000    # Maximum area for fruit detection
        self.min_circularity = 0.3     # Minimum circularity for fruit detection
        
        # Color range for greyish-white fruits (bad fruits)
        # HSV ranges for greyish-white detection
        self.lower_grey = np.array([0, 0, 50])
        self.upper_grey = np.array([180, 30, 200])
        
        # Morphological kernel
        self.kernel = np.ones((5, 5), np.uint8)

    def detect_bad_fruits(self, rgb_image):
        """
        Detect bad fruits using color segmentation
        
        Args:
            rgb_image: Input RGB image (BGR format)
            
        Returns:
            list: List of contours representing detected bad fruits
        """
        if rgb_image is None:
            return []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for greyish-white fruits
        mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)
        
        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to identify fruits
        fruit_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_fruit_area < area < self.max_fruit_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    if circularity > self.min_circularity:
                        fruit_contours.append(contour)
        
        return fruit_contours

    def get_fruit_centroid(self, contour):
        """
        Get centroid of a fruit contour
        
        Args:
            contour: OpenCV contour
            
        Returns:
            tuple: (x, y) centroid coordinates, or None if invalid
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def visualize_detections(self, rgb_image, contours):
        """
        Visualize detected fruits with contours and labels
        
        Args:
            rgb_image: Input RGB image
            contours: List of fruit contours
            
        Returns:
            numpy.ndarray: Image with visualizations
        """
        if rgb_image is None:
            return None
        
        # Create a copy for visualization
        vis_image = rgb_image.copy()
        
        # Draw contours and labels
        for i, contour in enumerate(contours):
            # Draw contour
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"bad_fruit_{i+1}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image

    def adjust_color_range(self, lower_h, lower_s, lower_v, upper_h, upper_s, upper_v):
        """
        Adjust color detection range
        
        Args:
            lower_h, lower_s, lower_v: Lower HSV bounds
            upper_h, upper_s, upper_v: Upper HSV bounds
        """
        self.lower_grey = np.array([lower_h, lower_s, lower_v])
        self.upper_grey = np.array([upper_h, upper_s, upper_v])

    def adjust_area_filters(self, min_area, max_area):
        """
        Adjust area filtering parameters
        
        Args:
            min_area: Minimum area for fruit detection
            max_area: Maximum area for fruit detection
        """
        self.min_fruit_area = min_area
        self.max_fruit_area = max_area

    def adjust_circularity(self, min_circularity):
        """
        Adjust circularity filter
        
        Args:
            min_circularity: Minimum circularity for fruit detection
        """
        self.min_circularity = min_circularity


# Example usage and testing
def test_detection():
    """Test the detection algorithm with a sample image"""
    
    # Create a test image with different colored circles
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw different colored circles
    cv2.circle(test_image, (100, 100), 50, (120, 120, 120), -1)  # Greyish-white (bad fruit)
    cv2.circle(test_image, (200, 100), 50, (200, 100, 100), -1)  # Reddish-pink (good fruit)
    cv2.circle(test_image, (300, 100), 50, (150, 150, 150), -1)  # Another greyish-white
    cv2.circle(test_image, (400, 100), 50, (180, 80, 80), -1)    # Another reddish-pink
    cv2.circle(test_image, (500, 100), 50, (110, 110, 110), -1)  # Another greyish-white
    
    # Create detector
    detector = FruitDetector()
    
    # Detect fruits
    fruit_contours = detector.detect_bad_fruits(test_image)
    
    # Visualize results
    result_image = detector.visualize_detections(test_image, fruit_contours)
    
    # Display results
    cv2.imshow('Original Image', test_image)
    cv2.imshow('Detection Result', result_image)
    
    print(f"Detected {len(fruit_contours)} bad fruits")
    
    # Print centroids
    for i, contour in enumerate(fruit_contours):
        centroid = detector.get_fruit_centroid(contour)
        if centroid:
            print(f"Fruit {i+1} centroid: {centroid}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_detection()
