import os
import requests
import numpy as np
from PIL import Image as PILImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

class AutoDriveByLine(Node):
    def __init__(self):
        super().__init__('drive_by_line')
        self.get_logger().info(f'{self.get_name()} created')

        # Parameters
        self.declare_parameter('image', "/mycamera/image_raw")
        self.declare_parameter('cmd', "/cmd_vel")
        self.declare_parameter('url', "http://localhost:9020/predict/")
        self.declare_parameter('x_vel', 0.5)
        self.declare_parameter('theta_vel', 0.5)
        self.declare_parameter('image_size', 64)

        # Get parameter values
        self._image_topic = self.get_parameter('image').get_parameter_value().string_value
        self._cmd_topic = self.get_parameter('cmd').get_parameter_value().string_value
        self._url = self.get_parameter('url').get_parameter_value().string_value
        self._x_vel = self.get_parameter('x_vel').get_parameter_value().double_value
        self._theta_vel = self.get_parameter('theta_vel').get_parameter_value().double_value
        self._image_size = self.get_parameter('image_size').get_parameter_value().integer_value

        # ROS 2 communication
        self.create_subscription(Image, self._image_topic, self._image_callback, 1)
        self._pub = self.create_publisher(Twist, self._cmd_topic, 1)

        # CvBridge for converting ROS images to OpenCV format
        self._bridge = CvBridge()
        self._auto_driving = False

    def _image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow('window', image)
            key = cv2.waitKey(3)

            if self._auto_driving:
                if key == 125:  # Stop auto-driving if key is pressed
                    self.get_logger().info(f"Auto driving ending")
                    self._auto_driving = False
                    return

                # Preprocess image for the API
                image_resized = cv2.resize(image, (self._image_size, self._image_size))
                image_pil = PILImage.fromarray(image_resized)  # Use PILImage to resolve conflict
                with open("temp_image.jpg", "wb") as f:
                    image_pil.save(f, format="JPEG")

                # Send image to the API
                with open("temp_image.jpg", "rb") as f:
                    files = {"file": f}
                    response = requests.post(self._url, files=files)

                if response.status_code == 200:
                    data = response.json()
                    predicted_class = data.get("predicted_class")

                    # Drive robot based on prediction
                    if predicted_class == "left":  # Turn left
                        self.turn_left()
                    elif predicted_class == "forward":  # Go straight
                        self.go_straight()
                    elif predicted_class == "right":  # Turn right
                        self.turn_right()
                else:
                    self.get_logger().error(f"API Error: {response.text}")

            else:
                # Manual controls
                if key == 106:  # 'j' - Turn left
                    self.turn_left()
                elif key == 107:  # 'k' - Go straight
                    self.go_straight()
                elif key == 108:  # 'l' - Turn right
                    self.turn_right()
                elif key == 32:  # Space - Stop
                    self.stop()
                elif key == 113:  # 'q' - Quit
                    self.get_logger().info(f"Closing node")
                    exit(0)
                elif key == 120:  # 'x' - Start auto-driving
                    self.get_logger().info(f"Auto driving starting")
                    self._auto_driving = True

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def _command(self, x_vel, theta_vel):
        twist = Twist()
        twist.linear.x = x_vel
        twist.angular.z = theta_vel
        self._pub.publish(twist)

    def go_straight(self):
        self._command(self._x_vel, 0.0)

    def turn_left(self):
        self._command(self._x_vel, self._theta_vel)

    def turn_right(self):
        self._command(self._x_vel, -self._theta_vel)

    def stop(self):
        self._command(0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    node = AutoDriveByLine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
