import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped
from sensor_msgs.msg import Image

import cv2
import time
import numpy as np
from cv_bridge import CvBridge

class WriteVideoNode(Node):
    def __init__(self):
        super().__init__('write_video_node')
        self.get_logger().info("Write Video Node has been started.")

        # ========== Subscribers ==========
        self.subscribe_image_ = self.create_subscription(Image, '/image_raw', self.image_callback, 2)

        # Video writer
        self.frame_width_ = 320
        self.frame_height_ = 240
        self.fps_ = 30
        self.bridge_ = CvBridge()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_filename = f"gimbal_angles_{timestamp}.avi"
        self.video_writer_ = cv2.VideoWriter(video_filename, fourcc, self.fps_, (self.frame_width_, self.frame_height_))
        self.get_logger().info(f"Started writing video: {video_filename}")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge_.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Received image frame, writing to video... shape: ", cv_image.shape)
        # Write frame to video
        self.video_writer_.write(cv_image)


def main(args=None):
    rclpy.init(args=args)
    write_video_node = WriteVideoNode()
    rclpy.spin(write_video_node)
    write_video_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()