import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from load_estimation.marker_detection.PoseEstimator import PoseEstimator
from load_estimation.marker_detection.MarkerTracker import MarkerTracker

import cv2
import time
import math
import numpy as np
#import cProfile # For profiling



class LoadEstimationNode(Node):
    def __init__(self):
        super().__init__('load_estimation_node')
        self.get_logger().info("Load Estimation Node has been started.")

        # ========== Publishers ==========
        self.publish_error_ = self.create_publisher(Vector3, '/gimbal/error', 2)
        self.publish_image_gray_ = self.create_publisher(Image, '/debug/image_gray', 2)
        self.publish_image_thres_ = self.create_publisher(Image, '/debug/image_thres', 2)
        self.publish_image_detected_ = self.create_publisher(Image, '/debug/image_detected', 2)
        self.publish_image_payload_ = self.create_publisher(Image, '/debug/image_payload', 2)

        # ========== Subscribers ==========
        self.subscribe_image_ = self.create_subscription(Image, '/image_raw', self.image_callback, 2)

        # ========== timers ==========
        timer_period = 1/30  # 30 Hz

        # Camera intrinsics and distortion
        #intrinsics = np.array([[835.4362078622368, 0, 323.0605420101571],
        #                        [0, 835.9483791851382, 232.14120929722597],
        #                        [0, 0, 1]], dtype=float)
        #dist_coeffs = np.array([-0.0999921394506428, 2.185188066835036, -0.005726667745540125, 0.00027787706601120816, -7.636164458366145], dtype=float)

        intrinsics = np.array([[410.5372217, 0, 165.56905082],
                               [0, 410.26619602, 119.30780434],
                               [0, 0, 1]], dtype=float)
        dist_coeffs = np.array([-2.92583245e-02, 1.00683957e+00, -2.29972697e-03, 9.16865223e-04,-3.11679202e+00], dtype=float)
        

        marker_ids = [17, 27, 39, 119]
        marker_placements = {marker_ids[0]: (-0.16, -0.16, 0.0),
                            marker_ids[1]: (0.16, -0.16, 0.0),
                            marker_ids[2]: (0.16, 0.16, 0.0),
                            marker_ids[3]: (-0.16, 0.16, 0.0)}

        self.downscale_factor = 1
        self.LP = PoseEstimator(intrinsics, dist_coeffs, marker_ids, marker_placements, alpha=0.5, max_reproj_error=10.0, downscale_factor=self.downscale_factor)

        marker_order = 5
        self.MT = MarkerTracker(marker_order, int(13/self.downscale_factor), 1000, marker_ids, self.downscale_factor)


        self.t0 = time.time()
        self.total_frames = 0
        self.total_time = 0

        self.bridge = CvBridge()

    def image_callback(self, msg: Image):
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Publish the grayscale image for debugging
        self.publish_image_gray_.publish(self.bridge.cv2_to_imgmsg(gray, encoding='mono8'))

        marker_positions, debug_image = self.MT.locate_marker(gray)
        self.publish_image_thres_.publish(self.bridge.cv2_to_imgmsg(debug_image.astype(np.uint8), encoding='mono8'))
        #self.publish_image_detected_.publish(self.bridge.cv2_to_imgmsg(marker_image.astype(np.uint8), encoding='mono8'))

        if marker_positions is not None:
            self.get_logger().info(f"Found {len(marker_positions)} markers.")

            if len(marker_positions) >= 3:
                pose = self.LP.estimate_load_pose(marker_positions)

                if pose is not None:
                    rvec = pose[0]
                    tvec = pose[1]
                    print(f"Estimated pose: rvec: {rvec.flatten()}, tvec: {tvec.flatten()}")

                    # calc roll and pitch from tvec
                    x, y, z = tvec[0], tvec[1], tvec[2] # in camera frame to payload frame
                    pitch = math.atan2(y, z) * (180 / math.pi) # pitch in gimbal, roll in camera frame
                    yaw = -math.atan2(x, z) * (180 / math.pi) # yaw in gimbal, -pitch in camera frame

                    # Publish the setpoint
                    error = Vector3(x=yaw, y=pitch, z=0.0)
                    self.publish_error_.publish(error)
                    self.get_logger().info(f"Published error: yaw: {yaw:.2f}, pitch: {pitch:.2f}")

                    display_frame = cv_image.copy()
                    for pose in marker_positions:
                        x = int(pose.x * self.downscale_factor)
                        y = int(pose.y * self.downscale_factor)
                        cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{pose.id}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    self.LP.display_pose(display_frame)
                    self.publish_image_payload_.publish(self.bridge.cv2_to_imgmsg(display_frame, encoding='bgr8'))
                else:
                    self.get_logger().info("No markers detected.")
                    error = Vector3(x=0.0, y=0.0, z=0.0)
                    self.publish_error_.publish(error)
            else:
                self.get_logger().info("No markers detected.")
                error = Vector3(x=0.0, y=0.0, z=0.0)
                self.publish_error_.publish(error)
        else:
            self.get_logger().info("No markers detected.")
            error = Vector3(x=0.0, y=0.0, z=0.0)
            self.publish_error_.publish(error)

                







def main(args=None):
    rclpy.init(args=args)
    load_estimation = LoadEstimationNode()
    rclpy.spin(load_estimation)
    load_estimation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()