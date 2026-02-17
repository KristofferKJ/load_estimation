import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

downscale_factor = 2
#cv2.setNumThreads(0) # Disable OpenCV multithreading to avoid conflicts with ROS2's multithreading


class CameraDriverNode(Node):
    def __init__(self):
        super().__init__('camera_driver_node')
        self.get_logger().info("Camera Driver Node has been started.")

        # ========== Publishers ==========
        self.publish_image_ = self.create_publisher(Image, '/camera/image_gray', 2)

        # ========== Timers ==========
        timer_period = 1/30  # 30 Hz
        self.timer_capture = self.create_timer(timer_period, self.capture_and_process)


        self.bridge = CvBridge()
        # Select the camera where the images should be grabbed from.
        self.camera = cv2.VideoCapture(2)
        #self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BGR3'))
        #self.set_camera_resolution()
        # Reduce buffering
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame
        # Disable auto exposure
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        # Fast shutter (requires lots of light!)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, -8)         # range: -1 .. -13 (lower = faster)
        # Keep gain low for less noise
        self.camera.set(cv2.CAP_PROP_GAIN, 200)
        if not self.camera.isOpened():
            print("Could not open video stream")
            exit()

    def capture_and_process(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0,0), fx=1/downscale_factor, fy=1/downscale_factor)
        self.get_logger().info(f"Captured and processed frame at {gray.shape[1]}x{gray.shape[0]} resolution")
        self.publish_image_.publish(self.bridge.cv2_to_imgmsg(gray, encoding='mono8'))



        






def main(args=None):
    rclpy.init(args=args)
    camera_driver = CameraDriverNode()
    rclpy.spin(camera_driver)
    camera_driver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()