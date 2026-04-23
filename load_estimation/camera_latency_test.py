import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import time
import gpiod


class LedImageTimerNode(Node):

    def __init__(self):
        super().__init__('led_image_timer_node')

        # ---------------- CONFIG ----------------
        self.frame_interval = 30     # blink every N frames
        self.brightness_threshold = 180  # 0–255 scale

        # ---------------- GPIO ----------------
        self.chip = gpiod.Chip('/dev/gpiochip4')
        self.led = self.chip.get_line(17)
        self.led.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT)

        # ---------------- ROS ----------------
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        # ---------------- STATE ----------------
        self.frame_count = 0
        self.led_state = False

        self.timer_running = False
        self.start_time = None

        self.get_logger().info("Brightness-based LED Timer Node started")

    # =========================================================
    # CALLBACK
    # =========================================================
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        self.frame_count += 1

        # ---------------- LED BLINK ----------------
        if self.frame_count % self.frame_interval == 0:

            self.led_state = not self.led_state
            self.led.set_value(int(self.led_state))

            if self.led_state and not self.timer_running:
                self.start_timer()

        # ---------------- MONITOR CENTER PIXEL ----------------
        if self.timer_running:

            if self.is_center_pixel_bright(frame):
                self.stop_timer()

    # =========================================================
    # TIMER
    # =========================================================
    def start_timer(self):
        self.start_time = time.time()
        self.timer_running = True
        self.get_logger().info("Timer started")

    def stop_timer(self):
        elapsed = time.time() - self.start_time

        self.timer_running = False
        self.start_time = None

        self.get_logger().info(f"BRIGHT DETECTED → elapsed time: {elapsed:.4f} sec")

    # =========================================================
    # BRIGHTNESS CHECK
    # =========================================================
    def is_center_pixel_bright(self, frame):

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2

        b, g, r = frame[cy, cx]

        # luminance (standard perceived brightness)
        brightness = 0.114 * b + 0.587 * g + 0.299 * r

        return brightness > self.brightness_threshold


# =========================================================
# MAIN
# =========================================================
def main(args=None):
    rclpy.init(args=args)

    node = LedImageTimerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.led.set_value(0)
        node.led.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
