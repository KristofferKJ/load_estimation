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
        self.get_logger().info("Initializing LED Image Timer Node")

        # ========== Subscribers ==========
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # ========== Timers ==========
        timer_period = 1/1000  # Check every 1/1000 sec for brightness (adjust as needed)
        self.timer_LED = self.create_timer(timer_period, self.led_timer_callback)

        # ========== GPIO Setup ==========
        self.chip = gpiod.Chip('/dev/gpiochip4')
        self.led = self.chip.get_line(17)
        self.led.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT)

        # ========== CONFIG ==========
        self.frame_interval = 30     # blink every N frames
        self.turn_LED_on_x_ms_after_frame = 20  # ms after frame to turn on LED
        self.brightness_threshold = 180  # 0–255 scale

        self.bridge = CvBridge()

        # ========== STATE ==========
        self.frame_count = 0
        self.LED_turn_on_counter = 0
        self.led_state = False
        self.turn_LED_on = False
        self.LED_turned_on = False

        self.timer_running = False
        self.start_time = None

        self.get_logger().info("Brightness-based LED Timer Node started")

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        self.frame_count += 1

        # ---------------- LED BLINK ----------------
        if self.frame_count % self.frame_interval == 0:
            self.turn_LED_on = True

        # ---------------- MONITOR CENTER PIXEL ----------------
        if self.timer_running:
            if self.is_center_pixel_bright(frame):
                self.stop_timer()

    def led_timer_callback(self):
        # turn on LED 20 after frame_interval frames.
        if self.turn_LED_on:
            self.LED_turn_on_counter += 1

            if self.LED_turn_on_counter == self.turn_LED_on_x_ms_after_frame:
                self.led.set_value(1)  # Turn on LED
                self.LED_turned_on = True
                self.start_timer()  # Start timer when LED is turned on
                self.get_logger().info("LED turned ON")


         

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
