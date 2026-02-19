import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped

import numpy as np

class TransformationNode(Node):
    def __init__(self):
        super().__init__('transformation_node')
        self.get_logger().info("Transformation Node has been started.")

        # ========== Publishers ==========
        self.publish_drone_payload_vector_ = self.create_publisher(Vector3, '/gimbal/error', 2)

        # ========== Subscribers ==========
        self.subscribe_camera_load_angles_ = self.create_subscription(Vector3, '/gimbal/camera_load_angle', self.camera_load_angles_callback, 2)
        self.subscribe_gimbal_angles_ = self.create_subscription(Vector3, '/gimbal/world_angles', self.gimbal_angles_callback, 2)
        self.subscribe_drone_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/protect_drone/pose', self.drone_pose_callback, 2)


        self.gimbal_angles = Vector3()
        self.camera_load_angles = Vector3()
        self.drone_pose = PoseStamped()

        self.distance_to_gimbal = 0.2 # 20 cm from drone to gimbal, with gimbal facing downwards


    def camera_load_angles_callback(self, msg):
        self.camera_load_angles = msg
        self.get_logger().info(f"Received camera load angles: {msg.x}, {msg.y}, {msg.z}")

    def drone_pose_callback(self, msg):
        self.drone_pose = msg
        self.get_logger().info(f"Received drone pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def gimbal_angles_callback(self, msg):
        self.gimbal_angles = msg
        self.get_logger().info(f"Received gimbal angles: {msg.x}, {msg.y}, {msg.z}")


        drone_to_gimbal_base_transform = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,-self.distance_to_gimbal],[0,0,0,1]]) # 20 cm from drone to gimbal, with gimbal facing downwards

        
        gimbal_base_to_gimbal_transform = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) # Identity transform for gimbal base to gimbal transform (no offset)
        # In a real application this would be computed from the gimbal's internal angles and offsets
        # For now we assume no offset between gimbal base and gimbal itself


def main(args=None):
    rclpy.init(args=args)
    transformation_node = TransformationNode()
    rclpy.spin(transformation_node)
    transformation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()