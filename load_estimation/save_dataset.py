import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose

from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R

class SaveDatasetNode(Node):
    def __init__(self):
        super().__init__('save_dataset_node')
        self.get_logger().info("Save Dataset Node has been started.")

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ========== Subscribers ==========
        self.subscribe_load_pose_ = self.create_subscription(Pose, '/payload/pose', self.load_pose_callback, 2)
        # create best effort QOS for the MoCap topics since they can be high frequency and we only care about the latest pose
        self.subscribe_MoCap_camera_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/camera_test_setup/pose', self.camera_pose_callback, qos_profile=self.qos_profile)
        self.subscribe_MoCap_load_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/protect_load/pose', self.mocap_load_pose_callback, qos_profile=self.qos_profile)

        self.camera_pose = None
        self.mocap_load_pose = None

    def camera_pose_callback(self, msg):
        self.camera_pose = msg

    def mocap_load_pose_callback(self, msg):
        self.mocap_load_pose = msg

    def load_pose_callback_old(self, msg):
        if self.camera_pose is not None and self.mocap_load_pose is not None:
            #transform the estimated load pose to the world frame using the camera pose

            #must account for the camera's orientation as well
            # Assuming the camera pose is in world frame and we want to transform the estimated load pose to world frame
            p_L_C = np.array([msg.position.x, msg.position.y, msg.position.z])

            # Extract camera orientation (quaternion)
            q_C_W = np.array([
                self.camera_pose.pose.orientation.x,
                self.camera_pose.pose.orientation.y,
                self.camera_pose.pose.orientation.z,
                self.camera_pose.pose.orientation.w
            ])

            # rotate camera frame 90 degrees around x to align with load estimation frame
            q_C_W = R.from_quat(q_C_W) #R.from_quat([-0.7071, 0, 0, 0.7071]) * R.from_quat(q_C_W) # 90 deg around x

            # Transform the estimated load pose from camera frame to world frame with quaternion rotation
            p_L_W = q_C_W.apply(p_L_C) + np.array([self.camera_pose.pose.position.x, self.camera_pose.pose.position.y, self.camera_pose.pose.position.z])

            with open('dataset.csv', 'a') as f:
                f.write(f"{self.mocap_load_pose.pose.position.x},{self.mocap_load_pose.pose.position.y},{self.mocap_load_pose.pose.position.z},{p_L_W[0]},{p_L_W[1]},{p_L_W[2]}\n")
            
            #self.get_logger().info(f"{estimated_load_pose.x, estimated_load_pose.y, estimated_load_pose.z} transformed to world frame: {p_W[0], p_W[1], p_W[2]}")

            #with open('dataset.csv', 'a') as f:
            #    f.write(f"{self.camera_pose.pose.position.x}, {self.camera_pose.pose.position.y}, {self.camera_pose.pose.position.z}, "
            #            f"{p_W[0]}, {p_W[1]}, {p_W[2]}\n")
            self.get_logger().info("Dataset entry saved.")
        else:
            self.get_logger().warning("Camera pose or MoCap load pose is not available yet.")
    
    def load_pose_callback(self, msg):
        if self.camera_pose is not None and self.mocap_load_pose is not None:
            #transform the estimated load pose to the world frame using the camera pose

            #must account for the camera's orientation as well
            # Assuming the camera pose is in world frame and we want to transform the estimated load pose to world frame
            p_L_C = np.array([msg.position.x, msg.position.y, msg.position.z])
            q_L_C = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

            # Extract camera orientation (quaternion)

            p_C_W = np.array([self.camera_pose.pose.position.x, self.camera_pose.pose.position.y, self.camera_pose.pose.position.z])
            q_C_W = np.array([
                self.camera_pose.pose.orientation.x,
                self.camera_pose.pose.orientation.y,
                self.camera_pose.pose.orientation.z,
                self.camera_pose.pose.orientation.w
            ])

            p_L_W = np.array([self.mocap_load_pose.pose.position.x, self.mocap_load_pose.pose.position.y, self.mocap_load_pose.pose.position.z])
            q_L_W = np.array([
                self.mocap_load_pose.pose.orientation.x,
                self.mocap_load_pose.pose.orientation.y,
                self.mocap_load_pose.pose.orientation.z,
                self.mocap_load_pose.pose.orientation.w
            ])

            #with open('dataset.csv', 'a') as f:
            #    f.write(f"{q_L_C[0]},{q_L_C[1]},{q_L_C[2]},{q_L_C[3]},{q_L_W[0]},{q_L_W[1]},{q_L_W[2]},{q_L_W[3]}\n")
            
            
            with open('dataset.csv', 'a') as f:
                f.write(f"{p_C_W[0]},{p_C_W[1]},{p_C_W[2]},{q_C_W[0]},{q_C_W[1]},{q_C_W[2]},{q_C_W[3]},{p_L_C[0]},{p_L_C[1]},{p_L_C[2]},{q_L_C[0]},{q_L_C[1]},{q_L_C[2]},{q_L_C[3]},{p_L_W[0]},{p_L_W[1]},{p_L_W[2]},{q_L_W[0]},{q_L_W[1]},{q_L_W[2]},{q_L_W[3]}\n")
            
            #self.get_logger().info(f"{estimated_load_pose.x, estimated_load_pose.y, estimated_load_pose.z} transformed to world frame: {p_W[0], p_W[1], p_W[2]}")

            #with open('dataset.csv', 'a') as f:
            #    f.write(f"{self.camera_pose.pose.position.x}, {self.camera_pose.pose.position.y}, {self.camera_pose.pose.position.z}, "
            #            f"{p_W[0]}, {p_W[1]}, {p_W[2]}\n")
            self.get_logger().info("Dataset entry saved.")
        else:
            self.get_logger().warning("Camera pose or MoCap load pose is not available yet.")


def main(args=None):
    rclpy.init(args=args)
    save_dataset = SaveDatasetNode()
    rclpy.spin(save_dataset)
    save_dataset.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()