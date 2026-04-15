import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3
from nmpc_interfaces.msg import State
from std_msgs.msg import Float64

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
        self.subscribe_load_pose_ = self.create_subscription(Pose, '/payload/pose', self.load_pose_world_no_transform_callback, 2)
        # create best effort QOS for the MoCap topics since they can be high frequency and we only care about the latest pose
        self.subscribe_MoCap_camera_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/camera_test_setup/pose', self.camera_pose_callback, qos_profile=self.qos_profile)
        self.subscribe_MoCap_drone_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/protect_drone/pose', self.mocap_drone_pose_callback, qos_profile=self.qos_profile)
        self.subscribe_MoCap_load_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/protect_load/pose', self.mocap_load_pose_callback, qos_profile=self.qos_profile)
        self.subscribe_gimbal_angles_ = self.create_subscription(Vector3, '/gimbal/angles', self.gimbal_angles_callback, 2)
        #self.subscribe_payload_world_pose_ = self.create_subscription(Pose, '/payload/world_pose', self.load_pose_world_callback, 2)

        self.subscribe_nmpc_estimated_state_ = self.create_subscription(State, '/nmpc/estimated_state', self.nmpc_estimated_state_callback, qos_profile=self.qos_profile)
        self.subscribe_nmpc_estimated_state_ukf_ = self.create_subscription(State, '/nmpc/estimated_state_ukf', self.nmpc_estimated_state_ukf_callback, qos_profile=self.qos_profile)
        self.subscribe_nmpc_estimated_bias_ = self.create_subscription(Float64, 'nmpc/estimated_bias', self.nmpc_estimated_bias_callback, 10)

        self.camera_pose = None
        self.mocap_load_pose = None
        self.mocap_drone_pose = None
        self.gimbal_angles = Vector3()
        self.nmpc_estimated_state = State()
        self.nmpc_estimated_bias = None


    def camera_pose_callback(self, msg):
        self.camera_pose = msg

    def mocap_drone_pose_callback(self, msg):
        self.mocap_drone_pose = msg

    def mocap_load_pose_callback(self, msg):
        self.mocap_load_pose = msg

    def gimbal_angles_callback(self, msg):
        self.gimbal_angles = msg
    
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

            

            with open('dataset.csv', 'a') as f:
                f.write(f"{p_C_W[0]},{p_C_W[1]},{p_C_W[2]},{q_C_W[0]},{q_C_W[1]},{q_C_W[2]},{q_C_W[3]},{p_L_C[0]},{p_L_C[1]},{p_L_C[2]},{q_L_C[0]},{q_L_C[1]},{q_L_C[2]},{q_L_C[3]},{p_L_W[0]},{p_L_W[1]},{p_L_W[2]},{q_L_W[0]},{q_L_W[1]},{q_L_W[2]},{q_L_W[3]}\n")
            
            #self.get_logger().info(f"{estimated_load_pose.x, estimated_load_pose.y, estimated_load_pose.z} transformed to world frame: {p_W[0], p_W[1], p_W[2]}")

            #with open('dataset.csv', 'a') as f:
            #    f.write(f"{self.camera_pose.pose.position.x}, {self.camera_pose.pose.position.y}, {self.camera_pose.pose.position.z}, "
            #            f"{p_W[0]}, {p_W[1]}, {p_W[2]}\n")
            self.get_logger().info("Dataset entry saved.")
        else:
            self.get_logger().warning("Camera pose or MoCap load pose is not available yet.")
    
    def load_pose_world_no_transform_callback(self, msg):
        if self.mocap_drone_pose is not None and self.mocap_load_pose is not None:
            #transform the estimated load pose to the world frame using the camera pose

            #must account for the camera's orientation as well
            # Assuming the camera pose is in world frame and we want to transform the estimated load pose to world frame
            p_L_C = np.array([msg.position.x, msg.position.y, msg.position.z])
            q_L_C = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

            # Extract camera orientation (quaternion)

            p_D_W = np.array([self.mocap_drone_pose.pose.position.x, self.mocap_drone_pose.pose.position.y, self.mocap_drone_pose.pose.position.z])
            q_D_W = np.array([
                self.mocap_drone_pose.pose.orientation.x,
                self.mocap_drone_pose.pose.orientation.y,
                self.mocap_drone_pose.pose.orientation.z,
                self.mocap_drone_pose.pose.orientation.w
            ])

            p_L_W = np.array([self.mocap_load_pose.pose.position.x, self.mocap_load_pose.pose.position.y, self.mocap_load_pose.pose.position.z])
            q_L_W = np.array([
                self.mocap_load_pose.pose.orientation.x,
                self.mocap_load_pose.pose.orientation.y,
                self.mocap_load_pose.pose.orientation.z,
                self.mocap_load_pose.pose.orientation.w
            ])

            R_G_G0 = np.array([self.gimbal_angles.x, self.gimbal_angles.y, 0]) # Assuming roll is 0

            with open('dataset.csv', 'a') as f:
                f.write(f"{p_D_W[0]},{p_D_W[1]},{p_D_W[2]},{q_D_W[0]},{q_D_W[1]},{q_D_W[2]},{q_D_W[3]},{p_L_C[0]},{p_L_C[1]},{p_L_C[2]},{q_L_C[0]},{q_L_C[1]},{q_L_C[2]},{q_L_C[3]},{p_L_W[0]},{p_L_W[1]},{p_L_W[2]},{q_L_W[0]},{q_L_W[1]},{q_L_W[2]},{q_L_W[3]},{R_G_G0[0]},{R_G_G0[1]},{R_G_G0[2]}\n")
            
            #self.get_logger().info(f"{estimated_load_pose.x, estimated_load_pose.y, estimated_load_pose.z} transformed to world frame: {p_W[0], p_W[1], p_W[2]}")

            #with open('dataset.csv', 'a') as f:
            #    f.write(f"{self.camera_pose.pose.position.x}, {self.camera_pose.pose.position.y}, {self.camera_pose.pose.position.z}, "
            #            f"{p_W[0]}, {p_W[1]}, {p_W[2]}\n")
            self.get_logger().info("Dataset entry saved.")
        else:
            self.get_logger().warning("Camera pose or MoCap load pose is not available yet.")
    
    def load_pose_world_callback(self, msg):
        if self.mocap_load_pose is not None:
            p_L_W = np.array([msg.position.x, msg.position.y, msg.position.z])
            q_L_W = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

            mocap_p_L_W = np.array([self.mocap_load_pose.pose.position.x, self.mocap_load_pose.pose.position.y, self.mocap_load_pose.pose.position.z])
            mocap_q_L_W = np.array([
                self.mocap_load_pose.pose.orientation.x,
                self.mocap_load_pose.pose.orientation.y,
                self.mocap_load_pose.pose.orientation.z,
                self.mocap_load_pose.pose.orientation.w
            ])

            with open('dataset_world.csv', 'a') as f:
                f.write(f"{p_L_W[0]},{p_L_W[1]},{p_L_W[2]},{q_L_W[0]},{q_L_W[1]},{q_L_W[2]},{q_L_W[3]},{mocap_p_L_W[0]},{mocap_p_L_W[1]},{mocap_p_L_W[2]},{mocap_q_L_W[0]},{mocap_q_L_W[1]},{mocap_q_L_W[2]},{mocap_q_L_W[3]}\n")
            
            self.get_logger().info("World dataset entry saved.")
        else:
            self.get_logger().warning("MoCap load pose is not available yet.")

    def nmpc_estimated_state_callback(self, msg):
        self.nmpc_estimated_state = msg
        
    def nmpc_estimated_bias_callback(self, msg):
        self.nmpc_estimated_bias = msg

    def nmpc_estimated_state_ukf_callback(self, msg):
        if self.nmpc_estimated_bias is not None:
            self.nmpc_estimated_state_ukf = msg

            nmpc = self.nmpc_estimated_state
            nmpc_ukf = self.nmpc_estimated_state_ukf
            nmpc_bias = self.nmpc_estimated_bias.data

            with open('dataset_nmpc.csv', 'a') as f:
                f.write(f"{nmpc.load_position[0]},{nmpc.load_position[1]},{nmpc.load_position[2]},"
                        f"{nmpc.load_velocity[0]},{nmpc.load_velocity[1]},{nmpc.load_velocity[2]},"
                        f"{nmpc.cable_vector[0]},{nmpc.cable_vector[1]},{nmpc.cable_vector[2]},"
                        f"{nmpc.load_angular_velocity[0]},{nmpc.load_angular_velocity[1]},{nmpc.load_angular_velocity[2]},"
                        f"{nmpc_ukf.load_position[0]},{nmpc_ukf.load_position[1]},{nmpc_ukf.load_position[2]},"
                        f"{nmpc_ukf.load_velocity[0]},{nmpc_ukf.load_velocity[1]},{nmpc_ukf.load_velocity[2]},"
                        f"{nmpc_ukf.cable_vector[0]},{nmpc_ukf.cable_vector[1]},{nmpc_ukf.cable_vector[2]},"
                        f"{nmpc_ukf.load_angular_velocity[0]},{nmpc_ukf.load_angular_velocity[1]},{nmpc_ukf.load_angular_velocity[2]},"
                        f"{nmpc_bias}\n")
            
            self.get_logger().info("NMPC dataset entry saved.")


def main(args=None):
    rclpy.init(args=args)
    save_dataset = SaveDatasetNode()
    rclpy.spin(save_dataset)
    save_dataset.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()