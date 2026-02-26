import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped

import numpy as np
from scipy.spatial.transform import Rotation as R

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

        self.d_drone_to_gimbal = 0.15 # 15 cm from drone to gimbal
        self.d_gimbal_to_camera = 0.025 # 2.5 cm from gimbal to camera


    def camera_load_angles_callback(self, msg):
        self.camera_load_angles = msg
        self.get_logger().info(f"Received camera load angles: {msg.x}, {msg.y}, {msg.z}")

    def drone_pose_callback(self, msg):
        self.drone_pose = msg
        self.get_logger().info(f"Received drone pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def gimbal_angles_callback(self, msg):
        self.gimbal_angles = msg
        self.get_logger().info(f"Received gimbal angles: {msg.x}, {msg.y}, {msg.z}")


        # 1. World -> Drone
        p_D_W = np.array([self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, self.drone_pose.pose.position.z])
        q_D_W = np.array([self.drone_pose.pose.orientation.x, self.drone_pose.pose.orientation.y, self.drone_pose.pose.orientation.z, self.drone_pose.pose.orientation.w])
        r_D_W = R.from_quat([self.drone_pose.pose.orientation.x, self.drone_pose.pose.orientation.y, self.drone_pose.pose.orientation.z, self.drone_pose.pose.orientation.w])
        
        # 2. Drone -> Gimbal base
        p_G0_D = np.array([0,0,-self.d_drone_to_gimbal])
        R_G0_D = np.array([[0,0,1],
                           [0,1,0],
                           [-1,0,0]])                           # Gimbal base is rotated 90 deg around Y relative to drone
        q_G0_D = R.from_matrix(R_G0_D).as_quat()  # [x,y,z,w]

        # 3. World -> Gimbal base
        p_G0_W = p_D_W + r_D_W.apply(p_G0_D)
        q_G0_W = r_D_W * R.from_quat(q_G0_D)
        self.get_logger().info(f"Gimbal base position in world: {p_G0_W}, Gimbal base orientation in world (quat): {q_G0_W.as_quat()}, angles (deg): {q_G0_W.as_euler('xyz', degrees=True)}")

        # 4. Extract drone yaw (around world Z)
        yaw_D = np.arctan2(2*(q_D_W[3]*q_D_W[2] + q_D_W[0]*q_D_W[1]), 1 - 2*(q_D_W[1]**2 + q_D_W[2]**2))
        self.get_logger().info(f"Drone yaw (deg): {np.rad2deg(yaw_D):.2f}, Gimbal angles (deg): x={self.gimbal_angles.x:.2f}, y={self.gimbal_angles.y:.2f}, z={self.gimbal_angles.z:.2f}")

        # 5. Compute X-axis compensation rotation based on drone yaw
        theta_x_eff = self.gimbal_angles.x + np.rad2deg(yaw_D)

        # 6. World -> Gimbal
        q_W_G = np.array([0.0, 0.70710678, 0.0, 0.70710678])  # 90 deg rotation for intermidiate frame between world and gimbal
        r = R.from_euler('zyx', [self.gimbal_angles.z, self.gimbal_angles.y, theta_x_eff], degrees=True)
        q_G = r.as_quat()  # [x,y,z,w]
        q_G_W = R.from_quat(q_W_G) * R.from_quat(q_G)

        # 7. Gimbal -> Camera
        p_C_G = np.array([self.d_gimbal_to_camera,0,0])
        R_C_G = np.array([[0,0,1],
                          [-1,0,0],
                          [0,-1,0]])                           # Camera is rotated 90 deg around Y and 90 deg around X relative to gimbal
        q_C_G = R.from_matrix(R_C_G)

        # 8. World -> Camera
        q_C_W = q_G_W * q_C_G
        p_C_W = p_G0_W + q_G_W.apply(p_C_G)
        self.get_logger().info(f"Camera position in world: {p_C_W}, Camera orientation in world (quat): {q_C_W.as_quat()}, angles (deg): {q_C_W.as_euler('xyz', degrees=True)}")

        




    def object_pose_world(
        p_D, q_D,                 # Drone position and quaternion in world (numpy array, quaternion as [x,y,z,w])
        theta_x, theta_y, theta_z, # Gimbal measured angles in degrees
        p_O_C, q_O_C               # Object pose in camera frame
    ):
        """
        Returns object pose in world frame given drone, gimbal, camera measurements.

        p_D: np.array(3,) drone position in world
        q_D: np.array(4,) drone quaternion in world [x,y,z,w]
        theta_x, theta_y, theta_z: gimbal measured angles (deg)
        p_O_C: np.array(3,) object position in camera frame
        q_O_C: np.array(4,) object orientation in camera frame [x,y,z,w]
        """

        # 1. Drone → Gimbal base
        p_G0_D = np.array([0,0,-0.15])  # Gimbal base relative to drone
        # Gimbal base rotation relative to drone (quaternion)
        R_G0_D = np.array([[0,0,1],
                        [0,1,0],
                        [-1,0,0]])
        q_G0_D = R.from_matrix(R_G0_D).as_quat()  # [x,y,z,w]

        # 2. Drone in world
        r_D = R.from_quat(q_D)

        # 3. Gimbal base in world
        p_G0_W = p_D + r_D.apply(p_G0_D)
        q_G0_W = r_D * R.from_quat(q_G0_D)
        print(f"gimbal base position in world: {p_G0_W}, gimbal base orientation in world (quat): {q_G0_W.as_quat()}, angles (deg): {q_G0_W.as_euler('xyz', degrees=True)}")

        # 4. Extract drone yaw (around world Z)
        yaw_D = np.arctan2(2*(q_D[3]*q_D[2] + q_D[0]*q_D[1]), 1 - 2*(q_D[1]**2 + q_D[2]**2))
        print(f"Drone yaw (deg): {np.rad2deg(yaw_D):.2f}, Gimbal angles (deg): x={theta_x:.2f}, y={theta_y:.2f}, z={theta_z:.2f}")

        # 5. Compute X-axis compensation rotation based on drone yaw
        theta_x_eff = theta_x + np.rad2deg(yaw_D)
        
        # 6. Build gimbal quaternions
        q_W_G = np.array([0.0, 0.70710678, 0.0, 0.70710678])  # 90 deg rotation around Y to align gimbal with drone
        r = R.from_euler('zyx', [theta_z, theta_y, theta_x_eff], degrees=True)
        q_G = r.as_quat()  # [x,y,z,w]
        q_G_W = R.from_quat(q_W_G) * R.from_quat(q_G)

        # 7. Camera offset
        p_C_G = np.array([1,0,0]) # 0.025
        # Camera rotation relative to gimbal
        R_C_G = np.array([[0,0,1],
                        [-1,0,0],
                        [0,-1,0]])
        q_C_G = R.from_matrix(R_C_G)

        # Camera pose in world
        q_C_W = q_G_W * q_C_G
        p_C_W = p_G0_W + q_G_W.apply(p_C_G)
        print(f"Camera position in world: {p_C_W}, Camera orientation in world (quat): {q_C_W.as_quat()}, angles (deg): {q_C_W.as_euler('xyz', degrees=True)}")

        # 8. Object pose in world
        r_O_C = R.from_quat(q_O_C)
        q_O_W = q_C_W * r_O_C
        p_O_W = p_C_W + q_C_W.apply(p_O_C)


def main(args=None):
    rclpy.init(args=args)
    transformation_node = TransformationNode()
    rclpy.spin(transformation_node)
    transformation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    