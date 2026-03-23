import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped, Pose

import numpy as np
from scipy.spatial.transform import Rotation as R

class TransformationNode(Node):
    def __init__(self):
        super().__init__('transformation_node')
        self.get_logger().info("Transformation Node has been started.")

        # ========== Publishers ==========
        self.publish_drone_payload_vector_ = self.create_publisher(Vector3, '/payload/vector', 2)
        self.publish_payload_world_pose_ = self.create_publisher(Pose, '/payload/world_pose', 2) 

        # ========== Subscribers ==========
        self.subscribe_payload_pose_ = self.create_subscription(Pose, '/payload/pose', self.payload_pose_callback, 2)
        self.subscribe_gimbal_angles_ = self.create_subscription(Vector3, '/gimbal/world_angles', self.gimbal_angles_callback, 2)
        self.subscribe_drone_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/protect_drone/pose', self.drone_pose_callback, 2)


        self.gimbal_angles = Vector3()
        self.payload_pose = Pose()
        self.drone_pose = PoseStamped()

        self.d_drone_to_gimbal = 0.15 # 15 cm from drone to gimbal
        self.d_gimbal_to_camera = 0.025 # 2.5 cm from gimbal to camera


    def payload_pose_callback(self, msg):
        self.payload_pose = msg
        self.get_logger().info(f"Received payload pose: {msg.position.x}, {msg.position.y}, {msg.position.z}")

    def drone_pose_callback(self, msg):
        self.drone_pose = msg
        self.get_logger().info(f"Received drone pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def gimbal_angles_callback(self, msg):
        self.gimbal_angles = msg
        self.get_logger().info(f"Received gimbal angles: {msg.x}, {msg.y}, {msg.z}")


        # 1. World -> Drone
        p_D_W = np.array([self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, self.drone_pose.pose.position.z])
        q_D_W = np.array([self.drone_pose.pose.orientation.x, self.drone_pose.pose.orientation.y, self.drone_pose.pose.orientation.z, self.drone_pose.pose.orientation.w])
        r_D_W = R.from_quat(q_D_W)
        
        # 2. Drone -> Gimbal base
        p_G0_D = np.array([0,0,-self.d_drone_to_gimbal])
        q_G0_D = R.from_euler('y', 90, degrees=True).as_quat()  # [x,y,z,w]

        # 3. World -> Gimbal base
        p_G0_W = p_D_W + r_D_W.apply(p_G0_D)
        q_G0_W = r_D_W * R.from_quat(q_G0_D)
        self.get_logger().info(f"Gimbal base position in world: {p_G0_W}, Gimbal base orientation in world (quat): {q_G0_W.as_quat()}, angles (deg): {q_G0_W.as_euler('xyz', degrees=True)}")

        # 4. Extract drone yaw (around world Z)
        yaw_D = np.arctan2(2*(q_D_W[3]*q_D_W[2] + q_D_W[0]*q_D_W[1]), 1 - 2*(q_D_W[1]**2 + q_D_W[2]**2))
        self.get_logger().info(f"Drone yaw (deg): {np.rad2deg(yaw_D):.2f}, Gimbal angles (deg): x={self.gimbal_angles.x:.2f}, y={self.gimbal_angles.y:.2f}, z={self.gimbal_angles.z:.2f}")

        # 5. Compute X-axis compensation rotation based on drone yaw
        theta_x_eff = self.gimbal_angles.x + np.rad2deg(yaw_D)

        # 6. Gimbal base -> Gimbal (pure rotation)
        q_G_G0 = R.from_euler('zyx', [self.gimbal_angles.z, self.gimbal_angles.y, theta_x_eff], degrees=True).as_quat()  # [x,y,z,w]

        # 7. World -> Gimbal
        p_G_W = p_G0_W# + R.from_quat(q_G0_W).apply(p_G_G0)
        q_G_W = R.from_euler('y', 90, degrees=True).as_quat()  # 90 deg rotation for intermidiate frame between world and gimbal
        q_G_W = R.from_quat(q_G_W) * R.from_quat(q_G_G0)

        # 8. Gimbal -> Camera
        p_C_G = np.array([self.d_gimbal_to_camera,0,0])
        q_C_G = R.from_euler('yx', [90, -90], degrees=True).as_quat() # Alternative way to get the same rotation

        # 9. World -> Camera
        p_C_W = p_G_W + q_G_W.apply(p_C_G)
        q_C_W = q_G_W * R.from_quat(q_C_G)
        angles = q_C_W.as_euler('xyz', degrees=True)
        self.get_logger().info(f"Camera position in world: {p_C_W[0]:.3f}, {p_C_W[1]:.3f}, {p_C_W[2]:.3f}, angles (deg): {angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}")

        # 10. Camera -> Payload
        p_P_C = np.array([self.payload_pose.position.x, self.payload_pose.position.y, self.payload_pose.position.z])
        q_P_C = np.array([self.payload_pose.orientation.x, self.payload_pose.orientation.y, self.payload_pose.orientation.z, self.payload_pose.orientation.w])

        # 11. World -> Payload
        p_P_W = p_C_W + q_C_W.apply(p_P_C)
        q_P_W = q_C_W * R.from_quat(q_P_C)
        angles = q_P_W.as_euler('xyz', degrees=True)
        self.get_logger().info(f"Payload position in world: {p_P_W[0]:.3f}, {p_P_W[1]:.3f}, {p_P_W[2]:.3f}, angles (deg): {angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}")

        payload_world_pose = Pose(position=Vector3(x=p_P_W[0], y=p_P_W[1], z=p_P_W[2]), orientation=Vector3(x=angles[0], y=angles[1], z=angles[2]))
        self.publish_payload_world_pose_.publish(payload_world_pose)

        drone_payload_vector = Vector3(x=p_P_W[0] - p_D_W[0], y=p_P_W[1] - p_D_W[1], z=p_P_W[2] - p_D_W[2])
        self.publish_drone_payload_vector_.publish(drone_payload_vector)


def main(args=None):
    rclpy.init(args=args)
    transformation_node = TransformationNode()
    rclpy.spin(transformation_node)
    transformation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    