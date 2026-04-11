#!/usr/bin/env python3
import numpy as np
import rospy
import PyKDL as kdl

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from kdl_parser_py.urdf import treeFromParam


class YumiLeftArmCartesianVelocityController:
    def __init__(self):
        self.base_link = rospy.get_param("~base_link", "yumi_robl_link_1")
        self.tip_link = rospy.get_param("~tip_link", "yumi_robl_link_7")
        self.robot_description_param = rospy.get_param(
            "~robot_description_param", "/yumi/robot_description"
        )

        self.command_timeout = rospy.get_param("~command_timeout", 0.2)
        self.publish_rate = rospy.get_param("~publish_rate", 100.0)
        self.max_joint_velocity = rospy.get_param("~max_joint_velocity", 0.2)
        self.damping = rospy.get_param("~damping", 0.03)

        self.left_joint_names = [
            "yumi_robl_joint_1",
            "yumi_robl_joint_2",
            "yumi_robl_joint_3",
            "yumi_robl_joint_4",
            "yumi_robl_joint_5",
            "yumi_robl_joint_6",
            "yumi_robl_joint_7",
        ]

        self.current_joint_map = {}
        self.latest_twist = np.zeros(6)
        self.latest_twist_time = rospy.Time(0)

        self.chain = self._build_kdl_chain()
        self.num_joints = self.chain.getNrOfJoints()

        if self.num_joints != 7:
            rospy.logwarn(f"Expected 7 joints in left arm chain, got {self.num_joints}")

        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        rospy.Subscriber(
            "/yumi/egm/joint_states",
            JointState,
            self.joint_state_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/robl/cartesian_velocity_command",
            TwistStamped,
            self.twist_cb,
            queue_size=1,
        )

        self.pub = rospy.Publisher(
            "/yumi/joint_group_velocity_command",
            Float64MultiArray,
            queue_size=1,
        )

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.update)

        rospy.loginfo("YuMi left arm Cartesian velocity controller started")

    def _build_kdl_chain(self):
        ok, tree = treeFromParam(self.robot_description_param)
        if not ok:
            raise RuntimeError(
                f"Could not parse URDF from parameter {self.robot_description_param}"
            )

        chain = tree.getChain(self.base_link, self.tip_link)
        return chain

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def twist_cb(self, msg):
        self.latest_twist = np.array(
            [
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z,
                msg.twist.angular.x,
                msg.twist.angular.y,
                msg.twist.angular.z,
            ],
            dtype=float,
        )
        self.latest_twist_time = rospy.Time.now()

    def get_left_joint_positions(self):
        q = []
        for joint_name in self.left_joint_names:
            if joint_name not in self.current_joint_map:
                return None
            q.append(self.current_joint_map[joint_name])
        return np.array(q, dtype=float)

    def compute_jacobian(self, q_np):
        q_kdl = kdl.JntArray(len(q_np))
        for i in range(len(q_np)):
            q_kdl[i] = q_np[i]

        jac_kdl = kdl.Jacobian(len(q_np))
        self.jac_solver.JntToJac(q_kdl, jac_kdl)

        J = np.zeros((6, len(q_np)))
        for r in range(6):
            for c in range(len(q_np)):
                J[r, c] = jac_kdl[r, c]
        return J

    def damped_pseudoinverse(self, J, damping):
        m, n = J.shape
        if m >= n:
            return np.linalg.inv(J.T @ J + (damping**2) * np.eye(n)) @ J.T
        return J.T @ np.linalg.inv(J @ J.T + (damping**2) * np.eye(m))

    def saturate(self, qdot, limit):
        qdot = np.asarray(qdot).copy()
        for i in range(len(qdot)):
            qdot[i] = np.clip(qdot[i], -limit, limit)
        return qdot

    def publish_joint_velocity_command(self, qdot_left):
        msg = Float64MultiArray()
        full_cmd = np.zeros(14, dtype=float)
        full_cmd[:7] = qdot_left
        msg.data = full_cmd.tolist()
        self.pub.publish(msg)

    def update(self, event):
        if event.last_real is None:
            return

        q_left = self.get_left_joint_positions()
        if q_left is None:
            rospy.logwarn_throttle(1.0, "Waiting for complete left arm joint states")
            return

        now = rospy.Time.now()
        if (now - self.latest_twist_time).to_sec() > self.command_timeout:
            desired_twist = np.zeros(6)
        else:
            desired_twist = self.latest_twist.copy()

        J = self.compute_jacobian(q_left)
        J_pinv = self.damped_pseudoinverse(J, self.damping)

        qdot_left = J_pinv @ desired_twist
        qdot_left = self.saturate(qdot_left, self.max_joint_velocity)

        self.publish_joint_velocity_command(qdot_left)


if __name__ == "__main__":
    rospy.init_node("yumi_left_arm_cartesian_velocity_controller")
    try:
        YumiLeftArmCartesianVelocityController()
        rospy.spin()
    except Exception as exc:
        rospy.logerr(f"Failed to start Cartesian velocity controller: {exc}")
