#!/usr/bin/env python3
import numpy as np
import rospy
import PyKDL as kdl
import tf

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from kdl_parser_py.urdf import treeFromParam


class SingleArmKinematics:
    def __init__(self, tree, arm_name, base_link, tip_link, joint_names, damping):
        self.arm_name = arm_name
        self.base_link = base_link
        self.tip_link = tip_link
        self.joint_names = joint_names
        self.damping = damping

        self.chain = tree.getChain(self.base_link, self.tip_link)
        self.num_joints = self.chain.getNrOfJoints()
        self.num_segments = self.chain.getNrOfSegments()
        self.chain_joint_names = self.get_chain_joint_names()
        self.command_to_chain_indices = None
        self.chain_to_command_indices = None

        rospy.loginfo(
            f"[{self.arm_name}] base_link={self.base_link}, tip_link={self.tip_link}, "
            f"segments={self.num_segments}, joints={self.num_joints}"
        )
        rospy.loginfo(f"[{self.arm_name}] KDL chain joints={self.chain_joint_names}")

        if self.num_joints != 7:
            rospy.logwarn(
                f"[{self.arm_name}] Expected 7 joints in chain, got {self.num_joints}"
            )
        if self.chain_joint_names != self.joint_names:
            rospy.logwarn(
                f"[{self.arm_name}] KDL joint order differs from command order: "
                f"KDL={self.chain_joint_names}, command={self.joint_names}"
            )
        if set(self.chain_joint_names) == set(self.joint_names):
            self.command_to_chain_indices = [
                self.joint_names.index(name) for name in self.chain_joint_names
            ]
            self.chain_to_command_indices = [
                self.chain_joint_names.index(name) for name in self.joint_names
            ]
        else:
            rospy.logerr(
                f"[{self.arm_name}] KDL and command joint sets differ; refusing IK"
            )

        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

    def get_chain_joint_names(self):
        names = []
        for i in range(self.num_segments):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getType() != kdl.Joint.Fixed:
                names.append(joint.getName())
        return names

    def compute_jacobian(self, q_chain):
        q_kdl = kdl.JntArray(len(q_chain))
        for i in range(len(q_chain)):
            q_kdl[i] = q_chain[i]

        jac_kdl = kdl.Jacobian(len(q_chain))
        self.jac_solver.JntToJac(q_kdl, jac_kdl)

        J = np.zeros((6, len(q_chain)))
        for r in range(6):
            for c in range(len(q_chain)):
                J[r, c] = jac_kdl[r, c]
        return J

    def damped_pseudoinverse(self, J):
        m, n = J.shape
        lam2 = self.damping**2
        if m >= n:
            return np.linalg.inv(J.T @ J + lam2 * np.eye(n)) @ J.T
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(m))

    def cartesian_to_joint_velocity(self, q_command, desired_twist):
        if self.num_joints != 7 or self.command_to_chain_indices is None:
            return np.zeros(7, dtype=float)

        q_chain = q_command[self.command_to_chain_indices]
        J = self.compute_jacobian(q_chain)
        J_pinv = self.damped_pseudoinverse(J)
        qdot_chain = J_pinv @ desired_twist
        return qdot_chain[self.chain_to_command_indices]

    def joint_velocity_to_cartesian(self, q_command, qdot_command):
        if self.num_joints != 7 or self.command_to_chain_indices is None:
            return np.zeros(6, dtype=float)

        q_chain = q_command[self.command_to_chain_indices]
        qdot_chain = qdot_command[self.command_to_chain_indices]
        J = self.compute_jacobian(q_chain)
        return J @ qdot_chain


class YumiDualArmCartesianVelocityController:
    def __init__(self):
        self.robot_description_param = rospy.get_param(
            "~robot_description_param", "/robot_description"
        )
        self.command_timeout = rospy.get_param("~command_timeout", 0.2)
        self.publish_rate = rospy.get_param("~publish_rate", 250.0)
        self.max_joint_velocity = rospy.get_param("~max_joint_velocity", 1.5)
        self.damping = rospy.get_param("~damping", 0.03)
        self.min_joint_velocity = rospy.get_param("~min_joint_velocity", 0.0)
        self.min_joint_velocity_eps = rospy.get_param("~min_joint_velocity_eps", 1e-4)
        self.diagnostics_enabled = rospy.get_param("~diagnostics_enabled", True)
        self.diagnostics_period = rospy.get_param("~diagnostics_period", 0.5)

        self.left_input_topic = rospy.get_param(
            "~left_input_topic", "/yumi/robl/cartesian_velocity_command"
        )
        self.right_input_topic = rospy.get_param(
            "~right_input_topic", "/yumi/robr/cartesian_velocity_command"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/yumi/joint_group_velocity_command"
        )
        self.publish_only_when_active = rospy.get_param(
            "~publish_only_when_active", True
        )

        rospy.loginfo(f"robot_description_param = {self.robot_description_param}")

        ok, tree = treeFromParam(self.robot_description_param)
        rospy.loginfo(f"treeFromParam ok = {ok}")
        if not ok:
            raise RuntimeError(
                f"Could not parse URDF from parameter {self.robot_description_param}"
            )

        self.left_joint_names = [
            "yumi_robl_joint_1",
            "yumi_robl_joint_2",
            "yumi_robl_joint_3",
            "yumi_robl_joint_4",
            "yumi_robl_joint_5",
            "yumi_robl_joint_6",
            "yumi_robl_joint_7",
        ]
        self.right_joint_names = [
            "yumi_robr_joint_1",
            "yumi_robr_joint_2",
            "yumi_robr_joint_3",
            "yumi_robr_joint_4",
            "yumi_robr_joint_5",
            "yumi_robr_joint_6",
            "yumi_robr_joint_7",
        ]

        self.left_arm = SingleArmKinematics(
            tree=tree,
            arm_name="left",
            base_link=rospy.get_param("~left_base_link", "yumi_base_link"),
            tip_link=rospy.get_param("~left_tip_link", "yumi_tcp_l"),
            joint_names=self.left_joint_names,
            damping=self.damping,
        )

        self.right_arm = SingleArmKinematics(
            tree=tree,
            arm_name="right",
            base_link=rospy.get_param("~right_base_link", "yumi_base_link"),
            tip_link=rospy.get_param("~right_tip_link", "yumi_tcp_r"),
            joint_names=self.right_joint_names,
            damping=self.damping,
        )

        self.current_joint_map = {}
        self.current_joint_velocity_map = {}

        self.latest_left_twist = np.zeros(6, dtype=float)
        self.latest_right_twist = np.zeros(6, dtype=float)
        self.latest_left_twist_time = rospy.Time(0)
        self.latest_right_twist_time = rospy.Time(0)
        self.tf_listener = tf.TransformListener()

        rospy.Subscriber(
            "/yumi/egm/joint_states",
            JointState,
            self.joint_state_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.left_input_topic,
            TwistStamped,
            self.left_twist_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.right_input_topic,
            TwistStamped,
            self.right_twist_cb,
            queue_size=1,
        )

        self.pub = rospy.Publisher(
            self.output_topic,
            Float64MultiArray,
            queue_size=1,
        )

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.update)

        rospy.loginfo("YuMi dual-arm Cartesian velocity controller started")

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos
        for name, vel in zip(msg.name, msg.velocity):
            self.current_joint_velocity_map[name] = vel

    def apply_min_joint_velocity(self, qdot, min_vel, eps):
        qdot = np.asarray(qdot).copy()
        if min_vel <= 0.0:
            return qdot

        max_abs = np.max(np.abs(qdot))
        if eps < max_abs < min_vel:
            qdot *= min_vel / max_abs
        return qdot

    def twist_msg_to_np(self, msg, target_frame):
        twist = np.array(
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

        source_frame = msg.header.frame_id.strip()
        if source_frame in ["", target_frame]:
            return twist

        try:
            self.tf_listener.waitForTransform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
            _, quat = self.tf_listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            rotation = tf.transformations.quaternion_matrix(quat)[:3, :3]
            twist[:3] = rotation @ twist[:3]
            twist[3:] = rotation @ twist[3:]
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                f"Could not transform twist from {source_frame} to {target_frame}: {exc}",
            )
            return np.zeros(6, dtype=float)

        return twist

    def left_twist_cb(self, msg):
        self.latest_left_twist = self.twist_msg_to_np(
            msg, self.left_arm.base_link
        )
        self.latest_left_twist_time = rospy.Time.now()

    def right_twist_cb(self, msg):
        self.latest_right_twist = self.twist_msg_to_np(
            msg, self.right_arm.base_link
        )
        self.latest_right_twist_time = rospy.Time.now()

    def get_joint_positions(self, joint_names):
        q = []
        for joint_name in joint_names:
            if joint_name not in self.current_joint_map:
                return None
            q.append(self.current_joint_map[joint_name])
        return np.array(q, dtype=float)

    def get_joint_velocities(self, joint_names):
        qdot = []
        for joint_name in joint_names:
            if joint_name not in self.current_joint_velocity_map:
                return None
            qdot.append(self.current_joint_velocity_map[joint_name])
        return np.array(qdot, dtype=float)

    def get_active_twist(self, latest_twist, latest_time):
        now = rospy.Time.now()
        if (now - latest_time).to_sec() > self.command_timeout:
            return np.zeros(6, dtype=float)
        return latest_twist.copy()

    def saturate_preserve_direction(self, qdot, limit):
        qdot = np.asarray(qdot).copy()
        max_abs = np.max(np.abs(qdot))
        if max_abs <= limit or max_abs < 1e-9:
            return qdot
        scale = limit / max_abs
        return qdot * scale

    def publish_joint_velocity_command(self, qdot_left, qdot_right):
        msg = Float64MultiArray()
        full_cmd = np.zeros(14, dtype=float)
        full_cmd[:7] = qdot_left
        full_cmd[7:] = qdot_right
        msg.data = full_cmd.tolist()
        self.pub.publish(msg)

    def format_vec(self, vec):
        return "[" + ", ".join(f"{v:+.4f}" for v in vec) + "]"

    def direction_cosine(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-9 or b_norm < 1e-9:
            return float("nan")
        return float(np.dot(a, b) / (a_norm * b_norm))

    def log_arm_diagnostics(
        self,
        arm_name,
        arm,
        q,
        qdot_feedback,
        desired_twist,
        qdot_command,
        arm_active,
    ):
        if not self.diagnostics_enabled or not arm_active or qdot_feedback is None:
            return

        desired_norm = np.linalg.norm(desired_twist)
        if desired_norm < 1e-9:
            return

        commanded_twist = arm.joint_velocity_to_cartesian(q, qdot_command)
        feedback_twist = arm.joint_velocity_to_cartesian(q, qdot_feedback)

        desired_linear = desired_twist[:3]
        commanded_linear = commanded_twist[:3]
        feedback_linear = feedback_twist[:3]

        cmd_cos = self.direction_cosine(desired_linear, commanded_linear)
        fb_cos = self.direction_cosine(desired_linear, feedback_linear)


    def update(self, event):
        if event.last_real is None:
            return

        # dont publish if neither arm has an active command. might mess with other controllers that are active
        left_active = (
            rospy.Time.now() - self.latest_left_twist_time
        ).to_sec() <= self.command_timeout
        right_active = (
            rospy.Time.now() - self.latest_right_twist_time
        ).to_sec() <= self.command_timeout

        desired_left_twist = self.get_active_twist(
            self.latest_left_twist, self.latest_left_twist_time
        )
        desired_right_twist = self.get_active_twist(
            self.latest_right_twist, self.latest_right_twist_time
        )

        if self.publish_only_when_active and not left_active and not right_active:
            return

        q_left = self.get_joint_positions(self.left_joint_names)
        q_right = self.get_joint_positions(self.right_joint_names)
        qdot_left_feedback = self.get_joint_velocities(self.left_joint_names)
        qdot_right_feedback = self.get_joint_velocities(self.right_joint_names)

        if q_left is None or q_right is None:
            rospy.logwarn_throttle(1.0, "Waiting for complete YuMi joint states")
            return

        desired_left_twist = self.get_active_twist(
            self.latest_left_twist, self.latest_left_twist_time
        )
        desired_right_twist = self.get_active_twist(
            self.latest_right_twist, self.latest_right_twist_time
        )

        qdot_left_raw = self.left_arm.cartesian_to_joint_velocity(
            q_left, desired_left_twist
        )
        qdot_right_raw = self.right_arm.cartesian_to_joint_velocity(
            q_right, desired_right_twist
        )

        qdot_left_sat = self.saturate_preserve_direction(
            qdot_left_raw, self.max_joint_velocity
        )
        qdot_right_sat = self.saturate_preserve_direction(
            qdot_right_raw, self.max_joint_velocity
        )

        qdot_left_final = self.apply_min_joint_velocity(
            qdot_left_sat, self.min_joint_velocity, self.min_joint_velocity_eps
        )
        qdot_right_final = self.apply_min_joint_velocity(
            qdot_right_sat, self.min_joint_velocity, self.min_joint_velocity_eps
        )

        self.log_arm_diagnostics(
            "left",
            self.left_arm,
            q_left,
            qdot_left_feedback,
            desired_left_twist,
            qdot_left_final,
            left_active,
        )
        self.log_arm_diagnostics(
            "right",
            self.right_arm,
            q_right,
            qdot_right_feedback,
            desired_right_twist,
            qdot_right_final,
            right_active,
        )

        self.publish_joint_velocity_command(qdot_left_final, qdot_right_final)


if __name__ == "__main__":
    rospy.init_node("yumi_dual_arm_cartesian_velocity_controller")
    try:
        YumiDualArmCartesianVelocityController()
        rospy.spin()
    except Exception as exc:
        rospy.logerr(
            f"Failed to start YuMi dual-arm Cartesian velocity controller: {exc}"
        )
