#!/usr/bin/env python3
import math
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF


class YumiJointVelocityGuard:
    def __init__(self):
        self.n_joints = 14
        self.rate_hz = rospy.get_param("~rate", 100.0)
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.2)
        self.robot_description_param = rospy.get_param(
            "~robot_description_param", "/robot_description"
        )

        self.joint_names_expected = [
            "yumi_robl_joint_1",
            "yumi_robl_joint_2",
            "yumi_robl_joint_3",
            "yumi_robl_joint_4",
            "yumi_robl_joint_5",
            "yumi_robl_joint_6",
            "yumi_robl_joint_7",
            "yumi_robr_joint_1",
            "yumi_robr_joint_2",
            "yumi_robr_joint_3",
            "yumi_robr_joint_4",
            "yumi_robr_joint_5",
            "yumi_robr_joint_6",
            "yumi_robr_joint_7",
        ]

        # Conservative default limits only as fallback
        self.v_max = rospy.get_param("~v_max", [0.40] * self.n_joints)
        self.a_max = rospy.get_param("~a_max", [0.40] * self.n_joints)
        self.limit_margin = rospy.get_param("~limit_margin", [0.03] * self.n_joints)

        self.q_min = [-3.14] * self.n_joints
        self.q_max = [3.14] * self.n_joints
        self.load_joint_limits_from_urdf()

        self.current_pos = None
        self.target_cmd = [0.0] * self.n_joints
        self.filtered_cmd = [0.0] * self.n_joints
        self.last_cmd_time = rospy.Time(0)

        self.pub = rospy.Publisher(
            "/yumi/egm/joint_group_velocity_controller/command",
            Float64MultiArray,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/joint_group_velocity_command",
            Float64MultiArray,
            self.command_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/egm/joint_states", JointState, self.joint_state_cb, queue_size=1
        )

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.update)

    def load_joint_limits_from_urdf(self):
        try:
            robot = URDF.from_parameter_server(key=self.robot_description_param)
            joint_map = {j.name: j for j in robot.joints}

            q_min = []
            q_max = []

            for joint_name in self.joint_names_expected:
                if joint_name not in joint_map:
                    rospy.logwarn(
                        f"Joint {joint_name} not found in URDF, using fallback limits"
                    )
                    q_min.append(-3.14)
                    q_max.append(3.14)
                    continue

                joint = joint_map[joint_name]
                if joint.limit is None:
                    rospy.logwarn(
                        f"Joint {joint_name} has no URDF limit, using fallback limits"
                    )
                    q_min.append(-3.14)
                    q_max.append(3.14)
                    continue

                q_min.append(float(joint.limit.lower))
                q_max.append(float(joint.limit.upper))

            self.q_min = q_min
            self.q_max = q_max

            rospy.loginfo("Loaded joint position limits from URDF:")
            for i, name in enumerate(self.joint_names_expected):
                rospy.loginfo(f"  {name}: [{self.q_min[i]:.4f}, {self.q_max[i]:.4f}]")

        except Exception as e:
            rospy.logwarn(f"Failed to load joint limits from URDF: {e}")
            rospy.logwarn("Using fallback joint limits")

    def command_cb(self, msg):
        if len(msg.data) != self.n_joints:
            rospy.logwarn_throttle(
                1.0, f"Expected {self.n_joints} joint velocities, got {len(msg.data)}"
            )
            return

        cmd = []
        for value in msg.data:
            if not math.isfinite(value):
                rospy.logwarn_throttle(
                    1.0, "Rejected command because it contains NaN or Inf"
                )
                return
            cmd.append(float(value))

        self.target_cmd = cmd
        self.last_cmd_time = rospy.Time.now()

    def joint_state_cb(self, msg):
        if len(msg.name) != len(msg.position):
            rospy.logwarn_throttle(1.0, "JointState name/position length mismatch")
            return

        joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}

        reordered = []
        for joint_name in self.joint_names_expected:
            if joint_name not in joint_map:
                rospy.logwarn_throttle(
                    1.0, f"Missing joint in JointState: {joint_name}"
                )
                return
            reordered.append(joint_map[joint_name])

        self.current_pos = reordered

    def clamp_velocity(self, cmd):
        out = [0.0] * self.n_joints
        for i in range(self.n_joints):
            out[i] = max(-self.v_max[i], min(self.v_max[i], cmd[i]))
        return out

    def apply_acceleration_limit(self, target, dt):
        out = [0.0] * self.n_joints
        for i in range(self.n_joints):
            dv_max = self.a_max[i] * dt
            dv = target[i] - self.filtered_cmd[i]
            dv = max(-dv_max, min(dv_max, dv))
            out[i] = self.filtered_cmd[i] + dv
        return out

    def apply_joint_limit_protection(self, cmd):
        if self.current_pos is None:
            return [0.0] * self.n_joints

        out = list(cmd)

        for i in range(self.n_joints):
            q = self.current_pos[i]
            q_min = self.q_min[i]
            q_max = self.q_max[i]
            margin = self.limit_margin[i]

            if q >= q_max - margin and out[i] > 0.0:
                rospy.logwarn_throttle(
                    0.5,
                    f"Blocking joint {self.joint_names_expected[i]} near upper limit: "
                    f"q={q:.4f}, q_max={q_max:.4f}, cmd={out[i]:.4f}",
                )
                out[i] = 0.0

            if q <= q_min + margin and out[i] < 0.0:
                rospy.logwarn_throttle(
                    0.5,
                    f"Blocking joint {self.joint_names_expected[i]} near lower limit: "
                    f"q={q:.4f}, q_min={q_min:.4f}, cmd={out[i]:.4f}",
                )
                out[i] = 0.0

        return out

    def publish_command(self, cmd):
        msg = Float64MultiArray()
        msg.data = cmd
        self.pub.publish(msg)

    def update(self, event):
        if event.last_real is None:
            return

        dt = (event.current_real - event.last_real).to_sec()
        if dt <= 0.0:
            return

        if self.current_pos is None:
            return

        now = rospy.Time.now()
        if (now - self.last_cmd_time).to_sec() > self.cmd_timeout:
            raw_cmd = [0.0] * self.n_joints
        else:
            raw_cmd = list(self.target_cmd)

        limited_cmd = self.clamp_velocity(raw_cmd)
        limited_cmd = self.apply_joint_limit_protection(limited_cmd)
        limited_cmd = self.apply_acceleration_limit(limited_cmd, dt)

        self.filtered_cmd = limited_cmd

        self.publish_command(self.filtered_cmd)


if __name__ == "__main__":
    rospy.init_node("yumi_joint_velocity_guard")
    YumiJointVelocityGuard()
    rospy.spin()
