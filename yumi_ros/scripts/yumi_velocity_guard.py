#!/usr/bin/env python3
import math
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class YumiJointVelocityGuard:
    def __init__(self):
        self.n_joints = 14
        self.rate_hz = rospy.get_param("~rate", 100.0)
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.2)

        # Conservative default limits
        self.v_max = rospy.get_param("~v_max", [0.20] * self.n_joints)
        self.a_max = rospy.get_param("~a_max", [0.40] * self.n_joints)

        # Joint position limits [rad]
        self.q_min = rospy.get_param("~q_min", [-3.14] * self.n_joints)
        self.q_max = rospy.get_param("~q_max", [3.14] * self.n_joints)
        self.limit_margin = rospy.get_param("~limit_margin", [0.05] * self.n_joints)

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

        self.current_pos = None
        self.current_joint_names = None

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
        if len(msg.name) < self.n_joints or len(msg.position) < self.n_joints:
            rospy.logwarn_throttle(1.0, "JointState message too short")
            return

        if self.current_joint_names is None:
            self.current_joint_names = list(msg.name[: self.n_joints])

            if self.current_joint_names != self.joint_names_expected:
                rospy.logwarn("Joint name order differs from expected order")
                rospy.logwarn(f"Received: {self.current_joint_names}")
                rospy.logwarn(f"Expected: {self.joint_names_expected}")
            else:
                rospy.loginfo("Joint name order matches expected YuMi EGM order")

        self.current_pos = list(msg.position[: self.n_joints])

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

            # Near upper limit: only allow motion back away from the limit
            if q >= q_max - margin:
                out[i] = min(out[i], 0.0)

            # Near lower limit: only allow motion back away from the limit
            if q <= q_min + margin:
                out[i] = max(out[i], 0.0)

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
