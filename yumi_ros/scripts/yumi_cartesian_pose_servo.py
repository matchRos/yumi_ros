#!/usr/bin/env python3
import math
import threading
import copy

import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vec_norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def quat_to_list(q):
    return [q.x, q.y, q.z, q.w]


def pose_to_pos_quat(pose):
    pos = [pose.position.x, pose.position.y, pose.position.z]
    quat = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    return pos, quat


def make_pose(pos, quat):
    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


def normalize_quat(q):
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [v / n for v in q]


def quat_multiply(q1, q2):
    return tf.transformations.quaternion_multiply(q1, q2)


def quat_inverse(q):
    return tf.transformations.quaternion_inverse(q)


def quat_slerp(q0, q1, s):
    return tf.transformations.quaternion_slerp(q0, q1, s)


def shortest_quat(q_from, q_to):
    # Ensure shortest-path interpolation
    dot = sum(q_from[i] * q_to[i] for i in range(4))
    if dot < 0.0:
        return [-q_to[0], -q_to[1], -q_to[2], -q_to[3]]
    return q_to


def quat_error_rotvec(q_current, q_target):
    """
    Returns orientation error as a rotation vector in the current/base frame.
    """
    q_current = normalize_quat(q_current)
    q_target = normalize_quat(q_target)
    q_delta = quat_multiply(q_target, quat_inverse(q_current))
    q_delta = normalize_quat(q_delta)

    # Keep shortest rotation
    if q_delta[3] < 0.0:
        q_delta = [-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3]]

    xyz_norm = math.sqrt(q_delta[0] ** 2 + q_delta[1] ** 2 + q_delta[2] ** 2)
    if xyz_norm < 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(xyz_norm, q_delta[3])
    axis = [q_delta[0] / xyz_norm, q_delta[1] / xyz_norm, q_delta[2] / xyz_norm]
    return vec_scale(axis, angle)


def angular_velocity_from_quats(q_prev, q_next, dt):
    """
    Approximate angular velocity from two nearby desired orientations.
    """
    if dt <= 0.0:
        return [0.0, 0.0, 0.0]

    q_prev = normalize_quat(q_prev)
    q_next = normalize_quat(q_next)
    q_rel = quat_multiply(q_next, quat_inverse(q_prev))
    q_rel = normalize_quat(q_rel)

    if q_rel[3] < 0.0:
        q_rel = [-q_rel[0], -q_rel[1], -q_rel[2], -q_rel[3]]

    xyz_norm = math.sqrt(q_rel[0] ** 2 + q_rel[1] ** 2 + q_rel[2] ** 2)
    if xyz_norm < 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(xyz_norm, q_rel[3])
    axis = [q_rel[0] / xyz_norm, q_rel[1] / xyz_norm, q_rel[2] / xyz_norm]
    return vec_scale(axis, angle / dt)


def quintic_time_scaling(t, T):
    """
    C2 time scaling:
        s(t)   = 10*tau^3 - 15*tau^4 + 6*tau^5
        ds/dt  = (30*tau^2 - 60*tau^3 + 30*tau^4) / T
    """
    if T <= 1e-9:
        return 1.0, 0.0

    tau = clamp(t / T, 0.0, 1.0)
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    ds_dt = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / T
    return s, ds_dt


class CartesianMotionProfile:
    def __init__(self, start_pose, target_pose, duration):
        self.start_pose = copy.deepcopy(start_pose)
        self.target_pose = copy.deepcopy(target_pose)
        self.duration = max(duration, 1e-3)
        self.start_time = rospy.Time.now()

        self.p0, self.q0 = pose_to_pos_quat(self.start_pose)
        self.p1, self.q1 = pose_to_pos_quat(self.target_pose)
        self.q1 = shortest_quat(self.q0, self.q1)

    def desired_pose_and_twist(self, now, dt_preview):
        t = max(0.0, (now - self.start_time).to_sec())
        s, ds_dt = quintic_time_scaling(t, self.duration)

        dp = vec_sub(self.p1, self.p0)
        p_des = vec_add(self.p0, vec_scale(dp, s))
        v_ff = vec_scale(dp, ds_dt)

        q_des = quat_slerp(self.q0, self.q1, s)

        # Preview a small step forward to derive angular feedforward
        t2 = min(t + dt_preview, self.duration)
        s2, _ = quintic_time_scaling(t2, self.duration)
        q_des_2 = quat_slerp(self.q0, self.q1, s2)
        w_ff = angular_velocity_from_quats(q_des, q_des_2, max(dt_preview, 1e-4))

        done = t >= self.duration
        return make_pose(p_des, q_des), v_ff, w_ff, done


class ArmServo:
    def __init__(
        self, name, pose_topic, twist_topic, tip_link, base_frame, tf_listener
    ):
        self.name = name
        self.pose_topic = pose_topic
        self.twist_topic = twist_topic
        self.tip_link = tip_link
        self.base_frame = base_frame
        self.tf_listener = tf_listener

        self.lock = threading.Lock()
        self.active_profile = None

        self.max_linear_speed = rospy.get_param(f"~{name}_max_linear_speed", 0.30)
        self.max_angular_speed = rospy.get_param(f"~{name}_max_angular_speed", 0.90)
        self.min_duration = rospy.get_param(f"~{name}_min_duration", 1.0)

        self.kp_pos = rospy.get_param(f"~{name}_kp_pos", 1.5)
        self.kp_rot = rospy.get_param(f"~{name}_kp_rot", 1.5)

        self.pos_tolerance = rospy.get_param(f"~{name}_pos_tolerance", 0.002)
        self.rot_tolerance = rospy.get_param(f"~{name}_rot_tolerance", 0.03)

        self.pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=1)
        self.sub = rospy.Subscriber(
            self.pose_topic, PoseStamped, self.target_cb, queue_size=1
        )

    def get_current_pose_tf(self):
        self.tf_listener.waitForTransform(
            self.base_frame,
            self.tip_link,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        trans, rot = self.tf_listener.lookupTransform(
            self.base_frame,
            self.tip_link,
            rospy.Time(0),
        )
        pose = Pose()
        pose.position.x = trans[0]
        pose.position.y = trans[1]
        pose.position.z = trans[2]
        pose.orientation.x = rot[0]
        pose.orientation.y = rot[1]
        pose.orientation.z = rot[2]
        pose.orientation.w = rot[3]
        return pose

    def transform_pose_to_base(self, pose_msg):
        if pose_msg.header.frame_id in ["", self.base_frame]:
            return pose_msg

        self.tf_listener.waitForTransform(
            self.base_frame,
            pose_msg.header.frame_id,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        return self.tf_listener.transformPose(self.base_frame, pose_msg)

    def estimate_duration(self, start_pose, target_pose):
        p0, q0 = pose_to_pos_quat(start_pose)
        p1, q1 = pose_to_pos_quat(target_pose)

        dpos = vec_norm(vec_sub(p1, p0))
        rotvec = quat_error_rotvec(q0, q1)
        dang = vec_norm(rotvec)

        T_lin = dpos / max(self.max_linear_speed * 0.5, 1e-3)
        T_rot = dang / max(self.max_angular_speed * 0.5, 1e-3)
        return max(self.min_duration, T_lin, T_rot)

    def target_cb(self, msg):
        try:
            msg_base = self.transform_pose_to_base(msg)
            current_pose = self.get_current_pose_tf()
            duration = self.estimate_duration(current_pose, msg_base.pose)

            with self.lock:
                self.active_profile = CartesianMotionProfile(
                    start_pose=current_pose,
                    target_pose=msg_base.pose,
                    duration=duration,
                )

            rospy.loginfo(
                f"[{self.name}] New Cartesian target accepted, duration={duration:.2f} s"
            )
        except Exception as e:
            rospy.logerr(f"[{self.name}] Failed to accept target pose: {e}")

    def publish_zero(self):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        self.pub.publish(msg)

    def update(self, dt):
        with self.lock:
            profile = self.active_profile

        if profile is None:
            return

        try:
            current_pose = self.get_current_pose_tf()
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[{self.name}] TF lookup failed: {e}")
            return

        now = rospy.Time.now()
        desired_pose, v_ff, w_ff, time_done = profile.desired_pose_and_twist(now, dt)

        p_cur, q_cur = pose_to_pos_quat(current_pose)
        p_des, q_des = pose_to_pos_quat(desired_pose)

        e_pos = vec_sub(p_des, p_cur)
        e_rot = quat_error_rotvec(q_cur, q_des)

        v_fb = vec_scale(e_pos, self.kp_pos)
        w_fb = vec_scale(e_rot, self.kp_rot)

        v_cmd = vec_add(v_ff, v_fb)
        w_cmd = vec_add(w_ff, w_fb)

        # Saturate linear velocity
        v_norm = vec_norm(v_cmd)
        if v_norm > self.max_linear_speed:
            v_cmd = vec_scale(v_cmd, self.max_linear_speed / v_norm)

        # Saturate angular velocity
        w_norm = vec_norm(w_cmd)
        if w_norm > self.max_angular_speed:
            w_cmd = vec_scale(w_cmd, self.max_angular_speed / w_norm)

        msg = TwistStamped()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = v_cmd[0]
        msg.twist.linear.y = v_cmd[1]
        msg.twist.linear.z = v_cmd[2]
        msg.twist.angular.x = w_cmd[0]
        msg.twist.angular.y = w_cmd[1]
        msg.twist.angular.z = w_cmd[2]
        self.pub.publish(msg)

        pos_done = vec_norm(vec_sub(p_des, p_cur)) < self.pos_tolerance
        rot_done = vec_norm(quat_error_rotvec(q_cur, q_des)) < self.rot_tolerance

        if time_done and pos_done and rot_done:
            self.publish_zero()
            with self.lock:
                self.active_profile = None
            rospy.loginfo(f"[{self.name}] Cartesian motion finished")


class YumiCartesianPoseServo:
    def __init__(self):
        rospy.init_node("yumi_cartesian_pose_servo")

        self.rate_hz = rospy.get_param("~rate_hz", 250.0)
        self.base_frame = rospy.get_param("~base_frame", "yumi_base_link")

        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)

        self.left_arm = ArmServo(
            name="left",
            pose_topic=rospy.get_param(
                "~left_pose_topic", "/yumi/robl/cartesian_pose_command"
            ),
            twist_topic=rospy.get_param(
                "~left_twist_topic", "/yumi/robl/cartesian_velocity_command"
            ),
            tip_link=rospy.get_param("~left_tip_link", "yumi_tcp_l"),
            base_frame=self.base_frame,
            tf_listener=self.tf_listener,
        )

        self.right_arm = ArmServo(
            name="right",
            pose_topic=rospy.get_param(
                "~right_pose_topic", "/yumi/robr/cartesian_pose_command"
            ),
            twist_topic=rospy.get_param(
                "~right_twist_topic", "/yumi/robr/cartesian_velocity_command"
            ),
            tip_link=rospy.get_param("~right_tip_link", "yumi_tcp_r"),
            base_frame=self.base_frame,
            tf_listener=self.tf_listener,
        )

        rospy.loginfo(f"YuMi Cartesian pose servo started at {self.rate_hz:.1f} Hz")

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz

        while not rospy.is_shutdown():
            self.left_arm.update(dt)
            self.right_arm.update(dt)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = YumiCartesianPoseServo()
        node.spin()
    except rospy.ROSInterruptException:
        pass
