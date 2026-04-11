#!/usr/bin/env python3
import bisect
import threading

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory


class YumiExecuteBridge:
    def __init__(self):
        self.rate_hz = rospy.get_param("~rate", 100.0)
        self.kp = rospy.get_param("~kp", 2.0)
        self.max_vel = rospy.get_param("~max_vel", 0.25)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.01)
        self.goal_settle_time = rospy.get_param("~goal_settle_time", 0.3)

        self.expected_joint_order = [
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
        self.left_joint_names = self.expected_joint_order[:7]
        self.right_joint_names = self.expected_joint_order[7:]
        self.n_joints = len(self.expected_joint_order)

        self.current_joint_pos = None
        self.current_joint_vel = None
        self.current_joint_names = None

        self.active_traj = None
        self.active_traj_start = None
        self.active_goal_reached_since = None
        self.lock = threading.Lock()

        self.cmd_pub = rospy.Publisher(
            "/yumi/joint_group_velocity_command",
            Float64MultiArray,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/egm/joint_states",
            JointState,
            self.joint_state_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/moveit_joint_trajectory",
            JointTrajectory,
            self.trajectory_cb,
            queue_size=1,
        )

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.update)

        rospy.loginfo("YuMi execute bridge started")

    def joint_state_cb(self, msg: JointState):
        if len(msg.name) < self.n_joints or len(msg.position) < self.n_joints:
            rospy.logwarn_throttle(1.0, "JointState message too short")
            return

        if self.current_joint_names is None:
            self.current_joint_names = list(msg.name[: self.n_joints])
            if self.current_joint_names != self.expected_joint_order:
                rospy.logwarn("JointState order differs from expected YuMi order")
                rospy.logwarn(f"Received: {self.current_joint_names}")
                rospy.logwarn(f"Expected: {self.expected_joint_order}")
            else:
                rospy.loginfo("JointState order matches expected YuMi order")

        name_to_idx = {name: i for i, name in enumerate(msg.name)}

        try:
            pos = [msg.position[name_to_idx[j]] for j in self.expected_joint_order]
            if len(msg.velocity) >= len(msg.name):
                vel = [msg.velocity[name_to_idx[j]] for j in self.expected_joint_order]
            else:
                vel = [0.0] * self.n_joints
        except KeyError as exc:
            rospy.logwarn_throttle(1.0, f"Missing joint in JointState: {exc}")
            return

        self.current_joint_pos = np.array(pos, dtype=float)
        self.current_joint_vel = np.array(vel, dtype=float)

    def trajectory_cb(self, msg: JointTrajectory):
        if len(msg.joint_names) == 0:
            rospy.logwarn("Received trajectory without joint names")
            return

        if len(msg.points) == 0:
            rospy.logwarn("Received empty trajectory")
            return

        if self.current_joint_pos is None:
            rospy.logwarn("Cannot accept trajectory yet: no current joint state")
            return

        mapped = self._map_trajectory_to_expected_order(msg)
        if mapped is None:
            return

        with self.lock:
            self.active_traj = mapped
            self.active_traj_start = rospy.Time.now()
            self.active_goal_reached_since = None

        rospy.loginfo(
            f"Accepted new trajectory with {len(mapped['points'])} points, "
            f"duration {mapped['times'][-1]:.3f} s, "
            f"mode={mapped['mode']}"
        )

    def _detect_mode(self, traj_joint_names):
        traj_set = set(traj_joint_names)
        left_set = set(self.left_joint_names)
        right_set = set(self.right_joint_names)
        full_set = set(self.expected_joint_order)

        if traj_set == full_set and len(traj_joint_names) == self.n_joints:
            return "both_arms"
        if traj_set == left_set and len(traj_joint_names) == 7:
            return "left_arm"
        if traj_set == right_set and len(traj_joint_names) == 7:
            return "right_arm"
        return None

    def _map_trajectory_to_expected_order(self, msg: JointTrajectory):
        mode = self._detect_mode(msg.joint_names)
        if mode is None:
            rospy.logwarn(
                "Trajectory joint set does not match left_arm, right_arm, or both_arms"
            )
            rospy.logwarn(f"Received joints: {msg.joint_names}")
            return None

        name_to_idx = {name: i for i, name in enumerate(msg.joint_names)}
        q_current = self.current_joint_pos.copy()

        points = []
        times = []

        for i, pt in enumerate(msg.points):
            if len(pt.positions) != len(msg.joint_names):
                rospy.logwarn(f"Point {i} has invalid positions length")
                return None

            q = q_current.copy()
            qd = np.zeros(self.n_joints, dtype=float)

            for joint_name in msg.joint_names:
                exp_idx = self.expected_joint_order.index(joint_name)
                src_idx = name_to_idx[joint_name]
                q[exp_idx] = pt.positions[src_idx]

                if len(pt.velocities) == len(msg.joint_names):
                    qd[exp_idx] = pt.velocities[src_idx]

            t = pt.time_from_start.to_sec()
            times.append(t)
            points.append({"q": q, "qd": qd})

        if any(times[i] < times[i - 1] for i in range(1, len(times))):
            rospy.logwarn("Trajectory time_from_start is not monotonic")
            return None

        if len(times) == 0:
            rospy.logwarn("Trajectory has no valid time points")
            return None

        if abs(times[0]) > 1e-9:
            rospy.logwarn("First trajectory point does not start at t=0.0; continuing")

        return {"times": times, "points": points, "mode": mode}

    def _interpolate(self, traj, t):
        times = traj["times"]
        points = traj["points"]

        if t <= times[0]:
            return points[0]["q"].copy(), points[0]["qd"].copy()

        if t >= times[-1]:
            return points[-1]["q"].copy(), points[-1]["qd"].copy()

        idx = bisect.bisect_right(times, t) - 1
        idx = max(0, min(idx, len(times) - 2))

        t0 = times[idx]
        t1 = times[idx + 1]
        p0 = points[idx]
        p1 = points[idx + 1]

        if t1 <= t0:
            return p1["q"].copy(), p1["qd"].copy()

        alpha = (t - t0) / (t1 - t0)

        q = (1.0 - alpha) * p0["q"] + alpha * p1["q"]
        qd = (1.0 - alpha) * p0["qd"] + alpha * p1["qd"]
        return q, qd

    def _clip(self, qdot):
        return np.clip(qdot, -self.max_vel, self.max_vel)

    def _publish_cmd(self, qdot):
        msg = Float64MultiArray()
        msg.data = qdot.tolist()
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._publish_cmd(np.zeros(self.n_joints, dtype=float))

    def update(self, event):
        if event.last_real is None:
            return

        if self.current_joint_pos is None:
            return

        with self.lock:
            traj = self.active_traj
            traj_start = self.active_traj_start

        if traj is None or traj_start is None:
            return

        t = (rospy.Time.now() - traj_start).to_sec()
        q_des, qd_ff = self._interpolate(traj, t)

        q_err = q_des - self.current_joint_pos
        qdot_cmd = qd_ff + self.kp * q_err
        qdot_cmd = self._clip(qdot_cmd)

        self._publish_cmd(qdot_cmd)

        final_q = traj["points"][-1]["q"]
        final_err = np.max(np.abs(final_q - self.current_joint_pos))
        traj_finished = t >= traj["times"][-1]

        with self.lock:
            if traj_finished and final_err < self.goal_tolerance:
                if self.active_goal_reached_since is None:
                    self.active_goal_reached_since = rospy.Time.now()
                elif (
                    rospy.Time.now() - self.active_goal_reached_since
                ).to_sec() >= self.goal_settle_time:
                    rospy.loginfo("Trajectory execution finished")
                    self.active_traj = None
                    self.active_traj_start = None
                    self.active_goal_reached_since = None
                    self._stop()
            else:
                self.active_goal_reached_since = None


if __name__ == "__main__":
    rospy.init_node("yumi_execute_bridge")
    YumiExecuteBridge()
    rospy.spin()
