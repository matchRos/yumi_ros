#!/usr/bin/env python3
import sys
import copy
import math

import rospy
import tf
import PyKDL as kdl
import moveit_commander
import numpy as np

from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Bool, String
from kdl_parser_py.urdf import treeFromParam
from urdf_parser_py.urdf import URDF


def _param_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class ArmModel:
    def __init__(
        self,
        name,
        group,
        base_link,
        tip_link,
        elbow_link,
        joint_names,
        joint_min,
        joint_max,
    ):
        self.name = name
        self.group = group
        self.base_link = base_link
        self.tip_link = tip_link
        self.elbow_link = elbow_link
        self.joint_names = joint_names
        self.joint_min = np.array(joint_min, dtype=float)
        self.joint_max = np.array(joint_max, dtype=float)


class YumiMoveItPoseTopics:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.base_frame = rospy.get_param("~base_frame", "yumi_base_link")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.robot_description_param = rospy.get_param(
            "~robot_description_param", "/robot_description"
        )

        # comparing all candidate plans can be very time-consuming,
        # so this option allows you to only use the first valid plan found
        self.compare_all_plans = rospy.get_param("~compare_all_plans", True)

        self.velocity_scaling = rospy.get_param("~velocity_scaling", 0.2)
        self.acceleration_scaling = rospy.get_param("~acceleration_scaling", 0.2)
        self.retime_trajectories = _param_bool(
            rospy.get_param("~retime_trajectories", True)
        )
        self.trajectory_time_scale = max(
            1.0, float(rospy.get_param("~trajectory_time_scale", 1.0))
        )
        self.fallback_joint_speed = max(
            1e-3, float(rospy.get_param("~fallback_joint_speed", 0.05))
        )
        self.min_segment_duration = max(
            0.0, float(rospy.get_param("~min_segment_duration", 0.05))
        )
        self.waypoint_velocity_scaling = rospy.get_param(
            "~waypoint_velocity_scaling", self.velocity_scaling
        )
        self.waypoint_acceleration_scaling = rospy.get_param(
            "~waypoint_acceleration_scaling", self.acceleration_scaling
        )
        self.waypoint_trajectory_time_scale = max(
            1.0,
            float(rospy.get_param("~waypoint_trajectory_time_scale", self.trajectory_time_scale)),
        )
        self.waypoint_fallback_joint_speed = max(
            1e-3,
            float(rospy.get_param("~waypoint_fallback_joint_speed", self.fallback_joint_speed)),
        )
        self.waypoint_min_segment_duration = max(
            0.0,
            float(rospy.get_param("~waypoint_min_segment_duration", self.min_segment_duration)),
        )
        self.log_plan_details_enabled = _param_bool(
            rospy.get_param("~log_plan_details", True)
        )
        self.plan_debug_sample_count = int(rospy.get_param("~plan_debug_sample_count", 8))
        self.planning_time = rospy.get_param("~planning_time", 3.0)
        self.num_planning_attempts = rospy.get_param("~num_planning_attempts", 5)
        self.num_candidate_plans = rospy.get_param("~num_candidate_plans", 6)
        self.cartesian_waypoint_eef_step = float(rospy.get_param("~cartesian_waypoint_eef_step", 0.01))
        self.cartesian_waypoint_avoid_collisions = _param_bool(
            rospy.get_param("~cartesian_waypoint_avoid_collisions", False)
        )
        self.cartesian_waypoint_min_fraction = float(
            rospy.get_param("~cartesian_waypoint_min_fraction", 0.85)
        )
        self.cartesian_waypoint_allow_partial_execution = _param_bool(
            rospy.get_param("~cartesian_waypoint_allow_partial_execution", False)
        )
        self.cartesian_waypoint_min_partial_fraction = float(
            rospy.get_param("~cartesian_waypoint_min_partial_fraction", 0.05)
        )
        self.cartesian_waypoint_debug_on_failure = _param_bool(
            rospy.get_param("~cartesian_waypoint_debug_on_failure", True)
        )
        self.cartesian_waypoint_debug_window = int(
            rospy.get_param("~cartesian_waypoint_debug_window", 4)
        )

        self.score_weight_elbow_z = rospy.get_param("~score_weight_elbow_z", 3.0)
        self.score_weight_joint_margin = rospy.get_param(
            "~score_weight_joint_margin", 1.5
        )
        self.score_weight_motion = rospy.get_param("~score_weight_motion", 0.4)

        self.left_facing_down_quat = rospy.get_param(
            "~left_facing_down_quat",
            [1.0, 0.0, 0.0, 0.0],
        )
        self.right_facing_down_quat = rospy.get_param(
            "~right_facing_down_quat",
            [1.0, 0.0, 0.0, 0.0],
        )

        # wait for move_group and other components to start up
        self.startup_delay = rospy.get_param("~startup_delay", 0.1)
        rospy.sleep(self.startup_delay)

        rospy.wait_for_message("/joint_states", JointState, timeout=5.0)

        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)

        self.robot = moveit_commander.RobotCommander()
        self.left_group = moveit_commander.MoveGroupCommander("left_arm")
        self.right_group = moveit_commander.MoveGroupCommander("right_arm")

        for group in [self.left_group, self.right_group]:
            group.set_max_velocity_scaling_factor(self.velocity_scaling)
            group.set_max_acceleration_scaling_factor(self.acceleration_scaling)
            group.set_planning_time(self.planning_time)
            group.set_num_planning_attempts(self.num_planning_attempts)

        self.current_joint_map = {}
        rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_state_cb,
            queue_size=1,
        )

        self.traj_pub = rospy.Publisher(
            "/yumi/moveit_joint_trajectory",
            JointTrajectory,
            queue_size=1,
            latch=True,
        )

        self.left_active_pub = rospy.Publisher(
            "/yumi/robl/moveit_active", Bool, queue_size=1, latch=True
        )
        self.left_arrived_pub = rospy.Publisher(
            "/yumi/robl/moveit_arrived", Bool, queue_size=1, latch=True
        )
        self.left_status_pub = rospy.Publisher(
            "/yumi/robl/moveit_status", String, queue_size=1, latch=True
        )
        self.right_active_pub = rospy.Publisher(
            "/yumi/robr/moveit_active", Bool, queue_size=1, latch=True
        )
        self.right_arrived_pub = rospy.Publisher(
            "/yumi/robr/moveit_arrived", Bool, queue_size=1, latch=True
        )
        self.right_status_pub = rospy.Publisher(
            "/yumi/robr/moveit_status", String, queue_size=1, latch=True
        )

        self.motion_joint_tolerance = rospy.get_param("~motion_joint_tolerance", 0.02)
        self.motion_timeout_margin = rospy.get_param("~motion_timeout_margin", 8.0)
        self.motion_timeout_scale = rospy.get_param("~motion_timeout_scale", 4.0)
        self.feedback_rate_hz = rospy.get_param("~feedback_rate_hz", 10.0)
        self.motion_watch = {
            "left": None,
            "right": None,
        }

        self._build_kdl()

        self.left_arm = ArmModel(
            name="left",
            group=self.left_group,
            base_link=self.base_frame,
            tip_link=rospy.get_param("~left_tip_link", "yumi_link_7_l"),
            elbow_link=rospy.get_param("~left_elbow_link", "yumi_link_4_l"),
            joint_names=[
                "yumi_robl_joint_1",
                "yumi_robl_joint_2",
                "yumi_robl_joint_3",
                "yumi_robl_joint_4",
                "yumi_robl_joint_5",
                "yumi_robl_joint_6",
                "yumi_robl_joint_7",
            ],
            joint_min=[-2.94, -2.50, -2.94, -2.16, -5.06, -1.54, -3.99],
            joint_max=[2.94, 0.76, 2.94, 1.40, 5.06, 2.41, 3.99],
        )

        self.right_arm = ArmModel(
            name="right",
            group=self.right_group,
            base_link=self.base_frame,
            tip_link=rospy.get_param("~right_tip_link", "yumi_link_7_r"),
            elbow_link=rospy.get_param("~right_elbow_link", "yumi_link_4_r"),
            joint_names=[
                "yumi_robr_joint_1",
                "yumi_robr_joint_2",
                "yumi_robr_joint_3",
                "yumi_robr_joint_4",
                "yumi_robr_joint_5",
                "yumi_robr_joint_6",
                "yumi_robr_joint_7",
            ],
            joint_min=[-2.94, -2.50, -2.94, -2.16, -5.06, -1.54, -3.99],
            joint_max=[2.94, 0.76, 2.94, 1.40, 5.06, 2.41, 3.99],
        )

        rospy.Subscriber(
            "/yumi/robl/moveit_target_position_current_orientation",
            PointStamped,
            self.left_position_current_orientation_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yumi/robl/moveit_target_position_facing_down",
            PointStamped,
            self.left_position_facing_down_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yumi/robl/moveit_target_pose",
            PoseStamped,
            self.left_pose_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yumi/robl/moveit_waypoints",
            PoseArray,
            self.left_waypoints_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/robr/moveit_target_position_current_orientation",
            PointStamped,
            self.right_position_current_orientation_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yumi/robr/moveit_target_position_facing_down",
            PointStamped,
            self.right_position_facing_down_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yumi/robr/moveit_target_pose",
            PoseStamped,
            self.right_pose_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yumi/robr/moveit_waypoints",
            PoseArray,
            self.right_waypoints_cb,
            queue_size=1,
        )

        rospy.Timer(rospy.Duration(1.0 / self.feedback_rate_hz), self.feedback_cb)
        self.publish_motion_state("left", False, False, "idle")
        self.publish_motion_state("right", False, False, "idle")

        rospy.loginfo("YuMi MoveIt pose topics node with scoring started")

    def _build_kdl(self):
        ok, tree = treeFromParam(self.robot_description_param)
        if not ok:
            raise RuntimeError(
                f"Could not parse URDF from parameter {self.robot_description_param}"
            )
        self.kdl_tree = tree

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def publish_motion_state(self, arm_name, active, arrived, status):
        if arm_name == "left":
            self.left_active_pub.publish(Bool(data=active))
            self.left_arrived_pub.publish(Bool(data=arrived))
            self.left_status_pub.publish(String(data=status))
        elif arm_name == "right":
            self.right_active_pub.publish(Bool(data=active))
            self.right_arrived_pub.publish(Bool(data=arrived))
            self.right_status_pub.publish(String(data=status))

    def _register_motion_watch(self, arm_model, plan, label):
        jt = plan.joint_trajectory
        joint_name_to_idx = {n: i for i, n in enumerate(jt.joint_names)}
        try:
            final_positions = np.array(
                [jt.points[-1].positions[joint_name_to_idx[j]] for j in arm_model.joint_names],
                dtype=float,
            )
        except KeyError:
            self.publish_motion_state(arm_model.name, False, False, "execution_failed: joint_name_mismatch")
            return

        duration = 0.0
        if jt.points:
            duration = jt.points[-1].time_from_start.to_sec()

        planned_window = max(duration * self.motion_timeout_scale, duration + self.motion_timeout_margin)
        self.motion_watch[arm_model.name] = {
            "target": final_positions,
            "deadline": rospy.Time.now().to_sec() + planned_window,
            "label": label,
            "planned_duration": duration,
        }
        self.publish_motion_state(arm_model.name, True, False, "executing")

    def feedback_cb(self, _event):
        for arm_name, watch in list(self.motion_watch.items()):
            if watch is None:
                continue
            arm_model = self.left_arm if arm_name == "left" else self.right_arm
            q_current = self.get_current_joint_values_for_arm(arm_model)
            if q_current is None:
                continue

            err = float(np.max(np.abs(q_current - watch["target"])))
            now = rospy.Time.now().to_sec()

            if err <= self.motion_joint_tolerance:
                self.motion_watch[arm_name] = None
                self.publish_motion_state(arm_name, False, True, "succeeded")
                continue

            if now > watch["deadline"]:
                self.motion_watch[arm_name] = None
                self.publish_motion_state(
                    arm_name,
                    False,
                    False,
                    f"timeout: joint_err={err:.4f} planned={watch.get('planned_duration', 0.0):.2f}s",
                )
                continue

            self.publish_motion_state(arm_name, True, False, f"executing: joint_err={err:.4f}")

    def get_current_joint_values_for_arm(self, arm_model):
        vals = []
        for name in arm_model.joint_names:
            if name not in self.current_joint_map:
                return None
            vals.append(self.current_joint_map[name])
        return np.array(vals, dtype=float)

    def get_current_pose_tf(self, tip_link):
        self.tf_listener.waitForTransform(
            self.base_frame, tip_link, rospy.Time(0), rospy.Duration(2.0)
        )
        trans, rot = self.tf_listener.lookupTransform(
            self.base_frame, tip_link, rospy.Time(0)
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

    def transform_point_to_base(self, point_msg):
        if point_msg.header.frame_id in ["", self.base_frame]:
            return point_msg

        self.tf_listener.waitForTransform(
            self.base_frame,
            point_msg.header.frame_id,
            rospy.Time(0),
            rospy.Duration(2.0),
        )
        return self.tf_listener.transformPoint(self.base_frame, point_msg)

    def transform_pose_to_base(self, pose_msg):
        if pose_msg.header.frame_id in ["", self.base_frame]:
            return pose_msg

        self.tf_listener.waitForTransform(
            self.base_frame,
            pose_msg.header.frame_id,
            rospy.Time(0),
            rospy.Duration(2.0),
        )
        return self.tf_listener.transformPose(self.base_frame, pose_msg)

    def build_pose_with_current_orientation(self, point_msg, tip_link):
        point_in_base = self.transform_point_to_base(point_msg)
        current_pose = self.get_current_pose_tf(tip_link)

        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x = point_in_base.point.x
        target_pose.position.y = point_in_base.point.y
        target_pose.position.z = point_in_base.point.z
        return target_pose

    def build_pose_with_fixed_orientation(self, point_msg, quat):
        point_in_base = self.transform_point_to_base(point_msg)
        target_pose = Pose()
        target_pose.position.x = point_in_base.point.x
        target_pose.position.y = point_in_base.point.y
        target_pose.position.z = point_in_base.point.z
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        return target_pose

    def build_pose_from_pose_msg(self, pose_msg):
        pose_in_base = self.transform_pose_to_base(pose_msg)
        return pose_in_base.pose

    def build_waypoints_from_pose_array(self, pose_array_msg):
        waypoints = []
        for pose in pose_array_msg.poses:
            stamped = PoseStamped()
            stamped.header = pose_array_msg.header
            stamped.pose = pose
            pose_in_base = self.transform_pose_to_base(stamped)
            waypoints.append(pose_in_base.pose)
        return waypoints

    def compute_fk_translation(self, base_link, tip_link, joint_values):
        chain = self.kdl_tree.getChain(base_link, tip_link)
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        q_kdl = kdl.JntArray(len(joint_values))
        for i, q in enumerate(joint_values):
            q_kdl[i] = float(q)

        frame = kdl.Frame()
        fk_solver.JntToCart(q_kdl, frame)

        return np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=float)

    def compute_joint_margin_score(self, arm_model, q):
        lower_margin = q - arm_model.joint_min
        upper_margin = arm_model.joint_max - q
        min_margin = np.minimum(lower_margin, upper_margin)
        min_margin = np.maximum(min_margin, 0.0)

        joint_ranges = np.maximum(arm_model.joint_max - arm_model.joint_min, 1e-6)
        normalized = min_margin / joint_ranges
        return float(np.mean(normalized))

    def score_plan(self, arm_model, plan, q_current):
        jt = plan.joint_trajectory
        if len(jt.points) == 0:
            return -1e9

        joint_name_to_idx = {n: i for i, n in enumerate(jt.joint_names)}
        try:
            q_final = np.array(
                [
                    jt.points[-1].positions[joint_name_to_idx[j]]
                    for j in arm_model.joint_names
                ],
                dtype=float,
            )
        except KeyError:
            rospy.logwarn("Trajectory joint names do not match expected arm joints")
            return -1e9

        elbow_pos = self.compute_fk_translation(
            arm_model.base_link,
            arm_model.elbow_link,
            q_final,
        )
        elbow_z = float(elbow_pos[2])

        joint_margin_score = self.compute_joint_margin_score(arm_model, q_final)
        motion_cost = float(np.linalg.norm(q_final - q_current))

        score = (
            self.score_weight_elbow_z * elbow_z
            + self.score_weight_joint_margin * joint_margin_score
            - self.score_weight_motion * motion_cost
        )

        return score

    def plan_best(self, arm_model, target_pose, label):
        q_current = self.get_current_joint_values_for_arm(arm_model)
        if q_current is None:
            rospy.logerr(f"No current joint state for {label}")
            return None

        best_plan = None
        best_score = -1e9

        if not self.compare_all_plans:
            self.num_candidate_plans = 1
        for i in range(self.num_candidate_plans):
            arm_model.group.clear_pose_targets()
            arm_model.group.set_start_state_to_current_state()
            arm_model.group.set_pose_target(target_pose)

            result = arm_model.group.plan()

            if isinstance(result, tuple):
                success, plan, planning_time, error_code = result
            else:
                plan = result
                success = (
                    hasattr(plan, "joint_trajectory")
                    and len(plan.joint_trajectory.points) > 0
                )

            if not success or len(plan.joint_trajectory.points) == 0:
                continue

            score = self.score_plan(arm_model, plan, q_current)
            rospy.loginfo(
                f"[{label}] candidate {i+1}/{self.num_candidate_plans} score = {score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_plan = plan

        arm_model.group.clear_pose_targets()

        if best_plan is None:
            rospy.logerr(f"Planning failed for {label}")
            return None

        rospy.loginfo(f"[{label}] selected best score = {best_score:.4f}")
        return best_plan

    def _trajectory_duration(self, plan):
        points = plan.joint_trajectory.points
        if not points:
            return 0.0
        return float(points[-1].time_from_start.to_sec())

    def _is_waypoint_plan(self, label):
        return "waypoints" in str(label).lower()

    def _execution_profile(self, label):
        if self._is_waypoint_plan(label):
            return {
                "velocity_scaling": float(self.waypoint_velocity_scaling),
                "acceleration_scaling": float(self.waypoint_acceleration_scaling),
                "trajectory_time_scale": float(self.waypoint_trajectory_time_scale),
                "fallback_joint_speed": float(self.waypoint_fallback_joint_speed),
                "min_segment_duration": float(self.waypoint_min_segment_duration),
                "name": "waypoint",
            }
        return {
            "velocity_scaling": float(self.velocity_scaling),
            "acceleration_scaling": float(self.acceleration_scaling),
            "trajectory_time_scale": float(self.trajectory_time_scale),
            "fallback_joint_speed": float(self.fallback_joint_speed),
            "min_segment_duration": float(self.min_segment_duration),
            "name": "normal",
        }

    def _retime_plan(self, arm_model, plan, label):
        if not self.retime_trajectories:
            return plan
        profile = self._execution_profile(label)
        try:
            current_state = self.robot.get_current_state()
            try:
                retimed = arm_model.group.retime_trajectory(
                    current_state,
                    plan,
                    profile["velocity_scaling"],
                    profile["acceleration_scaling"],
                )
            except TypeError:
                retimed = arm_model.group.retime_trajectory(
                    current_state,
                    plan,
                    profile["velocity_scaling"],
                )
            if (
                retimed is not None
                and hasattr(retimed, "joint_trajectory")
                and len(retimed.joint_trajectory.points) > 0
            ):
                rospy.loginfo(
                    f"[{label}] retimed trajectory duration "
                    f"{self._trajectory_duration(plan):.3f}s -> "
                    f"{self._trajectory_duration(retimed):.3f}s "
                    f"(profile={profile['name']}, "
                    f"vel_scale={profile['velocity_scaling']:.3f}, "
                    f"acc_scale={profile['acceleration_scaling']:.3f})"
                )
                return retimed
        except Exception as exc:
            rospy.logwarn(f"[{label}] trajectory retiming failed; using original timing: {exc}")
        return plan

    def _ensure_trajectory_timing(self, plan, label):
        out = copy.deepcopy(plan)
        points = out.joint_trajectory.points
        if not points:
            return out

        raw_times = [max(0.0, float(pt.time_from_start.to_sec())) for pt in points]
        first_time = raw_times[0]
        raw_times = [max(0.0, t - first_time) for t in raw_times]

        fixed = abs(first_time) > 1e-9
        profile = self._execution_profile(label)
        new_times = []
        prev_time = 0.0
        prev_pos = None
        for idx, pt in enumerate(points):
            current_pos = np.asarray(pt.positions, dtype=float) if len(pt.positions) > 0 else None
            if idx == 0:
                new_time = 0.0
            else:
                existing_dt = raw_times[idx] - raw_times[idx - 1]
                if existing_dt <= 1e-6:
                    if current_pos is not None and prev_pos is not None and len(current_pos) == len(prev_pos):
                        max_delta = float(np.max(np.abs(current_pos - prev_pos)))
                        fallback_dt = max_delta / profile["fallback_joint_speed"]
                    else:
                        fallback_dt = 0.0
                    new_time = prev_time + max(float(profile["min_segment_duration"]), fallback_dt)
                    fixed = True
                else:
                    new_time = raw_times[idx]
                    if profile["min_segment_duration"] > 0.0:
                        min_time = prev_time + float(profile["min_segment_duration"])
                        if new_time < min_time:
                            new_time = min_time
                            fixed = True
            pt.time_from_start = rospy.Duration.from_sec(new_time)
            new_times.append(new_time)
            prev_time = new_time
            prev_pos = current_pos

        if fixed:
            for pt in points:
                pt.velocities = []
                pt.accelerations = []
            rospy.logwarn(
                f"[{label}] adjusted trajectory timestamps; duration now {new_times[-1]:.3f}s; "
                "cleared feed-forward velocities"
            )
        return out

    def _scale_trajectory_time(self, plan, scale, label):
        scale = max(1.0, float(scale))
        if scale <= 1.0:
            return plan
        out = copy.deepcopy(plan)
        for pt in out.joint_trajectory.points:
            pt.time_from_start = rospy.Duration.from_sec(
                float(pt.time_from_start.to_sec()) * scale
            )
            if len(pt.velocities) > 0:
                pt.velocities = [float(v) / scale for v in pt.velocities]
            if len(pt.accelerations) > 0:
                pt.accelerations = [float(a) / (scale * scale) for a in pt.accelerations]
        rospy.logwarn(
            f"[{label}] scaled trajectory time by {scale:.2f}; "
            f"duration={self._trajectory_duration(out):.3f}s"
        )
        return out

    def _prepare_plan_for_execution(self, arm_model, plan, label):
        prepared = self._retime_plan(arm_model, plan, label)
        prepared = self._ensure_trajectory_timing(prepared, label)
        profile = self._execution_profile(label)
        prepared = self._scale_trajectory_time(
            prepared,
            profile["trajectory_time_scale"],
            label,
        )
        return prepared

    def log_plan_details(self, arm_model, plan, label):
        if not self.log_plan_details_enabled:
            return
        jt = plan.joint_trajectory
        points = jt.points
        if not points:
            return
        times = [float(pt.time_from_start.to_sec()) for pt in points]
        dts = np.diff(times) if len(times) > 1 else np.array([], dtype=float)
        profile = self._execution_profile(label)
        rospy.logwarn(
            "[%s] trajectory ready: points=%d duration=%.3fs min_dt=%.4fs max_dt=%.4fs "
            "profile=%s retime=%s time_scale=%.2f vel_scale=%.3f acc_scale=%.3f",
            label,
            len(points),
            times[-1],
            float(np.min(dts)) if dts.size else 0.0,
            float(np.max(dts)) if dts.size else 0.0,
            profile["name"],
            bool(self.retime_trajectories),
            float(profile["trajectory_time_scale"]),
            float(profile["velocity_scaling"]),
            float(profile["acceleration_scaling"]),
        )
        sample_count = max(1, min(int(self.plan_debug_sample_count), len(points)))
        sample_indices = sorted(set(np.linspace(0, len(points) - 1, sample_count, dtype=int).tolist()))
        name_to_idx = {name: idx for idx, name in enumerate(jt.joint_names)}
        for idx in sample_indices:
            pt = points[idx]
            q = []
            for joint_name in arm_model.joint_names:
                if joint_name in name_to_idx and len(pt.positions) > name_to_idx[joint_name]:
                    q.append(float(pt.positions[name_to_idx[joint_name]]))
            rospy.logwarn(
                "[%s] traj_pt[%03d] t=%.3fs q=%s",
                label,
                idx,
                float(pt.time_from_start.to_sec()),
                np.round(q, 4).tolist(),
            )

    def publish_plan(self, arm_model, plan, label):
        prepared = self._prepare_plan_for_execution(arm_model, plan, label)
        self.log_plan_details(arm_model, prepared, label)
        self.traj_pub.publish(prepared.joint_trajectory)
        self._register_motion_watch(arm_model, prepared, label)
        rospy.loginfo(
            f"Published trajectory for {label} with "
            f"{len(prepared.joint_trajectory.points)} points, "
            f"duration={self._trajectory_duration(prepared):.3f}s"
        )

    def plan_cartesian_waypoints(self, arm_model, waypoints, label):
        if len(waypoints) == 0:
            rospy.logerr(f"No Cartesian waypoints for {label}")
            return None

        arm_model.group.clear_pose_targets()
        arm_model.group.set_start_state_to_current_state()
        eef_step = float(self.cartesian_waypoint_eef_step)
        avoid_collisions = bool(self.cartesian_waypoint_avoid_collisions)
        result = arm_model.group.compute_cartesian_path(
            waypoints,
            eef_step,
            avoid_collisions,
        )
        if isinstance(result, tuple) and len(result) >= 2:
            plan, fraction = result[0], float(result[1])
        else:
            rospy.logerr(f"Unexpected compute_cartesian_path result for {label}")
            return None

        if fraction < self.cartesian_waypoint_min_fraction:
            rospy.logerr(
                f"Cartesian waypoint planning failed for {label}: "
                f"fraction={fraction:.3f} < {self.cartesian_waypoint_min_fraction:.3f}"
            )
            if self.cartesian_waypoint_debug_on_failure:
                self.log_cartesian_waypoint_failure(arm_model, plan, waypoints, fraction, label)
            if (
                self.cartesian_waypoint_allow_partial_execution
                and fraction >= self.cartesian_waypoint_min_partial_fraction
                and hasattr(plan, "joint_trajectory")
                and len(plan.joint_trajectory.points) > 0
            ):
                rospy.logwarn(
                    f"[{label}] executing partial Cartesian path for debugging: "
                    f"fraction={fraction:.3f}, points={len(plan.joint_trajectory.points)}"
                )
                return plan
            self.publish_motion_state(
                arm_model.name,
                False,
                False,
                f"planning_failed: cartesian_fraction={fraction:.3f}",
            )
            return None
        if not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
            rospy.logerr(f"Cartesian waypoint planning produced empty plan for {label}")
            self.publish_motion_state(arm_model.name, False, False, "planning_failed: empty_cartesian_plan")
            return None

        rospy.loginfo(
            f"[{label}] Cartesian waypoint path fraction={fraction:.3f}, "
            f"points={len(plan.joint_trajectory.points)}"
        )
        return plan

    def pose_position_np(self, pose):
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float,
        )

    def pose_quat_np(self, pose):
        return np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )

    def quat_angle_deg(self, quat_a, quat_b):
        qa = np.asarray(quat_a, dtype=float).reshape(4)
        qb = np.asarray(quat_b, dtype=float).reshape(4)
        qa_norm = float(np.linalg.norm(qa))
        qb_norm = float(np.linalg.norm(qb))
        if qa_norm < 1e-12 or qb_norm < 1e-12:
            return float("nan")
        qa = qa / qa_norm
        qb = qb / qb_norm
        dot = abs(float(np.dot(qa, qb)))
        dot = max(-1.0, min(1.0, dot))
        return float(np.degrees(2.0 * math.acos(dot)))

    def log_joint_margin_details(self, arm_model, q, label, prefix):
        q = np.asarray(q, dtype=float).reshape(-1)
        lower_margin = q - arm_model.joint_min
        upper_margin = arm_model.joint_max - q
        signed_margin = np.minimum(lower_margin, upper_margin)
        rospy.logwarn(
            "[%s] %s q=%s",
            label,
            prefix,
            np.round(q, 4).tolist(),
        )
        for idx, joint_name in enumerate(arm_model.joint_names):
            rospy.logwarn(
                "[%s] %s %s q=%.4f range=[%.4f, %.4f] margin=%.4f",
                label,
                prefix,
                joint_name,
                float(q[idx]),
                float(arm_model.joint_min[idx]),
                float(arm_model.joint_max[idx]),
                float(signed_margin[idx]),
            )

    def log_cartesian_waypoint_failure(self, arm_model, plan, waypoints, fraction, label):
        requested = len(waypoints)
        reached_est = int(math.floor(float(fraction) * float(requested)))
        reached_est = max(0, min(requested - 1, reached_est))
        next_est = min(requested - 1, reached_est + 1)
        window = max(1, int(self.cartesian_waypoint_debug_window))
        start = max(0, reached_est - window)
        end = min(requested, next_est + window + 1)

        rospy.logwarn(
            "[%s] Cartesian failure diagnostic: fraction=%.3f requested_waypoints=%d "
            "reached_est=%d next_est=%d eef_step=%.4f avoid_collisions=%s",
            label,
            float(fraction),
            requested,
            reached_est,
            next_est,
            float(self.cartesian_waypoint_eef_step),
            bool(self.cartesian_waypoint_avoid_collisions),
        )

        q_current = self.get_current_joint_values_for_arm(arm_model)
        if q_current is not None:
            self.log_joint_margin_details(arm_model, q_current, label, "current")

        if hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0:
            jt = plan.joint_trajectory
            name_to_idx = {name: idx for idx, name in enumerate(jt.joint_names)}
            try:
                q_last = np.array(
                    [jt.points[-1].positions[name_to_idx[j]] for j in arm_model.joint_names],
                    dtype=float,
                )
                self.log_joint_margin_details(arm_model, q_last, label, "last_cartesian")
            except KeyError:
                rospy.logwarn("[%s] Cannot map last Cartesian point to expected joint order", label)

        prev_pos = None
        prev_quat = None
        for idx in range(start, end):
            pose = waypoints[idx]
            pos = self.pose_position_np(pose)
            quat = self.pose_quat_np(pose)
            if prev_pos is None and idx > 0:
                prev_pose = waypoints[idx - 1]
                prev_pos = self.pose_position_np(prev_pose)
                prev_quat = self.pose_quat_np(prev_pose)
            seg_len = 0.0 if prev_pos is None else float(np.linalg.norm(pos - prev_pos))
            rot_delta = 0.0 if prev_quat is None else self.quat_angle_deg(prev_quat, quat)
            marker = "last_reached_est" if idx == reached_est else ("next_after_failure" if idx == next_est else "nearby")
            rospy.logwarn(
                "[%s] fail_wp[%03d] %s pos=%s seg_len=%.4f rot_delta_deg=%.2f quat_xyzw=%s",
                label,
                idx,
                marker,
                np.round(pos, 4).tolist(),
                seg_len,
                rot_delta,
                np.round(quat, 4).tolist(),
            )
            prev_pos = pos
            prev_quat = quat

    def log_received_waypoints(self, msg, label, max_count=8):
        rospy.logwarn(
            "[%s] received PoseArray frame='%s' waypoints=%d",
            label,
            msg.header.frame_id,
            len(msg.poses),
        )
        for idx, pose in enumerate(msg.poses[:max_count]):
            rospy.logwarn(
                "[%s] input_wp[%02d] pos=[%.4f, %.4f, %.4f] quat_xyzw=[%.4f, %.4f, %.4f, %.4f]",
                label,
                idx,
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        if len(msg.poses) > max_count:
            rospy.logwarn(
                "[%s] ... %d additional input waypoint(s) omitted",
                label,
                len(msg.poses) - max_count,
            )

    def log_base_waypoints(self, waypoints, label, max_count=8):
        rospy.logwarn("[%s] transformed base-frame waypoints=%d", label, len(waypoints))
        prev = None
        for idx, pose in enumerate(waypoints[:max_count]):
            pos = np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ],
                dtype=float,
            )
            if prev is None:
                seg_len = 0.0
                seg = np.zeros(3, dtype=float)
            else:
                seg = pos - prev
                seg_len = float(np.linalg.norm(seg))
            rospy.logwarn(
                "[%s] base_wp[%02d] pos=%s seg_len=%.4f seg=%s quat_xyzw=[%.4f, %.4f, %.4f, %.4f]",
                label,
                idx,
                np.round(pos, 4).tolist(),
                seg_len,
                np.round(seg, 4).tolist(),
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            prev = pos
        if len(waypoints) > max_count:
            rospy.logwarn(
                "[%s] ... %d additional transformed waypoint(s) omitted",
                label,
                len(waypoints) - max_count,
            )

    def left_position_current_orientation_cb(self, msg):
        try:
            pose = self.build_pose_with_current_orientation(msg, self.left_arm.tip_link)
            plan = self.plan_best(self.left_arm, pose, "left_arm current_orientation")
            if plan is not None:
                self.publish_plan(self.left_arm, plan, "left_arm current_orientation")
        except Exception as exc:
            rospy.logerr(f"Left current-orientation target failed: {exc}")

    def left_position_facing_down_cb(self, msg):
        try:
            pose = self.build_pose_with_fixed_orientation(
                msg,
                self.left_facing_down_quat,
            )
            plan = self.plan_best(self.left_arm, pose, "left_arm facing_down")
            if plan is not None:
                self.publish_plan(self.left_arm, plan, "left_arm facing_down")
        except Exception as exc:
            rospy.logerr(f"Left facing-down target failed: {exc}")

    def left_pose_cb(self, msg):
        try:
            pose = self.build_pose_from_pose_msg(msg)
            plan = self.plan_best(self.left_arm, pose, "left_arm full_pose")
            if plan is not None:
                self.publish_plan(self.left_arm, plan, "left_arm full_pose")
        except Exception as exc:
            rospy.logerr(f"Left full-pose target failed: {exc}")

    def left_waypoints_cb(self, msg):
        try:
            self.log_received_waypoints(msg, "left_arm waypoints")
            waypoints = self.build_waypoints_from_pose_array(msg)
            self.log_base_waypoints(waypoints, "left_arm waypoints")
            plan = self.plan_cartesian_waypoints(self.left_arm, waypoints, "left_arm waypoints")
            if plan is not None:
                self.publish_plan(self.left_arm, plan, "left_arm waypoints")
        except Exception as exc:
            rospy.logerr(f"Left waypoint target failed: {exc}")
            self.publish_motion_state("left", False, False, f"error: {exc}")

    def right_position_current_orientation_cb(self, msg):
        try:
            pose = self.build_pose_with_current_orientation(
                msg, self.right_arm.tip_link
            )
            plan = self.plan_best(self.right_arm, pose, "right_arm current_orientation")
            if plan is not None:
                self.publish_plan(self.right_arm, plan, "right_arm current_orientation")
        except Exception as exc:
            rospy.logerr(f"Right current-orientation target failed: {exc}")

    def right_position_facing_down_cb(self, msg):
        try:
            pose = self.build_pose_with_fixed_orientation(
                msg,
                self.right_facing_down_quat,
            )
            plan = self.plan_best(self.right_arm, pose, "right_arm facing_down")
            if plan is not None:
                self.publish_plan(self.right_arm, plan, "right_arm facing_down")
        except Exception as exc:
            rospy.logerr(f"Right facing-down target failed: {exc}")

    def right_pose_cb(self, msg):
        try:
            pose = self.build_pose_from_pose_msg(msg)
            plan = self.plan_best(self.right_arm, pose, "right_arm full_pose")
            if plan is not None:
                self.publish_plan(self.right_arm, plan, "right_arm full_pose")
        except Exception as exc:
            rospy.logerr(f"Right full-pose target failed: {exc}")

    def right_waypoints_cb(self, msg):
        try:
            self.log_received_waypoints(msg, "right_arm waypoints")
            waypoints = self.build_waypoints_from_pose_array(msg)
            self.log_base_waypoints(waypoints, "right_arm waypoints")
            plan = self.plan_cartesian_waypoints(self.right_arm, waypoints, "right_arm waypoints")
            if plan is not None:
                self.publish_plan(self.right_arm, plan, "right_arm waypoints")
        except Exception as exc:
            rospy.logerr(f"Right waypoint target failed: {exc}")
            self.publish_motion_state("right", False, False, f"error: {exc}")


if __name__ == "__main__":
    import numpy as np

    rospy.init_node("yumi_moveit_pose_topics")
    try:
        YumiMoveItPoseTopics()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
