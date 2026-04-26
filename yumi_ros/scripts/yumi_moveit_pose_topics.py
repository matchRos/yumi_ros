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

    def publish_plan(self, arm_model, plan, label):
        self.traj_pub.publish(plan.joint_trajectory)
        self._register_motion_watch(arm_model, plan, label)
        rospy.loginfo(
            f"Published trajectory for {label} with "
            f"{len(plan.joint_trajectory.points)} points"
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
