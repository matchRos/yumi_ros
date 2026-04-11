#!/usr/bin/env python3
import sys
import copy

import rospy
import tf
import moveit_commander

from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class YumiMoveItPoseTopics:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.base_frame = rospy.get_param("~base_frame", "yumi_base_link")
        self.left_tip_link = rospy.get_param("~left_tip_link", "yumi_link_7_l")
        self.right_tip_link = rospy.get_param("~right_tip_link", "yumi_link_7_r")

        self.velocity_scaling = rospy.get_param("~velocity_scaling", 0.2)
        self.acceleration_scaling = rospy.get_param("~acceleration_scaling", 0.2)
        self.planning_time = rospy.get_param("~planning_time", 3.0)
        self.num_planning_attempts = rospy.get_param("~num_planning_attempts", 5)
        self.startup_delay = rospy.get_param("~startup_delay", 0.1)

        rospy.sleep(self.startup_delay)

        # Facing-down orientation as quaternion [x, y, z, w]
        # This is a placeholder that may need adjustment depending on your tool frame.
        self.left_facing_down_quat = rospy.get_param(
            "~left_facing_down_quat",
            [1.0, 0.0, 0.0, 0.0],
        )
        self.right_facing_down_quat = rospy.get_param(
            "~right_facing_down_quat",
            [1.0, 0.0, 0.0, 0.0],
        )

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

        self.traj_pub = rospy.Publisher(
            "/yumi/moveit_joint_trajectory",
            JointTrajectory,
            queue_size=1,
            latch=True,
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

        rospy.loginfo("YuMi MoveIt pose topics node started")

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

    def build_pose_with_current_orientation(self, point_msg, tip_link):
        current_pose = self.get_current_pose_tf(tip_link)
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x = point_msg.point.x
        target_pose.position.y = point_msg.point.y
        target_pose.position.z = point_msg.point.z
        return target_pose

    def build_pose_with_fixed_orientation(self, point_msg, quat):
        target_pose = Pose()
        target_pose.position.x = point_msg.point.x
        target_pose.position.y = point_msg.point.y
        target_pose.position.z = point_msg.point.z
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        return target_pose

    def plan_and_publish(self, group, target_pose, label):
        group.clear_pose_targets()
        group.set_start_state_to_current_state()
        group.set_pose_target(target_pose)

        result = group.plan()

        if isinstance(result, tuple):
            success, plan, planning_time, error_code = result
        else:
            plan = result
            success = (
                hasattr(plan, "joint_trajectory")
                and len(plan.joint_trajectory.points) > 0
            )

        if not success or len(plan.joint_trajectory.points) == 0:
            rospy.logerr(f"Planning failed for {label}")
            group.clear_pose_targets()
            return

        self.traj_pub.publish(plan.joint_trajectory)
        rospy.loginfo(
            f"Published trajectory for {label} with "
            f"{len(plan.joint_trajectory.points)} points"
        )
        group.clear_pose_targets()

    def left_position_current_orientation_cb(self, msg):
        try:
            pose = self.build_pose_with_current_orientation(msg, self.left_tip_link)
            self.plan_and_publish(
                self.left_group,
                pose,
                "left_arm current_orientation",
            )
        except Exception as exc:
            rospy.logerr(f"Left current-orientation target failed: {exc}")

    def left_position_facing_down_cb(self, msg):
        try:
            pose = self.build_pose_with_fixed_orientation(
                msg,
                self.left_facing_down_quat,
            )
            self.plan_and_publish(
                self.left_group,
                pose,
                "left_arm facing_down",
            )
        except Exception as exc:
            rospy.logerr(f"Left facing-down target failed: {exc}")

    def left_pose_cb(self, msg):
        try:
            self.plan_and_publish(
                self.left_group,
                msg.pose,
                "left_arm full_pose",
            )
        except Exception as exc:
            rospy.logerr(f"Left full-pose target failed: {exc}")

    def right_position_current_orientation_cb(self, msg):
        try:
            pose = self.build_pose_with_current_orientation(msg, self.right_tip_link)
            self.plan_and_publish(
                self.right_group,
                pose,
                "right_arm current_orientation",
            )
        except Exception as exc:
            rospy.logerr(f"Right current-orientation target failed: {exc}")

    def right_position_facing_down_cb(self, msg):
        try:
            pose = self.build_pose_with_fixed_orientation(
                msg,
                self.right_facing_down_quat,
            )
            self.plan_and_publish(
                self.right_group,
                pose,
                "right_arm facing_down",
            )
        except Exception as exc:
            rospy.logerr(f"Right facing-down target failed: {exc}")

    def right_pose_cb(self, msg):
        try:
            self.plan_and_publish(
                self.right_group,
                msg.pose,
                "right_arm full_pose",
            )
        except Exception as exc:
            rospy.logerr(f"Right full-pose target failed: {exc}")


if __name__ == "__main__":
    rospy.init_node("yumi_moveit_pose_topics")
    try:
        YumiMoveItPoseTopics()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
