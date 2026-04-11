#!/usr/bin/env python3
import sys
import copy

import rospy
import tf
import moveit_commander

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


def get_current_pose_tf(listener, base_frame, tip_frame):
    listener.waitForTransform(base_frame, tip_frame, rospy.Time(0), rospy.Duration(2.0))
    trans, rot = listener.lookupTransform(base_frame, tip_frame, rospy.Time(0))

    pose = Pose()
    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]
    pose.orientation.x = rot[0]
    pose.orientation.y = rot[1]
    pose.orientation.z = rot[2]
    pose.orientation.w = rot[3]
    return pose


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("yumi_moveit_plan_and_send_left_tf")

    # Wait until real joint states are present
    rospy.wait_for_message("/joint_states", JointState, timeout=5.0)

    # TF listener for current TCP pose
    listener = tf.TransformListener()
    rospy.sleep(1.0)

    # MoveIt group
    group = moveit_commander.MoveGroupCommander("left_arm")
    group.set_max_velocity_scaling_factor(0.2)
    group.set_max_acceleration_scaling_factor(0.2)

    # Publisher to your execute bridge
    pub = rospy.Publisher(
        "/yumi/moveit_joint_trajectory",
        JointTrajectory,
        queue_size=1,
        latch=True,
    )
    rospy.sleep(0.5)

    rospy.loginfo(f"Planning group: {group.get_name()}")
    rospy.loginfo(f"Active joints: {group.get_active_joints()}")

    # Read current TCP pose from TF instead of MoveIt
    current_pose = get_current_pose_tf(
        listener,
        "yumi_base_link",
        "yumi_link_7_l",
    )

    rospy.loginfo(
        f"Current pose from TF: "
        f"x={current_pose.position.x:.3f}, "
        f"y={current_pose.position.y:.3f}, "
        f"z={current_pose.position.z:.3f}"
    )

    # Create a small Cartesian target
    target_pose = copy.deepcopy(current_pose)
    target_pose.position.z += 0.03

    # Optional: keep the current orientation exactly
    group.set_pose_target(target_pose)

    result = group.plan()

    # MoveIt versions differ in return format
    if isinstance(result, tuple):
        success, plan, planning_time, error_code = result
    else:
        plan = result
        success = (
            hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0
        )

    if not success or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning failed")
        return

    rospy.loginfo(
        f"Planning succeeded with {len(plan.joint_trajectory.points)} trajectory points"
    )

    rospy.sleep(0.5)
    pub.publish(plan.joint_trajectory)
    rospy.loginfo("Published trajectory to /yumi/moveit_joint_trajectory")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
