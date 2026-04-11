#!/usr/bin/env python3
import sys
import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("yumi_move_both_arms_home")

    rospy.wait_for_message("/joint_states", JointState, timeout=5.0)

    group = moveit_commander.MoveGroupCommander("both_arms")
    pub = rospy.Publisher(
        "/yumi/moveit_joint_trajectory",
        JointTrajectory,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(0.5)

    group.set_max_velocity_scaling_factor(0.2)
    group.set_max_acceleration_scaling_factor(0.2)

    group.set_named_target("home_both")
    result = group.plan()

    if isinstance(result, tuple):
        success, plan, planning_time, error_code = result
    else:
        plan = result
        success = (
            hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0
        )

    if not success or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning to named target 'both_home' failed")
        return

    rospy.loginfo(
        f"Publishing trajectory to 'both_home' with {len(plan.joint_trajectory.points)} points"
    )
    rospy.sleep(0.5)
    pub.publish(plan.joint_trajectory)
    rospy.loginfo("Trajectory published")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
