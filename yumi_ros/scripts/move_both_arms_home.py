#!/usr/bin/env python3
import sys
import threading

import rospy
import moveit_commander

from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class YumiHomeBothArmsService:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        rospy.init_node("yumi_home_both_arms_service")

        startup_delay = rospy.get_param("~startup_delay", 0.1)
        rospy.sleep(startup_delay)

        rospy.wait_for_message("/joint_states", JointState, timeout=5.0)

        self.lock = threading.Lock()

        self.group = moveit_commander.MoveGroupCommander("both_arms")
        self.group.set_max_velocity_scaling_factor(
            rospy.get_param("~velocity_scaling", 0.2)
        )
        self.group.set_max_acceleration_scaling_factor(
            rospy.get_param("~acceleration_scaling", 0.2)
        )

        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0)  # Allow time for the planning scene to initialize
        self.add_workspace_constraints()

        self.named_target = rospy.get_param("~named_target", "home_both")

        self.pub = rospy.Publisher(
            "/yumi/moveit_joint_trajectory",
            JointTrajectory,
            queue_size=1,
            latch=True,
        )

        rospy.sleep(0.5)

        self.service = rospy.Service(
            "/yumi/home_both_arms",
            Trigger,
            self.handle_home_both_arms,
        )

        rospy.loginfo("YuMi home-both-arms service ready: /yumi/home_both_arms")

    def handle_home_both_arms(self, _req):
        with self.lock:
            try:
                self.group.set_named_target(self.named_target)
                result = self.group.plan()

                if isinstance(result, tuple):
                    success, plan, planning_time, error_code = result
                else:
                    plan = result
                    success = (
                        hasattr(plan, "joint_trajectory")
                        and len(plan.joint_trajectory.points) > 0
                    )

                if not success or len(plan.joint_trajectory.points) == 0:
                    msg = f"Planning to named target '{self.named_target}' failed"
                    rospy.logerr(msg)
                    return TriggerResponse(success=False, message=msg)

                rospy.loginfo(
                    f"Publishing trajectory to '{self.named_target}' with "
                    f"{len(plan.joint_trajectory.points)} points"
                )

                rospy.sleep(0.5)
                self.pub.publish(plan.joint_trajectory)

                return TriggerResponse(
                    success=True,
                    message=f"Trajectory to '{self.named_target}' published",
                )

            except Exception as e:
                msg = f"Exception while planning home trajectory: {e}"
                rospy.logerr(msg)
                return TriggerResponse(success=False, message=msg)

    def add_workspace_constraints(self):
        from geometry_msgs.msg import PoseStamped

        box_pose = PoseStamped()
        box_pose.header.frame_id = "yumi_base_link"

        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 0.75
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 0.25

        self.scene.add_box("forbidden_zone", box_pose, size=(0.5, 0.6, 0.5))  # x, y, z

        rospy.loginfo("Added forbidden workspace zone")


if __name__ == "__main__":
    try:
        node = YumiHomeBothArmsService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
