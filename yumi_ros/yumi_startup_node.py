#!/usr/bin/env python3
import rospy
from abb_robot_msgs.srv import TriggerWithResultCode
from controller_manager_msgs.srv import SwitchController


def start_joint_group_vel_controller():
    service_name = "/yumi/egm/controller_manager/switch_controller"
    rospy.loginfo(f"Waiting for service: {service_name}")
    rospy.wait_for_service(service_name)

    try:
        switch_srv = rospy.ServiceProxy(service_name, SwitchController)
        resp = switch_srv(
            start_controllers=["joint_group_velocity_controller"],
            stop_controllers=[],
            strictness=1,
            start_asap=False,
            timeout=0.0,
        )
        rospy.loginfo(f"switch_controller ok={resp.ok}")
        return resp.ok
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to start controller: {e}")
        return False


def call_service(name):
    rospy.loginfo(f"Waiting for service: {name}")
    rospy.wait_for_service(name)

    try:
        srv = rospy.ServiceProxy(name, TriggerWithResultCode)
        resp = srv()
        rospy.loginfo(
            f"{name} -> result_code={resp.result_code}, message='{resp.message}'"
        )
        return resp
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {name} -> {e}")
        return None


if __name__ == "__main__":
    rospy.init_node("yumi_startup_node")
    rospy.sleep(2.0)

    services = [
        "/yumi/rws/stop_rapid",
        "/yumi/rws/pp_to_main",
        "/yumi/rws/start_rapid",
        "/yumi/rws/sm_addin/start_egm_joint",
    ]

    for srv_name in services:
        resp = call_service(srv_name)
        rospy.sleep(1.0)

        if resp is None:
            rospy.logerr(f"Aborting sequence because {srv_name} failed.")
            break

    start_joint_group_vel_controller()

    rospy.loginfo("YuMi startup sequence finished.")
