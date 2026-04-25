#!/usr/bin/env python3
import rospy
from abb_robot_msgs.srv import TriggerWithResultCode
from abb_rapid_sm_addin_msgs.srv import GetEGMSettings, SetEGMSettings
from controller_manager_msgs.srv import SwitchController


def configure_egm_settings():
    get_service_name = "/yumi/rws/sm_addin/get_egm_settings"
    set_service_name = "/yumi/rws/sm_addin/set_egm_settings"

    tasks = rospy.get_param("~egm_tasks", ["T_ROB_L", "T_ROB_R"])
    max_speed_deviation = rospy.get_param("~egm_max_speed_deviation", 100.0)
    ramp_in_time = rospy.get_param("~egm_ramp_in_time", 0.05)

    rospy.loginfo(f"Waiting for service: {get_service_name}")
    rospy.wait_for_service(get_service_name)
    rospy.loginfo(f"Waiting for service: {set_service_name}")
    rospy.wait_for_service(set_service_name)

    get_settings_srv = rospy.ServiceProxy(get_service_name, GetEGMSettings)
    set_settings_srv = rospy.ServiceProxy(set_service_name, SetEGMSettings)

    ok = True
    for task in tasks:
        try:
            get_resp = get_settings_srv(task)
            if get_resp.result_code != 1:
                rospy.logerr(
                    f"get_egm_settings({task}) failed: "
                    f"result_code={get_resp.result_code}, message='{get_resp.message}'"
                )
                ok = False
                continue

            settings = get_resp.settings
            old_max_speed_deviation = settings.activate.max_speed_deviation
            old_ramp_in_time = settings.run.ramp_in_time

            settings.activate.max_speed_deviation = max_speed_deviation
            settings.run.ramp_in_time = ramp_in_time

            set_resp = set_settings_srv(task, settings)
            if set_resp.result_code != 1:
                rospy.logerr(
                    f"set_egm_settings({task}) failed: "
                    f"result_code={set_resp.result_code}, message='{set_resp.message}'"
                )
                ok = False
                continue

            rospy.loginfo(
                f"Configured EGM settings for {task}: "
                f"max_speed_deviation {old_max_speed_deviation:.3f} -> "
                f"{max_speed_deviation:.3f} deg/s, "
                f"ramp_in_time {old_ramp_in_time:.3f} -> {ramp_in_time:.3f} s"
            )
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to configure EGM settings for {task}: {e}")
            ok = False

    return ok


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

        if srv_name == "/yumi/rws/start_rapid":
            if not configure_egm_settings():
                rospy.logerr("Failed to configure one or more EGM tasks.")
                break

    start_joint_group_vel_controller()

    rospy.loginfo("YuMi startup sequence finished.")
