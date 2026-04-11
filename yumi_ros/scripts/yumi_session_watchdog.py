#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
from abb_robot_msgs.srv import TriggerWithResultCode
from controller_manager_msgs.srv import ListControllers, SwitchController
from abb_egm_msgs.msg import EGMChannelState, EGMState


class YumiSessionWatchdog:
    def __init__(self):
        self.n_joints = 14

        self.check_rate = rospy.get_param("~check_rate", 5.0)
        self.restart_cooldown = rospy.get_param("~restart_cooldown", 5.0)
        self.startup_delay = rospy.get_param("~startup_delay", 2.0)

        self.egm_required_active_channels = rospy.get_param(
            "~egm_required_active_channels", 1
        )
        self.controller_name = rospy.get_param(
            "~controller_name", "joint_group_velocity_controller"
        )

        self.last_restart_time = rospy.Time(0)
        self.latest_egm_state = None

        self.zero_pub = rospy.Publisher(
            "/yumi/egm/joint_group_velocity_controller/command",
            Float64MultiArray,
            queue_size=1,
        )

        rospy.Subscriber(
            "/yumi/egm/egm_states", EGMState, self.egm_state_cb, queue_size=1
        )

        rospy.wait_for_service("/yumi/rws/sm_addin/start_egm_joint")
        rospy.wait_for_service("/yumi/egm/controller_manager/list_controllers")
        rospy.wait_for_service("/yumi/egm/controller_manager/switch_controller")

        self.start_egm_srv = rospy.ServiceProxy(
            "/yumi/rws/sm_addin/start_egm_joint", TriggerWithResultCode
        )

        self.list_ctrl_srv = rospy.ServiceProxy(
            "/yumi/egm/controller_manager/list_controllers", ListControllers
        )

        self.switch_ctrl_srv = rospy.ServiceProxy(
            "/yumi/egm/controller_manager/switch_controller", SwitchController
        )

        rospy.sleep(self.startup_delay)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.check_rate), self.update)

        rospy.loginfo("YuMi session watchdog started")

    def egm_state_cb(self, msg):
        self.latest_egm_state = msg

    def publish_zero(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.n_joints
        self.zero_pub.publish(msg)

    def egm_is_active(self):
        if self.latest_egm_state is None:
            return False

        active_count = 0
        for ch in self.latest_egm_state.egm_channels:
            if ch.active:
                active_count += 1

        return active_count >= self.egm_required_active_channels

    def controller_is_running(self):
        try:
            resp = self.list_ctrl_srv()
            for ctrl in resp.controller:
                if ctrl.name == self.controller_name and ctrl.state == "running":
                    return True
            return False
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, f"Failed to list controllers: {e}")
            return False

    def start_egm(self):
        try:
            resp = self.start_egm_srv()
            rospy.loginfo(
                f"start_egm_joint -> result_code={resp.result_code}, message='{resp.message}'"
            )
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call start_egm_joint: {e}")
            return False

    def start_controller(self):
        try:
            resp = self.switch_ctrl_srv(
                start_controllers=[self.controller_name],
                stop_controllers=[],
                strictness=1,
                start_asap=False,
                timeout=0.0,
            )
            rospy.loginfo(f"switch_controller -> ok={resp.ok}")
            return resp.ok
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to start controller: {e}")
            return False

    def restart_sequence(self):
        rospy.logwarn("EGM or controller inactive. Starting recovery sequence.")

        # Send a zero command before trying recovery
        self.publish_zero()
        rospy.sleep(0.1)

        egm_ok = self.start_egm()
        rospy.sleep(0.1)

        ctrl_ok = self.start_controller()
        rospy.sleep(0.1)

        self.publish_zero()

        if egm_ok and ctrl_ok:
            rospy.loginfo("Recovery sequence completed")
        else:
            rospy.logwarn(
                "Recovery sequence finished, but not all steps reported success"
            )

        # TODO: Investigate the root cause of why YuMi leaves 'Start Joint Motion'
        # and returns to idle after some time even with continuous valid commands.
        # This watchdog is only a recovery workaround, not the proper fix.

    def update(self, event):
        now = rospy.Time.now()

        if (now - self.last_restart_time).to_sec() < self.restart_cooldown:
            return

        egm_active = self.egm_is_active()
        controller_running = self.controller_is_running()

        if egm_active and controller_running:
            return

        self.last_restart_time = now
        self.restart_sequence()


if __name__ == "__main__":
    rospy.init_node("yumi_session_watchdog")
    YumiSessionWatchdog()
    rospy.spin()
