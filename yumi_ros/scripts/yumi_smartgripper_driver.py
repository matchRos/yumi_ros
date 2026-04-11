#!/usr/bin/env python3
import threading

import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState

from abb_rapid_sm_addin_msgs.srv import SetSGCommand, SetSGCommandRequest
from abb_robot_msgs.srv import TriggerWithResultCode


class SmartGripperSide:
    def __init__(self, parent, side_name, task_name, joint_name):
        self.parent = parent
        self.side_name = side_name
        self.task_name = task_name
        self.joint_name = joint_name

        self.lock = threading.Lock()
        self.ready = False
        self.last_pos_mm = 0.0

        self.command_topic = f"/yumi/gripper_{side_name}/command"
        self.state_topic = f"/yumi/gripper_{side_name}/joint_states"
        self.open_service = f"/yumi/gripper_{side_name}/open"
        self.close_service = f"/yumi/gripper_{side_name}/close"
        self.close_and_hold_service = f"/yumi/gripper_{side_name}/close_and_hold"

        self.state_pub = rospy.Publisher(self.state_topic, JointState, queue_size=10)
        rospy.Subscriber(self.command_topic, Float64, self.command_cb, queue_size=1)

        rospy.Service(self.open_service, Trigger, self.handle_open)
        rospy.Service(self.close_service, Trigger, self.handle_close)
        rospy.Service(self.close_and_hold_service, Trigger, self.handle_close_and_hold)

    def clamp_mm(self, value_mm):
        return max(self.parent.min_pos_mm, min(self.parent.max_pos_mm, float(value_mm)))

    def publish_state(self):
        with self.lock:
            pos_mm = self.last_pos_mm

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [self.joint_name]
        msg.position = [pos_mm * self.parent.joint_scale]
        msg.velocity = [0.0]
        msg.effort = [0.0]
        self.state_pub.publish(msg)

    def set_last_pos(self, pos_mm):
        with self.lock:
            self.last_pos_mm = float(pos_mm)

    def send(self, command, pos_mm=0.0):
        req = SetSGCommandRequest()
        req.task = self.task_name
        req.command = command
        req.target_position = float(pos_mm)

        try:
            resp = self.parent.set_cmd(req)
        except Exception as e:
            rospy.logerr(f"[gripper_{self.side_name}] set_sg_command failed: {e}")
            return False, f"set_sg_command exception: {e}"

        # On this ABB stack, result_code=1 can still correspond to successful execution.
        if resp.result_code not in [0, 1]:
            text = f"set_sg_command failed: code={resp.result_code}, message='{resp.message}'"
            rospy.logerr(f"[gripper_{self.side_name}] {text}")
            return False, text

        try:
            run_resp = self.parent.run_cmd()
        except Exception as e:
            rospy.logerr(f"[gripper_{self.side_name}] run_sg_routine failed: {e}")
            return False, f"run_sg_routine exception: {e}"

        if run_resp.result_code not in [0, 1]:
            text = f"run_sg_routine failed: code={run_resp.result_code}, message='{run_resp.message}'"
            rospy.logerr(f"[gripper_{self.side_name}] {text}")
            return False, text

        return True, "ok"

    def startup_sequence(self):
        rospy.loginfo(f"[gripper_{self.side_name}] initialize")
        ok, text = self.send(SetSGCommandRequest.SG_COMMAND_INITIALIZE, 0.0)
        if not ok:
            return False, f"initialize failed: {text}"

        rospy.sleep(self.parent.startup_pause_s)

        rospy.loginfo(f"[gripper_{self.side_name}] calibrate")
        ok, text = self.send(SetSGCommandRequest.SG_COMMAND_CALIBRATE, 0.0)
        if not ok:
            return False, f"calibrate failed: {text}"

        self.ready = True
        rospy.loginfo(f"[gripper_{self.side_name}] ready")
        return True, "ready"

    def move_to(self, target_mm):
        if not self.ready:
            return False, "gripper not ready"

        target_mm = self.clamp_mm(target_mm)
        rospy.loginfo(f"[gripper_{self.side_name}] move to {target_mm:.1f} mm")

        ok, text = self.send(SetSGCommandRequest.SG_COMMAND_MOVE_TO, target_mm)
        if ok:
            self.set_last_pos(target_mm)
            self.publish_state()
            return True, f"moved to {target_mm:.1f} mm"
        return False, text

    def open(self):
        return self.move_to(self.parent.max_pos_mm)

    def close(self):
        return self.move_to(self.parent.min_pos_mm)

    def close_and_hold(self):
        if not self.ready:
            return False, "gripper not ready"

        rospy.loginfo(f"[gripper_{self.side_name}] close and hold")

        ok, text = self.send(SetSGCommandRequest.SG_COMMAND_GRIP_IN, 0.0)
        if ok:
            self.set_last_pos(self.parent.min_pos_mm)
            self.publish_state()
            return True, "closed and holding"

        return False, text

    def command_cb(self, msg):
        ok, text = self.move_to(msg.data)
        if not ok:
            rospy.logerr(f"[gripper_{self.side_name}] command rejected: {text}")

    def handle_open(self, _req):
        ok, text = self.open()
        return TriggerResponse(success=ok, message=text)

    def handle_close(self, _req):
        ok, text = self.close()
        return TriggerResponse(success=ok, message=text)

    def handle_close_and_hold(self, _req):
        ok, text = self.close_and_hold()
        return TriggerResponse(success=ok, message=text)


class YumiSmartGripperDriver:
    def __init__(self):
        rospy.wait_for_service("/yumi/rws/sm_addin/set_sg_command")
        rospy.wait_for_service("/yumi/rws/sm_addin/run_sg_routine")

        self.set_cmd = rospy.ServiceProxy(
            "/yumi/rws/sm_addin/set_sg_command",
            SetSGCommand,
        )
        self.run_cmd = rospy.ServiceProxy(
            "/yumi/rws/sm_addin/run_sg_routine",
            TriggerWithResultCode,
        )

        self.publish_rate = rospy.get_param("~publish_rate", 10.0)
        self.startup_pause_s = rospy.get_param("~startup_pause_s", 0.5)

        self.min_pos_mm = rospy.get_param("~min_pos_mm", 0.0)
        # TODO: later raise back to 25.0 mm
        self.max_pos_mm = rospy.get_param("~max_pos_mm", 19.0)

        # If the URDF joint is prismatic in meters, 0.001 is correct.
        self.joint_scale = rospy.get_param("~joint_scale", 0.001)

        self.left = SmartGripperSide(
            parent=self,
            side_name="l",
            task_name=rospy.get_param("~left_task", "T_ROB_L"),
            joint_name=rospy.get_param("~left_joint_name", "gripper_l_joint"),
        )
        self.right = SmartGripperSide(
            parent=self,
            side_name="r",
            task_name=rospy.get_param("~right_task", "T_ROB_R"),
            joint_name=rospy.get_param("~right_joint_name", "gripper_r_joint"),
        )

        rospy.loginfo("YuMi SmartGripper driver starting")
        self.left.startup_sequence()
        self.right.startup_sequence()

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_cb)
        rospy.loginfo("YuMi SmartGripper driver ready")

    def timer_cb(self, _event):
        self.left.publish_state()
        self.right.publish_state()


if __name__ == "__main__":
    rospy.init_node("yumi_smartgripper_driver")
    YumiSmartGripperDriver()
    rospy.spin()
