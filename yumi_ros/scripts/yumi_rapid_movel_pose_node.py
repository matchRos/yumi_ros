#!/usr/bin/env python3
import math
import threading

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String

from abb_robot_msgs.msg import RAPIDSymbolPath
from abb_robot_msgs.srv import SetRAPIDSymbol, SetRAPIDNum, GetRAPIDBool, SetRAPIDBool
from abb_rapid_sm_addin_msgs.srv import SetRAPIDRoutine
from abb_robot_msgs.srv import TriggerWithResultCode


class YumiTRobRapidMoveLNode:
    def __init__(self):
        self.task = rospy.get_param("~task", "T_ROB_R")
        self.module = rospy.get_param("~module", "TRobRAPID")
        self.base_frame = rospy.get_param("~base_frame", "yumi_base_link")
        self.tip_link = rospy.get_param("~tip_link", "yumi_tcp_r")
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/yumi/robr/rapid_movel_target_pose"
        )

        self.default_speed_tcp = rospy.get_param("~speed_tcp", 100.0)
        self.default_speed_ori = rospy.get_param("~speed_ori", 10.0)
        self.default_speed_leax = rospy.get_param("~speed_leax", 100.0)
        self.default_speed_reax = rospy.get_param("~speed_reax", 10.0)

        self.arrival_pos_tol_m = rospy.get_param("~arrival_pos_tol_m", 0.003)
        self.arrival_rot_tol_rad = rospy.get_param("~arrival_rot_tol_rad", 0.05)
        self.feedback_rate_hz = rospy.get_param("~feedback_rate_hz", 10.0)
        self.command_timeout = rospy.get_param("~command_timeout", 120.0)

        self.target_symbol = rospy.get_param("~target_symbol", "ros_movel_target")
        self.busy_symbol = rospy.get_param("~busy_symbol", "ros_movel_busy")
        self.done_symbol = rospy.get_param("~done_symbol", "ros_movel_done")
        self.speed_symbol = rospy.get_param("~speed_symbol", "move_speed_input")
        self.routine_name = rospy.get_param("~routine_name", "runMoveL")

        self.listener = tf.TransformListener()
        rospy.sleep(0.5)

        rospy.wait_for_service("/yumi/rws/set_rapid_symbol")
        rospy.wait_for_service("/yumi/rws/set_rapid_num")
        rospy.wait_for_service("/yumi/rws/get_rapid_bool")
        rospy.wait_for_service("/yumi/rws/sm_addin/set_rapid_routine")
        rospy.wait_for_service("/yumi/rws/sm_addin/run_rapid_routine")
        rospy.wait_for_service("/yumi/rws/set_rapid_bool")

        self.set_rapid_symbol = rospy.ServiceProxy(
            "/yumi/rws/set_rapid_symbol", SetRAPIDSymbol
        )
        self.set_rapid_num = rospy.ServiceProxy("/yumi/rws/set_rapid_num", SetRAPIDNum)
        self.get_rapid_bool = rospy.ServiceProxy(
            "/yumi/rws/get_rapid_bool", GetRAPIDBool
        )
        self.set_rapid_bool = rospy.ServiceProxy(
            "/yumi/rws/set_rapid_bool", SetRAPIDBool
        )
        self.set_rapid_routine = rospy.ServiceProxy(
            "/yumi/rws/sm_addin/set_rapid_routine", SetRAPIDRoutine
        )
        self.run_rapid_routine = rospy.ServiceProxy(
            "/yumi/rws/sm_addin/run_rapid_routine", TriggerWithResultCode
        )

        self.arrived_pub = rospy.Publisher("~arrived", Bool, queue_size=1, latch=True)
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)

        self.lock = threading.Lock()
        self.active = False
        self.target_pose_base = None
        self.command_start_time = None

        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.feedback_rate_hz), self.feedback_cb)

        self.publish_state(False, False, "idle")
        rospy.loginfo("YuMi TRobRAPID MoveL node ready")

    def make_path(self, symbol_name):
        path = RAPIDSymbolPath()
        path.task = self.task
        path.module = self.module
        path.symbol = symbol_name
        return path

    def check_result(self, resp, label):
        if resp.result_code not in [0, 1]:
            raise RuntimeError(
                f"{label} failed: code={resp.result_code}, msg='{resp.message}'"
            )

    def set_num(self, symbol_name, value):
        resp = self.set_rapid_num(self.make_path(symbol_name), float(value))
        self.check_result(resp, f"set num {symbol_name}")

    def set_symbol_raw(self, symbol_name, raw_value):
        resp = self.set_rapid_symbol(self.make_path(symbol_name), raw_value)
        self.check_result(resp, f"set symbol {symbol_name}")

    def set_bool(self, symbol_name, value):
        resp = self.set_rapid_bool(self.make_path(symbol_name), bool(value))
        self.check_result(resp, f"set bool {symbol_name}")

    def get_bool(self, symbol_name):
        resp = self.get_rapid_bool(self.make_path(symbol_name))
        self.check_result(resp, f"get bool {symbol_name}")
        return resp.value

    def set_speeddata(self, v_tcp, v_ori, v_leax, v_reax):
        raw = f"[{float(v_tcp):.6f},{float(v_ori):.6f},{float(v_leax):.6f},{float(v_reax):.6f}]"
        self.set_symbol_raw(self.speed_symbol, raw)

    def normalize_quat(self, q):
        n = math.sqrt(sum(v * v for v in q))
        if n < 1e-12:
            return [0.0, 0.0, 0.0, 1.0]
        return [v / n for v in q]

    def pose_to_abb_robtarget_raw(self, pose):
        x_mm = 1000.0 * pose.position.x
        y_mm = 1000.0 * pose.position.y
        z_mm = 1000.0 * pose.position.z

        q_ros = self.normalize_quat(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )
        qx, qy, qz, qw = q_ros
        # ABB robtarget quaternion order is [q1,q2,q3,q4] = [w,x,y,z]
        abb_q = [qw, qx, qy, qz]

        conf = [0, 0, 0, 0]
        extax = [9e9, 9e9, 9e9, 9e9, 9e9, 9e9]

        return (
            f"[[{x_mm:.3f},{y_mm:.3f},{z_mm:.3f}],"
            f"[{abb_q[0]:.8f},{abb_q[1]:.8f},{abb_q[2]:.8f},{abb_q[3]:.8f}],"
            f"[{conf[0]},{conf[1]},{conf[2]},{conf[3]}],"
            f"[{extax[0]:.1E},{extax[1]:.1E},{extax[2]:.1E},{extax[3]:.1E},{extax[4]:.1E},{extax[5]:.1E}]]"
        )

    def transform_pose_to_base(self, msg):
        if msg.header.frame_id in ["", self.base_frame]:
            return msg
        self.listener.waitForTransform(
            self.base_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(2.0)
        )
        return self.listener.transformPose(self.base_frame, msg)

    def get_current_pose_tf(self):
        self.listener.waitForTransform(
            self.base_frame, self.tip_link, rospy.Time(0), rospy.Duration(1.0)
        )
        trans, rot = self.listener.lookupTransform(
            self.base_frame, self.tip_link, rospy.Time(0)
        )
        return trans, rot

    def quat_angle(self, q1, q2):
        dot = abs(sum(a * b for a, b in zip(q1, q2)))
        dot = max(-1.0, min(1.0, dot))
        return 2.0 * math.acos(dot)

    def publish_state(self, active, arrived, status):
        self.active_pub.publish(Bool(data=active))
        self.arrived_pub.publish(Bool(data=arrived))
        self.status_pub.publish(String(data=status))

    def pose_cb(self, msg):
        try:
            msg_base = self.transform_pose_to_base(msg)
            raw_target = self.pose_to_abb_robtarget_raw(msg_base.pose)

            self.set_symbol_raw(self.target_symbol, raw_target)
            self.set_bool(self.done_symbol, False)
            self.set_bool(self.busy_symbol, False)
            self.set_speeddata(
                self.default_speed_tcp,
                self.default_speed_ori,
                self.default_speed_leax,
                self.default_speed_reax,
            )

            resp = self.set_rapid_routine(self.task, self.routine_name)
            self.check_result(resp, "set rapid routine")

            resp = self.run_rapid_routine()
            self.check_result(resp, "run rapid routine")

            with self.lock:
                self.active = True
                self.target_pose_base = msg_base.pose
                self.command_start_time = rospy.Time.now()

            self.publish_state(True, False, "moving")
            rospy.loginfo(
                "Triggered RAPID MoveL to x=%.3f y=%.3f z=%.3f in %s",
                msg_base.pose.position.x,
                msg_base.pose.position.y,
                msg_base.pose.position.z,
                self.base_frame,
            )
        except Exception as e:
            with self.lock:
                self.active = False
                self.target_pose_base = None
                self.command_start_time = None
            self.publish_state(False, False, f"error: {e}")
            rospy.logerr("Failed to trigger RAPID MoveL: %s", e)

    def feedback_cb(self, _event):
        with self.lock:
            active = self.active
            target = self.target_pose_base
            start_time = self.command_start_time

        if not active or target is None or start_time is None:
            return

        try:
            busy = self.get_bool(self.busy_symbol)
            done = self.get_bool(self.done_symbol)

            trans, rot = self.get_current_pose_tf()
            dx = target.position.x - trans[0]
            dy = target.position.y - trans[1]
            dz = target.position.z - trans[2]
            pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)

            q_target = self.normalize_quat(
                [
                    target.orientation.x,
                    target.orientation.y,
                    target.orientation.z,
                    target.orientation.w,
                ]
            )
            q_current = self.normalize_quat(list(rot))
            rot_err = self.quat_angle(q_target, q_current)

            if (
                done
                and pos_err <= self.arrival_pos_tol_m
                and rot_err <= self.arrival_rot_tol_rad
            ):
                with self.lock:
                    self.active = False
                self.publish_state(False, True, "arrived")
                return

            if (rospy.Time.now() - start_time).to_sec() > self.command_timeout:
                with self.lock:
                    self.active = False
                self.publish_state(False, False, "timeout")
                return

            if busy:
                self.publish_state(
                    True, False, f"moving pos_err={pos_err:.4f} rot_err={rot_err:.4f}"
                )
            else:
                self.publish_state(
                    True, False, f"waiting pos_err={pos_err:.4f} rot_err={rot_err:.4f}"
                )

        except Exception as e:
            with self.lock:
                self.active = False
            self.publish_state(False, False, f"error: {e}")
            rospy.logerr_throttle(1.0, "Feedback error: %s", e)


if __name__ == "__main__":
    rospy.init_node("yumi_trrapid_movel_node")
    YumiTRobRapidMoveLNode()
    rospy.spin()
