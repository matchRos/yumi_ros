#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState


class JointStateRepublisher:
    def __init__(self):
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.use_input_stamp = rospy.get_param("~use_input_stamp", False)
        self.force_monotonic_stamps = rospy.get_param("~force_monotonic_stamps", True)
        self.publish_on_gripper_update = rospy.get_param("~publish_on_gripper_update", False)
        self.last_published_stamp = rospy.Time(0)

        self.latest_arm_state = None
        self.latest_left_gripper_state = JointState()  # None
        self.latest_right_gripper_state = JointState()  # None

        rospy.Subscriber(
            "/yumi/egm/joint_states", JointState, self.arm_cb, queue_size=10
        )
        rospy.Subscriber(
            "/yumi/gripper_l/joint_states",
            JointState,
            self.left_gripper_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            "/yumi/gripper_r/joint_states",
            JointState,
            self.right_gripper_cb,
            queue_size=10,
        )

    def arm_cb(self, msg):
        self.latest_arm_state = msg
        stamp = msg.header.stamp if self.use_input_stamp else rospy.Time.now()
        self.publish_merged(stamp)

    def left_gripper_cb(self, msg):
        self.latest_left_gripper_state = msg
        if self.publish_on_gripper_update:
            self.publish_merged(rospy.Time.now())

    def right_gripper_cb(self, msg):
        self.latest_right_gripper_state = msg
        if self.publish_on_gripper_update:
            self.publish_merged(rospy.Time.now())

    def monotonic_stamp(self, stamp):
        if stamp is None or stamp == rospy.Time(0):
            stamp = rospy.Time.now()
        if self.force_monotonic_stamps and stamp <= self.last_published_stamp:
            stamp = rospy.Time.from_sec(self.last_published_stamp.to_sec() + 1e-9)
        self.last_published_stamp = stamp
        return stamp

    def publish_merged(self, stamp=None):
        if self.latest_arm_state is None:
            return

        out = JointState()
        out.header.stamp = self.monotonic_stamp(stamp)
        out.header.frame_id = self.latest_arm_state.header.frame_id

        out.name = list(self.latest_arm_state.name)
        out.position = list(self.latest_arm_state.position)
        out.velocity = list(self.latest_arm_state.velocity)
        out.effort = list(self.latest_arm_state.effort)

        self.append_joint_state(out, self.latest_left_gripper_state)
        self.append_joint_state(out, self.latest_right_gripper_state)

        self.pub.publish(out)

    def append_joint_state(self, merged, part):
        if part is None:
            return

        for i, name in enumerate(part.name):
            if name in merged.name:
                continue

            merged.name.append(name)
            merged.position.append(part.position[i] if i < len(part.position) else 0.0)
            merged.velocity.append(part.velocity[i] if i < len(part.velocity) else 0.0)
            merged.effort.append(part.effort[i] if i < len(part.effort) else 0.0)


if __name__ == "__main__":
    rospy.init_node("yumi_joint_state_republisher")
    JointStateRepublisher()
    rospy.spin()
