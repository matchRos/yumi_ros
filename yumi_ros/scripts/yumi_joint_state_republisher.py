#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState


class JointStateRepublisher:
    def __init__(self):
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

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
        self.publish_merged()

    def left_gripper_cb(self, msg):
        self.latest_left_gripper_state = msg
        self.publish_merged()

    def right_gripper_cb(self, msg):
        self.latest_right_gripper_state = msg
        self.publish_merged()

    def publish_merged(self):
        if self.latest_arm_state is None:
            return

        out = JointState()
        out.header.stamp = rospy.Time.now()
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
