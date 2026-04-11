#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState


class JointStateRepublisher:
    def __init__(self):
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        rospy.Subscriber("/yumi/egm/joint_states", JointState, self.cb, queue_size=10)

    def cb(self, msg):
        out = JointState()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = msg.header.frame_id
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("yumi_joint_state_republisher")
    JointStateRepublisher()
    rospy.spin()
