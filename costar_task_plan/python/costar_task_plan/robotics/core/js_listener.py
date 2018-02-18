
import rospy
from sensor_msgs.msg import JointState
import numpy as np


class JointStateListener(object):

    def __init__(self, config):
        self.topic = config['joint_states_topic']
        self.sub = rospy.Subscriber(self.topic, JointState, self.js_cb)
        self.joints = config['joints']
        self.dof = config['dof']
        self.q0 = None
        self.dq = None
        self.old_q0 = [0] * self.dof

        if self.joints is not None:
            assert self.dof == len(self.joints)

    def js_cb(self, msg):
        if len(msg.position) is self.dof:
            self.old_q0 = self.q0
            self.q0 = np.array(msg.position)
        elif self.joints is not None:
            entries = {}
            for name, pos in zip(msg.name, msg.position):
                if name in self.joints:
                    entries[name] = pos

            self.old_q0 = self.q0
            self.q0 = [entries[name] for name in self.joints]
            self.dq = [0.] * len(self.q0)
        else:
            rospy.logwarn('Incorrect joint state message dimensionality!')
