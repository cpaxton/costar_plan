# Model credit goes to Jennifer Bueller and co

from abstract import AbstractRobotInterface

import gym
from gym import spaces
import numpy as np
import os
import pybullet as pb
import rospkg
import subprocess


class JacoRobotiqInterface(AbstractRobotInterface):

    '''
    Defines action space for the Jaco arm.
    '''

    xacro_filename = 'robot/jaco_robot.urdf.xacro'
    urdf_filename = 'jaco_robot.urdf'

    arm_name = "jaco_arm"
    gripper_name = "None"
    base_name = None

    left_knuckle = 8
    left_finger = 9
    left_inner_knuckle = 12
    left_fingertip = 13

    right_knuckle = 10
    right_finger = 11
    right_inner_knuckle = 14
    right_fingertip = 15

    def __init__(self, *args, **kwargs):
        super(JacoRobotiqInterface, self).__init__(*args, **kwargs)

    def load(self):
        '''
        This is an example of a function that allows you to load a robot from
        file based on command line arguments. It just needs to find the
        appropriate directory, use xacro to create a temporary robot urdf,
        and then load that urdf with PyBullet.
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_simulation')
        filename = os.path.join(path, self.xacro_filename)
        urdf_filename = os.path.join(path, 'robot', self.urdf_filename)
        urdf = open(urdf_filename, "r")
        # urdf =
        # open("/home/albert/costar_ws/src/costar_plan/costar_simulation/robot/jaco_robot.urdf",
        # "r")

        # Recompile the URDF to make sure it's up to date
        # subprocess.call(['rosrun', 'xacro', 'xacro.py', filename],
        # stdout=urdf)

        self.handle = pb.loadURDF(urdf_filename)
        self.grasp_idx = self.findGraspFrame()
        self.loadKinematicsFromURDF(urdf_filename, "robot_root")

        return self.handle

    def place(self, pos, rot, joints):
        pb.resetBasePositionAndOrientation(self.handle, pos, rot)
        pb.createConstraint(
            self.handle, -1, -1, -1, pb.JOINT_FIXED, pos, [0, 0, 0], rot)
        for i, q in enumerate(joints):
            pb.resetJointState(self.handle, i, q)

        # gripper
        pb.resetJointState(self.handle, self.left_knuckle, 0)
        pb.resetJointState(self.handle, self.right_knuckle, 0)

        pb.resetJointState(self.handle, self.left_finger, 0)
        pb.resetJointState(self.handle, self.right_finger, 0)

        pb.resetJointState(self.handle, self.left_fingertip, 0)
        pb.resetJointState(self.handle, self.right_fingertip, 0)

        self.arm(joints,)
        self.gripper(0)

    def arm(self, cmd, mode=pb.POSITION_CONTROL):
        '''
        Set joint commands for the robot arm.
        '''
        if len(cmd) > 6:
            raise RuntimeError('too many joint positions')
        for i, q in enumerate(cmd):
            pb.setJointMotorControl2(self.handle, i, mode, q)

    def gripper(self, cmd, mode=pb.POSITION_CONTROL):
        '''
        Gripper commands need to be mirrored to simulate behavior of the actual
        UR5.
        '''
        pb.setJointMotorControl2(self.handle, self.left_knuckle, mode,  -cmd)
        pb.setJointMotorControl2(
            self.handle, self.left_inner_knuckle, mode,  -cmd)
        pb.setJointMotorControl2(self.handle, self.left_finger, mode,  cmd)
        pb.setJointMotorControl2(self.handle, self.left_fingertip, mode,  cmd)

        pb.setJointMotorControl2(self.handle, self.right_knuckle, mode,  -cmd)
        pb.setJointMotorControl2(
            self.handle, self.right_inner_knuckle, mode,  -cmd)
        pb.setJointMotorControl2(self.handle, self.right_finger, mode,  cmd)
        pb.setJointMotorControl2(self.handle, self.right_fingertip, mode,  cmd)

    def act(self, action):
        '''
        Parse a list of continuous commands and send it off to the robot.
        '''
        assert(len(action) == 7)
        self.arm(action[:6])
        self.gripper(action[6])

    def getActionSpace(self):
        return spaces.Tuple((spaces.Box(-np.pi, np.pi, 6), spaces.Box(-0.6, 0.6, 1)))

    def _getArmPosition(self):
        q = [0.] * 6
        for i in xrange(6):
            q[i] = pb.getJointState(self.handle, i)[0]
        return q

    def _getGripper(self):
        return pb.getJointState(self.handle, self.left_finger)
