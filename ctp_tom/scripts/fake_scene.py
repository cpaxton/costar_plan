#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import rospy
import PyKDL as kdl
import tf
import tf_conversions.posemath as pm

from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse
from trajectory_msgs.msg import JointTrajectoryPoint

class FakeScenePublisher(object):

    def __init__(self):
        self.seq = 0
        self.tf_pub = tf.TransformBroadcaster()
        self.orange1 = (0.641782207489,
                  -0.224464386702,
                  -0.363829042912)
        self.orange2 = (0.69,
                  -0.31,
                  -0.363829042912)
        self.orange3 = (0.68,
                  -0.10,
                  -0.363829042912)

        self.blue1 = (0.641782207489,
                  -0.224464386702,
                  -0.363829042912)
        self.red1 = (0.69,
                  -0.31,
                  -0.363829042912)
        self.green1 = (0.68,
                  -0.10,
                  -0.363829042912)
        self.yellow1 = (0.68,
                  -0.20,
                  -0.363829042912)

        table_pos = (0.5, 0., 0.863-0.5)
        self.table_pos, self.table_rot = pm.toTf(kdl.Frame(
            kdl.Rotation.RotZ(-np.pi/2.),
            kdl.Vector(*table_pos)))
        box_pos = (0.82, -0.4, 0.863-0.5+0.1025)
        self.box_pos, self.box_rot = pm.toTf(kdl.Frame(
            kdl.Rotation.RotZ(1.5),
            kdl.Vector(*box_pos)))
        box_pos = (0.82, -0.4, 0.863-0.5+0.1025)
        self.box_pos, self.box_rot = pm.toTf(kdl.Frame(
            kdl.Rotation.RotZ(1.5),
            kdl.Vector(*box_pos)))
        b1_pos = (0.78, -0.03, 0.863-0.5+0.0435)
        self.block1_pos, self.block1_rot = pm.toTf(kdl.Frame(
            kdl.Rotation.RotZ(-np.pi/2.),
            kdl.Vector(*b1_pos)))
        b1_pos = (0.73, 0.12, 0.863-0.5+0.0435)
        self.block2_pos, self.block2_rot = pm.toTf(kdl.Frame(
            kdl.Rotation.RotZ(-np.pi/2.+0.07),
            kdl.Vector(*b1_pos)))

    def tick(self):
        self.tf_pub.sendTransform(self.orange1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange_1",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange2,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange_2",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange3,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange_3",
                "/torso_link")
        self.tf_pub.sendTransform(self.table_pos,
                self.table_rot,
                rospy.Time.now(),
                "/tom_table",
                "/base_link")
        self.tf_pub.sendTransform(self.box_pos,
                self.box_rot,
                rospy.Time.now(),
                "/box",
                "/base_link")
        self.tf_pub.sendTransform(self.block1_pos,
                self.block1_rot,
                rospy.Time.now(),
                "/block_1",
                "/base_link")
        self.tf_pub.sendTransform(self.block2_pos,
                self.block2_rot,
                rospy.Time.now(),
                "/block_2",
                "/base_link")
        # =============================================
        # For debugging the cubes
        self.tf_pub.sendTransform(self.red1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/red1",
                "/torso_link")
        self.tf_pub.sendTransform(self.green1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/green1",
                "/torso_link")
        self.tf_pub.sendTransform(self.blue1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/blue1",
                "/torso_link")
        self.tf_pub.sendTransform(self.yellow1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/yellow1",
                "/torso_link")

if __name__ == '__main__':

    rospy.init_node('costar_tom_sim')

    # set up sim
    sim = FakeScenePublisher()

    # loop
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
          sim.tick()
          rate.sleep()
    except rospy.ROSInterruptException, e:
        pass
