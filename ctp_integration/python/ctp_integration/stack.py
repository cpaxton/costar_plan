from __future__ import print_function

import PyKDL as kdl
import rospy
import tf_conversions.posemath as pm

from costar_robot_msgs.srv import SmartMove
from geometry_msgs.msg import Pose

from costar_task_plan.abstract.task import *


def GetPoses():
    '''
    All poses have been recorded relative to /base_link. If the robot moves
    they may no longer work.

    This creates the poses necessary to make certain colorful patterns on the
    bottom of the white tray.
    '''
    pose1_left = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector(0.493, -0.202, 0.216))
    pose2_left = kdl.Frame(
            kdl.Rotation.Quaternion(0.610, 0.318, -0.549, 0.474),
            kdl.Vector(0.450, -0.304, 0.216))
    pose3_left = kdl.Frame(
            kdl.Rotation.Quaternion(0.603, 0.320, -0.528, 0.505),
            kdl.Vector(0.557, -0.336, 0.198))
    pose4_left = kdl.Frame(
            kdl.Rotation.Quaternion(0.627, 0.320, -0.518, 0.486),
            kdl.Vector(0.594, -0.228, 0.205))
    pose1_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.650, 0.300, -0.451, 0.533),
            kdl.Vector(0.492, 0.013, 0.214))
    pose2_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.645, 0.304, -0.467, 0.523),
            kdl.Vector(0.480, -0.089, 0.210))
    pose3_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.657, 0.283, -0.472, 0.514),
            kdl.Vector(0.569, -0.110, 0.198))
    pose4_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.638, 0.330, -0.421, 0.553),
            kdl.Vector(0.596, -0.014, 0.203))
    pose_home = kdl.Frame(
            kdl.Rotation.Quaternion(0.711, -0.143, -0.078, 0.684),
            kdl.Vector(0.174, -0.157, 0.682))
    poses = {"home": pose_home,
             "pose1_left": pose1_left,
             "pose2_left": pose2_left,
             "pose3_left": pose3_left,
             "pose4_left": pose4_left,
             "pose1_right": pose1_right,
             "pose2_right": pose2_right,
             "pose3_right": pose3_right,
             "pose4_right": pose4_right,}
    return poses


def _makeSmartPlaceRequest(poses, name):
    '''
    Helper function for making the place call
    '''
    req = SmartMove()
    req.pose = pm.toMsg(poses[name])
    req.name = name
    req.obj_class = "place"
    return req

def _makeSmartGraspRequest(color):
    '''
    Helper function to create a grasp request via smartmove.
    '''
    req = SmartMove()
    req.pose = None
    if not color in ["red", "blue", "green", "yellow"]:
        raise RuntimeError("color %s not recognized" % color)
    req.obj_class = "%s_cube" % color

def MakeStackTask():
    '''
    Create a version of the robot task for stacking two blocks.
    '''

    task = Task()

    # Make services
    rospy.wait_for_service("/costar/SmartPlace")
    rospy.wait_for_service("/costar/SmartGrasp")
    place = rospy.ServiceProxy("/costar/SmartPlace", SmartMove)
    grasp = rospy.ServiceProxy("/costar/SmartGrasp", SmartMove)

    # Create sub-tasks for left and right
    pickup_left = _makePickupLeft()
    pickup_right = _makePickupRight()
    place_left = _makePlaceLeft()
    place_right = _makePlaceRight()

    # Create the task: pick up any one block and put it down in a legal
    # position somewhere on the other side of the bin.
    task = Task()
    task.add("pickup_left", None, pickup_left)
    task.add("pickup_right", None, pickup_right)
    task.add("place_left", "pickup_right", place_left)
    task.add("place_right", "pickup_left", place_right)
    task.add("DONE", ["place_right", "place_left"], {})

    return task

def _makePickupLeft():
    pickup = TaskTemplate("pickup_left", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return {"task": pickup, "args": ["object"]}

def _makePickupRight():
    pickup = TaskTemplate("pickup_right", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return {"task": pickup, "args": ["object"]}

def _makePlaceLeft():
    place = TaskTemplate("place_left", ["pickup_right"])
    place.add("home", None, _homeArgs())
    place.add("detect_objects", "home", _detectObjectsArgs())
    return {"task": place, "args": ["frame"]}

def _makePlaceRight():
    place = TaskTemplate("place_right", ["pickup_left"])
    place.add("home", None, _homeArgs())
    place.add("detect_objects", "home", _detectObjectsArgs())
    return {"task": place, "args": ["frame"]}

def _pickupLeftArgs():
    # Create args for pickup from left task
    return {
        "task": pickup_left,
        "args": ["block1"],
    }

def _pickupRightArgs():
    # And create args for pickup from right task
    return {
        "task": pickup_right,
        "args": ["block1"],
    }

def _homeArgs():
    return {}

def _detectObjectsArgs():
    return {}

def _checkBlocks1And2(block1,block2,**kwargs):
    '''
    Simple function that is passed as a callable "check" when creating the task
    execution graph. This makes sure we don't build branches that just make no
    sense -- like trying to put a blue block on top of itself.

    Parameters:
    -----------
    block1: unique block name, e.g. "red_block"
    block2: second unique block name, e.g. "blue_block"
    '''
    return not block1 == block2
