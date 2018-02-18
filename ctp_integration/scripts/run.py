#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import rospy
import tf2_ros as tf2

from costar_task_plan.mcts import PlanExecutionManager, DefaultExecute
from costar_task_plan.mcts import OptionsExecutionManager
from costar_task_plan.robotics.core import RosTaskParser
from costar_task_plan.robotics.core import CostarWorld
from costar_task_plan.robotics.workshop import UR5_C_MODEL_CONFIG
from sensor_msgs.msg import JointState

from ctp_integration import MakeStackTask
from ctp_integration.observer import IdentityObserver, Observer
from ctp_integration.collector import DataCollector
from ctp_integration.util import GetDetectObjectsService

def getArgs():
    '''
    Get argument parser and call it to get information from the command line.

    Parameters:
    -----------
    none

    Returns:
    --------
    args: command-line arguments
    '''
    parser = argparse.ArgumentParser(add_help=True, description="Parse rosbag into graph.")
    parser.add_argument("--fake",
                        action="store_true",
                        help="create some fake options for stuff")
    parser.add_argument("--show",
                        action="store_true",
                        help="show a graph of the compiled task")
    parser.add_argument("--plan",
                        action="store_true",
                        help="set if you want the robot to generate a task plan")
    parser.add_argument("--execute",
                        type=int,
                        help="execute this many loops",
                        default=1)
    parser.add_argument("--iter","-i",
                        default=0,
                        type=int,
                        help="number of samples to draw")
    parser.add_argument("--mode",
                        choices=["collect","test"],
                        default="collect",
                        help="Choose which mode to run in.")
    parser.add_argument("--verbose", "-v",
                        type=int,
                        default=0,
                        help="verbosity level")

    return parser.parse_args()

def fakeTaskArgs():
  '''
  Set up a simplified set of arguments. These are for the optimization loop, 
  where we expect to only have one object to grasp at a time.
  '''
  args = {
    'block': ['block_1', 'block_2'],
    'endpoint': ['r_ee_link'],
    'high_table': ['ar_marker_2'],
    'Cube_blue': ['blue1'],
    'Cube_red': ['red1'],
    'Cube_green': ['green1'],
    'Cube_yellow': ['yellow1'],
  }
  return args

def main():

    # Create node and read options from command line
    rospy.init_node("ctp_integration_runner")
    args = getArgs()

    # Default joints for the motion planner when it needs to go to the home
    # position - this will take us out of the way of the camera.
    try:
        q0 = rospy.get_param('/costar/robot/home')
    except KeyError as e:
        rospy.logwarn("CoSTAR home position not set, using default.")
        q0 = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    # Create the task model, world, and other tools
    task = MakeStackTask()
    world = CostarWorld(robot_config=UR5_C_MODEL_CONFIG)
    tf_buffer = tf2.Buffer()
    tf_listener = tf2.TransformListener(tf_buffer)
    rospy.sleep(0.5) # wait to cache incoming transforms

    if args.fake:
        world.addObjects(fakeTaskArgs())
        filled_args = task.compile(fakeTaskArgs())
        observe = IdentityObserver(world, task)
    else:
        objects = GetDetectObjectsService()
        observe = Observer(world=world,
                task=task,
                detect_srv=objects,
                topic="/costar_sp_segmenter/detected_object_list",
                tf_listener=tf_buffer)

    # print out task info
    if args.verbose > 0:
        print(task.nodeSummary())
        print(task.children['ROOT()'])

    collector = None
    if args.mode == "show":
        from costar_task_plan.tools import showTask
        showTask(task)
    elif args.mode == "collect":
        collector = DataCollector(
                data_root="~/.costar/data",
                rate=10,
                data_type="h5f",
                robot_config=UR5_C_MODEL_CONFIG,
                camera_frame="camera_link",
                tf_listener=tf_buffer)

    for i in range(args.execute):
        print("Executing trial %d..."%(i))
        task, world = observe()
        names, options = task.sampleSequence()
        plan = OptionsExecutionManager(options)

        # Update the plan and the collector in synchrony.
        while not rospy.is_shutdown():
            # Note: this will be "dummied out" for most of 
            control = plan.apply(world)
            ok = collector.tick()

        if collector is not None:
            collector.save(i, 1.)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        pass
