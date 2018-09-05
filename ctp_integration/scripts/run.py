#!/usr/bin/env python

import os
import sys
import copy
import time
from threading import Thread
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import PyKDL as kdl
import rospy
import roslaunch
import tf_conversions.posemath as pm
import tf2_ros as tf2
import traceback

from costar_task_plan.mcts import PlanExecutionManager, DefaultExecute
from costar_task_plan.mcts import OptionsExecutionManager
from costar_task_plan.robotics.core import RosTaskParser
from costar_task_plan.robotics.core import CostarWorld
from costar_task_plan.robotics.workshop import UR5_C_MODEL_CONFIG
from sensor_msgs.msg import JointState
from ctp_integration.observer import IdentityObserver, Observer
from ctp_integration.collector import DataCollector
from ctp_integration.util import GetDetectObjectsService
from ctp_integration.util import GetOpenGripperService
from ctp_integration.util import GetCloseGripperService
from ctp_integration.stack import GetGraspPose
from ctp_integration.stack import GetStackPose
from ctp_integration.stack import GetMoveToPose
from ctp_integration.stack import GetHome
from ctp_integration.stack import GetRandomHome
from ctp_integration.stack import GetUpdate
from ctp_integration.stack import GetStackManager
from ctp_integration.constants import GetHomeJointSpace
from ctp_integration.launcher import launch_main

import faulthandler

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
    parser = argparse.ArgumentParser(add_help=True, description="Run data collection for costar plan.")
    parser.add_argument("--fake",
                        action="store_true",
                        help="create some fake options for stuff")
    parser.add_argument("--show",
                        action="store_true",
                        help="show a graph of the compiled task")
    parser.add_argument("--plan",
                        action="store_true",
                        help="set if you want the robot to generate a task plan")
    parser.add_argument("--launch",
                        action="store_true",
                        help="Starts ROS automatically. Set to false if this script should assume ros is already launched.")
    parser.add_argument("--execute",
                        type=int,
                        help="execute this many loops",
                        default=100)
    parser.add_argument("--start",
                        type=int,
                        help="start from this index",
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
    parser.add_argument("--rate", "-r",
                        default=10,
                        type=int,
                        help="rate at which data will be collected in hertz")

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

def collect_data(args):
    if args.launch:
        time.sleep(5)
    # Create node and read options from command line
    rospy.init_node("ctp_data_collection_runner")

    # Default joints for the motion planner when it needs to go to the home
    # position - this will take us out of the way of the camera.
    q0 = GetHomeJointSpace()
    # Create the task model, world, and other tools
    # task = MakeStackTask()
    rospy.loginfo("Making world...")
    world = CostarWorld(robot_config=UR5_C_MODEL_CONFIG)
    rospy.loginfo("Aggregating TF data...")
    tf_buffer = tf2.Buffer(rospy.Duration(120))
    # the tf_listener fills out the buffers
    tf_listener = tf2.TransformListener(tf_buffer)

    rospy.loginfo("Node started, waiting for transform data...")
    rospy.sleep(0.5) # wait to cache incoming transforms

    rospy.loginfo("Making stack manager...")
    stack_task = GetStackManager()
    if args.fake:
        world.addObjects(fakeTaskArgs())
        filled_args = stack_task.compile(fakeTaskArgs())
        observe = IdentityObserver(world, stack_task)
    else:
        objects = GetDetectObjectsService()
        observe = Observer(world=world,
                task=stack_task,
                detect_srv=objects,
                topic="/costar_sp_segmenter/detected_object_list",
                tf_buffer=tf_buffer,
                tf_listener=tf_listener)

    # print out task info
    # TODO(ahundt) re-enable summary for this task
    # if args.verbose > 0:
    #     print(task.nodeSummary())
    #     print(task.children['ROOT()'])

    collector = DataCollector(
            task=stack_task,
            data_root="~/.costar/data",
            rate=args.rate,
            data_type="h5f",
            robot_config=UR5_C_MODEL_CONFIG,
            camera_frame="camera_link",
            tf_buffer=tf_buffer,
            tf_listener=tf_listener)
    # set fn to call after actions
    home, rate, move_to_pose, close_gripper, open_gripper = \
        initialize_collection_objects(args, observe, collector, stack_task)

    # How we verify the objet
    def verify(object_name):
        '''
        Simple verify functor. This is designed to work if we have one object
        of each color, and is not guaranteed to work otherwise.

        Parameters:
        -----------
        object_name: name of the object being manipulated
        '''
        pose = None
        for i in range(50):
            try:
                t = rospy.Time(0)
                pose = collector.tf_buffer.lookup_transform(collector.base_link, object_name, t)
                break
            except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                rospy.sleep(0.1)
        if pose is None:
            rospy.logwarn("Failed lookup: %s to %s"%(collector.base_link, object_name))
            return False
        print("object name and pose: ", object_name, pose)
        return pose.transform.translation.z > 0.095

    consecutive_bad_rounds = 0
    max_consecutive_bad_rounds = 5
    start = max(0, args.start-1)
    i = start
    idx = i + 1
    # go home on startup outside of try block so program
    # won't save files if ROS is not active and ready
    home()
    # start collecting data
    try:
        # start main execution loop, should run at specified rate
        while i < args.execute:
            home_q, home_pose = home()
            collector.set_home_pose(home_pose)

            # perform initial rate sleep to
            # initialize duration remaining time counter
            rate.sleep()
            t = rospy.Time(0)
            home_pose = collector.tf_buffer.lookup_transform(collector.base_link, 'ee_link', t)
            print("home_pose: " + str(home_pose))
            # rospy.sleep(0.5) # Make sure nothing weird happens with timing
            idx = i + 1
            rospy.loginfo("Executing trial %d" % (idx))
            _, world = observe()
            # NOTE: not using CTP task execution framework right now
            # It's just overkill
            #names, options = task.sampleSequence()
            #plan = OptionsExecutionManager(options)

            # Reset the task manager
            reward = 0.
            stack_task.reset()
            rospy.loginfo("Starting loop...")

            poses = []
            cur_pose = None
            frame_count = 0
            # Update the plan and the collector in synchrony.
            # This loop is exited upon completion with a break statement.
            while not rospy.is_shutdown():

                cur_pose = collector.current_ee_pose

                # Note: this will be "dummied out" for most of
                start_time = time.clock()
                done = stack_task.tick()
                tick_time = time.clock()
                if not collector.update(stack_task.current_action, done):
                    raise RuntimeError('could not handle data collection. '
                                       'There may be an inconsistency in the system state '
                                       'so try shutting all of ROS down and starting up again. '
                                       'Alternately, run this program in a debugger '
                                       'to try and diagnose the issue.')
                update_time = time.clock()

                # figure out where the time has gone
                time_str = ('Total tick + log time: {:04} sec, '
                            'Robot Tick: {:04} sec, '
                            'Data Logging: {:04} sec'.format(update_time - start_time, tick_time - start_time, update_time - tick_time))

                # Check if this script is running quickly enough,
                # and print a warning if it isn't
                verify_update_rate(update_time_remaining=rate.remaining(), update_rate=args.rate, info=time_str)
                rate.sleep()
                # if len(collector.data['image']) > 5:
                #     collector.save(idx, reward)
                #     exit()
                frame_count += 1

                if stack_task.finished_action:

                    object_was_placed = (collector.prev_action is not None and
                                        "place" in collector.prev_action.split(':')[-1])
                    if object_was_placed:
                        # We finished one step in the task,
                        # save the most recent pose update
                        rospy.loginfo("Remembering " + str(collector.prev_action))
                        poses.append(cur_pose)

                if done:
                    if stack_task.ok:
                        savestr = "WE WILL SAVE TO DISK"
                    else:
                        savestr = "BUT THERE A PROBLEM WAS DETECTED SO WE ARE SAVING TO DISK AS AN ERROR + FAILURE"
                    rospy.logwarn("DONE COLLECTING THIS ROUND, " + savestr)
                    if stack_task.ok:
                        # Increase count
                        i += 1

                        # We should actually check results here
                        # home and observe are now built into the actions
                        # home(); observe()
                        # rospy.sleep(0.5)
                        # Get the second to last object,
                        # since the final one is none.
                        if verify(collector.prev_objects[-2]):
                            reward = 1.
                        else:
                            reward = 0.
                        rospy.loginfo("reward = " + str(reward))
                    break

            if stack_task.ok:
                collector.save(idx, reward)
                print("------------------------------------------------------------")
                print("Finished one round of data collection. Attempting to automatically ")
                print("reset the test environment to continue.")
                print("")
                print("Example number:", idx, "/", args.execute)
                print("Success:", reward)
                print("")
                consecutive_bad_rounds = 0
            else:
                # Both an error and a failure!
                # We are saving the failure info now because
                # some of the most interesting failure cases
                # lead to errors.
                collector.save(idx, "error.failure")
                print("------------------------------------------------------------")
                print("Bad data collection round, " + str(consecutive_bad_rounds) + " consecutive. Attempting to automatically reset.")
                print("If this happens repeatedly try restarting the program or loading in a debugger.")
                collector.reset()
                stack_task.reset()
                consecutive_bad_rounds += 1
                if consecutive_bad_rounds > 5:
                    print("Hit limit of " + str(max_consecutive_bad_rounds) + "max consecutive bad rounds. ")
                    raise RuntimeError("Killing the program... you may want to debug this or "
                                       "hopefully somebody will restart it automatically! "
                                       "You can try the following bash line for auto restarts: "
                                       "while true; do ./scripts/run.py --execute 1000; done")

            rospy.loginfo('Attempting to unstack the blocks')

            for count_from_top, drop_pose in enumerate(reversed(poses)):
                if drop_pose is None:
                    continue
                # Determine destination spot above the block
                unstack_one_block(drop_pose, move_to_pose, close_gripper, open_gripper, i=count_from_top)

            if len(poses) > 0 and drop_pose is not None:
                # one extra unstack step, try to get the block on the bottom.
                count_from_top += 1
                # move vertically down in the z axis,
                # but by slightly less than a
                # whole block length which is 0.05 meters
                drop_pose.p[2] -= 0.035
                result = None
                max_tries = 1
                tries = 0
                # sometimes this tries to go below the floor,
                # so go up a bit on errors, then just give
                # up and move on if it still doesn't work
                # because the block will likely already
                # be in a reasonable position.
                while tries < max_tries and result is None:
                    try:
                        result = unstack_one_block(drop_pose, move_to_pose, close_gripper, open_gripper, i=count_from_top)
                    except RuntimeError as e:
                        drop_pose.p[2] += 0.025
                        tries +=1
        rospy.loginfo("Done one loop.")
    except RuntimeError as ex:
        ex_type, ex2, tb = sys.exc_info()
        # save the current data if we can
        message = ('error.failure due to RuntimeError:\n' +
                    ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=tb)))
        rospy.logerr(message)
        collector.save(idx, 'error.failure', log=message)
        # deletion must be explicit to prevent leaks
        # https://stackoverflow.com/a/16946886/99379
        del tb
        # re-raise the caught exception https://stackoverflow.com/a/4825279/99379
        raise
    except KeyboardInterrupt as ex:
        # save the current data if we can
        message = ('error.failure due to KeyboardInterrupt, '
                    'collection canceled based on a user request.')
        rospy.logerr(message)
        collector.save(idx, 'error.failure', log=message)
        # re-raise the caught exception https://stackoverflow.com/a/4825279/99379
        raise


def verify_update_rate(update_time_remaining, update_rate=10, minimum_update_rate_fraction_allowed=0.1, info=''):
    """
    make sure at least 10% of time is remaining when updates are performed.
    we are converting to nanoseconds here since
    that is the unit in which all reasonable
    rates are expected to be measured
    """
    update_duration_sec = 1.0 / update_rate
    minimum_allowed_remaining_time = update_duration_sec * minimum_update_rate_fraction_allowed
    min_remaining_duration = rospy.Duration(minimum_allowed_remaining_time)
    if update_time_remaining < min_remaining_duration:
        rospy.logwarn_throttle(1.0, 'Not maintaining requested update rate, there may be gaps in the data log!\n'
                               '    Update rate is: ' + str(update_rate) + 'Hz, Duration is '+ str(update_duration_sec) +' sec\n' +
                               '    Minimum time allowed time remaining is: ' + str(minimum_allowed_remaining_time) + ' sec\n'  +
                               '    Actual remaining on this update was: ' + str(float(str(update_time_remaining))/1.0e9) + ' sec\n' +
                               '    ' + info)

def unstack_one_block(drop_pose, move_to_pose, close_gripper, open_gripper,
                      block_width=0.025, backoff_distance=0.075, i="", verbose=0):
    """ drop_pose is the top position of a block

    This function will go above that block, open the gripper, grasp the block,
    then place the block at a random location.
    """
    if drop_pose is None:
        return
    # Determine destination spot above the block
    grasp_pose = copy.deepcopy(drop_pose)
    grasp_pose.p[2] -= backoff_distance # should be smart release backoff distance
    # Determine destination spot where the gripper will be closed on the block
    grasp_pose2 = copy.deepcopy(drop_pose)
    grasp_pose2.p[2] += block_width # should be smart release backoff distance
    # drop in a random spot that isn't on top of the current location
    pose_random = random_drop_coordinate(grasp_pose, drop_pose)
    if verbose:
        rospy.loginfo('unstack block ' + str(i) + ' from top to bottom drop_pose:\n' + str(drop_pose))
    # go to where the object was originally dropped from
    move_to_pose(drop_pose)
    if verbose:
        rospy.loginfo('unstack block ' + str(i) + ' from top to bottom  grasp_pose:\n' + str(grasp_pose))
    # move down to grasp an object
    if grasp_pose is not None:
        move_to_pose(grasp_pose)
    else:
        rospy.logwarn('unstack_one_block() grasp_pose was None! this needs to be debugged')
    close_gripper()
    if verbose:
        rospy.loginfo('unstack_one_block() ' + str(i) + ' from top to bottom  grasp_pose2:\n' + str(grasp_pose2))
    # move up a small amount so there won't be a collision
    if grasp_pose2 is not None:
        move_to_pose(grasp_pose2)
    else:
        rospy.logwarn('unstack_one_block() grasp_pose2 was None! this needs to be debugged')

    # move to random drop location
    if verbose:
        rospy.loginfo('unstack_one_block() random drop pose: \n' + str(pose_random))
    if grasp_pose2 is not None:
        move_to_pose(pose_random)
    else:
        rospy.logwarn('unstack_one_block() pose_random was None! this needs to be debugged')

    # release the object
    open_gripper()
    return grasp_pose2

def random_drop_coordinate(grasp_pose, drop_pose, z=0.3):
    """ Determine a random drop coordinate that isn't on top of the current object location
    axis_range: range of allowable coordinates in meters
    """
    x_random = random_drop_axis_coordinate(grasp_pose, axis_idx=0, axis_corner=0.43, axis_range=0.3)
    y_random = random_drop_axis_coordinate(grasp_pose, axis_idx=1, axis_corner=-0.00, axis_range=-0.22)
    pose_random = kdl.Frame(drop_pose.M,
            kdl.Vector(x_random,y_random,z))
    return pose_random

def random_drop_axis_coordinate(grasp_pose, axis_idx, axis_corner, axis_range, min_diff_from_current=0.025):
    """ Determine a random drop coordinate axis value that isn't on top of the current object location

    min_diff_from_current: how far must the new random coordinate be from the current one
    """
    x_random = grasp_pose.p[axis_idx]
    while np.abs(x_random - grasp_pose.p[axis_idx]) < min_diff_from_current:
        x_random = axis_corner + np.random.random() * axis_range
    return x_random

def initialize_collection_objects(args, observe, collector, stack_task):
    rate = rospy.Rate(args.rate)
    home = GetRandomHome()
    # home = GetHome()
    move_to_pose = GetMoveToPose()
    open_gripper = GetOpenGripperService()
    close_gripper = GetCloseGripperService()
    update = GetUpdate(observe, collector) # uses collector because it listens for js

    # Set the function which sends the robot to home and
    # gets an update of all the object poses.
    # set fn to call after each action
    stack_task.setUpdate(update)
    return home, rate, move_to_pose, close_gripper, open_gripper

def main():
    args = getArgs()
    faulthandler.enable()

    if args.launch:
        launch_main(argv=['roslaunch', 'ctp_integration', 'bringup.launch'],
                    real_args=args,
                    fn_to_call=collect_data)
    else:
        # assume ros was already running and start collecting data
        return collect_data(args)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        pass
