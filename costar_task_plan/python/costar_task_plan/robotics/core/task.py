from __future__ import print_function

from costar_task_plan.abstract.task_parser import TaskParser
from costar_task_plan.abstract.task_parser import ObjectInfo
from costar_task_plan.abstract.task_parser import ActionInfo
from .lfd import LfD
from .dmp_option import DmpOption
from .dmp_policy import CartesianDmpPolicy

from learning_planning_msgs.msg import HandInfo

import numpy as np
import tf_conversions.posemath as pm

import rosbag

class RosTaskParser(TaskParser):

    def __init__(self,
            filename=None,
            demo_topic="demonstration",
            alias_topic="alias",
            from_unity=True,
            *args, **kwargs):
        '''
        Create the ROS version of the task parser -- represents a given task
        as a set of DMPs of various sorts.

        There are a couple frames that we ignore by default: the "NONE" object,
        the "surveillance camera" that represents a view of the scene, and
        possibly others in the future.

        Parameters:
        -----------
        filename: name of the bag file to load when parsing messages
        demo_topic: topic on which messages were published
        alias_topic: topic on which messages renaming granular actions were
                     published.
        from_unity: true if messages were published with poses specified by
                    the Unity VR system, which uses left-hand y-up notation for
                    rotations and poses.
        '''
        super(RosTaskParser, self).__init__(*args,**kwargs)
        self.ignore = ["NONE","none","surveillance_camera"]
        self.addIdle("IdleMotion")
        self.addUnknown("UnknownActivity")
        self.demo_topic = demo_topic
        self.alias_topic = alias_topic
        self.from_unity = from_unity
        self.lfd = LfD(self.configs[0])
        if filename is not None:
            self.loadFromFile(filename)

    def loadFromFile(self, filename):
        filenames = filename.split(',')
        for i, f in enumerate(filenames):
            bag = rosbag.Bag(f, 'r')
            self.loadFromBag(bag, seq=i)

    def loadFromBag(self, bag, seq=0):
        '''
        Parse an individual bag.
        '''

        # call whenever adding a new rosbag or data source for a particular
        # trial.
        for topic, msg, _ in bag:
            # We do not trust the timestamps associated with the bag since
            # these may be written separately from when the data was actually
            # collected.
            if topic == self.demo_topic:
                # Demo topic: read object information and add
                t = self._getTime(msg)
                objs = self._getObjects(msg)
                left = self._getHand(msg.left, ActionInfo.ARM_LEFT)
                right = self._getHand(msg.right, ActionInfo.ARM_RIGHT)
                self.addExample(t, objs, [left, right], seq)
            elif topic == self.alias_topic:
                self.addAlias(msg.old_name, msg.new_name)

    def _getArgs(self, skill_name):
        '''
        Get the args for a DMP option for creating a task graph
        
        Parameters:
        -----------
        skill_name: name of the action/skill to insert into graph
        '''

        # NOTE: hard coded for now; take the last skill. This should be either
        # a real object or the trajectory endpoint.
        obj = self.trajectory_features[skill_name][-1]

        # Create a function that will make a Cartesian skill instance.
        dmp_maker = lambda goal: DmpOption(
                goal_object=goal,
                config=self.configs[0],
                skill_name=skill_name,
                feature_model=self.lfd.getSkillModel(skill_name),
                kinematics=self.lfd.kdl_kin,
                traj_dist=self.lfd.getParamDistribution(skill_name),
                policy_type=CartesianDmpPolicy)

        return {
                "constructor": dmp_maker,
                "args": [obj],
                "remap": {obj: "goal"},
                }

    def _getHand(self, msg, id):
        '''
        Get the robot hand and create all appropiate fields here
        '''
        action_name = msg.activity
        if action_name is None:
            raise RuntimeError('unnamed action')
        obj_acted_on = msg.object_acted_on
        obj_in_gripper = msg.object_in_hand
        if (obj_acted_on == HandInfo.NO_OBJECT
                or len(obj_acted_on) == 0
                or action_name in self.idle_tags
                or obj_acted_on in self.ignore):
            obj_acted_on = None
        if (obj_in_gripper == HandInfo.NO_OBJECT
                or len(obj_in_gripper) == 0
                or action_name in self.idle_tags
                or obj_in_gripper in self.ignore):
            obj_in_gripper = None
        pose = self._makePose(msg.pose)
        gripper_state = msg.gripper_state
        return ActionInfo(id, action_name, obj_acted_on, obj_in_gripper, pose,
                gripper_state)

    def _getTime(self, demo):
        '''
        Compute time in seconds from header of a ROS message.

        Parameters:
        -----------
        demo: ros message of a demonstration

        Returns:
        --------
        t: float time in seconds
        '''
        t = demo.header.stamp
        t = t.to_sec()
        return t

    def _getObjects(self, demo):
        '''
        Read in the demonstration and update object knowledge for this specific
        message instance. This mostly converts into a standard message format.

        Parameters:
        -----------
        demo: a DemonstrationInfo message
        '''
        objs = {}
        for obj in demo.object:
            
            obj.name = obj.name.rstrip()
            obj.object_class = obj.object_class.rstrip()
            if obj.object_class in self.ignore:
                continue

            pose = self._makePose(obj.pose)
            objs[obj.name] = ObjectInfo(pose=pose,
                        obj_class=obj.object_class,
                        id=obj.id,
                        name=obj.name)
        return objs

    def _makePose(self, pose_msg):
        '''
        Read in a pose message and convert it to a standard data format. We
        use KDL Frames for fast operations down the line.

        If we previously set the `from_unity` flag to true, then we need to
        adjust our frames so that they make sense as well.

        Parameters:
        -----------
        pose_msg: a ROS pose message

        Returns:
        --------
        a KDL pose in right hand, z-up notation
        '''
        pose = pm.fromMsg(pose_msg)
        if self.from_unity:
            H = np.array([
                [0,0,1,0],
                [-1,0,0,0],
                [0,1,0,0],
                [0,0,0,1]])
            x = pm.toMatrix(pose)
            x = H.dot(x)
            pose = pm.fromMatrix(x)

        return pose

    def train(self):
        '''
        Create the task by:
          - calling self.lfd.train() with the appropriate data
          - constructing a task model based on transitions
          - populating those transitions with the _getDmpArgs function

        Resulting task plan can be compiled and visualized as normal.
        '''
        print("===============================")
        print("-------------------------------")
        print('Observed transitions:')
        print("-------------------------------")
        for key, value in self.transitions.items():
            print (key, "has parents", list(value))
        print("-------------------------------")
        print('Observed transition counts:')
        print("-------------------------------")
        for (a, b), count in self.transition_counts.items():
            print (a, "-->", b, "seen", count, "times")
        print("-------------------------------")
        print("Number of example trajectories:")
        print("-------------------------------")
        for key, traj in self.trajectories.items():
            print("%s:"%key, len(traj), "with", self.trajectory_features[key])
        trajectories, data, features, params = self.collectTrajectories()
        print("-------------------------------")
        print("Number of parent trajectories:")
        print("-------------------------------")
        for key, traj in trajectories.items():
            print("%s:"%key, len(traj), "with", features[key])
        print("===============================")
        self.lfd.train(trajectories, data, features, params)

    def debug(self, world):
        self.lfd.debug(world)
