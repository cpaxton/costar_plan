from __future__ import print_function

from collections import namedtuple
from learning_planning_msgs.msg import TaskInfo
from learning_planning_msgs.msg import DemonstrationInfo

from .task import Task


NAME_STYLE_UNIQUE = 0
NAME_STYLE_SAME = 1

'''
ObjectInfo
Describe world observations of objects
'''
ObjectInfo = namedtuple(
        'ObjectInfo',
        field_names=['pose','obj_class', 'id', 'name'],
        verbose=False)

class ActionInfo(object):
    '''
    ActionInfo
    Store information about what action was being performed and by which arm. 
    This information must be provided in order to compute what type of skill
    representation we will use, and to actually create the task graph.
    '''

    ARM_LEFT = 0
    ARM_RIGHT = 1
    ARM_BOTH = 2
    ARM_POSES = [1, 1, 2]
    def __init__(self, arm, name, object_acted_on, object_in_hand, pose,
            gripper_state):
        if isinstance(arm, str):
            if arm.lower() == "left":
                arm = ARM_LEFT
            elif arm.lower() == "right":
                arm = ARM_RIGHT
            elif arm.lower() == "both":
                arm = ARM_BOTH
            else:
                raise RuntimeError('activity parse failed: arm %s not understood'%arm)
        elif not arm in [self.ARM_LEFT, self.ARM_RIGHT, self.ARM_BOTH]:
            raise RuntimeError('options are limited to LEFT, RIGHT, and BOTH.')
        self.arm = arm
        self.base_name = name
        self.name = None
        self.object_acted_on = object_acted_on
        self.object_in_hand = object_in_hand
        self.pose = pose
        self.gripper_state = gripper_state


class TaskParser(object):

    class Example:
        '''
        Lightweight class to track observed trajectories for one hand or
        another.
        '''
        def __init__(self):
            self.reset()

        def reset(self, *args):
            self.traj = []
            self.data = {}
            self.obj_classes = list(args)

        def addData(self, obj_class, pose):
            if obj_class not in self.data:
                self.data[obj_class] = []
            self.data[obj_class].append(pose)

        def addPoint(self, t, pose, gripper):
            self.traj.append((t,pose,gripper))

        def resetEndpointRelative(self, pose):
            self.reset('endpoint')
            self.data['endpoint'] = []

        def addEndpointRelative(self, pose):
            if len(self.data['endpoint']) == 0:
                self.addData('endpoint', pose)
            else:
                self.addData('endpoint', self.data['endpoint'][0])

    def __init__(self,
            action_naming_style=NAME_STYLE_UNIQUE,
            min_action_length=5, # ignore anything below this length
            unknown_apply_before=4,
            configs=[],
            *args, **kwargs):
        '''
        Create a task parser. This lets you load one demonstration in at a
        time, and will parse all the necessary information to create a 
        particular object.
        '''
        self.transition_counts = {}
        self.transitions = {}
        self.object_classes = set()
        self.objects_by_class = {}
        self.classes_by_object = {}
        self.object_parent_classes = {}
        self.parent_action = {}
        self.action_naming_style = action_naming_style
        self.idle_tags = []
        self.unknown_tags = []
        self.resetDemonstration()
        self.configs = configs
        self.num_arms = len(self.configs)
        self.unknown_apply_before = unknown_apply_before
        self.min_action_length = min_action_length
        self._done = "DONE"

        # Store alias information for converting unknown activities into
        # others partway through a parse
        self.alias = {}

        self.trajectories = {}
        self.trajectory_data = {}
        self.trajectory_features = {}

        self.parent_trajectories = {}

        self.sequence_ends = {}

    def setDoneName(self, done="DONE"):
        self._done = done

    def addAlias(self, old_name, new_name):
        '''
        This adds a mapping from old_name to a new action name. Note that 
        these are all going to be processed in a single "pass", so if something
        is listed twice you will get weird behavior.

        Parameters:
        -----------
        old_name: previous name in dataset that should be removed
        new_name: newer name in dataset
        '''
        self.alias[old_name] = new_name

    def addObjectClass(self, object_class):
        self.object_classes.add(object_class)

    def addObject(self, obj, obj_class):
        self.addObjectClass(obj_class)
        if not obj_class in self.objects_by_class:
            self.objects_by_class[obj_class] = set()
        self.objects_by_class[obj_class].add(obj)
        if (obj in self.classes_by_object and not obj_class == self.classes_by_object[obj]):
            raise RuntimeError("object %s has inconsistent class in data: %s vs %s"%(obj, obj_class, self.classes_by_object[obj]))
        self.classes_by_object[obj] = obj_class

    def _getActionName(self, action):
        '''
        Get a long-form name of a particular action
        '''
        if action.base_name in self.unknown_tags:
            return None
        elif self.action_naming_style == NAME_STYLE_SAME:
            return action.base_name
        elif self.action_naming_style == NAME_STYLE_UNIQUE:
            if action.arm == ActionInfo.ARM_LEFT:
                arm = "left"
            elif action.arm == ActionInfo.ARM_RIGHT:
                arm = "right"
            name = action.base_name
            if action.object_in_hand is not None:
                name += "_with_%s"%(self.classes_by_object[action.object_in_hand])
            if action.object_acted_on is not None:
                name += "_to_%s"%(self.classes_by_object[action.object_acted_on])
            return name

    def _getObjectClassParent(self, obj):
        '''
        Return the parent class of a particular object if it has one. This is
        used when computing the "parent" class of a particular action.
        '''
        if obj in self.object_parent_classes:
            return self.object_parent_classes[obj]
        else:
            return obj

    def _getParentActionName(self, action):
        '''
        Get the name of the "parent" action which shares all its parameters
        '''
        if action.base_name in self.unknown_tags:
            return None
        elif self.action_naming_style == NAME_STYLE_SAME:
            return action.base_name
        elif self.action_naming_style == NAME_STYLE_UNIQUE:
            if action.arm == ActionInfo.ARM_LEFT:
                arm = "left"
            elif action.arm == ActionInfo.ARM_RIGHT:
                arm = "right"
            name = action.base_name
            if action.object_in_hand is not None:
                name += "_with_%s" % (self._getObjectClassParent(
                    self.classes_by_object[action.object_in_hand]))
            if action.object_acted_on is not None:
                name += "_to_%s" % (self._getObjectClassParent(
                    self.classes_by_object[action.object_acted_on]))
            return name

    def resetDemonstration(self):
        self.data = []

    def addIdle(self, *args):
        self.idle_tags += list(args)

    def addObjectClassParent(self, obj_class, parent_class):
        '''
        Used as a way of indicating that multiple objects all belong to the
        same parent object class. So for example if we see many different
        cubes, this should be fine.
        '''
        if obj_class in self.object_parent_classes:
            raise RuntimeError('Object class %s already has a'
                               'parent class' % obj_class)
        else:
            self.object_parent_classes[obj_class] = parent_class

    def addUnknown(self, *args):
        self.unknown_tags += list(args)

    def _addTransition(self, from_action, to_action):
        '''
        Helper function to populate helper data structures storing transition
        information.

        Parameters:
        ----------
        from_action: action/activity that we just executed/finished
        to_action: action/activity that we just began
        '''
        if not to_action in self.transitions:
            self.transitions[to_action] = set()
        self.transitions[to_action].add(from_action)
        key = (from_action, to_action)
        if not key in self.transition_counts:
            self.transition_counts[key] = 0
        self.transition_counts[key] += 1

    def addExample(self, t, objs, actions, seq):
        '''
        Call this to parse a single time step. This assumes you have properly
        called resetDemonstration() above.
        '''
        for i, a in enumerate(actions):
            if a.base_name is None:
                raise RuntimeError('unnamed action')
            if a.name is not None:
                raise RuntimeError('instance name should not be set')
            if (a.base_name in self.idle_tags):
                a.object_acted_on = None
                a.object_in_gripper = None
            actions[i] = a
        self.data.append((t, objs, actions, seq))

    def process(self):
        '''
        This function runs through the whole data set and processes it.
        '''
        prev_t = [None] * self.num_arms
        prev = [None] * self.num_arms
        action_start_t = [None] * self.num_arms
        
        # Preprocess to remove "Unknown" symbols
        ACTION_IDX = 2
        for i, (t, objs, actions, seq) in enumerate(self.data):
            for j, a in enumerate(actions):
                if a.base_name in self.alias:
                    a.base_name = self.alias[a.base_name]
                if a.base_name in self.unknown_tags:
                    reset_name = False
                    for k in range(self.unknown_apply_before):
                        if i+k < len(self.data):
                            next_name = self.data[i+k][ACTION_IDX][j].base_name
                            if next_name not in self.unknown_tags:
                                a.base_name = next_name
                                reset_name = True
                                break
                    #if not reset_name:
                    #    a.base_name = prev[j]
                else:
                    prev[j] = a.base_name
                if a.base_name is None or a.base_name in self.unknown_tags:
                    print('WARNING: was not able to preprocess unknown tag at %d'%i)

        # Reset previous tags again
        prev = [None] * self.num_arms
        prev_added = [None] * self.num_arms
        counts = [0] * self.num_arms

        examples = []
        for _ in range(self.num_arms):
            examples.append(TaskParser.Example())

        prev_seq = None

        # Loop over all time steps
        for i, (t, objs, actions, seq) in enumerate(self.data):

            # Reset everything when parsing from multiple bags
            if seq is not prev_seq:
                prev = [None] * self.num_arms
                prev_added = [None] * self.num_arms
                counts = [0] * self.num_arms
                for ex in examples:
                    ex.reset()
            prev_seq = seq

            for name, obj in objs.items():
                self.addObject(obj.name, obj.obj_class)

            if seq not in self.sequence_ends:
                self.sequence_ends[seq] = [
                        self._getActionName(a)
                        for a
                        in actions]

            # In order - all actions specified
            for j, action in enumerate(actions):

                if action.object_acted_on is not None:
                    obj_class = self.classes_by_object[action.object_acted_on]
                else:
                    obj_class = None

                name = self._getActionName(action)
                parent_name = self._getParentActionName(action)
                if name not in self.parent_action:
                    self.parent_action[name] = parent_name
                if name is None:
                    # "None" is only acceptable as the root of a tree -- not
                    # anywhere as an action node. Whenever we see it, we are
                    # observing junk data and should ignore it.
                    continue
                elif not prev[j] == name:
                    # Finish up by adding this trajectory to the data set
                    if (len(examples[j].traj) > self.min_action_length
                            and not prev_added[j] == name):
                        self.addTrajectory(
                                prev[j],
                                examples[j].traj,
                                examples[j].data,
                                ["time"] + examples[j].obj_classes)
                        self._addTransition(prev_added[j], name)
                        prev_added[j] = name

                        # compute final action
                        if (self.sequence_ends[seq][j] in self.idle_tags
                            or name not in self.idle_tags):
                            self.sequence_ends[seq][j] = name
                    else:
                        print("WARNING: trajectory %s of length %d was too short"%(prev[j],len(examples[j].traj)))

                    # Update feature list
                    # TODO: enable this if we need to
                    #if action.object_in_hand is not None:
                    #    objs.append(action.object_in_hand)
                    action_start_t[j] = t
                    if action.object_acted_on is not None:
                        examples[j].reset(obj_class)
                    else:
                        examples[j].resetEndpointRelative(action.pose)
                elif prev_t[j] is not None:
                    # Trigger a sanity check to make sure we do not have any weird jumps in our file.
                    dt = abs(prev_t[j] - t)
                    if dt > 1:
                        print("WARNING: large time jump from %f to %f; did you reset between demonstrations?"%(prev_t[j],t))

                if obj_class is not None:
                    examples[j].addData(obj_class, objs[action.object_acted_on].pose)
                else:
                    # add position of EE at beginning of the action
                    examples[j].addEndpointRelative(action.pose)
                examples[j].addPoint(t - action_start_t[j], action.pose, action.gripper_state)

                prev[j] = name
                prev_t[j] = t
        for seq, actions in self.sequence_ends.items():
            for action in actions:
                self._addTransition(action, self._done)

    def addTrajectory(self, name, traj, data, objs):
        '''
        Helper function that adds a list of end effector poses, object
        positions, and associated relevant feature objects to the current
        collected data set.

        Parameters:
        -----------
        name: name of the action
        traj: end effector pose trajectory
        data: dict of object poses relevant to this action
        objs: list of object/feature names for this action
        '''
        parent_name = self.parent_action[name]
        skill_name = name
        if not name in self.trajectories:
            self.trajectories[skill_name] = []
            self.trajectory_data[skill_name] = []
            self.trajectory_features[skill_name] = objs

        self.trajectories[skill_name].append(traj)
        self.trajectory_data[skill_name].append(data)

    def collectTrajectories(self):
        trajs = {}
        data = {}
        features = {}
        params = {}
        print("Collecting trajectories for abstraction training.")
        for name, traj in self.trajectories.items():
            parent_name = self.parent_action[name]
            if parent_name not in features:
                parent_features = []
                print(">", name, "has features:", self.trajectory_features[name])
                for f in self.trajectory_features[name]:
                    if f in self.object_parent_classes:
                        pf = self.object_parent_classes[f]
                        print(f, "-->", pf)
                        parent_features.append(pf)
                    else:
                        parent_features.append(f)
                print("Parent features:", parent_features)
                features[parent_name] = parent_features

            # Instantiate the new set up for this model
            if parent_name not in trajs:
                trajs[parent_name] = []
                data[parent_name] = []
                params[parent_name] = []
            trajs[parent_name] += traj
            instance_data = self.trajectory_data[name]
            num_params = len(instance_data)
            data[parent_name] += instance_data
            params[parent_name] += num_params * [(name, self.trajectory_features[name])]
        return trajs, data, features, params

    def _getArgs(self, action_name):
        raise NotImplementedError('Create arguments for graph node')
    
    def train(self):
        '''
        Implement application-specific logic for training here.
        '''
        raise NotImplementedError('train() not implemented for this application')

    def makeTask(self):
        self.train()
        task = Task()
        for node, parents in self.transitions.items():
            if node == self._done:
                special_node = True
            else:
                generic_node = self.parent_action[node]
                special_node = False
            if not special_node:
                if not node in self.trajectory_features:
                    continue
                args = self._getArgs(node)
            counts = [self.transition_counts[(parent,node)] for parent in parents]
            task.add(node, list(parents), args, counts)
        return task
