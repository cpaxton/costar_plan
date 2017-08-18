
from costar_task_plan.abstract import *

import pybullet as pb
import PyKDL as kdl


class SimulationWorld(AbstractWorld):

    def __init__(self, dt=0.1, simulation_step=0.0001, task_name="", cameras=[], *args, **kwargs):
        super(SimulationWorld, self).__init__(NullReward(), *args, **kwargs)
        self.task_name = task_name
        self.cameras = cameras

        self.dt = dt
        self.simulation_step = simulation_step
        self.num_steps = int(dt / simulation_step)


        # stores object handles and names
        self.class_by_object = {}
        self.object_by_class = {}
        self.id_by_object = {}

    def addObject(self, obj_name, obj_class, handle, state):
        '''
        Wraps add actor function for objects. Make sure they have the right
        policy and are added so we can easily look them up later on.
        '''

        obj_id = self.addActor(SimulationObjectActor(
            name=obj_name,
            handle=handle,
            dynamics=SimulationDynamics(self),
            policy=NullPolicy(),
            state=state))
        self.class_by_object[obj_name] = obj_class
        self.id_by_object[obj_name] = obj_id
        if obj_class not in self.object_by_class:
            self.object_by_class[obj_class] = [obj_name]
        else:
            self.object_by_class[obj_class].append(obj_name)

        return obj_id

    def getObjectId(self, obj_name):
        return self.id_by_object[obj_name]

    def getObjects(self):
        '''
        Return information about specific objects in the world. This should tell us
        for some semantic identifier which entities in the world correspond to that.
        As an example:
            {
                "goal": ["goal1", "goal2"]
            }
        Would be a reasonable response, saying that there are two goals called
        goal1 and goal2.
        '''
        return self.object_by_class

    def getObject(self, name):
        '''
        Look up the particular actor we are interested in for this world.

        Params:
        -------
        name: name of the actor (string)
        '''
        idx = self.id_by_object[name]
        return self.actors[idx]

    def _update_environment(self):
        '''
        Step the simulation forward after all actors have given their comments
        to associated simulated robots. Then update all actors' states.
        '''

        # Loop through the given number of steps
        for i in xrange(self.num_steps):
            pb.stepSimulation()

        # Update the states of all actors.
        for actor in self.actors:
            actor.state = actor.getState()
            actor.state.t = self.ticks * self.dt

    def zeroAction(self, actor=0):
        return SimulationRobotAction()

    def _reset(self):
        # Update the states of all actors.
        for actor in self.actors:
            actor.state = actor.getState()
            actor.state.t = self.ticks * self.dt


class SimulationDynamics(AbstractDynamics):

    '''
    Send robot's command over to the actor in the current simulation.
    This assumes the world is in the correct configuration, as represented
    by "state."
    '''

    def __call__(self, state, action, dt):
        if state.robot is not None:
            state.robot.command(action)

class SimulationObjectState(AbstractState):

    '''
    Represents state and position of an arbitrary rigid object, and any
    associated predicates.
    '''

    def __init__(self, handle,
                 base_pos=(0, 0, 0),
                 base_rot=(0, 0, 0, 1),
                 t=0.):
        self.predicates = []
        self.base_pos = base_pos
        self.base_rot = base_rot
        p = kdl.Vector(*base_pos)
        R = kdl.Rotation.Quaternion(*base_rot)
        self.T = kdl.Frame(R, p)
        self.t = t
        self.robot = None


class SimulationObjectActor(AbstractActor):

    '''
    Not currently any different from the default actor.
    '''

    def __init__(self, name, handle, *args, **kwargs):
        super(SimulationObjectActor, self).__init__(*args, **kwargs)
        self.name = name
        self.handle = handle
        self.getState = lambda: GetObjectState(self.handle)


class SimulationRobotState(AbstractState):

    '''
    Includes full state necessary for this robot, including gripper, base, and
    arm position.
    '''

    def __init__(self, robot,
                 base_pos=(0, 0, 0),
                 base_rot=(0, 0, 0, 1),
                 arm=[],
                 arm_v=[],
                 gripper=0.,
                 T=None,
                 t=0.,):

        self.predicates = []
        self.arm = arm
        self.arm_v = arm_v
        self.gripper = gripper
        self.base_pos = base_pos
        self.base_rot = base_rot
        self.robot = robot
        self.T = T
        self.t = t

    def toParams(self, action):
        '''
        Wrapper for robot toParams()
        '''
        return self.robot.toParams(action)


class SimulationRobotAction(AbstractAction):

    '''
    Includes the command that gets sent to robot.act(). This pretty much just
    holds the tuple for arm_cmd, gripper_cmd, etc.
    '''

    def __init__(self, arm_cmd=None, gripper_cmd=None, base_cmd=None, code=None):
        self.arm_cmd = arm_cmd
        self.gripper_cmd = gripper_cmd
        self.base_cmd = base_cmd

        # This is used to track which high-level action is being executed at
        # any given time.
        self.code = code

    def getDescription(cls):
        return "arm_cmd", "gripper_cmd", "base_cmd"

class SimulationRobotActor(AbstractActor):

    def __init__(self, robot, *args, **kwargs):
        super(SimulationRobotActor, self).__init__(*args, **kwargs)
        self.robot = robot
        self.getState = self.robot.getState

class NullPolicy(AbstractPolicy):

    def evaluate(self, world, state, actor=None):
        return SimulationRobotAction()

# =============================================================================
# Helper Fucntions


def GetObjectState(handle):
    '''
    Look up the handle and get its base position and eventually other
    information, if we find that necessary.
    '''
    pos, rot = pb.getBasePositionAndOrientation(handle)
    return SimulationObjectState(handle,
                                 base_pos=pos,
                                 base_rot=rot)
