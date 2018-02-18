from __future__ import print_function

from costar_task_plan.abstract import AbstractOption

from .arm_policies import CostarArmMotionPolicy

class MotionOption(AbstractOption):
    '''
    This option calls out to the current segmenter and waits until termination.
    '''

    def __init__(self, world, goal, pose, pose_tolerance=(1e-4,1e-4)):
        '''
        Create a goal-directed motion.
        '''
        super(MotionOption, self).__init__("motion", True)
        self.goal = goal # move to this particular object if it exists
        if goal is not None:
            # Look up the specific ID of the object in the world
            self.goal_id = world.getObjectId(goal)
        else:
            # Use negative numbers to specify missing goal frame
            self.goal_id = -1
        if pose is not None:
            self.position, self.rotation = pose
            self.position_tolerance, self.rotation_tolerance = pose_tolerance
        else:
            raise RuntimeError('Must specify pose.')

    def makeWorld(self, *args, **kwargs):
        '''
        Create a world for testing this specific option
        '''
        raise Exception('cannot make training world for this option')

    def makePolicy(self, world, *args, **kwargs):
        '''
        Get policy for performing this option.
        Get the gating condition for a specific option.
        - execution should continue until such time as this condition is true.
        '''

        policy = CostarArmMotionPolicy(goal, pose)
        condition = policy.running
        return policy, condition

