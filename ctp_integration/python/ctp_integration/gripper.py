from __future__ import print_function

from costar_task_plan.abstract import AbstractOption
from costar_task_plan.abstract import AbstractPolicy

class GripperOption(AbstractOption):
    '''
    This option calls out to the current segmenter and waits until termination.
    '''

    def __init__(self, name="gripper"):
        super(GripperOption, self).__init__(name, True)

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
        raise Exception('option.makePolicy not implemented!')


class CostarGripperPolicy(AbstractPolicy):

    '''
    This simple policy just looks at robot internals to send the appropriate
    "open gripper" command.
    '''

    def evaluate(self, world, state, actor):
        return CostarRobotAction(gripper_cmd=state.robot.gripperOpenCommand())


