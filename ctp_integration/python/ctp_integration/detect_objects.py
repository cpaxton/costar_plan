from __future__ import print_function

from costar_task_plan.abstract import AbstractOption

class DetectObjectsOption(AbstractOption):
    '''
    This option calls out to the current segmenter and waits until termination.
    '''

    def __init__(self, name="detect_objects"):
        super(DetectObjectsOption, self).__init__(name, True)

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


