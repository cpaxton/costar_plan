
from tasks import *
from robots import *


def GetAvailableTasks():
    return [
            "blocks", # move 1 block from A to B
            "stack1", # stack 2 blocks
            "drl_blocks", # Kapil's DRL environment
            "clutter", # spawn a bunch of random objects
            "sorting",
            "tower",
            "oranges", # NOT YET IMPLEMENTED
            "obstacles", # Albert: spawn a bunch of random objects on top of
                         # blocks
            "trays",
            "obstructions",
            "sorting",
            "sorting2",
            "explore"]


def GetAvailableRobots():
    return ["jaco", "ur5_2_finger", "ur5", "ur5_robotiq", "turtlebot", "husky"]


def GetAvailableAlgorithms():
    return [None, "ddpg", "cdqn"]


def GetTaskDefinition(task, robot, features, *args, **kwargs):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
            'blocks': lambda: BlocksTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'stack1': lambda: BlocksTaskDefinition(1, robot=robot, features=features, *args, **kwargs),
            'trays': lambda: BlocksTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'obstacles': lambda: ObstaclesTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'obstructions': lambda: ObstructionsTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'drl_blocks': lambda: DRLBlocksTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'tower': lambda: BlocksTaskDefinition(None, robot, features=features, *args, **kwargs),
            'clutter': lambda: ClutterTaskDefinition(robot, features=features, *args, **kwargs),
            'sorting': lambda: SortingTaskDefinition(robot, features=features, *args, **kwargs),
            'sorting2': lambda: Sorting2TaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'oranges': lambda: OrangesTaskDefinition(robot, features=features, *args, **kwargs),
            'explore': lambda: ExploreTaskDefinition(robot, features=features, *args, **kwargs),
        }[task.lower()]()
    except KeyError, e:
        raise NotImplementedError('Task %s not implemented!' % task)


def GetRobotInterface(robot, *args, **kwargs):
    '''
    Returns a simple interface representing a particular robot in the
    simulation.
    '''
    try:
        return {
            'ur5_2_finger': lambda: Ur5RobotiqInterface(*args, **kwargs),
            'ur5_robotiq': lambda: Ur5RobotiqInterface(*args, **kwargs),
            'ur5': lambda: Ur5RobotiqInterface(*args, **kwargs),
            'jaco': lambda: JacoRobotiqInterface(*args, **kwargs),
            'turtlebot': lambda: TurtlebotInterface(*args, **kwargs),
            #'iiwa_3_finger': IiwaRobotiq3FingerInterface(*args, **kwargs),
            #'iiwa': IiwaRobotiq3FingerInterface(*args, **kwargs),
        }[robot.lower()]()
    except KeyError, e:
        raise NotImplementedError('Robot %s not implemented!' % robot)
