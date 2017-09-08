from abstract import AbstractTaskDefinition
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.option import *
from costar_task_plan.simulation.reward import *
from costar_task_plan.simulation.condition import *

import pybullet as pb
from costar_task_plan.simulation.camera import Camera



class ExploreTaskDefinition(AbstractTaskDefinition):

    '''
    Robot must move to a goal position within a house, despite clutter and 
    closed doorways.
    '''

    joint_positions = [0.0, -1.33, -1.80, 0.0, 1.50, 1.60]

    def __init__(self, *args, **kwargs):
        '''
        Arguments define how to create a house and populate it with rabdom
        objects.
        '''
        super(ExploreTaskDefinition, self).__init__(*args, **kwargs)
        self.addCamera(
            #Camera("right", [-0.5, 0., 0.15], distance=0.7, roll=0.0,
        #Camera("top", [-0.4, 0., 0.25], distance=0.7, roll=0.0,
        Camera("top", [-0.0, 0., 1.00], distance=0.7, roll=0.0,
                image_width=64,
                image_height=64,
                pitch=-90,
                yaw=0,
                fov=40))


    def _setup(self):
        '''
        Create task by adding objects to the scene.
        '''

    def _setupRobot(self, handle):
        '''
        Robot must be mobile.
        '''
        if not self.robot.mobile():
            raise RuntimeError('Exploration task does not even make sense' \
                               + 'without a mobile robot.')
        self.robot.place([0,0,0],[0,0,0,1],self.joint_positions)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)# -*- coding: utf-8 -*-
        
    def _makeTask(self):
        task = Task()
        return task
    
    def getName(self):
        return "explore"
    
    def reset(self):
        pass        

