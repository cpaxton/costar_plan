from abstract import AbstractAgent

import numpy as np
import PyKDL as kdl

from costar_models import MakeModel
from costar_task_plan.abstract import *
from costar_task_plan.robotics.core import CostarAction

class NeuralNetworkPlannerAgent(AbstractAgent):
    '''
    Simple feed forward agent. Loads everything based on model definition and
    executes in the environment.

    This does not perform any checks for what kind of model you create -- the
    only thing is that it will create the model and use model.predict() to get
    two outputs.
    
    One output is expected to be the arm position command, the other the
    gripper position command.
    '''

    name = "random"

    approved_models = ["predictor"]

    def __init__(self, env, model, *args, **kwargs):
        super(NeuralNetworkPlannerAgent, self).__init__(env=env, *args, **kwargs)

        if model not in self.approved_models:
            raise RuntimeError('Model type %s not approved.'%model)

        self.model = MakeModel(taskdef=self.env.task, model=model, **kwargs)
        self.model.load(self.env.world)

    def fit(self, num_iter):

        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            features = self.env.reset()

            # Make sure the model's state is clear before moving on to the next
            # step.
            self.model.reset()

            policy = None
            j = 0
            while not self._break:
                if policy is None or j % 10 == 0:
                    arm_goal, gripper_goal= self.model.predict(self.env.world)
                    policy = self.SimpleMotionPolicy(
                            arm_goal[:3],
                            arm_goal[3:],
                            gripper_goal)
                control = policy.evaluate(
                        self.env.world,
                        self.env.world.actors[0].state,
                        self.env.world.actors[0],)
                features, reward, done, info = self.env.step(control)
                self._addToDataset(self.env.world,
                        control,
                        features,
                        reward,
                        done,
                        i,
                        self.model.name)
                if done:
                    break
                j += 1

            if self._break:
                return
        
    class SimpleMotionPolicy(AbstractPolicy):

        def __init__(self, pos, rot, gripper, goal=None, cartesian_vel=0.35, angular_vel=0.5):
            self.pos = pos
            self.rot = [r * np.pi for r in rot]
            self.goal = goal
            self.gripper = gripper
            self.cartesian_vel = cartesian_vel
            self.angular_vel = angular_vel

            print (self.pos, self.rot)

            pg = kdl.Vector(*self.pos)
            Rg = kdl.Rotation.RPY(*self.rot)
            self.T = kdl.Frame(Rg, pg)

        def evaluate(self, world, state, actor):
            '''
            Compute IK to goal pose for actor.
            Goal pose is computed based on the position of a goal object, if one
            has been specified; otherwise we assume the goal has been specified
            in global coordinates.
            '''

            if self.goal is not None:
                # Get position of the object we are grasping. Since we compute a
                # KDL transform whenever we update the world's state, we can use
                # that for computing positions and stuff like that.
                obj = world.getObject(self.goal)
                T = obj.state.T * self.T
            else:
                # We can just use the cached position, since this is a known world
                # position and not something special.
                T = self.T

            if actor.robot.grasp_idx is None:
                raise RuntimeError(
                    'Did you properly set up the robot URDF to specify grasp frame?')

            # =====================================================================
            # Compute transformation from current to goal frame
            T_r_goal = state.T.Inverse() * T

            # Interpolate in position alone
            dist = T_r_goal.p.Norm()
            step = min(self.cartesian_vel*world.dt, dist)
            p = T_r_goal.p / dist * step

            # Interpolate in rotation alone
            angle, axis = T_r_goal.M.GetRotAngle()

            angle = min(self.angular_vel*world.dt, angle)
            R = kdl.Rotation.Rot(axis, angle)
            T_step = state.T * kdl.Frame(R, p)

            # =====================================================================
            # Compute motion goak and send
            q_goal = actor.robot.ik(T_step, state.arm)
            print (q_goal, state.arm, self.gripper)
            if q_goal is None:
                error = True
            else:
                error = False
            # print q_goal, state.arm, state.arm_v
            return CostarAction(q=q_goal,
                    gripper_cmd=self.gripper,
                    error=error)
