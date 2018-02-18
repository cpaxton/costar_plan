from __future__ import print_function

from costar_task_plan.abstract import AbstractPolicy

import PyKDL as kdl
import tf_conversions.posemath as pm

class CostarArmMotionPolicy(AbstractPolicy):

    def __init__(self, pos, rot, goal=None, cartesian_vel=0.25, angular_vel=0.65):
        self.pos = pos
        self.rot = rot
        self.goal = goal
        self.cartesian_vel = cartesian_vel
        self.angular_vel = angular_vel

        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
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
        # Issue computing inverse kinematics
        #compos, comorn, ifpos, iforn, lwpos, lworn = pb.getLinkState(actor.robot.handle, actor.robot.grasp_idx)
        # print lwpos, lworn
        #q = pb.calculateInverseKinematics(actor.robot.handle,
        #                                  actor.robot.grasp_idx,
        #                                  targetPosition=list(T_step.p),
        #                                  targetOrientation=list(T_step.M.GetQuaternion()))
        # from tf_conversions import posemath as pm
        # mat = pm.toMatrix(T)
        # print mat
        # print actor.robot.kinematics.forward(state.arm)

        # =====================================================================
        # Compute motion goak and send
        q_goal = actor.robot.ik(T_step, state.arm)
        #print q_goal, state.arm
        if q_goal is None:
            error = True
        else:
            error = False
        # print q_goal, state.arm, state.arm_v
        return SimulationRobotAction(arm_cmd=q_goal, error=error)


