from __future__ import print_function

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from dmp_policy import CartesianDmpPolicy
from dmp_option import DmpOption
from dynamics import SimulatedDynamics
from geometry_msgs.msg import PoseArray
from os.path import join
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf_conversions import posemath as pm
from urdf_parser_py.urdf import URDF

# Include representations for either (a) the generic skill or (b) an instance
# of a skill represented as a goal-directed set of motion primitives.
from costar_task_plan.robotics.representation import RobotFeatures
from costar_task_plan.robotics.representation import CartesianSkillInstance
from costar_task_plan.robotics.representation import GMM
from costar_task_plan.robotics.representation import Distribution
from costar_task_plan.robotics.representation import RequestActiveDMP, PlanDMP

# Important libraries
import numpy as np
import os
import rospy
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class LfD(object):

    '''
    Computes features and representations which given trajectories, objects, etc.
    This must take a world object to use for computing objects and positions.
    Note that this LfD object currently assumes supervised (labeled) data, which
    indicates exactly which objects are important for each action and which are
    not.
    '''

    def __init__(self, config):

        base_link = config['base_link']
        end_link = config['end_link']

        if 'robot_description_param' in config:
            self.robot = URDF.from_parameter_server(
                config['robot_description_param'])
        else:
            self.robot = URDF.from_parameter_server()

        self.config = config
        self.base_link = base_link

        # set up kinematics stuff
        self.tree = kdl_tree_from_urdf_model(self.robot)
        self.chain = self.tree.getChain(base_link, end_link)
        self.kdl_kin = KDLKinematics(self.robot, base_link, end_link)

        self.base_link = base_link
        self.end_link = end_link

        self.skill_instances = {}
        self.skill_features = {}
        self.skill_models = {}
        self.parent_skills = {}

        self.pubs = {}

    def train(self, trajectories, trajectory_data, objs, instance_params=None):
        '''
        Generate DMPs and GMMs associated with different labeled actions.

        Parameters:
        -----------
        trajectories: trajectory data by high level skill
        trajectory_data: training data (object poses)
        objs: specific data needed for each skill
        skill_instances: mid-level class info about objects (red, blue, etc.)
        '''

        for name, trajs in trajectories.items():

            # Create publisher for debugging purposes
            self.pubs[name] = rospy.Publisher(
                join('costar', 'lfd', name), PoseArray, queue_size=1000)

            data = trajectory_data[name]
            features = RobotFeatures(self.config, self.kdl_kin)

            if not name in self.skill_instances:
                self.skill_instances[name] = []

            # Each world here is an observation of a particular frame in this scene
            assert(len(trajs) == len(data))
            for i, (traj, world) in enumerate(zip(trajs, data)):

                if instance_params is not None:
                    sub_name, skill_objs = instance_params[name][i]
                    self.parent_skills[sub_name] = name
                else:
                    sub_name = None
                    skill_objs = objs[name]

                # Add a publisher for the sub-model if one does not exist yet
                if sub_name is not None and sub_name not in self.pubs:
                    self.pubs[sub_name] = rospy.Publisher(
                        join('costar', 'lfd', sub_name), PoseArray, queue_size=1000)

                ts = [t for t, _, _ in traj]
                dt = np.mean(np.diff(ts))

                #ee = [pm.fromMsg(pose) for _, pose, _ in traj]
                ee = [pose for _, pose, _ in traj]
                gripper = [gopen for _, _, gopen in traj]

                if not len(ee) == len(gripper) and len(gripper) == len(world):
                    raise RuntimeError('counting error')

                # compute features?
                f, g = features.GetFeaturesForTrajectory(ee, world, skill_objs)
                instance = CartesianSkillInstance(self.config,
                                                  dt=dt,
                                                  objs=skill_objs)
                instance.fit(ee_frames=ee, worlds=world)

                self.skill_instances[name].append(instance)
                if ((not sub_name == name) and (sub_name is not None)):
                    if sub_name not in self.skill_instances:
                        self.skill_instances[sub_name] = []
                    self.skill_instances[sub_name].append(instance)

                if not name in self.skill_features:
                    self.skill_features[name] = f
                else:
                    self.skill_features[name] = np.concatenate((self.skill_features[name], f), axis=0)

            # only fit models if we have an example of that skill
            if name in self.skill_features:
                try:
                    self.skill_models[name] = GMM(
                        self.config['gmm_k'], self.skill_features[name])
                except np.linalg.LinAlgError as e:
                    simple_conf = {
                        'mu': np.mean(self.skill_features[name], axis=-1),
                        'k': 1,
                        'pi': np.ones((1,1)),
                        'sigma': np.expand_dims(np.diag(np.std(self.skill_features[name], axis=-1)),axis=0),
                    }
                    self.skill_models[name] = GMM(config=simple_conf)
                print( "> Skill", name, "extracted with dataset of shape", self.skill_features[name].shape, "k =", self.config['gmm_k'])
            else:
                print(" ... skipping skill", name, "(no data)")

        return self.skill_models

    def debug(self, world, verbose=False):
        '''
        Publish a bunch of ROS messages showing trajectories. This is a helpful
        tool for debugging problems with training data, DMP learning, and DMP
        segmentation.

        Parameters:
        -----------
        '''

        for name, instances in self.skill_instances.items():

            goal_type = instances[0].objs[-1]
            goals = world.getObjects(goal_type)
            if goals is None:
                continue
            else:
                goal = goals[0]
            goal_pose = world.getPose(goal)
            if verbose:
                print(name,"goal is",goal_type,"and chose",goal)
            if goal_pose is None:
                continue
            if name in self.parent_skills:
                parent_name = self.parent_skills[name]
            else:
                parent_name = name

            model = self.skill_models[parent_name]

            option = DmpOption(
                policy_type=CartesianDmpPolicy,
                config=self.config,
                kinematics=self.kdl_kin,
                goal_object=goal,
                skill_name=name,
                feature_model=model,
                traj_dist=self.getParamDistribution(parent_name))

            policy, condition = option.makePolicy(world)
            dynamics = SimulatedDynamics()

            state = world.actors[0].state

            RequestActiveDMP(instances[0].dmp_list)

            q = state.q
            if q is None:
                continue
            T = pm.fromMatrix(self.kdl_kin.forward(q))
            ee_rpy = T.M.GetRPY()
            relative_goal = goal_pose * instances[0].goal_pose
            rpy = relative_goal.M.GetRPY()
            adj_rpy = [0, 0, 0]

            # Fix rotations
            for j, (lvar, var) in enumerate(zip(ee_rpy, rpy)):
                if lvar < 0 and var > lvar + np.pi:
                    adj_rpy[j] = var - 2 * np.pi
                elif lvar > 0 and var < lvar - np.pi:
                    adj_rpy[j] = var + 2 * np.pi
                else:
                    adj_rpy[j] = var

            # Create start and goal poses
            x = [T.p[0], T.p[1], T.p[2], ee_rpy[0], ee_rpy[1], ee_rpy[2]]
            g = [relative_goal.p[0], relative_goal.p[1], relative_goal.p[2],
                 adj_rpy[0], adj_rpy[1], adj_rpy[2]]
            x0 = [0.] * 6
            g_threshold = [1e-1] * 6
            integrate_iter = 10

            # Get DMP result
            res = PlanDMP(x, x0, 0., g, g_threshold, 2 * instances[
                          0].tau, 1.0, world.dt, integrate_iter)

            # Convert to poses
            poses = []
            for i, pt in enumerate(res.plan.points):
                T = pm.Frame(
                    pm.Rotation.RPY(pt.positions[3], pt.positions[4], pt.positions[5]))
                T.p[0] = pt.positions[0]
                T.p[1] = pt.positions[1]
                T.p[2] = pt.positions[2]
                poses.append(pm.toMsg(T))
                new_q = self.kdl_kin.inverse(pm.toMatrix(T), q)
                q = new_q

            msg = PoseArray(poses=poses)
            msg.header.frame_id = self.base_link
            self.pubs[name].publish(msg)

    def save(self, project_name):
        '''
        Save models after we fit them. Models for skill instances and for
        expected feature observations are recorded separately.
        '''
        skills_dir = os.path.join(project_name, 'skill_models')
        models_dir = os.path.join(project_name, 'feature_models')
        if not os.path.exists(project_name):
            os.mkdir(project_name)
        if not os.path.exists(skills_dir):
            os.mkdir(skills_dir)
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        skill_counts = {}

        for name in self.skill_instances.keys():
            skill_counts[name] = len(self.skill_instances[name])
            for i, skill in enumerate(self.skill_instances[name]):
                filename = os.path.join(skills_dir, '%s%02d.yml' % (name, i))
                yaml_save(skill, filename)
            model_filename = os.path.join(models_dir, '%s_gmm.yml' % name)
            yaml_save(self.skill_models[name], model_filename)

        skill_filename = os.path.join(project_name, "skills.yml")
        yaml_save(skill_counts, skill_filename)

    def load(self, project_name):
        '''
        Load saved models into this object.
        '''
        skills_dir = os.path.join(project_name, 'skill_models')
        models_dir = os.path.join(project_name, 'feature_models')
        skill_filename = os.path.join(project_name, "skills.yml")
        skills = yaml_load(skill_filename)
        for name, count in skills.items():
            print("Loading skill %s/%s - %d examples"%(project_name, name, count))

            # For debugging only
            if name not in self.pubs:
                self.pubs[name] = rospy.Publisher(
                    join('costar', 'lfd', name), PoseArray, queue_size=1000)

            self.skill_instances[name] = []
            for i in xrange(count):
                filename = os.path.join(skills_dir, '%s%02d.yml' % (name, i))
                dmp = yaml_load(filename)
                self.skill_instances[name].append(dmp)

            model_filename = os.path.join(models_dir, '%s_gmm.yml' % name)
            self.skill_models[name] = yaml_load(model_filename)

    def getSkillModel(self, skill):
        '''
        Return the appropriate model for this particular skill.
        '''
        skill = self.getParent(skill)
        return self.skill_models[skill]

    def getParent(self, skill):
        while skill in self.parent_skills:
            if skill == self.parent_skills[skill]:
                break
            skill = self.parent_skills[skill]
        return skill

    def getParamDistribution(self, skill):
        '''
        Get the mean and covariance associated with our observed expert
        policies. We can then use these together with our expected feature
        counts to optimize to a new environment.
        '''

        # Get the parent skill - this is the highest level skill of the type
        # in case we have more than one.
        skill = self.getParent(skill)

        # Aggregate features used when training the parent skill model
        params = []
        for instance in self.skill_instances[skill]:
            params.append(instance.params())

        # get mean and get std dev
        params = np.array(params)
        mu = np.mean(params,axis=0)
        
        if params.T.shape[1] < 2:
            #raise RuntimeError("Cannot create a distribution from one example!")
            rospy.logwarn("%s: Cannot create distribution from one example!" % skill)
            rospy.logwarn("skill =" + str(skill) + " shape =" + str(params.T.shape))
            size = mu.shape[0]
            sigma = 1e-6 * np.eye(size)
        else:
            sigma = np.cov(params.T)

        assert mu.shape[0] == sigma.shape[0]
        return Distribution(mu, sigma)

def yaml_save(obj, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(obj, outfile, default_flow_style=False)


def yaml_load(filename):
    with open(filename, 'r') as infile:
        return yaml.load(infile)#, Loader=Loader)

