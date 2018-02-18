
# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

# ROS stuff
import rospy
import rosbag
from urdf_parser_py.urdf import URDF
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import copy

import numpy as np
#from scipy.stats import multivariate_normal as mvn

# KDL utilities
import PyKDL
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

# tf stuff
import tf
import tf_conversions.posemath as pm

# for fast feature computation
from numba import jit

# input message types 
import sensor_msgs
import trajectory_msgs
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

# output message types
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray

TIME = 'time'
GRIPPER = 'gripper'
JOINT = 'joint' # features indicating total joint velocity/effort

NUM_OBJ_VARS = 8
NUM_OBJ_DIFF_VARS = 1
NUM_GRIPPER_VARS = 1
NUM_GRIPPER_DIFF_VARS = 0
NUM_TIME_VARS = 1

@jit
def P_Gauss(x,mu,inv,det,wts):
    '''
    P_Gauss
    Compute the Gaussian probability of something
    '''

    nvar = mu.shape[1]
    p = np.zeros(x.shape[0])

    for i in range(wts.shape[0]):
        res = (x - mu).dot(inv[0]) * (x - mu)
        res = -0.5 * np.sum(res,axis=1)
        p += wts[i] * np.exp(res) / (np.sqrt((2*np.pi)**nvar * np.abs(det[i])))

    return np.log(p)

class RobotFeatures:
    '''
    Old class that holds and represents a robot -- for one skill and one skill
    alone. Goals of this class:
    - collect data based on the list of specified objects
    - fit models for policies like DMPs
    '''

    def __init__(self,
            config, kinematics=None,
            objects={}, indices={}, diff_indices={},
            filename=None
            ):
        '''
        Create a robot
        loads robot description and kinematics from parameter server if available
        configured as a kinematic chain; uses KDLKinematics for robot forward
        kinematics by default if no kinematics were provided.
        '''

        self.sync_gripper = False
        self.is_recording = False

        self.dof = config['dof'];
        self.obj_frame = config['obj_frame']
        self.base_link = config['base_link']
        self.end_link = config['end_link']
        self.robot_description_param = config['robot_description_param']
        self.js_topic = config['joint_states_topic']
        self.gripper_topic = config['gripper_topic']

        if kinematics is None:
          robot = URDF.from_parameter_server(self.robot_description_param)
          self.kinematics = KDLKinematics(robot, base_link, end_link)
        else:
          self.kinematics = kinematics
          
        self.objects = objects
        self.world = {}

        self.last_gripper_msg = rospy.Time(0)
        self.gripper_t_threshold = 0.05

        self.times = []
        self.joint_states = []
        self.gripper_cmds = []
        self.world_states = []

        self.indices = indices
        self.diff_indices = diff_indices
        self.max_index = 0
        self.max_diff_index = 0

        self.feature_model = None
        self.sub_model = None

        self.manip_obj = None
        self.manip_frame = None

        self.action_inv = []
        self.goal_inv = []
        self.action_det = []
        self.goal_det = []
        self.action_pdf = []
        self.goal_pdf = []
        self.action_mean = None
        self.action_std = None
        self.goal_mean = None
        self.goal_std = None
        self.action_mean_ng = None
        self.action_std_ng = None
        self.goal_mean_ng = None
        self.goal_std_ng = None

        self.recorded = False
        self.quiet = True # by default hide TF error messages

        if not filename == None:
            stream = file(filename,'r')
            data = yaml.load(stream,Loader=Loader)
            self.joint_states = data['joint_states']
            self.world_states = data['world_states']
            self.times = data['times']
            self.base_tform = data['base_tform']
            self.manip_obj = data['manip_obj']
            self.obj_frame = data['obj_frame']
            self.base_link = data['base_link']
            self.end_link = data['end_link']

            if data.has_key('indices') and data.has_key('diff_indices'):
                self.indices = data['indices']
                self.diff_indices = data['diff_indices']
                self.max_index = data['max_index']
            else: # initialize the indices
                for obj in self.world_states[0].keys():
                    self.AddObject(obj)

            self.recorded = True
            self.gripper_cmds = data['gripper_cmds']

    def ConfigureSkill(self,action,goal):

        self.action_inv = np.zeros(action.covars_.shape)

        self.action_def = []
        self.goal_det = []

        #self.action_pdf = mvn(mean=action.means_[0],cov=action.covars_[0])
        #self.goal_pdf = mvn(mean=goal.means_[0],cov=goal.covars_[0])

        for i in range(action.n_components):
            self.action_inv[i,:,:] = np.linalg.inv(action.covars_[i,:,:])
            self.action_det.append(np.linalg.det(action.covars_[i,:,:]))

        if not goal is None:
            self.goal_inv = np.zeros(goal.covars_.shape)
            for i in range(goal.n_components):
                self.goal_inv[i,:,:] = np.linalg.inv(goal.covars_[i,:,:])
                self.goal_det.append(np.linalg.det(goal.covars_[i,:,:]))

        self.traj_model = action;
        self.goal_model = goal;

    def P_Action(self,X):
        if self.traj_model.n_components == 1:
            return P_Gauss(X,self.traj_model.means_,self.action_inv,self.action_det,self.traj_model.weights_)
        else:
            return self.traj_model.score(X)

    def P_Goal(self,X):
        if self.goal_model.n_components == 1:
            return P_Gauss(X,self.goal_model.means_,self.goal_inv,self.goal_det,self.goal_model.weights_)
        else:
            return self.goal_model.score(X)

    '''
    Start recording on the specified topics.
    '''
    def StartRecording(self):
        self.is_recording = True
        if self.recorded:
            return

        self.recorded = True
        if self.js_sub is None:
          self.js_sub = rospy.Subscriber(self.js_topic,sensor_msgs.msg.JointState,self._js_cb)
          self.gripper_sub = rospy.Subscriber(self.gripper_topic,GripperCmd,self._gripper_cb)

    '''
    Stop recording and clear.
    '''
    def StopRecording(self):

        self.is_recording = False
        self.recorded = False

        self.world_states = []
        self.joint_states = []
        self.gripper_cmds = []
        self.times = []


    '''
    stop recording, save data to disk.
    '''
    def Save(self,filename):

        self.is_recording = False

        if self.recorded:
            stream = file(filename,'w')

            data = {}
            data['times'] = self.times
            data['obj_frame'] = self.obj_frame
            data['gripper_cmds'] = self.gripper_cmds
            data['joint_states'] = self.joint_states
            data['world_states'] = self.world_states
            data['base_link'] = self.base_link
            data['end_link'] = self.end_link
            data['robot_description_param'] = self.robot_description_param
            data['base_tform'] = self.base_tform
            data['objects'] = self.objects
            data['indices'] = self.indices
            data['diff_indices'] = self.diff_indices
            data['max_index'] = self.max_index
            data['manip_obj'] = self.manip_obj

            yaml.dump(data,stream)

        else:
            rospy.logerr("Could not save; no recording done!")

    def ToRosBag(self,filename) :
        with rosbag.Bag(filename,'w') as outbag:
            try:
                for i in range(len(self.times)):
                    print "%d / joint states = %d / times = %d / world states = %d"%(i, len(self.joint_states), len(self.times), len(self.world_states))
                    outbag.write('joint_states',self.joint_states[i],self.times[i])
                    if i >= len(self.world_states):
                        print "Done early."
                        break
                    for frame in self.world_states[0].keys():
                        outbag.write('world/%s'%frame,pm.toMsg(self.world_states[i][frame]),self.times[i])
                    outbag.write('base_tform',pm.toMsg(self.base_tform),self.times[i])
                    if len(self.gripper_cmds) > 0:
                        outbag.write('gripper_msg',self.gripper_cmds[i],self.times[i])

            finally:
                outbag.close()


    def _js_cb(self,msg):

        if not self.is_recording:
            #print "[JOINTS] Not currently recording!"
            return

        updated = self.TfUpdateWorld()
        if updated and not self.sync_gripper:
            # record joints
            self.times.append(rospy.Time.now())
            self.joint_states.append(msg)
            print msg.position
            self.world_states.append(copy.deepcopy(self.world))
        elif updated and (rospy.Time.now() - self.last_gripper_msg).to_sec() < self.gripper_t_threshold:
            # record joints
            self.times.append(rospy.Time.now())
            self.joint_states.append(msg)
            self.gripper_cmds.append(self.gripper_cmd)
            self.world_states.append(copy.deepcopy(self.world))
        else:
            print "[JOINTS] Waiting for TF (updated=%d) and gripper..."%(updated)

    def _gripper_cb(self,msg):

        self.gripper_cmd = msg
        self.last_gripper_msg = rospy.Time.now()

    def updateManipObj(self,manip_objs):
        '''
        Set up manipulation object frame
        '''
        if len(manip_objs) > 0:
            self.manip_obj = manip_objs[0]
        else:
            self.manip_obj = None

    def resetIndices(self):
        self.indices = {}
        self.diff_indices = {}
    
    def AddObject(self,obj,frame=""):
        '''
        Add an object we can use as a reference
        for now number of gripper, object variables are all hard coded
        
        Objects are all TF coordinate frames.
        '''

        self.max_index = max([0]+[v[1] for k,v in self.indices.items()])

        if obj == TIME:
            nvars = NUM_TIME_VARS
            ndvars = 0
        elif obj == GRIPPER:
            nvars = NUM_GRIPPER_VARS
            ndvars = NUM_GRIPPER_DIFF_VARS
        else:
            nvars = NUM_OBJ_VARS
            ndvars = NUM_OBJ_DIFF_VARS
            self.objects[obj] = frame

        if not obj in self.indices:
            self.indices[obj] = (self.max_index,self.max_index+nvars)
            self.diff_indices[obj] = (self.max_index,self.max_index+nvars)
            self.max_index += nvars
            self.max_diff_index += nvars + ndvars

            if ndvars > 0:
                self.diff_indices["diff_" + obj] = (self.max_index+nvars,self.max_index+ndvars)

    '''
    GetForward
    Returns the position of the gripper from a given set of joint positions
    Also gets relative positions to objects at different frames of reference
    '''
    def GetForward(self,q):

        #q = [0,0,0,0,0,0,0]
        mat = self.kinematics.forward(q)
        f = pm.fromMatrix(mat)

        if not self.manip_frame is None:
            f = f * self.manip_frame #* self.manip_frame.Inverse()

        return f

    '''
    SetWorld
    Sets locations of different objects at the beginning of the action.
    '''
    def SetWorld(self,frames):
        for obj in self.objects.keys():
            self.world[obj] = frames[obj]

    '''
    Get an actual trajectory: the things we are trying to learn how to reproduce
    '''
    def GetTrajectory(self):
        traj = []
        gripper = []
        for i in range(len(self.times)):
            pt = [j for j in self.joint_states[i].position[:self.dof]]
            traj.append(pt)

            if i < len(self.gripper_cmds):
                    g = [k for k in self.gripper_cmds[i].cmd[:NUM_GRIPPER_VARS]]
            else:
                    g = None
            gripper.append(g)
        return traj,gripper

    def GetWorldPoseMsg(self,frame):

        msg = PoseArray()
        msg.header.frame_id = self.obj_frame

        for i in range(len(self.world_states)): 
            #pmsg = pm.toMsg(self.world_states[i][frame] * PyKDL.Frame(PyKDL.Rotation.RotY(-1*np.pi/2)))
            pmsg = pm.toMsg(self.world_states[i][frame])
            msg.poses.append(pmsg)

        return msg

    '''
    GetFwdPoseMsg
    Create a pose array from forward kinematics
    '''
    def GetFwdPoseMsg(self):

        msg = PoseArray()
        msg.header.frame_id = self.obj_frame

        for i in range(len(self.world_states)): 
            mat = self.kinematics.forward(self.joint_states[i].position[:self.dof])
            ee_frame = self.base_tform * pm.fromMatrix(mat)
            pmsg = pm.toMsg(ee_frame * PyKDL.Frame(PyKDL.Rotation.RotY(-1*np.pi/2)))
            msg.poses.append(pmsg)
#
        return msg


    def GetTrajectoryWeight(self,traj,world,objs,p_z,p_obs=0,t_lambda=0.5):
        '''
        GetTrajectoryWeight()
        - z is the trajectory params
        - Z is the trajectory distribution
        - p_obs is the probability of these feature observations (fixed at one)
        '''

        weights = [0.0]*len(traj)

        ee_frame = [self.base_tform * self.GetForward(q[:self.dof]) for q in traj]
        features, goal_features = self.GetFeaturesForTrajectory(ee_frame,world,objs)

        features = self.NormalizeActionNG(features)
        if not self.goal_model is None:
            goal_features = self.NormalizeGoalNG(goal_features)

        N = features.shape[0]
        pa = self.P_Action(features)

        avg = np.mean(pa)
        denom = p_obs + p_z

        if not self.goal_model is None:
            goal_prob = self.P_Goal(goal_features)[0]
            print "a=%g / g=%g / %g"%(avg,goal_prob,avg + goal_prob)
        else:
            goal_prob = 0;
            print "a=%g //"%(avg)

        return np.exp(avg + goal_prob - denom), np.exp(avg + goal_prob)

    def GetTrajectoryLikelihood(self,traj,world,objs,step=1.,sigma=0.000):
        '''
        GetTrajectoryLikelihood
        slow computation of trajectory likelihood...
        Computes the same features as before
        Will then score them as per usual
        '''

        ee_frame = [self.base_tform * self.GetForward(q[:self.dof]) for q in traj]
        features,goal_features = self.GetFeaturesForTrajectory(ee_frame,world,objs)
        isum = np.sum(range(len(features)))
        scores = self.traj_model.score(features)

        # average score
        avg = np.mean(scores)

        return self.goal_model.score(goal_features) + avg

    def GetFeaturesForTrajectory(self,ee_frame,world,objs,gripper=None):
        '''
        GetFeaturesForTrajectory
        '''

        #features = [[]]*(len(traj)-1)
        npts = len(ee_frame)-1
        features = [[]]*(npts)

        #diffs = self.GetDiffFeatures(ee_frame,world,objs)

        if not gripper is None:
            for i in range(npts):
                t = float(i+1) / npts
                features[i] = self.GetFeatures(ee_frame[i],t,world,objs,i,gripper[i]) #+ diffs[i]
            goal_features = self.GetFeatures(ee_frame[-1],0.0,world,objs,i,gripper[i-1])
        else:
            for i in range(npts):
                t = float(i+1) / npts
                features[i] = self.GetFeatures(ee_frame[i],t,world,objs,i)
            goal_features = self.GetFeatures(ee_frame[-1],0.0,world,objs,i)

        return np.array(features), np.array([goal_features])

    def GetFeatures(self,ee_frame,t,world,objs,idx,gripper=[0]*NUM_GRIPPER_VARS):
        '''
        GetFeatures
        Gets the features for a particular combination of world, time, and point.
        '''

        # initialize empty features list
        # TODO: allocate this more intelligently
        features = []
    
        for obj in objs:

            if obj == TIME:
                features += [t]
            elif obj == GRIPPER:
                features += gripper
            else:

                # we care about this world object...
                obj_frame = world[obj][idx]

                # ... so get object offset to end effector ...
                #offset = (obj_frame*PyKDL.Frame(PyKDL.Rotation.RotY(np.pi/2))).Inverse() * (ee_frame)
                offset = obj_frame.Inverse() * (ee_frame)
                #offset = obj_frame.Inverse() * (self.base_tform * ee_frame)

                # ... use position offset and distance ...
                features += offset.p
                features += [offset.p.Norm()]

                # ... and use axis/angle representation
                #(theta,w) = offset.M.GetRotAngle()
                #features += [theta*w[0],theta*w[1],theta*w[2],theta]
                #(theta,w) = offset.M.GetRotAngle()
                (x,y,z,w) = offset.M.GetQuaternion()
                #print offset.M.GetQuaternion()
                #print PyKDL.Rotation.Quaternion(x,y,z,w).GetQuaternion()
                #print '----'
                #raw_input()
                features += [x,y,z,w]

        return features

    '''
    GetDiffIndices
    '''
    def GetDiffIndices(self,objs=None):

        if objs == None:
            objs = self.diff_indices.keys()

        idx = []
        for obj in objs:
            if obj in self.diff_indices:
                idx += range(*self.diff_indices[obj])
            else:
                Exception('Missing object!')

        return idx

    '''
    GetIndices
    '''
    def GetIndices(self,objs=None):

        if objs == None:
            objs = self.indices.keys()

        idx = []
        for obj in objs:
            if obj in self.indices:
                idx += range(*self.indices[obj])
            else:
                Exception('Missing object!')

        return idx

    '''
    GetTrainingFeatures
    Takes a joint-space trajectory (with times) and produces an output vector of (expected) features based on known object positions
    '''
    def GetTrainingFeatures(self,objs=None):

        if objs == None:
            objs = self.indices.keys()

        traj,gripper = self.GetTrajectory()

        if not self.manip_obj == None:
            self.manip_frame = None
            manip_frame = (self.base_tform * self.GetForward(traj[0])).Inverse() * self.world_states[0][self.manip_obj]
            ee_frame = [x[self.manip_obj] for x in self.world_states]
        else:
            ee_frame = [self.base_tform * self.GetForward(q[:self.dof]) for q in traj]

        return self.GetFeaturesForTrajectory(ee_frame,self.world_states[0],objs,gripper)

    '''
    GetFeatureLabels()
    Gets a set of string labels, one for each feature.
    This is to make debugging/visualizing results a little bit easier.
    '''
    def GetFeatureLabels(self,objs=None):
        
        if objs==None:
            objs = self.indices.keys()

        labels = []

        for obj in objs: #self.world_states[0].keys():

            if obj == TIME:
                labels += ["time"]
            elif obj == GRIPPER:
                labels += ["gripper1", "gripper2", "gripper3"]
            else:
                labels += ["%s_ee_x"%obj]
                labels += ["%s_ee_y"%obj]
                labels += ["%s_ee_z"%obj]
                labels += ["%s_ee_dist"%obj]
                labels += ["%s_ee_wx"%obj]
                labels += ["%s_ee_wy"%obj]
                labels += ["%s_ee_wz"%obj]
                labels += ["%s_ee_theta"%obj]

        return labels

    '''
    Get the diff features we are using
    These are the translation, distance, and axis-angle rotation between frames
    '''
    def GetDiffFeatures(self,ees,world,objs):
        diffs = [[]]*(len(ees)-1)
        dists = [0]*(len(ees))

        for i in range(len(ees)):
            for obj in objs:
                if not (obj == TIME or obj == GRIPPER):
                    dists[i] = ((ees[i]).p - world[obj].p).Norm()
                    if i > 0:
                        #df = ees[i-1].Inverse() * ees[i]
                        #(theta,w) = df.M.GetRotAngle()
                        #diffs[i-1] = [dists[i] - dists[i-1],df.p.x(),df.p.y(),df.p.z(),df.p.Norm(),theta*w[0],theta*w[1],theta*w[2],theta]
                        diffs[i-1] = [dists[i] - dists[i-1]]

        return diffs

    '''
    GetJointPositions()
    Just get the positions of each joint
    '''
    def GetJointPositions(self):
        return [pt.position for pt in self.joint_states]

    def Dims(self):
        return len(self.joint_states[0].position) + 3

    '''
    Set up mean/std to normalize incoming action data
    '''
    def SetActionNormalizer(self,skill):
        self.action_mean = skill.action_mean
        self.action_std = skill.action_std
        self.action_mean_ng = skill.action_mean_ng
        self.action_std_ng = skill.action_std_ng
    def NormalizeAction(self,features):
        return (features - self.action_mean) / self.action_std
    def NormalizeActionNG(self,features):
        return (features - self.action_mean_ng) / self.action_std_ng
    def DenormalizeAction(self,features):
        return (features * self.action_std) + self.action_mean
    def DenormalizeActionNG(self,features):
        return (features * self.action_std_ng) + self.action_mean_ng

    '''
    Set up mean/std to normalize incoming goal data
    '''
    def SetGoalNormalizer(self,skill):
        #self.goal_mean = skill.goal_mean
        #self.goal_std = skill.goal_std
        #self.goal_mean_ng = skill.goal_mean_ng
        #self.goal_std_ng = skill.goal_std_ng
        self.goal_mean = skill.action_mean
        self.goal_std = skill.action_std
        self.goal_mean_ng = skill.action_mean_ng
        self.goal_std_ng = skill.action_std_ng
    def NormalizeGoal(self,features):
        return (features - self.goal_mean) / self.goal_std
    def NormalizeGoalNG(self,features):
        return (features - self.goal_mean_ng) / self.goal_std_ng
    def DenormalizeGoal(self,features):
        return (features * self.goal_std) + self.goal_mean
    def DenormalizeGoalNG(self,features):
        return (features * self.goal_std_ng) + self.goal_mean_ng

def LoadRobotFeatures(filename):

    stream = file(filename,'r')
    data = yaml.load(stream,Loader=Loader)

    r = RobotFeatures(base_link=data['base_link'],
            end_link=data['end_link']
            ,obj_frame=data['obj_frame'],
            robot_description_param=data['robot_description_param'],preset=None)

    r.gripper_cmds = data['gripper_cmds']
    r.joint_states = data['joint_states']
    r.world_states = data['world_states']
    r.times = data['times']
    r.base_tform = data['base_tform']
    r.obj_frame = data['obj_frame']
    r.base_link = data['base_link']

    if data.has_key('indices'):
        r.indices = data['indices']
        r.max_index = data['max_index']
    else: # initialize the indices
        for obj in r.world_states[0].keys():
            r.AddObject(obj)

    r.recorded = True

    return r

