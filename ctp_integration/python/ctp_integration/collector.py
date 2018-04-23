
import matplotlib.pyplot as plt
import numpy as np
import PyKDL as kdl
import rospy
import tf2_ros as tf2
import tf_conversions.posemath as pm

from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.h5f import H5fDataset
from costar_models.datasets.image import GetJpeg
from costar_models.datasets.image import GetPng
from costar_models.datasets.image import JpegToNumpy
from costar_models.datasets.image import ConvertJpegListToNumpy
from costar_models.datasets.depth_image_encoding import FloatArrayToRgbImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from robotiq_c_model_control.msg import CModel_robot_input as GripperMsg
from ar_track_alvar_msgs import AlvarMarkers

class DataCollector(object):
    '''
    Manages data collection. Will consume:
    - images from camera
    - depth data (optional)
    - current end effector pose
    - current joint states
    - current gripper status
    '''

    def __init__(
        self, robot_config,
        task,
        data_type="h5f",
        rate=10,
        data_root=".",
        img_shape=(128,128),
        camera_frame="camera_link",
        tf_listener=None):


        self.js_topic = "joint_states"
        self.rgb_topic = "/camera/rgb/image_rect_color"
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect_raw"
        self.ee = "endpoint"
        self.base_link = "base_link"
        self.description = "/robot_description"
        self.data_types = ["h5f", "npz"]
        self.info_topic = "/costar/info"
        self.object_topic = "/costar/SmartMove/object"
        self.gripper_topic = "/CModelRobotInput"
        self.ar_pose_topic = "/camera/ar_pose_marker"
        self.camera_depth_info_topic = "/camera/rgb/camera_info"
        self.camera_rgb_info_topic = "/camera/depth/camera_info"
        self.camera_rgb_optical_frame = "camera_rgb_optical_frame"
        self.camera_depth_optical_frame = "camera_depth_optical_frame"

        '''
        Set up the writer (to save trials to disk) and subscribers (to process
        input from ROS and store the current state).
        '''
        self.verbosity = 0

        if tf_listener is not None:
            self.tf_listener = tf_listener
        else:
            self.tf_listener = tf.TransformListener()

        if isinstance(rate, int) or isinstance(rate, float):
            self.rate = rospy.Rate(rate)
        elif isinstance(rate, rospy.Rate):
            self.rate = rate
        else:
            raise RuntimeError("rate data type not supported: %s" % type(rate))

        self.root = data_root
        self.data_type = data_type
        rospy.logwarn("Dataset root set to " + str(self.root))
        if self.data_type == "h5f":
            self.writer = H5fDataset(self.root)
        elif self.data_type == "npz":
            self.writer = NpzDataset(self.root)
        else:
            raise RuntimeError("data type %s not supported" % data_type)

        self.T_world_ee = None
        self.T_world_camera = None
        self.camera_frame = camera_frame
        self.ee_frame = robot_config['end_link']

        self.q = None
        self.dq = None
        self.pc = None
        self.camera_depth_info = None
        self.camera_rgb_info = None
        self.depth_img = None
        self.rgb_img = None
        self.gripper_msg = None

        self._bridge = CvBridge()
        self.task = task
        self.reset()

        self._camera_depth_info_sub = rospy.Subscriber(self.camera_depth_info_topic, CameraInfo, self._depthInfoCb)
        self._camera_rgb_info_sub = rospy.Subscriber(self.camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
        self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
        self._depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depthCb)
        self._joints_sub = rospy.Subscriber(self.js_topic,
                JointState,
                self._jointsCb)
        self._info_sub = rospy.Subscriber(self.info_topic,
                String,
                self._infoCb)
        self._smartmove_object_sub = rospy.Subscriber(self.object_topic,
                String,
                self._objectCb)
        self._gripper_sub = rospy.Subscriber(self.gripper_topic,
                GripperMsg,
                self._gripperCb)
        self._ar_sub = rospy.Subscriber(self.ar_pose_topic,
                AlvarMarkers,
                self._arPoseCb)

        self.verbosity = 1

    def _rgbCb(self, msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self.rgb_img = np.asarray(cv_image)
            #print(self.rgb_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def _infoCb(self, msg):
        self.info = msg.data

    def _depthInfoCb(self, msg):
        self.depth_info = msg

    def _rgbInfoCb(self, msg):
        self.rgb_info = msg

    def _objectCb(self, msg):
        self.object = msg.data

    def _gripperCb(self, msg):
        self.gripper_msg = msg

    def _arPoseCb(self, msg):
        self.ar_pose_msg = msg

    def _depthCb(self, msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg)
            self.depth_img = np.asarray(cv_image)
            #print (self.depth_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def setTask(self, task):
        self.task = task

    def reset(self):
        self.data = {}
        self.data["q"] = []
        self.data["dq"] = []
        self.data["pose"] = []
        self.data["camera"] = []
        self.data["image"] = []
        self.data["depth_image"] = []
        self.data["goal_idx"] = []
        self.data["gripper"] = []
        self.data["ar_pose"] = []
        self.data["label"] = []
        self.data["info"] = []
        self.data["depth_info"] = []
        self.data["rgb_info"] = []
        self.data["object"] = []
        self.data["object_pose"] = []
        self.data["labels_to_name"] = list(self.task.labels)
        self.data["rgb_info_D"] = []
        self.data["rgb_info_K"] = []
        self.data["rgb_info_R"] = []
        self.data["rgb_info_P"] = []
        self.data["rgb_info_distortion_model"] = []
        self.data["depth_info_D"] = []
        self.data["depth_info_K"] = []
        self.data["depth_info_R"] = []
        self.data["depth_info_P"] = []
        self.data["depth_distortion_model"] = []
        self.data["ar_pose_marker"] = []
        self.data["visualization_marker"] = []
        #self.data["depth"] = []

        self.info = None
        self.object = None
        self.prev_object = None
        self.action = None
        self.prev_action = None
        self.current_ee_pose = None
        self.last_goal = 0
        self.prev_last_goal = 0

    def _jointsCb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity
        if self.verbosity > 3:
            rospy.loginfo(self.q, self.dq)

    def save(self, seed, result):
        '''
        Save function that wraps data set access.
        '''
        for k, v in self.data.items():
            print(k, np.array(v).shape)
        print(self.data["labels_to_name"])
        print("Labels and goals:")
        print(self.data["label"])
        print(self.data["goal_idx"])

        # for now all examples are considered a success
        self.writer.write(self.data, seed, result, image_type="jpeg")
        self.reset()

    def update(self, action_label, is_done):
        '''
        Compute endpoint positions and update data. Should happen at some
        fixed frequency like 10 hz.

        Parameters:
        -----------
        action: name of high level action being executed
        '''

        switched = False
        if not self.action == action_label:
            if not self.action is None:
                switched = True
            self.prev_action = self.action
            self.action = action_label
            self.prev_object = self.object
            self.object = None
        if switched or is_done:
            self.prev_last_goal = self.last_goal
            self.last_goal = len(self.data["label"])
            len_label = len(self.data["label"])

            # Count one more if this is the last frame -- since our goal could
            # not be the beginning of a new action
            if is_done:
                len_label += 1
                extra = 1
            else:
                extra = 0

            rospy.loginfo("Starting new action: "
                    + str(action_label)
                    + ", prev was from "
                    + str(self.prev_last_goal)
                    + " to " + str(self.last_goal))
            self.data["goal_idx"] += (self.last_goal - self.prev_last_goal + extra) * [self.last_goal]

            len_idx = len(self.data["goal_idx"])
            if not len_idx  == len_label:
                rospy.logerr("lens = " + str(len_idx) + ", " + str(len_label))
                raise RuntimeError("incorrectly set goal idx")
        if self.object is None:
            rospy.logwarn("passing -- has not yet started executing motion")
            return True

        rospy.loginfo("Logging: " + str(self.action) +
                ", obj = " + str(self.object) +
                ", prev = " + str(self.prev_object))

        have_data = False
        attempts = 0
        max_attempts = 10
        while not have_data:
            try:
                t = rospy.Time(0)
                c_pose = self.tf_listener.lookup_transform(self.base_link, self.camera_frame, t)
                ee_pose = self.tf_listener.lookup_transform(self.base_link, self.ee_frame, t)
                obj_pose = self.tf_listener.lookup_transform(self.base_link, self.object, t)
                rgb_optical_pose = self.tf_listener.lookup_transform(self.base_link, self.camera_rgb_optical_frame, t)
                depth_optical_pose = self.tf_listener.lookup_transform(self.base_link, self.camera_depth_optical_frame, t)
                have_data = True
            except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                rospy.logwarn("Failed lookup: %s to %s, %s, %s" %
                        (self.base_link, self.camera_frame, self.ee_frame, str(self.object)))
                have_data = False
                attempts += 1
                if attempts > max_attempts:
                    # Could not look up one of the transforms -- either could
                    # not look up camera, endpoint, or object.
                    raise e

        c_xyz = [c_pose.transform.translation.x,
                 c_pose.transform.translation.y,
                 c_pose.transform.translation.z,]
        c_quat = [c_pose.transform.rotation.x,
                  c_pose.transform.rotation.y,
                  c_pose.transform.rotation.z,
                  c_pose.transform.rotation.w,]
        rgb_optical_xyz = [rgb_optical_pose.transform.translation.x,
                           rgb_optical_pose.transform.translation.y,
                           rgb_optical_pose.transform.translation.z,]
        rgb_optical_quat = [rgb_optical_pose.transform.rotation.x,
                  rgb_optical_pose.transform.rotation.y,
                  rgb_optical_pose.transform.rotation.z,
                  rgb_optical_pose.transform.rotation.w,]
        depth_optical_xyz = [depth_optical_pose.transform.translation.x,
                             depth_optical_pose.transform.translation.y,
                             depth_optical_pose.transform.translation.z,]
        depth_optical_quat = [depth_optical_pose.transform.rotation.x,
                              depth_optical_pose.transform.rotation.y,
                              depth_optical_pose.transform.rotation.z,
                              depth_optical_pose.transform.rotation.w,]
        ee_xyz = [ee_pose.transform.translation.x,
                 ee_pose.transform.translation.y,
                 ee_pose.transform.translation.z,]
        ee_quat = [ee_pose.transform.rotation.x,
                  ee_pose.transform.rotation.y,
                  ee_pose.transform.rotation.z,
                  ee_pose.transform.rotation.w,]
        obj_xyz = [obj_pose.transform.translation.x,
                 obj_pose.transform.translation.y,
                 obj_pose.transform.translation.z,]
        obj_quat = [obj_pose.transform.rotation.x,
                  obj_pose.transform.rotation.y,
                  obj_pose.transform.rotation.z,
                  obj_pose.transform.rotation.w,]

        self.current_ee_pose = pm.fromTf((ee_xyz, ee_quat))

        self.data["q"].append(np.copy(self.q)) # joint position
        self.data["dq"].append(np.copy(self.dq)) # joint velocuity
        self.data["pose"].append(ee_xyz + ee_quat) # end effector pose (6 DOF)
        self.data["camera"].append(c_xyz + c_quat) # camera pose (6 DOF)
        self.data["object_pose"].append(obj_xyz + obj_quat)
        self.data["camera_rgb_optical_frame_pose"].append(rgb_optical_xyz + rgb_optical_quat)
        self.data["camera_depth_optical_frame_pose"].append(depth_optical_xyz + depth_optical_quat)
        #plt.figure()
        #plt.imshow(self.rgb_img)
        #plt.show()
        self.data["image"].append(GetJpeg(self.rgb_img)) # encoded as JPEG
        self.data["depth_image"].append(GetPng(FloatArrayToRgbImage(self.depth_img)))
        self.data["gripper"].append(self.gripper_msg.gPO / 255.)
        self.data["ar_pose"].append(self.ar_pose_msg)

        # TODO(cpaxton): verify
        if not self.task.validLabel(action_label):
            raise RuntimeError("action not recognized: " + str(action_label))

        action = self.task.index(action_label)
        self.data["label"].append(action)  # integer code for high-level action
        self.data["info"].append(self.info)  # string description of current step
        self.data["rgb_info_D"].append(self.rgb_info.D)
        self.data["rgb_info_K"].append(self.rgb_info.K)
        self.data["rgb_info_R"].append(self.rgb_info.R)
        self.data["rgb_info_P"].append(self.rgb_info.P)
        self.data["rgb_info_distortion_model"].append(self.rgb_info.distortion_model)
        self.data["depth_info_D"].append(self.depth_info.D)
        self.data["depth_info_K"].append(self.depth_info.K)
        self.data["depth_info_R"].append(self.depth_info.R)
        self.data["depth_info_P"].append(self.depth_info.P)
        self.data["depth_distortion_model"].append(self.depth_info.distortion_model)
        self.data["object"].append(self.object)

        # TODO(cpaxton): add pose of manipulated object
        self.data["object_pose"].append(obj_xyz + obj_quat)

        #self.data["depth"].append(GetJpeg(self.depth_img))

        return True

if __name__ == '__main__':
    pass


