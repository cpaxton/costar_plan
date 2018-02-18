
import numpy as np
import PyKDL as kdl
import rospy 
import tf2_ros as tf2
import tf_conversions.posemath as pm

from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.h5f import H5fDataset

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

class DataCollector(object):
    '''
    Manages data collection. Will consume:
    - images from camera
    - depth data (optional)
    - current end effector pose
    - current joint states
    - current gripper status
    '''
    js_topic = "joint_states"
    rgb_topic = "camera/rgb/image_raw"
    depth_topic = "camera/depth_registered/image_raw"
    ee = "endpoint"
    base_link = "base_link"
    description = "/robot_description"
    data_types = ["h5f", "npz"]

    def __init__(self, robot_config,
            data_type="h5f",
            rate=10,
            data_root=".",
            img_shape=(128,128),
            camera_frame = "camera_link",
            tf_listener=None):

        '''
        Set up the writer (to save trials to disk) and subscribers (to process
        input from ROS and store the current state).
        '''


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

        #self._camera_depth_info_sub = rospy.Subscriber(camera_depth_info_topic, CameraInfo, self._depthInfoCb)
        #self._camera_rgb_info_sub = rospy.Subscriber(camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
        self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
        self._depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depthCb)
        self._joints_sub = rospy.Subscriber(self.js_topic, JointState, self._jointsCb)

        self._bridge = CvBridge()
 
        self._resetData()

        self.verbosity = 1

    def _rgbCb(self, msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(str(e))

        self.rgb_img = np.asarray(cv_image)

    def _depthCb(self, msg):
        self.depth_img = msg

    def _resetData(self):
        self.data = {}
        self.data["q"] = []
        self.data["dq"] = []
        self.data["pose"] = []
        self.data["camera"] = []
        self.data["image"] = []

    def _jointsCb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity
        if self.verbosity > 3:
            rospy.loginfo(self.q, self.dq)

    def save(self, seed, result):
        '''
        Save function that wraps data set access.
        '''

        # for now all examples are considered a success
        self.writer.write(self.data, seed, result)
        self._resetData()

    def tick(self, action=None):
        '''
        Compute endpoint positions and update data. Should happen at some
        fixed frequency like 10 hz.

        Parameters:
        -----------
        action: name of high level action being executed
        '''
        try:
            t = rospy.Time(0)
            c_pose = self.tf_listener.lookup_transform(self.base_link, self.camera_frame, t)
            ee_pose = self.tf_listener.lookup_transform(self.base_link, self.ee_frame, t)
        except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
            rospy.logwarn("Failed lookup: %s to %s, %s"%(self.base_link, self.camera_frame, self.ee_frame))
            return False

        c_xyz = [c_pose.transform.translation.x,
                 c_pose.transform.translation.y,
                 c_pose.transform.translation.z,]
        c_quat = [c_pose.transform.rotation.x,
                  c_pose.transform.rotation.y,
                  c_pose.transform.rotation.z,
                  c_pose.transform.rotation.w,]
        ee_xyz = [ee_pose.transform.translation.x,
                 ee_pose.transform.translation.y,
                 ee_pose.transform.translation.z,]
        ee_quat = [ee_pose.transform.rotation.x,
                  ee_pose.transform.rotation.y,
                  ee_pose.transform.rotation.z,
                  ee_pose.transform.rotation.w,]

        self.data["q"].append(np.copy(self.q))
        self.data["dq"].append(np.copy(self.dq))
        self.data["pose"].append(ee_xyz + ee_quat)
        self.data["camera"].append(c_xyz + c_quat)
        self.data["image"].append(np.copy(self.rgb_img))

        return True

if __name__ == '__main__':
    pass


