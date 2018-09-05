
import sys
import os
import numpy as np
import tensorflow as tf
import PyKDL as kdl
import rospy
import tf2_ros as tf2
import tf_conversions.posemath as pm
import cv2
from tensorflow.python.platform import flags
from costar_hyper import grasp_utilities
from costar_hyper import cornell_grasp_train
from costar_hyper import grasp_model
from costar_hyper import block_stacking_reader
from costar_hyper import grasp_metrics
from threading import Lock
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ctp_integration.ros_geometry import pose_to_vec_quat_list
from costar_task_plan.robotics.workshop import UR5_C_MODEL_CONFIG
from skimage.transform import resize

FLAGS = flags.FLAGS


class CostarHyperPosePredictor(object):

    def __init__(
            self,
            label_features_to_extract=None,
            data_features_to_extract=None,
            total_actions_available=41,
            feature_combo_name='image_preprocessed',
            problem_name='semantic_grasp_regression',
            load_weights=None,
            load_hyperparams=None,
            top='classification',
            robot_config=UR5_C_MODEL_CONFIG,
            tf_buffer=None,
            tf_listener=None,
            image_shape=(224, 224, 3),
            verbose=0):

        self.total_actions_available = total_actions_available
        self.label_features_to_extract = label_features_to_extract
        self.data_features_to_extract = data_features_to_extract
        self.need_clear_view_rgb_img = True
        self.clear_view_rgb_img = None
        self._initialize_hypertree_model_for_inference(
                feature_combo_name=feature_combo_name,
                problem_name=problem_name,
                load_weights=load_weights,
                load_hyperparams=load_hyperparams,
                top=top)
        self.verbose = verbose
        self.mutex = Lock()
        self._initialize_ros(robot_config, tf_buffer, tf_listener)
        self.image_shape = image_shape

    def _initialize_hypertree_model_for_inference(
            self,
            feature_combo_name='image_preprocessed',
            problem_name='semantic_grasp_regression',
            load_weights=None,
            load_hyperparams=None,
            top='classification'):

        if load_weights is None:
            load_weights = FLAGS.load_weights
        if load_hyperparams is None:
            load_hyperparams = FLAGS.load_hyperparams

        if load_weights is None or load_weights == '':
            raise ValueError('A weights file must be specified with: --load_weights path/to/weights.h5f')
        if load_hyperparams is None or load_hyperparams == '':
            raise ValueError('A hyperparams file must be specified with: --load_hyperparams path/to/hyperparams.json')

        # load hyperparams from a file
        hyperparams = grasp_utilities.load_hyperparams_json(
            FLAGS.load_hyperparams, FLAGS.fine_tuning, FLAGS.learning_rate,
            feature_combo_name=feature_combo_name)

        [image_shapes, vector_shapes, data_features, model_name,
         monitor_loss_name, label_features, monitor_metric_name,
         loss, metrics, classes, success_only] = cornell_grasp_train.choose_features_and_metrics(feature_combo_name, problem_name)

        # TODO(ahundt) may need to strip out hyperparams that don't affect the model

        model = grasp_model.choose_hypertree_model(
            image_shapes=image_shapes,
            vector_shapes=vector_shapes,
            top=top,
            classes=classes,
            **hyperparams)

        model.load_weights(load_weights)
        # we don't use the optimizer, so just choose a default
        model.compile(
            optimizer='sgd',
            loss=loss,
            metrics=metrics)

        self.model = model

    def _initialize_ros(self, robot_config, tf_buffer, tf_listener):
        if tf_buffer is None:
            self.tf_buffer = tf2.Buffer()
        else:
            self.tf_buffer = tf_buffer
        if tf_listener is None:
            self.tf_listener = tf2.TransformListener(self.tf_buffer)
        else:
            self.tf_listener = tf_listener

        # http://wiki.ros.org/depth_image_proc
        # http://www.ros.org/reps/rep-0118.html
        # http://wiki.ros.org/rgbd_launch
        # we will be getting 16 bit integer values in milimeters
        self.rgb_topic = "/camera/rgb/image_rect_color"
        # raw means it is in the format provided by the openi drivers, 16 bit int
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect"
        self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
        self._bridge = CvBridge()
        self.base_link = "base_link"
        self.rgb_time = None
        self.ee_frame = robot_config['end_link']

    def _rgbCb(self, msg):
        if msg is None:
            rospy.logwarn("_rgbCb: msg is None !!!!!!!!!")
        try:
            # max out at 10 hz assuming 30hz data source
            if msg.header.seq % 3 == 0:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
                # decode the data, this will take some time

                # rospy.loginfo('rgb color cv_image shape: ' + str(cv_image.shape) + ' depth sequence number: ' + str(msg.header.seq))
                # print('rgb color cv_image shape: ' + str(cv_image.shape))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                # Convert image to numpy format
                rgb_img = np.asarray(cv_image, dtype=np.uint8)
                # Resize to match what is expected of the neural network
                if self.image_shape is not None:
                    rgb_img = resize(rgb_img, self.image_shape, mode='constant', preserve_range=True)

                with self.mutex:
                    self.rgb_time = msg.header.stamp
                    self.rgb_img = rgb_img
                    if self.need_clear_view_rgb_img or self.clear_view_rgb_img is None:
                        self.clear_view_rgb_img = rgb_img
                        self.clear_view_rgb_time = msg.header.stamp
                        self.need_clear_view_rgb_img = False

        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def _actionCb(self, msg):
        """ Get the list of actions, and encode the current action for the prediction step.
        """
        if msg is None:
            rospy.logwarn("_actionCb")
        else:
            action_labels = []
            # the index of the current action in the
            # TODO(ahundt) set up a message to initialize & update the real action labels, this is not done yet
            action_index = 1
            if (self.data_features_to_extract is not None and
                    'image_0_image_n_vec_xyz_aaxyz_nsc_15' in self.data_features_to_extract):
                # normalized floating point encoding of action vector
                # from 0 to 1 in a single float which still becomes
                # a 2d array of dimension batch_size x 1
                action = [float(action_index / self.total_actions_available)]
                action_labels.append(action)
            else:
                # generate the action label one-hot encoding
                action = np.zeros(self.total_actions_available)
                action[action_index] = 1
                action_labels.append(action)
            with self.mutex:
                self.action_labels = action_labels

    def _clearViewCb(self, msg):
        """ We have a clear view, save the next image as an update to the clear view image.
        """
        with self.mutex:
            self.need_clear_view_rgb_img = True

    def __call__(self):

        local_time = rospy.Time.now()
        # this will get the latest available time
        latest_available_time_lookup = rospy.Time(0)

        ##### BEGIN MUTEX
        with self.mutex:
            # get the time for this data sample
            if self.rgb_time is not None:
                t = self.rgb_time
            else:
                t = local_time

        have_data = False
        # how many times have we tried to get the transforms
        attempts = 0
        max_attempts = 10
        # the number attempts that should
        # use the backup timestamps
        backup_timestamp_attempts = 4
        while not have_data:
            try:
                ee_pose = self.tf_buffer.lookup_transform(self.base_link, self.ee_frame, t)

                have_data = True
            except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                rospy.logwarn_throttle(
                    10.0,
                    'CostarHyperPosePredictor transform lookup Failed: %s to %s,'
                    ' at image time: %s and local time: %s '
                    '\nNote: This message may print >1000x less often than the problem occurs.' %
                    (self.base_link, self.ee_frame, str(t), str(latest_available_time_lookup)))

                have_data = False
                attempts += 1
                # rospy.sleep(0.0)
                if attempts > max_attempts - backup_timestamp_attempts:
                    rospy.logwarn_throttle(
                        10.0,
                        'CostarHyperPosePredictor failed to use the rgb image rosmsg timestamp, '
                        'trying latest available time as backup. '
                        'Note: This message may print >1000x less often than the problem occurs.')
                    # try the backup timestamp even though it will be less accurate
                    t = latest_available_time_lookup
                if attempts > max_attempts:
                    # Could not look up one of the transforms -- either could
                    # not look up camera, endpoint, or object.
                    raise e

        ee_xyz_quat = pose_to_vec_quat_list(ee_pose)
        with self.mutex:
            rgb_images = [self.rgb_img]
            clear_view_rgb_images = [self.clear_view_rgb_img]
            action_labels = np.copy(self.action_labels)

        # encode the prediction information
        X = block_stacking_reader.encode_action_and_images(
                self.data_features_to_extract,
                poses=[ee_xyz_quat],
                action_labels=action_labels,
                init_images=clear_view_rgb_images,
                current_images=rgb_images)

        predictions = self.model.predict_on_batch(X)
        prediction_xyz_qxyzw = grasp_metrics.decode_xyz_aaxyz_nsc_to_xyz_qxyzw(predictions[0])

        # prediction_kdl = kdl.Frame(
        #         kdl.Rotation.Quaternion(prediction_xyz_qxyzw[3], prediction_xyz_qxyzw[4], prediction_xyz_qxyzw[5], prediction_xyz_qxyzw[6]),
        #         kdl.Vector(prediction_xyz_qxyzw[0], prediction_xyz_qxyzw[1], prediction_xyz_qxyzw[2]))
        return prediction_xyz_qxyzw




def main(_):
    """ Main function, starts predictor.
    """
    print('main')
    predictor = CostarHyperPosePredictor()

    while True:
        prediction_xyz_qxyzw = predictor()
        # TODO(ahundt) broadcast prediction on TF.

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)