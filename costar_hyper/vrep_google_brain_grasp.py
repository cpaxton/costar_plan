# -*- coding: utf-8 -*-
"""Code for visualizing grasp attempt examples from the google brain robotics grasping dataset.

https://sites.google.com/site/brainrobotdata/home/grasping-dataset

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0
"""
import vrep_grasp
import os
import errno
import traceback

import numpy as np
import six  # compatibility between python 2 + 3 = six
import matplotlib.pyplot as plt

try:
    import vrep
except Exception as e:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in PYTHONPATH folder relative to this file,')
    print ('or appropriately adjust the file "vrep.py. Also follow the"')
    print ('ReadMe.txt in the vrep remote API folder')
    print ('--------------------------------------------------------------')
    print ('')
    raise e

import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from keras.utils import get_file
from ply import write_xyz_rgb_as_ply
from PIL import Image

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from grasp_dataset import GraspDataset
import grasp_geometry
import grasp_geometry_tf
from grasp_train import GraspTrain
from grasp_train import choose_make_model_fn
from depth_image_encoding import ClipFloatValues
from depth_image_encoding import FloatArrayToRgbImage
from depth_image_encoding import FloatArrayToRawRGB
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import img_as_uint
from skimage.color import grey2rgb

try:
    import eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
    import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
except ImportError:
    print('eigen and sva python modules are not available. To install run the script at:'
          'https://github.com/ahundt/robotics_setup/blob/master/robotics_tasks.sh'
          'or follow the instructions at https://github.com/jrl-umi3218/Eigen3ToPython'
          'and https://github.com/jrl-umi3218/SpaceVecAlg. '
          'When you build the modules make sure python bindings are enabled.')

from vrep_grasp import VREPGraspVisualization

# the following line is needed for tf versions before 1.5
# flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS


class VREPGoogleBrainGraspVisualization(vrep_grasp.VREPGraspVisualization):
    """ Visualize the google brain robot data grasp dataset in the V-REP robot simulator.
    """

    def __init__(self):
        """Start the connection to the remote V-REP simulation

           Once initialized, call visualize().
        """
        super(VREPGoogleBrainGraspVisualization, self).__init__()

    def visualize_tensorflow(self, tf_session=None, dataset=FLAGS.grasp_dataset, batch_size=1, parent_name=FLAGS.vrepParentName,
                             visualization_dir=FLAGS.visualization_dir, verbose=0):
        """Visualize one dataset in V-REP from performing all preprocessing in tensorflow.

            tensorflow loads the raw data from the dataset and also calculates all
            features before they are rendered with vrep via python,
        """
        if tf_session is None:
            raise ValueError('Google Brain Dataset requires tf_session to be defined, such as with session = tf.Session().')
        batch_size = 1
        grasp_dataset_object = GraspDataset(dataset=dataset)
        if FLAGS.vrepVisualizePredictions is True:
            make_model_fn = choose_make_model_fn()
            gt = GraspTrain()
            (pred_model, pregrasp_op_batch, grasp_step_op_batch,
             simplified_grasp_command_op_batch,
             grasp_success_op_batch, feature_op_dicts,
             features_complete_list,
             time_ordered_feature_name_dict,
             num_samples) = gt.get_compiled_model(
                dataset=grasp_dataset_object,
                make_model_fn=make_model_fn)
        else:
            (feature_op_dicts, features_complete_list,
             time_ordered_feature_name_dict,
             num_samples) = grasp_dataset_object.get_training_dictionaries(batch_size=batch_size)

        if verbose > 0:
            print('visualize_tensorflow.time_ordered_feature_name_dict', time_ordered_feature_name_dict, 'feature_op_dicts:', feature_op_dicts)
        tf_session.run(tf.global_variables_initializer())

        error_code, parent_handle = vrep.vrep.simxGetObjectHandle(self.client_id, parent_name, vrep.vrep.simx_opmode_blocking)
        if error_code is -1:
            parent_handle = -1
            print('could not find object with the specified name, so putting objects in world frame:', parent_name)

        for attempt_num in tqdm(range(num_samples / batch_size), desc='dataset'):
            attempt_num_string = 'attempt_' + str(attempt_num).zfill(4) + '_'
            vrep.visualization.vrepPrint(self.client_id, 'dataset_' + dataset + '_' + attempt_num_string + 'starting')
            # use fetches arguments to get tensors explicitly
            if FLAGS.vrepVisualizePredictions is True:
                # x should be passed through internal calls
                predictions, _, output_features_dicts = pred_model.predict_on_batch(x=None)
                output_features_dicts = [(output_features_dicts[0], output_features_dicts[1])]
                print(predictions.shape)
            else:
                # batch shize should actually always be 1 for this visualization
                predictions = None
                output_features_dicts = tf_session.run(feature_op_dicts)
            # reorganize is grasp attempt so it is easy to walk through
            [time_ordered_feature_data_dict] = grasp_dataset_object.to_tensors(output_features_dicts, time_ordered_feature_name_dict)
            # features_dict_np contains fixed dimension features, sequence_dict_np contains variable length sequences of data
            # We're assuming the batch size is 1, which is why there are only two elements in the list.
            [(features_dict_np, sequence_dict_np)] = output_features_dicts

            if (attempt_num > FLAGS.vrepVisualizeGraspAttempt_max and not FLAGS.vrepVisualizeGraspAttempt_max == -1):
                # stop running if we've gone through all the relevant attempts.
                break

            # check if this attempt is one the user requested, if not get the next one
            if not ((attempt_num >= FLAGS.vrepVisualizeGraspAttempt_min or FLAGS.vrepVisualizeGraspAttempt_min == -1) and
                    (attempt_num < FLAGS.vrepVisualizeGraspAttempt_max or FLAGS.vrepVisualizeGraspAttempt_max == -1)):
                continue

            # Add the camera frame transform and all transforms that start at the base
            for name, value in six.iteritems(time_ordered_feature_data_dict):
                print('feature: ' + name)
                if 'base_T_camera/vec_quat_7' in name:
                    base_to_camera_vec_quat_7 = value[0]
                    base_T_camera_handle = vrep.visualization.create_dummy(
                        self.client_id, 'base_T_camera', base_to_camera_vec_quat_7,
                        parent_handle, operation_mode=vrep.vrep.simx_opmode_blocking)
                elif 'base_T' in name and 'vec_quat_7' in name:
                    for i, base_transform in enumerate(value):
                        vrep.visualization.create_dummy(
                            self.client_id, str(i).zfill(2) + '_' + '_'.join(name.split('/')[-2:]),
                            base_transform, parent_handle, operation_mode=vrep.vrep.simx_opmode_blocking)

            # Add all transforms that start at the camera frame
            camera_to_depth_name = None
            depth_to_ee_name = None
            for name, value in six.iteritems(time_ordered_feature_data_dict):
                if 'camera_T_depth_pixel/vec_quat_7' in name:
                    # save name to draw separately
                    camera_to_depth_name = name
                elif 'endeffector_clear_view_depth_pixel_T_endeffector/vec_quat_7' in name:
                    depth_to_ee_name = name
                elif 'camera_T' in name and 'vec_quat_7' in name:
                    for i, transform in enumerate(value):
                        vrep.visualization.create_dummy(
                            self.client_id, str(i).zfill(2) + '_' + '_'.join(name.split('/')[-2:]),
                            transform, base_T_camera_handle, operation_mode=vrep.vrep.simx_opmode_blocking)
                elif 'depth_pixel_T_endeffector_final/image_coordinate/yx_2' in name:
                    final_coordinate_name = name
                elif 'endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/yx_2' in name:
                    current_coordinate_name = name

            if camera_to_depth_name is not None and depth_to_ee_name is not None:
                if FLAGS.vrepVisualizeSurfaceRelativeTransformLines:
                    # if visualizing lines, get camera position in world frame
                    ret, camera_world_position = vrep.vrep.simxGetObjectPosition(
                        self.client_id, base_T_camera_handle, -1, vrep.vrep.simx_opmode_oneshot_wait)
                # display all camera to depth and depth to end effector transforms
                for i, (ctd_vq7, dtee_vq7) in enumerate(zip(time_ordered_feature_data_dict[camera_to_depth_name],
                                                            time_ordered_feature_data_dict[depth_to_ee_name])):
                    time_step_name = str(i).zfill(2) + '_'
                    depth_point_dummy_handle = vrep.visualization.create_dummy(
                        self.client_id, time_step_name + str(i).zfill(2) + '_' + '_'.join(camera_to_depth_name.split('/')[-2:]),
                        ctd_vq7, base_T_camera_handle)
                    surface_relative_transform_dummy_handle = vrep.visualization.create_dummy(
                        self.client_id, time_step_name + 'depth_point_T_endeffector',
                        dtee_vq7, depth_point_dummy_handle)
                    if FLAGS.vrepVisualizeSurfaceRelativeTransformLines:
                        # Draw lines from the camera through the gripper pose to the depth pixel in the clear view frame used for surface transforms
                        ret, depth_world_position = vrep.vrep.simxGetObjectPosition(
                            self.client_id, depth_point_dummy_handle, -1, vrep.vrep.simx_opmode_oneshot_wait)
                        ret, surface_relative_gripper_world_position = vrep.vrep.simxGetObjectPosition(
                            self.client_id, surface_relative_transform_dummy_handle, -1, vrep.vrep.simx_opmode_oneshot_wait)
                        vrep.visualization.drawLines(self.client_id, 'camera_to_depth_lines',
                                  np.append(camera_world_position, depth_world_position),
                                  base_T_camera_handle, operation_mode=vrep.vrep.simx_opmode_blocking)
                        vrep.visualization.drawLines(self.client_id, 'camera_to_depth_lines',
                                  np.append(depth_world_position, surface_relative_gripper_world_position),
                                  base_T_camera_handle, operation_mode=vrep.vrep.simx_opmode_blocking)
            # grasp attempt is complete
            vrep.visualization.vrepPrint(self.client_id, attempt_num_string + 'complete, success: ' + str(int(features_dict_np['grasp_success'])))

            # Visualize point clouds
            if FLAGS.vrepVisualizeRGBD:
                # display clear view point cloud
                # show the median filtered or the raw xyz + depth images depending on user selection
                if FLAGS.median_filter:
                    clear_view_depth_image = time_ordered_feature_data_dict['move_to_grasp/time_ordered/clear_view/depth_image/median_filtered'][0]
                else:
                    clear_view_depth_image = time_ordered_feature_data_dict['move_to_grasp/time_ordered/clear_view/depth_image/decoded'][0]
                clear_view_depth_image = np.copy(clear_view_depth_image)
                vrep.visualization.create_point_cloud(
                    self.client_id, 'clear_view_cloud',
                    transform=base_to_camera_vec_quat_7,
                    depth_image=clear_view_depth_image,
                    color_image=time_ordered_feature_data_dict['move_to_grasp/time_ordered/clear_view/rgb_image/decoded'][0],
                    parent_handle=parent_handle,
                    rgb_sensor_display_name='kcam_rgb_clear_view',
                    depth_sensor_display_name='kcam_depth_clear_view',
                    point_cloud=time_ordered_feature_data_dict['move_to_grasp/time_ordered/clear_view/xyz_image/decoded'][0])

                # display the close gripper step rgb image
                close_gripper_rgb_image = features_dict_np['gripper/image/decoded']
                # TODO(ahundt) make sure rot180 + fliplr is applied upstream in the dataset and to the depth images
                # gripper/image/decoded is unusual because there is no depth image and the orientation is rotated 180 degrees from the others
                # it might also only be available captured in some of the more recent datasets.
                vrep.visualization.set_vision_sensor_image(
                    self.client_id, 'kcam_rgb_close_gripper', close_gripper_rgb_image, convert=FLAGS.vrepVisualizeRGBFormat)

                # Walk through all the other images from initial time step to final time step
                rgb_images = time_ordered_feature_data_dict['move_to_grasp/time_ordered/rgb_image/decoded']
                # show the median filtered or the raw xyz + depth images depending on user selection
                if FLAGS.median_filter:
                    depth_images = time_ordered_feature_data_dict['move_to_grasp/time_ordered/depth_image/median_filtered']
                    xyz_images = time_ordered_feature_data_dict['move_to_grasp/time_ordered/xyz_image/median_filtered']
                else:
                    depth_images = time_ordered_feature_data_dict['move_to_grasp/time_ordered/depth_image/decoded']
                    xyz_images = time_ordered_feature_data_dict['move_to_grasp/time_ordered/xyz_image/decoded']
                current_coordinates = time_ordered_feature_data_dict[current_coordinate_name]
                final_coordinates = time_ordered_feature_data_dict[final_coordinate_name]

                # workaround for when predictions aren't enabled
                if predictions is None:
                    preds = [None] * len(rgb_images)
                else:
                    preds = predictions

                # get crop offset and size from dict
                crop_offset = features_dict_np['random_crop_offset'] # 3-dim
                crop_size = features_dict_np['rgb_random_crop_dimensions'] # 3-dim
                # create color map
                color_map = plt.cm.RdBu
                # Display each point cloud
                for img_num, (rgb, depth, xyz, current_coordinate,
                              final_coordinate, prediction) in enumerate(zip(rgb_images, depth_images,
                                                                             xyz_images, current_coordinates,
                                                                             final_coordinates, preds)):
                    # depth = grasp_geometry.draw_circle(grasp_geometry.draw_circle(depth, current_coordinate), final_coordinate)
                    rgb = grasp_geometry.draw_circle(grasp_geometry.draw_circle(rgb, current_coordinate,
                                                     color=(0, 255, 255)), final_coordinate, color=(255, 255, 0))
                    print('rgb_image_dtype:  ' + str(rgb.dtype))
                    print('rgb_image_shape:  ' + str(rgb.shape))
                    vrep.visualization.create_point_cloud(
                        self.client_id, 'current_point_cloud',
                        transform=base_to_camera_vec_quat_7,
                        depth_image=depth,
                        color_image=np.copy(rgb),
                        parent_handle=parent_handle,
                        rgb_sensor_display_name='kcam_rgb_current',
                        depth_sensor_display_name='kcam_depth_current',
                        point_cloud=xyz)

                    if prediction is not None:
                        print('original prediction_shape: ', prediction.shape, 'pred_max: ', np.max(prediction), ' pred_min:', np.min(prediction))
                        prediction = prediction - 0.5
                        prediction = prediction * 100
                        prediction = prediction + 0.5
                        print('scaled prediction_shape: ', prediction.shape, 'pred_max: ', np.max(prediction), ' pred_min:', np.min(prediction))
                        prediction = np.squeeze(prediction)
                        # if FLAGS.vrepVisualizeMatPlotLib:
                        #     plt.imshow(prediction)
                        #     plt.show()
                        print('grey_prediction_shape: ', prediction.shape, 'pred_max: ', np.max(prediction), ' pred_min:', np.min(prediction))
                        # To resize a one-channel image, need to squeeze singleton dim first
                        fullsize_prediction = vrep.visualization.restore_cropped(prediction, crop_size, crop_offset, rgb.shape)
                        # return RGBA, cut alpha channel
                        rgb_prediction = ((color_map(fullsize_prediction)*255).astype('uint8'))[:,:,:-1]
                        if FLAGS.vrepVisualizeMatPlotLib:
                            plt.imshow(rgb_prediction)
                            plt.show()
                        rgb_prediction = grasp_geometry.draw_circle(grasp_geometry.draw_circle(rgb_prediction, current_coordinate,
                                                                    color=(0, 255, 255)), final_coordinate, color=(255, 255, 0))
                        # Adjust the depth by the delta depth offset for visualization
                        gdtf_delta_depth_sin_cos_3 = time_ordered_feature_data_dict['move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3']
                        depth_pred_offset = np.copy(clear_view_depth_image) + gdtf_delta_depth_sin_cos_3[img_num][0]
                        vrep.visualization.create_point_cloud(
                            self.client_id, 'prediction_point_cloud',
                            transform=base_to_camera_vec_quat_7,
                            depth_image=depth_pred_offset,
                            color_image=rgb_prediction,
                            parent_handle=parent_handle,
                            rgb_sensor_display_name='kcam_rgb_prediction',
                            depth_sensor_display_name='kcam_depth_prediction',
                            point_cloud=xyz)

    def visualize_python(self, tf_session, dataset=FLAGS.grasp_dataset, batch_size=1, parent_name=FLAGS.vrepParentName,
                         visualization_dir=FLAGS.visualization_dir):
        """Visualize one dataset in V-REP from raw dataset features, performing all preprocessing manually in this function.
        """
        grasp_dataset_object = GraspDataset(dataset=dataset)
        batch_size = 1
        feature_op_dicts, features_complete_list, num_samples = grasp_dataset_object._get_simple_parallel_dataset_ops(
            batch_size=batch_size)

        tf_session.run(tf.global_variables_initializer())

        error_code, parent_handle = vrep.vrep.simxGetObjectHandle(self.client_id, parent_name, vrep.vrep.simx_opmode_blocking)
        if error_code is -1:
            parent_handle = -1
            print('could not find object with the specified name, so putting objects in world frame:', parent_name)

        features_complete_list_time_ordered = grasp_dataset_object.get_time_ordered_features(features_complete_list)
        print('fixed features time ordered: ', features_complete_list_time_ordered)

        clear_frame_depth_image_feature = grasp_dataset_object.get_time_ordered_features(
            features_complete_list_time_ordered,
            feature_type='depth_image/decoded',
            step='view_clear_scene'
        )[0]

        clear_frame_rgb_image_feature = grasp_dataset_object.get_time_ordered_features(
            features_complete_list_time_ordered,
            feature_type='/image/decoded',
            step='view_clear_scene'
        )[0]

        depth_image_features = grasp_dataset_object.get_time_ordered_features(
            features_complete_list_time_ordered,
            feature_type='depth_image/decoded',
            step='move_to_grasp'
        )

        rgb_image_features = grasp_dataset_object.get_time_ordered_features(
            features_complete_list_time_ordered,
            feature_type='/image/decoded',
            step='move_to_grasp'
        )

        grasp_success_feature_name = 'grasp_success'

        for attempt_num in range(num_samples / batch_size):
            # load data from the next grasp attempt
            if FLAGS.vrepVisualizeDilation:
                depth_image_tensor = feature_op_dicts[0][0][clear_frame_depth_image_feature]
                dilated_tensor = tf.nn.dilation2d(input=depth_image_tensor,
                                                  filter=tf.zeros([50, 50, 1]),
                                                  strides=[1, 1, 1, 1],
                                                  rates=[1, 10, 10, 1],
                                                  padding='VALID')
                feature_op_dicts[0][0][clear_frame_depth_image_feature] = dilated_tensor
            output_features_dict = tf_session.run(feature_op_dicts)
            if ((attempt_num >= FLAGS.vrepVisualizeGraspAttempt_min or FLAGS.vrepVisualizeGraspAttempt_min == -1) and
                    (attempt_num < FLAGS.vrepVisualizeGraspAttempt_max or FLAGS.vrepVisualizeGraspAttempt_max == -1)):
                for features_dict_np, sequence_dict_np in output_features_dict:
                    # Visualize the data from a single grasp attempt
                    self._visualize_one_grasp_attempt(
                        grasp_dataset_object, features_complete_list, features_dict_np, parent_handle,
                        clear_frame_depth_image_feature,
                        clear_frame_rgb_image_feature,
                        depth_image_features,
                        rgb_image_features,
                        grasp_success_feature_name,
                        dataset_name=dataset,
                        attempt_num=attempt_num)
            if (attempt_num > FLAGS.vrepVisualizeGraspAttempt_max and not FLAGS.vrepVisualizeGraspAttempt_max == -1):
                # stop running if we've gone through all the relevant attempts.
                break

    def _visualize_one_grasp_attempt(self, grasp_dataset_object, features_complete_list, features_dict_np, parent_handle,
                                     clear_frame_depth_image_feature,
                                     clear_frame_rgb_image_feature,
                                     depth_image_features,
                                     rgb_image_features,
                                     grasp_success_feature_name,
                                     dataset_name=FLAGS.grasp_dataset,
                                     attempt_num=0,
                                     grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
                                     grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
                                     visualization_dir=FLAGS.visualization_dir,
                                     vrepDebugMode=FLAGS.vrepDebugMode,
                                     vrepVisualizeRGBD=FLAGS.vrepVisualizeRGBD,
                                     vrepVisualizeSurfaceRelativeTransform=FLAGS.vrepVisualizeSurfaceRelativeTransform):
        """Take an extracted grasp attempt tfrecord numpy dictionary and visualize it in vrep

        # Params

        parent_handle: the frame in which to display transforms, defaults to base frame of 'LBR_iiwa_14_R820'

        It is important to note that both V-REP and the grasp dataset use the xyzw quaternion format.
        """
        # workaround for V-REP bug where handles may not be correctly deleted
        res, lines_handle = vrep.vrep.simxGetObjectHandle(self.client_id, 'camera_to_depth_lines', vrep.vrep.simx_opmode_oneshot_wait)
        print('lines handle:', lines_handle)
        # if res == vrep.vrep.simx_return_ok and lines_handle is not -1:
        #     vrep.vrep.simxRemoveObject(self.client_id, lines_handle, vrep.vrep.simx_opmode_oneshot)
        # grasp attempt string for showing status
        attempt_num_string = 'attempt_' + str(attempt_num).zfill(4) + '_'
        vrep.visualization.vrepPrint(self.client_id, attempt_num_string + ' success: ' + str(int(features_dict_np[grasp_success_feature_name])) + ' has started')
        # get param strings for every single gripper position
        base_to_endeffector_transforms = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            # feature_type='transforms/base_T_endeffector/vec_quat_7')  # display only commanded transforms
            # feature_type='vec_quat_7')  # display all transforms
            feature_type='reached_pose',
            step='move_to_grasp')
        print(features_complete_list)
        print(base_to_endeffector_transforms)
        camera_to_base_transform_name = 'camera/transforms/camera_T_base/matrix44'
        camera_intrinsics_name = 'camera/intrinsics/matrix33'

        # Create repeated values for the final grasp position where the gripper closed
        base_T_endeffector_final_close_gripper_name = base_to_endeffector_transforms[-1]
        base_T_endeffector_final_close_gripper = features_dict_np[base_T_endeffector_final_close_gripper_name]

        # get the camera intrinsics matrix and camera extrinsics matrix
        camera_intrinsics_matrix = features_dict_np[camera_intrinsics_name]
        camera_to_base_4x4matrix = features_dict_np[camera_to_base_transform_name]
        if 'print_transform' in vrepDebugMode:
            print('camera/transforms/camera_T_base/matrix44: \n', camera_to_base_4x4matrix)
        camera_to_base_vec_quat_7 = grasp_geometry.matrix_to_vector_quaternion_array(camera_to_base_4x4matrix)
        # verify that another transform path gets the same result
        camera_T_base_ptrans = grasp_geometry.matrix_to_ptransform(camera_to_base_4x4matrix)
        camera_to_base_vec_quat_7_ptransform_conversion_test = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_base_ptrans)
        assert(grasp_geometry.vector_quaternion_arrays_allclose(camera_to_base_vec_quat_7, camera_to_base_vec_quat_7_ptransform_conversion_test))
        # verify that another transform path gets the same result
        base_T_camera_ptrans = camera_T_base_ptrans.inv()
        base_to_camera_vec_quat_7 = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_camera_ptrans)
        base_T_camera_handle = vrep.visualization.create_dummy(self.client_id, 'base_T_camera', base_to_camera_vec_quat_7, parent_handle)
        camera_T_base_handle = vrep.visualization.create_dummy(self.client_id, 'camera_T_base', camera_to_base_vec_quat_7, base_T_camera_handle)

        # TODO(ahundt) check that ptransform times its inverse is identity, or very close to it
        identity = sva.PTransformd.Identity()
        should_be_identity = base_T_camera_ptrans * camera_T_base_ptrans
        # Make sure converting to a ptransform and back to a quaternion generates a sensible transform
        base_to_camera_vec_quat_7_ptransform_conversion_test = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_camera_ptrans)
        vrep.visualization.create_dummy(self.client_id, 'base_to_camera_vec_quat_7_ptransform_conversion_test', base_to_camera_vec_quat_7_ptransform_conversion_test, parent_handle)
        assert(grasp_geometry.vector_quaternion_arrays_allclose(base_to_camera_vec_quat_7, base_to_camera_vec_quat_7_ptransform_conversion_test))

        clear_frame_depth_image = np.squeeze(features_dict_np[clear_frame_depth_image_feature])
        clear_frame_rgb_image = np.squeeze(features_dict_np[clear_frame_rgb_image_feature])
        # Visualize clear view point cloud
        if FLAGS.vrepVisualizeRGBD:
            vrep.visualization.create_point_cloud(
                self.client_id, 'clear_view_cloud',
                depth_image=np.copy(clear_frame_depth_image),
                camera_intrinsics_matrix=camera_intrinsics_matrix,
                transform=base_to_camera_vec_quat_7,
                color_image=clear_frame_rgb_image, parent_handle=parent_handle,
                rgb_sensor_display_name='kcam_rgb_clear_view',
                depth_sensor_display_name='kcam_depth_clear_view')

            close_gripper_rgb_image = features_dict_np['gripper/image/decoded']
            # TODO(ahundt) make sure rot180 + fliplr is applied upstream in the dataset and to the depth images
            # gripper/image/decoded is unusual because there is no depth image and the orientation is rotated 180 degrees from the others
            # it might also only be available captured in some of the more recent datasets.
            cg_rgb = 256 - np.fliplr(close_gripper_rgb_image)
            vrep.visualization.set_vision_sensor_image(self.client_id, 'kcam_rgb_close_gripper', cg_rgb, convert=None)

        # loop through each time step
        for i, base_T_endeffector_vec_quat_feature_name, depth_name, rgb_name in zip(range(len(base_to_endeffector_transforms)),
                                                                                     base_to_endeffector_transforms,
                                                                                     depth_image_features, rgb_image_features):
            # prefix with time step so vrep data visualization is in order
            time_step_name = str(i).zfill(2) + '_'
            # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
            # 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            base_T_endeffector_vec_quat_feature = features_dict_np[base_T_endeffector_vec_quat_feature_name]
            # display the raw base to endeffector feature
            bTe_display_name = time_step_name + base_T_endeffector_vec_quat_feature_name.replace('/', '_')
            bTe_handle = vrep.visualization.create_dummy(self.client_id, bTe_display_name, base_T_endeffector_vec_quat_feature, parent_handle)

            # do the conversion needed for training
            camera_T_endeffector_ptrans, base_T_endeffector_ptrans, base_T_camera_ptrans = grasp_geometry.grasp_dataset_to_ptransform(
                camera_to_base_4x4matrix,
                base_T_endeffector_vec_quat_feature
            )
            # update the camera to base transform so we can visually ensure consistency
            # while this is run above, this second run validates the correctness of grasp_dataset_to_ptransform()
            vrep.visualization.create_dummy(self.client_id, 'camera_T_base_vec_quat_7_ptransform_conversion_test', camera_to_base_vec_quat_7_ptransform_conversion_test, base_T_camera_handle)
            base_T_camera_handle = vrep.visualization.create_dummy(self.client_id, 'base_T_camera', base_to_camera_vec_quat_7, parent_handle)
            camera_T_base_handle = vrep.visualization.create_dummy(self.client_id, 'camera_T_base', camera_to_base_vec_quat_7, base_T_camera_handle)

            # test that the base_T_endeffector -> ptransform -> vec_quat_7 roundtrip returns the same transform
            base_T_endeffector_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_endeffector_ptrans)
            bTe_display_name = time_step_name + 'base_T_endeffector_ptransform_conversion_test_' + base_T_endeffector_vec_quat_feature_name.replace('/', '_')
            vrep.visualization.create_dummy(self.client_id, bTe_display_name, base_T_endeffector_vec_quat, parent_handle)
            assert(grasp_geometry.vector_quaternion_arrays_allclose(base_T_endeffector_vec_quat_feature, base_T_endeffector_vec_quat))

            # verify that another transform path gets the same result
            # camera_to_base_vec_quat_7_ptransform_conversion_test = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_base_ptrans)
            # display_name = time_step_name + 'camera_to_base_vec_quat_7_ptransform_conversion_test'
            # vrep.visualization.create_dummy(self.client_id, display_name, camera_to_base_vec_quat_7_ptransform_conversion_test, parent_handle)

            cTe_display_name = time_step_name + 'camera_T_endeffector_' + base_T_endeffector_vec_quat_feature_name.replace(
                '/transforms/base_T_endeffector/vec_quat_7', '').replace('/', '_')
            cTe_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_endeffector_ptrans)
            vrep.visualization.create_dummy(self.client_id, cTe_display_name, cTe_vec_quat, base_T_camera_handle)

            # format the dummy string nicely for display
            transform_display_name = time_step_name + base_T_endeffector_vec_quat_feature_name.replace(
                '/transforms/base_T_endeffector/vec_quat_7', '').replace('/', '_')
            if 'print_transform' in vrepDebugMode:
                print(base_T_endeffector_vec_quat_feature_name, transform_display_name, base_T_endeffector_vec_quat_feature)
            # display the gripper pose
            vrep.visualization.create_dummy(self.client_id, transform_display_name, base_T_endeffector_vec_quat_feature, parent_handle)
            # Perform some consistency checks based on the above
            assert(grasp_geometry.vector_quaternion_arrays_allclose(base_T_endeffector_vec_quat, base_T_endeffector_vec_quat_feature))

            #############################
            # get the transform from the current endeffector pose to the final
            transform_display_name = time_step_name + 'current_T_end'
            current_to_end = grasp_geometry.current_endeffector_to_final_endeffector_feature(
                base_T_endeffector_vec_quat_feature, base_T_endeffector_final_close_gripper, feature_type='vec_quat_7')
            current_to_end_ptransform = grasp_geometry.vector_quaternion_array_to_ptransform(current_to_end)
            current_to_end_rotation = current_to_end_ptransform.rotation()
            theta = grasp_geometry.grasp_dataset_rotation_to_theta(current_to_end_rotation, verbose=0)
            # compare these printed theta values in the visualization to what is documented in
            # see grasp_dataset_rotation_to_theta() this printout will let you verify that
            # theta is estimated correctly for training.
            if 'print_transform' in vrepDebugMode:
                print('current to end estimated theta ', transform_display_name, theta)
            vrep.visualization.create_dummy(self.client_id, transform_display_name, current_to_end, bTe_handle)

            # TODO(ahundt) check that transform from end step to itself should be identity, or very close to it
            # if base_T_endeffector_final_close_gripper_name == base_T_endeffector_vec_quat_feature_name:
            #     transform from end step to itself should be identity.
            #     identity = sva.PTransformd.Identity()
            #     assert(identity == current_to_end)

            # Use the function that gets all the features at once for comparison to manusal computation
            [gdtf_current_base_T_camera_vec_quat_7_array,
             gdtf_eectf_vec_quat_7_array,
             gdtf_camera_T_endeffector_current_vec_quat_7_array,
             gdtf_camera_T_depth_pixel_current_vec_quat_7_array,
             gdtf_camera_T_endeffector_final_vec_quat_7_array,
             gdtf_camera_T_depth_pixel_final_vec_quat_7_array,
             gdtf_depth_pixel_T_endeffector_current_vec_quat_7_array,
             gdtf_image_coordinate_current,
             gdtf_depth_pixel_T_endeffector_final_vec_quat_7_array,
             gdtf_image_coordinatee_final,
             gdtf_sin_cos_2,
             gdtf_vec_sin_cos_5,
             gdtf_delta_depth_sin_cos_3,
             gdtf_delta_depth_quat_5] = grasp_geometry.grasp_dataset_to_transforms_and_features(
                    clear_frame_depth_image,
                    camera_intrinsics_matrix,
                    camera_to_base_4x4matrix,
                    base_T_endeffector_vec_quat_feature,
                    base_T_endeffector_final_close_gripper)
            if 'print_transform' in vrepDebugMode:
                print('gdtf_current_base_T_camera_vec_quat_7_array', gdtf_current_base_T_camera_vec_quat_7_array)
                print('gdtf_eectf_vec_quat_7_array', gdtf_eectf_vec_quat_7_array)
                print('gdtf_camera_T_endeffector_current_vec_quat_7_array', gdtf_camera_T_endeffector_current_vec_quat_7_array)
                print('gdtf_camera_T_endeffector_final_vec_quat_7_array', gdtf_camera_T_endeffector_final_vec_quat_7_array)
                print('gdtf_camera_T_depth_pixel_final_vec_quat_7_array', gdtf_camera_T_depth_pixel_final_vec_quat_7_array)
                print('gdtf_depth_pixel_T_endeffector_current_vec_quat_7_array', gdtf_depth_pixel_T_endeffector_current_vec_quat_7_array)
                print('gdtf_image_coordinate_current', gdtf_image_coordinate_current)
                print('gdtf_depth_pixel_T_endeffector_final_vec_quat_7_array', gdtf_depth_pixel_T_endeffector_final_vec_quat_7_array)
                print('gdtf_image_coordinatee_final', gdtf_image_coordinatee_final)
                print('gdtf_sin_cos_2', gdtf_sin_cos_2)
                print('gdtf_vec_sin_cos_5', gdtf_vec_sin_cos_5)
                print('gdtf_delta_depth_sin_cos_3', gdtf_delta_depth_sin_cos_3)
                print('gdtf_delta_depth_quat_5', gdtf_delta_depth_quat_5)
                print('gdtf_vec_sin_cos_5', gdtf_vec_sin_cos_5)
            #############################
            # visualize surface relative transform
            if vrepVisualizeSurfaceRelativeTransform:
                ee_cloud_point, ee_image_coordinate = grasp_geometry.endeffector_image_coordinate_and_cloud_point(
                    clear_frame_depth_image, camera_intrinsics_matrix, camera_T_endeffector_ptrans)

                # Create a dummy for the key depth point and display it
                depth_point_dummy_ptrans = grasp_geometry.vector_to_ptransform(ee_cloud_point)
                depth_point_display_name = time_step_name + 'depth_point'
                print(depth_point_display_name + ': ' + str(ee_cloud_point) + ' end_effector image coordinate: ' + str(ee_image_coordinate))
                depth_point_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(depth_point_dummy_ptrans)
                depth_point_dummy_handle = vrep.visualization.create_dummy(self.client_id, depth_point_display_name, depth_point_vec_quat, base_T_camera_handle)
                depth_point_dummy_handle = vrep.visualization.create_dummy(self.client_id, depth_point_display_name + '_gdtf', gdtf_camera_T_depth_pixel_current_vec_quat_7_array, base_T_camera_handle)

                # Get the transform for the gripper relative to the key depth point and display it.
                # Dummy should coincide with the gripper pose if done correctly
                depth_pixel_T_endeffector_final_ptrans, pixel_coordinate_of_endeffector, camera_T_cloud_point_ptrans = grasp_geometry.surface_relative_transform(
                    clear_frame_depth_image, camera_intrinsics_matrix, camera_T_endeffector_ptrans)
                assert np.allclose(pixel_coordinate_of_endeffector, gdtf_image_coordinate_current)
                surface_relative_transform_vq7 = grasp_geometry.ptransform_to_vector_quaternion_array(depth_pixel_T_endeffector_final_ptrans)
                surface_relative_transform_dummy_handle = vrep.visualization.create_dummy(self.client_id, time_step_name + 'depth_point_T_endeffector',
                                                                       surface_relative_transform_vq7,
                                                                       depth_point_dummy_handle)
                assert np.allclose(surface_relative_transform_vq7, gdtf_depth_pixel_T_endeffector_current_vec_quat_7_array)

                if FLAGS.vrepVisualizeSurfaceRelativeTransformLines:
                    # Draw lines from the camera through the gripper pose to the depth pixel in the clear view frame used for surface transforms
                    ret, camera_world_position = vrep.vrep.simxGetObjectPosition(self.client_id, base_T_camera_handle, -1, vrep.vrep.simx_opmode_oneshot_wait)
                    ret, depth_world_position = vrep.vrep.simxGetObjectPosition(self.client_id, depth_point_dummy_handle, -1, vrep.vrep.simx_opmode_oneshot_wait)
                    ret, surface_relative_gripper_world_position = vrep.vrep.simxGetObjectPosition(
                        self.client_id, surface_relative_transform_dummy_handle, -1, vrep.vrep.simx_opmode_oneshot_wait)
                    vrep.visualization.drawLines(self.client_id, 'camera_to_depth_lines', np.append(camera_world_position, depth_world_position), base_T_camera_handle)
                    vrep.visualization.drawLines(self.client_id, 'camera_to_depth_lines', np.append(depth_world_position, surface_relative_gripper_world_position), base_T_camera_handle)

            # only visualize the RGBD point clouds if they are within the user specified range
            if(vrepVisualizeRGBD and (attempt_num >= FLAGS.vrepVisualizeRGBD_min or FLAGS.vrepVisualizeRGBD_min == -1) and
               (attempt_num < FLAGS.vrepVisualizeRGBD_max or FLAGS.vrepVisualizeRGBD_max == -1)):
                self.visualize_rgbd(features_dict_np, rgb_name, depth_name, grasp_sequence_min_time_step, i, grasp_sequence_max_time_step,
                                    camera_intrinsics_matrix, vrepDebugMode, dataset_name, attempt_num, grasp_success_feature_name,
                                    visualization_dir, base_to_camera_vec_quat_7, parent_handle, time_step_name)
            # time step is complete
            vrep.visualization.vrepPrint(self.client_id, attempt_num_string + 'time_step_' + time_step_name + 'complete')
        # grasp attempt is complete
        vrep.visualization.vrepPrint(self.client_id, attempt_num_string + 'complete, success: ' + str(int(features_dict_np[grasp_success_feature_name])))

    def visualize_rgbd(self, features_dict_np, rgb_name, depth_name, grasp_sequence_min_time_step, i,
                       grasp_sequence_max_time_step, camera_intrinsics_matrix, vrepDebugMode, dataset_name, attempt_num,
                       grasp_success_feature_name, visualization_dir, base_to_camera_vec_quat_7, parent_handle,
                       time_step_name):
        """Display rgbd image for a specific time step and generate relevant custom strings
        """

        # TODO(ahundt) move squeeze steps into dataset api if possible
        depth_image_float_format = np.squeeze(features_dict_np[depth_name])
        rgb_image = np.squeeze(features_dict_np[rgb_name])
        # print rgb_name, rgb_image.shape, rgb_image
        if np.count_nonzero(depth_image_float_format) is 0:
            print('WARNING: DEPTH IMAGE IS ALL ZEROS')
        status_string = 'displaying rgb: ' + rgb_name + ' depth: ' + depth_name + ' shape: ' + str(depth_image_float_format.shape)
        print(status_string)
        vrep.visualization.vrepPrint(self.client_id, status_string)
        if ((grasp_sequence_min_time_step is None or i >= grasp_sequence_min_time_step) and
                (grasp_sequence_max_time_step is None or i <= grasp_sequence_max_time_step)):
            # only output one depth image while debugging
            # mp.pyplot.imshow(depth_image_float_format, block=True)
            # print 'plot done'
            if 'fixed_depth' in vrepDebugMode:
                # fixed depth is to help if you're having problems getting your point cloud to display properly
                point_cloud = grasp_geometry.depth_image_to_point_cloud(np.ones(depth_image_float_format.shape), camera_intrinsics_matrix)
            point_cloud_detailed_name = ('point_cloud_' + str(dataset_name) + '_' + str(attempt_num) + '_' + time_step_name +
                                         'rgbd_' + depth_name.replace('/depth_image/decoded', '').replace('/', '_') +
                                         '_success_' + str(int(features_dict_np[grasp_success_feature_name])))
            print(point_cloud_detailed_name)

            path = os.path.join(visualization_dir, point_cloud_detailed_name + '.ply')
            if 'fixed_depth' in vrepDebugMode:
                depth_image_float_format = np.ones(depth_image.shape)

            # Save out Point cloud
            if 'save_ply' not in vrepDebugMode:
                point_cloud_detailed_name = None
            # TODO(ahundt) should displaying all clouds be a configurable option?
            point_cloud_display_name = 'current_point_cloud'
            vrep.visualization.create_point_cloud(
                self.client_id,
                point_cloud_display_name,
                depth_image=depth_image_float_format,
                camera_intrinsics_matrix=camera_intrinsics_matrix,
                transform=base_to_camera_vec_quat_7,
                color_image=rgb_image, save_ply_path=path,
                parent_handle=parent_handle,
                rgb_sensor_display_name='kcam_rgb_current',
                depth_sensor_display_name='kcam_depth_current')

    def display_images(self, rgb, depth_image_float_format):
        """Display the rgb and depth image in V-REP (not yet working)

        Reference code: https://github.com/nemilya/vrep-api-python-opencv/blob/master/handle_vision_sensor.py
        V-REP Docs: http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm#simxSetVisionSensorImage
        """
        res, kcam_rgb_handle = vrep.vrep.simxGetObjectHandle(self.client_id, 'kcam_rgb_current', vrep.vrep.simx_opmode_oneshot_wait)
        print('kcam_rgb_current_handle: ', kcam_rgb_handle)
        rgb_for_display = rgb.astype('uint8')
        rgb_for_display = rgb_for_display.ravel()
        is_color = 1
        res = vrep.vrep.simxSetVisionSensorImage(self.client_id, kcam_rgb_handle, rgb_for_display, is_color, vrep.vrep.simx_opmode_oneshot_wait)
        print('simxSetVisionSensorImage rgb result: ', res, ' rgb shape: ', rgb.shape)
        res, kcam_depth_handle = vrep.vrep.simxGetObjectHandle(self.client_id, 'kcam_depth_current', vrep.vrep.simx_opmode_oneshot_wait)
        normalized_depth = depth_image_float_format * 255 / depth_image_float_format.max()
        normalized_depth = normalized_depth.astype('uint8')
        normalized_depth = normalized_depth.ravel()
        is_color = 0
        res = vrep.vrep.simxSetVisionSensorImage(self.client_id, kcam_depth_handle, normalized_depth, is_color, vrep.vrep.simx_opmode_oneshot_wait)
        print('simxSetVisionSensorImage depth result: ', res, ' depth shape: ', depth_image_float_format.shape)

    def __del__(self):
        vrep.vrep.simxFinish(-1)


def vrep_grasp_main(_):
    with tf.Session() as sess:
        viz = VREPGoogleBrainGraspVisualization()
        viz.visualize(sess)

if __name__ == '__main__':
    tf.app.run(main=vrep_grasp_main)

