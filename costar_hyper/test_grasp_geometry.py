# test 3D geometry algorithms for calculating deep learning grasp algorithm input parameters.
#
# Copyright 2017 Andrew Hundt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
try:
    import eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
    import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
except ImportError:
    print('eigen and sva not available, skipping components utilizing 3D geometry algorithms.'
          'To install run the script at'
          'https://github.com/ahundt/robotics_setup/blob/master/robotics_tasks.sh'
          'or follow the instructions at https://github.com/jrl-umi3218/Eigen3ToPython'
          'and https://github.com/jrl-umi3218/SpaceVecAlg and make sure python bindings'
          'are enabled.')
    eigen = None
    sva = None
import pytest
import random_crop as rcp
from grasp_geometry import grasp_dataset_to_transforms_and_features
from grasp_geometry import depth_image_to_point_cloud
import grasp_geometry_tf
import tensorflow as tf
from tqdm import tqdm
import numba


# @guvectorize(["void(float64[:,:], float64, float64, float64, float64, float64, float64, float64[:,:,:])"],
#              "(y,x),(),(),(),(),(),()->(y,x,3)")
@numba.jit(nopython=True, nogil=True, parallel=False, cache=True)
def depth_image_to_point_cloud_numba(depth, intrinsics_matrix, XYZ):
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    # center of image x coordinate
    center_x = intrinsics_matrix[2, 0]
    # center of image y coordinate
    center_y = intrinsics_matrix[2, 1]
    y_range = depth.shape[0]
    x_range = depth.shape[1]
    for y in range(y_range):
        for x in range(x_range):
            Z = depth[y, x]
            X = (x - center_x) * Z / fx
            Y = (y - center_y) * Z / fy
            XYZ[y, x] = [X, Y, Z]
    return XYZ

# @guvectorize(["void(float64[:,:], float64, float64, float64, float64, float64, float64, float64[:,:,:])"],
#              "(y,x),(),(),(),(),(),()->(y,x,3)")
@numba.jit(nopython=True, nogil=True, parallel=False, cache=True)
def depth_image_to_point_cloud_numba2(depth, y_range, x_range, center_y, center_x, fy, fx, XYZ):
    for y in range(y_range):
        for x in range(x_range):
            Z = depth[y, x]
            X = (x - center_x) * Z / fx
            Y = (y - center_y) * Z / fy
            XYZ[y, x] = [X, Y, Z]
    return XYZ


def depth_image_to_point_cloud3(depth, intrinsics_matrix, dtype=np.float32):
    """Depth images become an XYZ point cloud in the camera frame with shape (depth.shape[0], depth.shape[1], 3).

    Transform a depth image into a point cloud in the camera frame with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    Based on:
    https://github.com/tensorflow/models/blob/master/research/cognitive_mapping_and_planning/src/depth_utils.py
    https://codereview.stackexchange.com/a/84990/10101

    also see grasp_geometry_tf.depth_image_to_point_cloud().

    # Arguments

      depth: is a 2-D ndarray with shape (rows, cols) containing
          32bit floating point depths in meters. The result is a 3-D array with
          shape (rows, cols, 3). Pixels with invalid depth in the input have
          NaN or 0 for the z-coordinate in the result.

      intrinsics_matrix: 3x3 matrix for projecting depth values to z values
      in the point cloud frame. http://ksimek.github.io/2013/08/13/intrinsic/
      In this case x0, y0 are at index [2, 0] and [2, 1], respectively.

      transform: 4x4 Rt matrix for rotating and translating the point cloud
    """

    depth = np.squeeze(depth)
    XYZ = np.zeros(depth.shape + (3,))
    XYZ = depth_image_to_point_cloud_numba(depth, intrinsics_matrix, XYZ)

    return XYZ.astype(dtype)


def depth_image_to_point_cloud2(depth, intrinsics_matrix, dtype=np.float32):
    """Depth images become an XYZ point cloud in the camera frame with shape (depth.shape[0], depth.shape[1], 3).

    Transform a depth image into a point cloud in the camera frame with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    Based on:
    https://github.com/tensorflow/models/blob/master/research/cognitive_mapping_and_planning/src/depth_utils.py
    https://codereview.stackexchange.com/a/84990/10101

    also see grasp_geometry_tf.depth_image_to_point_cloud().

    # Arguments

      depth: is a 2-D ndarray with shape (rows, cols) containing
          32bit floating point depths in meters. The result is a 3-D array with
          shape (rows, cols, 3). Pixels with invalid depth in the input have
          NaN or 0 for the z-coordinate in the result.

      intrinsics_matrix: 3x3 matrix for projecting depth values to z values
      in the point cloud frame. http://ksimek.github.io/2013/08/13/intrinsic/
      In this case x0, y0 are at index [2, 0] and [2, 1], respectively.

      transform: 4x4 Rt matrix for rotating and translating the point cloud
    """

    depth = np.squeeze(depth)
    XYZ = np.zeros(depth.shape + (3,))

    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    # center of image x coordinate
    center_x = intrinsics_matrix[2, 0]
    # center of image y coordinate
    center_y = intrinsics_matrix[2, 1]
    y_range = depth.shape[0]
    x_range = depth.shape[1]
    XYZ = depth_image_to_point_cloud_numba2(depth, y_range, x_range, center_y, center_x, fy, fx, XYZ)

    return XYZ.astype(dtype)


def test_eigen_and_sva():
    """ Test/demo of 3D geometry libraries eigen and sva

        eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
        sva  # https://github.com/jrl-umi3218/SpaceVecAlg
    """
    print('eigen and sva test')
    if eigen is not None and sva is not None:
        qa = np.array([0, 0, 0, 1])
        v = eigen.Vector3d([1, 1, 1])

        qa4 = eigen.Vector4d(qa)

        q = eigen.Quaterniond(qa4)

        pt = sva.PTransformd(q, v)

        print(str(np.array(pt.rotation())))
        rot = [[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]]
        assert np.allclose(np.array(rot), np.array(pt.rotation()))
        pt.translation()

        trans = pt.translation()

        assert trans.x() == 1.0
        assert trans.y() == 1.0
        assert trans.z() == 1.0
        trans_np = np.array(trans)
        assert np.allclose(trans_np, v)


def test_grasp_dataset_to_transforms_and_features():

    def evaluate_grasp_dataset_to_transforms_and_features(depth_image, intrinsics, camera_to_base, base_to_end_current, base_to_end_final):
        [current_base_T_camera_vec_quat_7_array,
         eectf_vec_quat_7_array,
         camera_T_endeffector_current_vec_quat_7_array,
         camera_T_depth_pixel_current_vec_quat_7_array,
         camera_T_endeffector_final_vec_quat_7_array,
         camera_T_depth_pixel_final_vec_quat_7_array,
         depth_pixel_T_endeffector_current_vec_quat_7_array,
         image_coordinate_current,
         depth_pixel_T_endeffector_final_vec_quat_7_array,
         image_coordinate_final,
         sin_cos_2,
         vec_sin_cos_5,
         delta_depth_sin_cos_3,
         delta_depth_quat_5] = grasp_dataset_to_transforms_and_features(
                                     depth_image,
                                     intrinsics,
                                     camera_to_base,
                                     base_to_end_current,
                                     base_to_end_final)

        assert len(delta_depth_sin_cos_3) == 3
        assert np.allclose(delta_depth_sin_cos_3[0], delta_depth_quat_5[0])
        # TODO(ahundt) check if only a tolerance increase is needed for the following error:
        # >       assert np.allclose(delta_depth_sin_cos_3[0], np.array([1]))
        # E       assert False
        # E        +  where False = <function allclose at 0x7f65996cd668>(1.05, array([1]))
        # E        +    where <function allclose at 0x7f65996cd668> = np.allclose
        # E        +    and   array([1]) = <built-in function array>([1])
        # E        +      where <built-in function array> = np.array
        assert np.allclose(delta_depth_sin_cos_3[0], np.array([1]))

    depth_image = np.ones([10, 20, 1])
    intrinsics = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [5., 10., 1.]])
    camera_to_base = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
    # vector and quaternion
    base_to_end_current = np.array([0., 0., 1., 0., 0., 0., 1.])
    # vector and quaternion
    base_to_end_final = np.array([1., 1., 2., 0., 0., 0., 1.])

    evaluate_grasp_dataset_to_transforms_and_features(depth_image, intrinsics, camera_to_base, base_to_end_current, base_to_end_final)

    # test if the end effector is outside the frame
    base_to_end_current = np.array([20., 1., 1., 0., 0., 0., 1.])
    evaluate_grasp_dataset_to_transforms_and_features(depth_image, intrinsics, camera_to_base, base_to_end_current, base_to_end_final)


def test_depth_image_to_point_cloud():
    steps = range(1)
    depth = np.random.rand(3, 4)
    intrinsics = np.random.rand(3, 3)

    XYZ_numba = depth_image_to_point_cloud3(depth, intrinsics)
    XYZ_np = depth_image_to_point_cloud(depth, intrinsics)
    XYZ_tf = grasp_geometry_tf.depth_image_to_point_cloud(
                tf.convert_to_tensor(depth), tf.convert_to_tensor(intrinsics))
    with tf.Session() as sess:
        XYZ_tf = sess.run(XYZ_tf)

    assert np.allclose(XYZ_numba, XYZ_np)
    assert np.allclose(XYZ_numba, XYZ_tf)
    assert np.allclose(np.squeeze(XYZ_numba[:, :, 2]), np.squeeze(depth))
    assert np.allclose(np.squeeze(XYZ_np[:, :, 2]), np.squeeze(depth))
    depth = np.random.rand(21, 10, 1)
    intrinsics = np.random.rand(3, 3)
    XYZ_numba = depth_image_to_point_cloud2(depth, intrinsics)
    XYZ_np = depth_image_to_point_cloud(depth, intrinsics)
    XYZ_tf = grasp_geometry_tf.depth_image_to_point_cloud(
                tf.convert_to_tensor(depth), tf.convert_to_tensor(intrinsics))
    with tf.Session() as sess:
        XYZ_tf = sess.run(XYZ_tf)
    assert np.allclose(XYZ_numba, XYZ_np)
    assert np.allclose(np.squeeze(XYZ_numba[:, :, 2]), np.squeeze(depth))
    assert np.allclose(np.squeeze(XYZ_np[:, :, 2]), np.squeeze(depth))
    assert np.allclose(np.squeeze(XYZ_tf[:, :, 2]), np.squeeze(depth))

    depth = np.random.rand(640, 512, 1)
    intrinsics = np.random.rand(3, 3)
    for _ in tqdm(steps, desc='depth_image_to_point_cloud_numba_intrinsics_matrix_wrapped'):
        XYZ_numba = depth_image_to_point_cloud3(depth, intrinsics)
    for _ in tqdm(steps, desc='depth_image_to_point_cloud2_numba_scalars'):
        XYZ_numba = depth_image_to_point_cloud2(depth, intrinsics)

    for _ in tqdm(steps, desc='depth_image_to_point_cloud_np'):
        XYZ_np = depth_image_to_point_cloud(depth, intrinsics)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    # center of image x coordinate
    center_x = intrinsics[2, 0]
    # center of image y coordinate
    center_y = intrinsics[2, 1]
    y_range = depth.shape[0]
    x_range = depth.shape[1]
    XYZ = np.zeros((y_range, x_range, 3))

    for _ in tqdm(steps, desc='depth_image_to_point_cloud_numba_intrinsics_matrix_direct'):
        sdepth = np.squeeze(depth)
        XYZ = depth_image_to_point_cloud_numba(sdepth, intrinsics, XYZ)
    assert np.allclose(XYZ, XYZ_np)
    assert np.allclose(np.squeeze(XYZ[:, :, 2]), np.squeeze(depth))

    with tf.Session() as sess:
        depth_tf = tf.convert_to_tensor(depth)
        intrinsics_tf = tf.convert_to_tensor(intrinsics)
        XYZ_tf = grasp_geometry_tf.depth_image_to_point_cloud(depth_tf, intrinsics_tf)
        for _ in tqdm(steps, desc='depth_image_to_point_cloud_tf'):
            XYZ = sess.run(XYZ_tf)
        print('xyz', XYZ, 'XYZ_np', XYZ_np)

        assert XYZ.shape == XYZ_np.shape
        assert np.allclose(XYZ, XYZ_np)
        assert np.allclose(np.squeeze(XYZ[:, :, 2]), np.squeeze(depth))


def test_crop_pointcloud():
    """ Test pointcloud use random crop of tensor
    """
    with tf.Session() as sess:
        test_input = np.random.rand(30, 20, 1)
        test_input_size_tf = tf.constant([30, 20, 1])
        intrinsics = np.random.rand(3, 3)
        intrinsics_tf = tf.convert_to_tensor(intrinsics)
        crop_size_np = np.array([5, 4, 1])
        cropped_size_tf = tf.constant([5, 4, 1])
        XYZ_np = depth_image_to_point_cloud(test_input, intrinsics)
        depth_tf = tf.convert_to_tensor(test_input)
        offset_tf = rcp.random_crop_offset(test_input_size_tf, cropped_size_tf)
        rcp_crop_tf = rcp.crop_images(depth_tf, offset_tf, cropped_size_tf)
        cropped_intrinsics_tf = rcp.crop_image_intrinsics(intrinsics_tf, offset_tf)
        cropped_intrinsics_np = sess.run(cropped_intrinsics_tf)
        rcp_crop_np = sess.run(rcp_crop_tf)
        offset_np = sess.run(offset_tf)
        crop_XYZ_np = depth_image_to_point_cloud(rcp_crop_np, cropped_intrinsics_np)
        assert np.allclose(np.squeeze(crop_XYZ_np[:, :, 0]),
            np.squeeze(XYZ_np[offset_np[1]:offset_np[1] + crop_size_np[1],
                              offset_np[0]:offset_np[0] + crop_size_np[0], 0]))

if __name__ == '__main__':
    pytest.main([__file__])