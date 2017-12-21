# 3D geometry algorithms for calculating deep learning grasp algorithm input parameters.
#
# Copyright 2017 Andrew Hundt 2017.
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

import tensorflow as tf
from keras import backend as K


def depth_image_to_point_cloud(depth, intrinsics_matrix, dtype=tf.float32):
    """Depth images become an XYZ point cloud in the camera frame with shape (depth.shape[0], depth.shape[1], 3).

    Transform a depth image into a point cloud in the camera frame with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    Based on:
    numpy version of depth_image_to_point_cloud() in grasp_geometry.py
    https://github.com/tensorflow/models/blob/master/research/cognitive_mapping_and_planning/src/depth_utils.py
    https://codereview.stackexchange.com/a/84990/10101

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
    with K.name_scope('depth_image_to_point_cloud'):
        intrinsics_matrix = tf.to_float(intrinsics_matrix)
        fy = intrinsics_matrix[1, 1]
        fx = intrinsics_matrix[0, 0]
        # center of image y coordinate
        center_y = intrinsics_matrix[2, 1]
        # center of image x coordinate
        center_x = intrinsics_matrix[2, 0]
        depth = tf.to_float(tf.squeeze(depth))
        # y, x
        y_shape, x_shape = K.int_shape(depth)

        y, x = tf.meshgrid(K.arange(y_shape),
                           K.arange(x_shape),
                           indexing='ij')

        x = tf.to_float(K.flatten(x))
        y = tf.to_float(K.flatten(y))
        depth = K.flatten(depth)

        assert K.int_shape(y) == K.int_shape(x)
        assert K.int_shape(y) == K.int_shape(depth)

        X = (x - center_x) * depth / fx
        Y = (y - center_y) * depth / fy

        assert K.int_shape(y) == K.int_shape(x)
        assert K.int_shape(y) == K.int_shape(depth)

        XYZ = K.stack([X, Y, depth], axis=-1)

        assert K.int_shape(XYZ) == (y_shape * x_shape, 3)

        XYZ = K.reshape(XYZ, [y_shape, x_shape, 3])
        return XYZ