import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops


def random_crop_parameters(input_shape, output_shape, seed=None, name=None):
    """ Generate crop parameter by random.
        # Params
        input_shape: shape of input image need to crop.
        output_shape: shape of image after cropping.
        seed: Python integer. Used to create a random seed.
        See tf.set_random_seed for behavior.
        name: A name for this operation.
    """
    with tf.name_scope(name, "random_crop_parameters",
                       [input_shape, output_shape]) as name:
        input_shape = ops.convert_to_tensor(
            input_shape, dtype=dtypes.int32, name="input_shape")
        output_shape = ops.convert_to_tensor(
            output_shape, dtype=dtypes.int32, name="output_shape")
        check = control_flow_ops.Assert(
            math_ops.reduce_all(input_shape >= output_shape),
            ["Need input_shape >= output_shape ", input_shape, output_shape],
            summarize=1000)
        input_shape = control_flow_ops.with_dependencies([check], input_shape)
        limit = input_shape - output_shape + 1
        offset = tf.random_uniform(
            array_ops.shape(input_shape),
            dtype=output_shape.dtype,
            maxval=output_shape.dtype.max,
            seed=seed) % limit

    return offset


def crop_images(image_list, offset, size, name=None):
    """ Crop color image and depth image by random.
        # Params
        image_list: input images need to crop.
        image_intrinsics: 3*3 matirx with focal lengthes, and principle points,
        size: output size of image after cropping.
        seed: Python integer. Used to create a random seed.
        See tf.set_random_seed for behavior.
        name: A name for this operation.
    """
    with tf.name_scope(name, "crop_images", [image_list, size]) as name:
        if isinstance(image_list, list):
            cropped_image_list = []
            size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
            for image in image_list:
                image = ops.convert_to_tensor(image, name="image")
                image = array_ops.slice(image, offset, size, name=name)
                cropped_image_list.append(image)
        else:
            size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
            image_list = ops.convert_to_tensor(image_list, name="image")
            cropped_image_list = array_ops.slice(
                image_list, offset, size, name=name)

    return cropped_image_list


def crop_image_intrinsics(camera_intrinsics_matrix, offset, name=None):
    """ Calculate the intrinsic after crop.
        # Params
        camera_intrinsics_matrix: intrinsic matrix before cropping.
        offset: offset used in cropping.
    """
    # offset should be array can be access by index
    with tf.name_scope(name, "crop_image_intrinsics",
                       [camera_intrinsics_matrix, offset]) as name:
        offset_x = tf.gather(offset, tf.constant([0]))
        offset_x = tf.pad(offset_x, tf.constant([[2, 0], [0, 2]]))
        offset_y = tf.gather(offset, tf.constant([1]))
        offset_y = tf.pad(offset_x, tf.constant([[2, 0], [1, 1]]))
        camera_intrinsics_matrix = camera_intrinsics_matrix - offset_x
        - offset_y

    return camera_intrinsics_matrix
