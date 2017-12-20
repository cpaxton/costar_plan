import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops


def random_crop_offset(input_shape, output_shape, seed=None, name=None):
    """ Generate a random image corner offset to randomly crop an image.

        The return value should be supplied to crop_images().

        # Arguments

        input_shape: shape of input image to crop.
        output_shape: shape of image after cropping.
        seed: Python integer. Used to create a random seed.
            See tf.set_random_seed for behavior.
        name: A name for this operation.
    """
    with tf.name_scope(name, "random_crop",
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


def crop_images(image_list, offset, size, name=None, verbose=0):
    """ Crop color image and depth image to specified offset and size.

        random_crop_offset() can be utilized to generate the offset parameter.
        Any associated image intrinsics matrix, can be updated with crop_image_intrinsics().

        # Arguments

        image_list: input images need to crop.
        image_intrinsics: 3*3 matrix with focal lengths, and principle points,
        size: output size of image after cropping, must be less than or equal to the image size.
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
            offset = ops.convert_to_tensor(offset, dtype=dtypes.int32, name="offset")
            if verbose > 0:
                print("crop_images offset:", offset, 'size', size, 'img_list_shape', image_list.shape)
                offset = tf.Print(offset, [offset, size, image_list.shape])
            cropped_image_list = array_ops.slice(
                image_list, offset, size, name=name)

    return cropped_image_list


def crop_image_intrinsics(camera_intrinsics_matrix, offset, name=None):
    """ Calculate an updated image intrinsics matrix after a cropping with the specified offset.

        # Arguments

        camera_intrinsics_matrix: intrinsic matrix before cropping.
        offset: offset used in cropping.
    """
    # offset should be array can be access by index
    with tf.name_scope(name, "crop_image_intrinsics", [camera_intrinsics_matrix, offset]) as name:
        offset = tf.cast(offset, camera_intrinsics_matrix.dtype)
        offset_matrix = tf.convert_to_tensor(
            [[0., 0., 0.],
             [0., 0., 0.],
             [offset[0], offset[1], 0.]])
        return camera_intrinsics_matrix - offset_matrix
