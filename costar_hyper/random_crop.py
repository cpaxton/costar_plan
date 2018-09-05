import math
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

        To update an image coordinate after the crop simply do:

            new_image_coordinate = original_image_coordinate - offset

        # Arguments

        offset: The coordinate offset from the origin at which the cropped image should begin.
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
            if not tf.contrib.framework.is_tensor(offset):
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


def random_projection_transform(
        input_shape=None, output_shape=None,
        translation=False, scale=False, rotation=False,
        horizontal_flip=False, vertical_flip=False,
        name=None, seed=None):
    """ Generate a random projection data augmentation transform.

    Please note this is not truly random in SE(3), and shear is not yet supported.

    # Arguments

        input_shape: The shape of the image to which the projection will be applied.
        scale: False means no scaling will be applied, True calculates the range of scaling
            to be +/- the change in scale from the input shape to the output shape.
            Range of valid scaling multiples, for example (0.5, 1.5)
            is 2x zoom in to 1.5x zoom out the Default None is (1.0, 1.0).
        rotation: Range of valid random rotation angles. The default None is (0.0, 0.0)
        required_coordinate: A coordinate which must be included in the output image.
        crop_size: The final dimensions expected for the output image
            a random translation will be created so you can make a central
            crop to the crop size. The crop size must be equal to or
            smaller than the input size.
        seed: A random seed.

    # Returns
        A dictionary containing features tensors representing the exact chosen random configuration.
            'random_transform': An 8x1 random projection transform, this is always present.
        Possible feature strings include:
            'random_theta': an angle in radians
            'random_scale': floatin point scale multiplier where 1.0 is constant scale
            'random_horizontal_flip': 0 for no flip 1 for flip
            'random_vertical_flip': 0 for no flip 1 for flip
            'random_translation_offset': A 2D or 3D offset from the origin of the new image corner.
    """
    with tf.name_scope(name, "random_projection_transform",
                       [input_shape, output_shape]) as name:
        input_shape = ops.convert_to_tensor(
            input_shape, dtype=dtypes.int32, name="input_shape")
        if output_shape is None:
            output_shape = input_shape
        output_shape = ops.convert_to_tensor(
            output_shape, dtype=dtypes.int32, name="output_shape")
        check = control_flow_ops.Assert(
            math_ops.reduce_all(input_shape >= output_shape),
            ["Need input_shape >= output_shape ", input_shape, output_shape],
            summarize=1000)
        input_shape = control_flow_ops.with_dependencies([check], input_shape)

        features = {}
        transforms = []
        input_height_f = tf.cast(output_shape[0], tf.float32)
        input_width_f = tf.cast(output_shape[1], tf.float32)

        if translation is not None and translation is not False:
            if isinstance(translation, bool) and translation:
                offset = random_crop_offset(input_shape, output_shape, seed=seed)
            else:
                offset = translation
            # The transform is float32, which differs from
            # random_crop_offset which is int32
            offset = tf.cast(offset, tf.float32)
        else:
            offset = tf.constant([0., 0.], dtype=tf.float32)

        # there should always be some offset, even if it is (0, 0)
        # these transforms are inverted
        # so we pass in the negative offset
        transforms += [tf.contrib.image.translations_to_projective_transforms(offset)]
        features['random_translation_offset'] = offset
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if rotation is not None and rotation is not False:
            if isinstance(rotation, bool) and rotation:
                rotation = ops.convert_to_tensor(
                    [-math.pi, math.pi], dtype=tf.float32, name="input_shape")

            theta = tf.random_uniform([1], minval=rotation[0], maxval=rotation[1], seed=seed, dtype=tf.float32)
            transforms += [tf.contrib.image.angles_to_projective_transforms(theta, input_height_f, input_width_f)]
            features['random_rotation'] = theta

        if scale is not None and scale is not False:
            if isinstance(scale, bool) and scale:
                # choose the scale from the input to output image size difference
                max_input_dim = input_shape[tf.argmax(input_shape)]
                max_crop_dim = output_shape[tf.argmax(output_shape)]
                s0 = tf.cast(max_input_dim, dtype=tf.float32) / tf.cast(max_crop_dim, dtype=tf.float32)
                s1 = tf.cast(max_crop_dim, dtype=tf.float32) / tf.cast(max_input_dim, dtype=tf.float32)
                scale = [s0, s1]

            s = tf.random_uniform([1], minval=scale[0], maxval=scale[1], seed=seed, dtype=tf.float32)
            scale_matrix = [s,  array_ops.zeros((1), dtypes.float32), array_ops.zeros((1), dtypes.float32),
                            array_ops.zeros((1), dtypes.float32), s, array_ops.zeros((1), dtypes.float32),
                            array_ops.zeros((1), dtypes.float32), array_ops.zeros((1), dtypes.float32)]
            scale_matrix = tf.stack(scale_matrix, axis=-1)
            transforms += [scale_matrix]
            features['random_scale'] = s
        else:
            features['random_scale'] = tf.constant(1.0, dtype=tf.float32)

        batch_size = 1
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [-1., 0., input_width_f, 0., 1., 0., 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            no_flip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            random_flip_transform = tf.where(coin, flip, no_flip)
            transforms.append(random_flip_transform)
            features['random_horizontal_flip'] = coin

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [1., 0., 0., 0., -1., input_height_f, 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            no_flip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            random_flip_transform = tf.where(coin, flip, no_flip)
            transforms.append(random_flip_transform)
            features['random_vertical_flip'] = coin

        composed_transforms = tf.contrib.image.compose_transforms(*transforms)
        features['random_projection_transform'] = composed_transforms
        return composed_transforms, features


def transform_and_crop_coordinate(coordinate, transform=None, offset=None, theta=None, inverse=True, name=None):
    """ Transforms a single coordinate then applies a crop offset.

     You probably don't need to use this, just call random_projection_transform()
     and then transform_and_crop_image().

    Please note that when calling this function you should be very careful
    about your transform matrix, because the projection function expects the matrix
    to be inverted relative to what you would normally expect. Instead of taking a
    point in the current frame and projecting it out to the final frame it does the inverse.

    # Arguments

        coordinate: A 2D image coordinate.
        transform: A 3x3 homogenous 2D image transform matrix stored in shape [1, 8].
        offset: A crop offset which is the location of (0,0) in the post-crop image.
            offset is in y, x order.
        theta: An optional orientation angle for the 2D image coordinate.

    # Returns

        The coordinate after a tranform and crop is applied.
    """
    with tf.name_scope(name, "transform_and_crop_coordinate",
                       [coordinate, transform, offset, theta]) as name:

        if transform is not None:
            projection_matrix = _flat_transforms_to_matrices(transform)
            # TODO(ahundt) replace above with the following once flat_transforms_to_matrices becomes public in tf
            # projection_matrix = tf.contrib.image._flat_transforms_to_matrices(transform)

            # Very important: most TF code expects y, x coordinates
            # but the projection matrix is generated uing x, y coordinates
            # so we swap it from (y,x) to (x,y) here.
            if theta is None:
                if not tf.contrib.framework.is_tensor(coordinate):
                    coordinate = tf.transpose(tf.convert_to_tensor(
                        [coordinate[1],
                         coordinate[0],
                         1]
                    ))
                else:
                    coordinate = tf.stack([tf.reshape(coordinate[1], (1,)),
                                           tf.reshape(coordinate[0], (1,)),
                                           tf.constant([1], tf.float32)], axis=-1)

                coordinate = tf.transpose(coordinate)
                if inverse:
                    projection_matrix = tf.matrix_inverse(projection_matrix)
                projection_matrix = tf.squeeze(projection_matrix)
                coordinate = tf.matmul(projection_matrix,
                                       coordinate)
                coordinate = tf.squeeze(tf.transpose(coordinate))
                coordinate = tf.stack([coordinate[1], coordinate[0]])

                # Very important: most TF code expects y, x coordinates
                # but the projection matrix is generated uing x, y coordinates
                # so we swap it back from (x,y) to (y,x) here.
            else:
                if not tf.contrib.framework.is_tensor(coordinate):
                    raise NotImplementedError
                else:
                    # TODO(ahundt) should theta be positive or negative?
                    oriented_coordinate_matrix = array_ops.concat(
                        values=[
                            math_ops.cos(theta),
                            -math_ops.sin(theta),
                            coordinate[1],  # x_offset
                            math_ops.sin(theta),
                            math_ops.cos(theta),
                            coordinate[0],  # y_offset
                            array_ops.zeros((1, 2), dtypes.float32),
                        ],
                        axis=1)
                    oriented_coordinate_matrix = tf.contrib.image.compose_transforms([transform, oriented_coordinate_matrix])
                    # Very important: most TF code expects y, x coordinates
                    # but the projection matrix is generated uing x, y coordinates
                    # so we swap it back from (x,y) to (y,x) here.
                    coordinate = tf.stack([oriented_coordinate_matrix[5], oriented_coordinate_matrix[2]])
                    theta = tf.atan2(oriented_coordinate_matrix[3], oriented_coordinate_matrix[4])

        if offset is not None:
            if isinstance(offset, list):
                offset = tf.constant([[offset[0]], [offset[1]]], tf.float32)
            coordinate = coordinate - tf.cast(offset[:2], tf.float32)
        if theta is None:
            return coordinate
        else:
            return coordinate, theta


def resize_coordinate(coordinate, input_shape, output_shape, dtype=tf.float32, name=None):
    """ Update a coordinate that changed with a tf.image.resize_images call.

    Update is made based on the current input shape and a new updated output shape.
    Warning: Do not use this with crop! This is strictly designed to work with tf.image.resize_images().
    """
    with tf.name_scope(name, "resize_coordinate",
                       [coordinate, input_shape, output_shape]) as name:
        input_shape = tf.cast(input_shape, tf.float32)
        output_shape = tf.cast(output_shape, tf.float32)
        if isinstance(input_shape, list):
            input_shape = tf.constant([[input_shape[0]][input_shape[1]]], tf.float32)
            proportional_dimension_change = output_shape / input_shape
        else:
            proportional_dimension_change = output_shape / input_shape[:2]

        resized_coordinate = tf.squeeze(tf.cast(coordinate, tf.float32)) * proportional_dimension_change
    return tf.cast(resized_coordinate, dtype)


def transform_crop_and_resize_image(
        image, offset=None, crop_shape=None, transform=None,
        interpolation='BILINEAR', central_crop=False,
        resize_shape=None, coordinate=None, coordinate_theta=None,
        name=None):
    """ Project the image with a 3x3 htransform, crop, then resize the image to the output shape.

    A projection, then crop, then a separate resize is performed. This allows
    random cropping to be applied with as little or as much translation as needed.
    The resize is then applied afterward so the image can still be
    output with whatever arbitrary dimensions the user desires.

    Typically you will need to call random_projection_transform(),
    to get the augmentation parameters, including random_translation_offset.
    Then transform_and_crop_image() to actually perform the crop. Optionally supply
    an image coordinate which will be updated and returned according to the changes
    made to the image.

    Please note this function does not yet have an equivalent to crop_image_intrinsics.

    # Arguments
        offset: an offset to perform after the transform is applied,
           if not defined it defaults to 0 offset, which means cropping at the origin.
           This is because the transform can already include the translation, and we
           want to keep the coordinate system the same.
        central_crop: False by default, when central_crop is True and offset is None,
           a central crop offset will be calculated, which is half the
           size difference between the input image and offset.
        crop_shape: The output image shape, default None is the input image shape.
        transform: An 8 element homogeneous projective transformation matrix.
        interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
        coordinate: An optional specific coordinate to which a transform should also be applied.
        coordinate_theta: An optional orientation for the coordinate.

    # Returns

       cropped_image if coordinate is None, otherwise [cropped_image, new_coordinate]
    """
    # TODO(ahundt) projective grasp coordinate transform with xy sin cos rot and multiply also see crop_to_gripper_transform.
    with tf.name_scope(name, "transform_crop_and_resize_image",
                       [image]) as name:
        if crop_shape is None and offset is not None:
            raise ValueError('If crop_shape is None offset must also be None.')

        if transform is not None:
            image = tf.contrib.image.transform(image, transforms=transform, interpolation=interpolation)

        if crop_shape is not None or offset is not None:
            if crop_shape is None:
                crop_shape = tf.shape(image)

            if offset is None and central_crop:
                offset = (tf.shape(image) - crop_shape) // 2
            else:
                # in this case the random part of the
                # random crop is built into the transform
                offset = tf.constant([0, 0, 0], tf.int32)

            image = crop_images(image, offset, crop_shape)

            if coordinate is not None and (transform is not None or offset is not None):
                if coordinate_theta is None:
                    coordinate = transform_and_crop_coordinate(coordinate, transform, offset)
                else:
                    coordinate, coordinate_theta = transform_and_crop_coordinate(coordinate, transform, offset, theta=coordinate_theta)
        if resize_shape is not None:
            if coordinate is not None:
                coordinate = resize_coordinate(coordinate, tf.shape(image), resize_shape)
            image = tf.image.resize_images(image, resize_shape)

        if coordinate is None:
            return image
        else:
            if coordinate_theta is None:
                return image, coordinate
            else:
                return image, coordinate, coordinate_theta


def random_translation_in_bounds(grasp_center_coordinate, output_shape, max_pixels, seed=None):
    max_pixels = tf.cast(max_pixels, tf.float32)
    bottom_y = (grasp_center_coordinate[0] - tf.cast(output_shape[0] // 2, tf.float32))
    bottom_x = (grasp_center_coordinate[1] - tf.cast(output_shape[1] // 2, tf.float32))
    min_y_offset = (bottom_y - max_pixels)
    max_y_offset = (bottom_y + max_pixels)
    min_x_offset = (bottom_x - max_pixels)
    max_x_offset = (bottom_x + max_pixels)
    off_y = tf.random_uniform([1], minval=min_y_offset, maxval=max_y_offset, seed=seed, dtype=tf.float32)
    off_x = tf.random_uniform([1], minval=min_x_offset, maxval=max_x_offset, seed=seed, dtype=tf.float32)
    random_offset = tf.concat([-off_y, -off_x, [0.0]], axis=-1)
    return random_offset


def random_translation_offset(max_pixels, seed=None):
    max_pixels = tf.cast(max_pixels, tf.float32)
    off_y = tf.random_uniform([1], minval=-max_pixels, maxval=max_pixels, seed=seed, dtype=tf.float32)
    off_x = tf.random_uniform([1], minval=-max_pixels, maxval=max_pixels, seed=seed, dtype=tf.float32)
    random_offset = tf.concat([-off_y, -off_x, [0.0]], axis=-1)
    return random_offset


def _flat_transforms_to_matrices(transforms):
    # For converting between transformation matrices and homogeneous transforms see:
    # transform_and_crop_coordinate()
    # https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/image/python/ops/image_ops.py

    # tf.contrib.image._flat_transforms_to_matrices()
    # tf.contrib.image._transform_matrices_to_flat()
    # TODO(ahundt) remove this when flat_transforms_to_matrices becomes public in tf
    # Make the transform(s) 2D in case the input is a single transform.
    transforms = array_ops.reshape(transforms, tf.constant([-1, 8]))
    num_transforms = array_ops.shape(transforms)[0]
    # Add a column of ones for the implicit last entry in the matrix.
    return array_ops.reshape(
        array_ops.concat(
            [transforms, array_ops.ones([num_transforms, 1])], axis=1),
             tf.constant([-1, 3, 3]))


# def _transform_matrices_to_flat(transform_matrices):
#   # Flatten each matrix.
#   transforms = array_ops.reshape(transform_matrices,
#                                  constant_op.constant([-1, 9]))
#   # Divide each matrix by the last entry (normally 1).
#   transforms /= transforms[:, 8:9]
#   return transforms[:, :8]
