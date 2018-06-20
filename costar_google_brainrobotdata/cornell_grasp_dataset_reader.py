#!/usr/local/bin/python
'''
Image processing of cornell grasping dataset for detecting grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code based on:
    https://github.com/tnikolla/robot-grasp-detection

'''
import os
import copy
import math
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import random
import glob

import tensorflow as tf
from tensorflow.python.platform import flags

import keras
from keras import backend as K
import keras_contrib

from grasp_loss import gaussian_kernel_2D
from inception_preprocessing import preprocess_image
import random_crop as rcp
import grasp_visualization

flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
flags.DEFINE_string('grasp_dataset', 'all', 'TODO(ahundt): integrate with brainrobotdata or allow subsets to be specified')
flags.DEFINE_boolean('grasp_download', False,
                     """Download the grasp_dataset to data_dir if it is not already present.""")
flags.DEFINE_string(
    'tfrecord_filename_base', 'cornell-grasping-dataset-multigrasp',
    """base of the filename used for the dataset tfrecords and csv files

    Works best with options: cornell-grasping-dataset-multigrasp, cornell-grasping-dataset-singlegrasp,
        or for older deprecated conversions cornell-grasping-dataset.

    The tfrecord file names will contain the text multigrasp if there are not redundant images
    and singlegrasp to indicate if there are redundant images.

    Older versions of the tfrecord files may need: cornell-grasping-dataset.
    """)
flags.DEFINE_integer('image_size', 224,
                     """DEPRECATED - this doesn't do anything right now. Provide square images of this size.""")
flags.DEFINE_integer('num_preprocess_threads', 12,
                     """Number of preprocessing threads per tower. """
                     """Please make this a multiple of 4.""")
flags.DEFINE_integer('num_readers', 20,
                     """Number of parallel threads reading from the dataset.""")
flags.DEFINE_integer('input_queue_memory_factor', 12,
                     """Size of the queue of preprocessed images. """
                     """Default is ideal but try smaller values, e.g. """
                     """4, 2 or 1, if host memory is constrained. See """
                     """comments in code for more details.""")
flags.DEFINE_boolean(
    'redundant_images', True,
    """Duplicate images for every bounding box so dataset is easier to traverse.
       Please note that this substantially affects the output file size,
       but the dataset parsing code becomes much easier to write.
       The downside of the above is you can't only compare against the nearest
       grasp label, since there can be multiple successful grasp poses in a single image.
       If redundant_images is false there will be multiple bounding boxes per image,
       so you'll need to traverse each grasp label, which vary in number on every image.

       The tfrecord file names will contain the text multigrasp if there are not redundant images
       and singlegrasp to indicate if there are redundant images.
    """)
flags.DEFINE_boolean('showTextBox', True,
                     """Display textBox with bbox info on image.
                     """)
flags.DEFINE_integer('sigma_divisor', 10,
                     """Sigma divisor for grasp success 2d labels.""")
flags.DEFINE_integer(
    'sensor_image_height',
    480,
    'The height of the dataset images'
)
flags.DEFINE_integer(
    'sensor_image_width',
    640,
    'The width of the dataset images'
)
flags.DEFINE_integer(
    'sensor_color_channels',
    3,
    'The width of the dataset images'
)
flags.DEFINE_integer('crop_height', 331,
                     """Height to crop images, resize_width and resize_height is applied next""")
flags.DEFINE_integer('crop_width', 331,
                     """Width to crop images, resize_width and resize_height is applied next""")
flags.DEFINE_string(
    'crop_to', 'center_on_gripper_grasp_box_and_rotate_upright',
    """ Choose the data augmentation projective transform configuration.

    Options are:
       'resize_height_and_resize_width' ensures your image will be the right size
           but when training with random translations and rotations any part of the
           image might be chosen.
       'image_contains_grasp_box_center' ensures you will always have the grasp box center somewhere in the image
           during training, and then a central crop is performed during testing.
       'center_on_gripper_grasp_box' ensures the grasp box will always be
           in the center of the image and if random translation is enabled
           it will remain within the intersection over union jaccard metric limits.
       'center_on_gripper_grasp_box_and_rotate_upright' ensures the grasp box
           will always be centered, visible, and rotated upright, plus any random rotations
           will be within the intersection over union jaccard metric and the
           angle distance limits.

    TODO(ahundt) consider adding image_contains_grasp_box which ensures the whole grasp box is in the image.
    """)
flags.DEFINE_boolean(
    'random_translation', True,
    """random_translation will apply the tf random crop function with
       the parameters specified by crop_width and crop_height.

       If random crop is disabled, a fixed crop will be applied
       to a box on the far right of the image which is vertically
       centered. If no crop is desired simply set the crop_width
       and crop_height equal to the sensor_image_width and
       sensor_image_height. However, it is important to ensure
       that you crop images to the same size during both training
       and test time.

       If crop_to_gripper and random_translation are enabled at the same
       time, the translation will be limited to the width and height
       of the grasp rectangle bounding box.
    """)
flags.DEFINE_boolean(
    'random_rotation', True,
    """ Apply a random rotation to the input data subject to the current setting for crop_to.

        The random rotation can be any arbitrary rotation amount.
        However, when crop_to requires rotating upright, the random rotation
        will be limited to 15 degrees in either direction.
    """
)
flags.DEFINE_integer('resize_height', 224,
                     """Height to resize images before prediction, if enabled.""")
flags.DEFINE_integer('resize_width', 224,
                     """Width to resize images before prediction, if enabled.""")
flags.DEFINE_boolean('resize', False,
                     """resize will resize the input images to the desired dimensions specified by the
                        resize_width and resize_height flags. It is suggested that an exact factor of 2 be used
                        relative to the input image directions if random_translation is disabled or the crop dimensions otherwise.
                     """)
# TODO(ahundt) load bbox_padded_count from csv file
flags.DEFINE_integer(
    'bbox_padded_count', 50,
    """
    Must be the same as bbox_padded_count in cornell_grasp_dataset_writer.py.
    When redundant_images is False, this is the fixed number of entries every example will contain
    regardless of the number of entries in the original dataset, zero padding all data to have equal length.
    Must be greater than or equal to the maximum number of labels for any image in the dataset. If this value
    is 0 or None, no padding will be added and all labels of success, gripper poses, etc will have variable length.
    """)

FLAGS = flags.FLAGS


def parse_example_proto(examples_serialized, have_image_id=False):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64),
        # if there is an error on bbox/count here,
        # regenerate the dataset and make sure you are parsing
        # single image multi grasp box example data and not
        # redundant single image single grasp box example data
        'bbox/count': tf.FixedLenFeature([], dtype=tf.int64)
    }

    # TODO(ahundt) remove boolean once we are set up with k-fold cross validation of images and objects
    if have_image_id:
        feature_map['object/id'] = tf.FixedLenFeature([], dtype=tf.int64)
    variable_length_keys = [
        'bbox/cy',
        'bbox/cx',
        'bbox/tan',
        'bbox/theta',
        'bbox/sin_theta',
        'bbox/cos_theta',
        'bbox/width',
        'bbox/height',
        'bbox/grasp_success'
    ]
    for i in range(4):
        y_key = 'bbox/y' + str(i)
        x_key = 'bbox/x' + str(i)
        variable_length_keys += [y_key]
        variable_length_keys += [x_key]
        feature_map[y_key] = tf.VarLenFeature(dtype=tf.float32)
        feature_map[x_key] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cy'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cx'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/tan'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/sin_theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cos_theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/width'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/height'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/grasp_success'] = tf.VarLenFeature(dtype=tf.int64)
    # feature_map['bbox/sin_2_theta'] = tf.sin(feature_map['bbox/theta'] * 2.0)
    # feature_map['bbox/cos_2_theta'] = tf.cos(feature_map['bbox/theta'] * 2.0)

    features = tf.parse_single_example(examples_serialized, feature_map)

    features['bbox/count'] = tf.Print(features['bbox/count'], [features['bbox/count']], 'bbox/count')
    bbox_feature_shape = tf.stack([features['bbox/count'], tf.cast(1, tf.int64)])
    # set the length of all the keys based on bbox/count
    # source: https://stackoverflow.com/a/42603692/99379
    for key in variable_length_keys:
        # convert the feature to dense and set the shape as defined in bbox/count
        val = features[key]
        val = tf.sparse_tensor_to_dense(val, default_value=0)
        val = tf.reshape(val, bbox_feature_shape)
        features[key] = val
    return features


def parse_example_proto_redundant_images(examples_serialized, have_image_id=False, bbox_padded_count=1):
    """ Parse data from the tfrecord

    See also: _create_examples_redundant_images()
    """
    feature_map = {
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64),
        'bbox/count': tf.FixedLenFeature([], dtype=tf.int64, default_value=1)
    }

    # TODO(ahundt) remove boolean once we are set up with k-fold cross validation of images and objects
    if have_image_id:
        feature_map['image/id'] = tf.FixedLenFeature([], dtype=tf.int64)

    if bbox_padded_count > 1:
        # if there is an error on bbox/count here,
        # regenerate the dataset and make sure you are parsing
        # single image multi grasp box example data and not
        # redundant single image single grasp box example data
        feature_map['bbox/count'] = tf.FixedLenFeature([], dtype=tf.int64)
        # constant length padded number of data labels in this image
        feature_map['bbox/padded_count'] = tf.FixedLenFeature([], dtype=tf.int64)
        # number of values that are actually just padding
        feature_map['bbox/padding_amount'] = tf.FixedLenFeature([], dtype=tf.int64)

    for i in range(4):
        y_key = 'bbox/y' + str(i)
        x_key = 'bbox/x' + str(i)
        feature_map[y_key] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
        feature_map[x_key] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/cy'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/cx'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/tan'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/theta'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/sin_theta'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/cos_theta'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/width'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/height'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.float32)
    feature_map['bbox/grasp_success'] = tf.FixedLenFeature([bbox_padded_count], dtype=tf.int64)

    features = tf.parse_single_example(examples_serialized, feature_map)

    return features


def sin_cos_height_width_4(height=None, width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/height'], features['bbox/width']]
    else:
        sin_cos_height_width = [sin_theta, cos_theta, height, width]
    return K.stack(sin_cos_height_width, axis=-1)


def sin_cos_width_3(width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/width']]
    else:
        sin_cos_height_width = [sin_theta, cos_theta, width]
    return K.stack(sin_cos_height_width, axis=-1)


def sin2_cos2_height_width_4(height=None, width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/height'], features['bbox/width']]
    else:
        sin_cos_height_width = [sin_theta, cos_theta, height, width]
    return K.stack(sin_cos_height_width, axis=-1)


def sin2_cos2_width_3(width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/width']]
    else:
        sin_cos_height_width = [sin_theta, cos_theta, width]
    return K.stack(sin_cos_height_width, axis=-1)


def approximate_gaussian_ground_truth_image(image_shape, center, grasp_theta, grasp_width, grasp_height, label, sigma_divisor=None):
    """ Gaussian "ground truth" image approximation for a single proposed grasp at a time.

        For use with the Cornell grasping dataset

       see also: ground_truth_images() in cornell_grasp_dataset_writer.py
    """
    if sigma_divisor is None:
        sigma_divisor = FLAGS.sigma_divisor
    grasp_dims = keras.backend.concatenate([grasp_width, grasp_height])
    sigma = keras.backend.max(grasp_dims) / sigma_divisor

    # make sure center value for gaussian is 0.5
    gaussian = gaussian_kernel_2D(image_shape[:2], center=center, sigma=sigma)
    # label 0 is grasp failure, label 1 is grasp success, label 0.5 will have "no effect".
    # gaussian center with label 0 should be subtracting 0.5
    # gaussian center with label 1 should be adding 0.5
    gaussian = ((label * 2) - 1.0) * gaussian
    max_num = K.max(K.max(gaussian), K.placeholder(1.0))
    min_num = K.min(K.min(gaussian), K.placeholder(-1.0))
    gaussian = (gaussian - min_num) / (max_num - min_num)
    return gaussian


def crop_to_gripper_transform(
        input_image_shape, grasp_center_coordinate, grasp_center_rotation_theta, cropped_image_shape,
        random_translation_max_pixels=None, random_rotation=None, seed=None):
    """ Transform and rotate image to be centered and aligned with proposed grasp.

        Given a gripper center coodinate and rotation angle,
        transform and rotate the image so it is centered with 0 theta.
    """
    # TODO(ahundt) projective grasp coordinate transform with xy sin cos rot and multiply also see transform_crop_and_resize_image.
    transforms = []
    input_image_shape_float = tf.cast(input_image_shape, tf.float32)
    cropped_image_shape = tf.cast(cropped_image_shape, tf.float32)
    half_image_shape = (cropped_image_shape / 2)[:2]

    crop_offset = - grasp_center_coordinate + half_image_shape

    crop_offset = crop_offset[::-1]
    if random_translation_max_pixels is not None:
        crop_offset = crop_offset + rcp.random_translation_offset(random_translation_max_pixels)[:2]
    # reverse yx to xy
    transforms += [tf.contrib.image.translations_to_projective_transforms(crop_offset)]
    input_height_f = cropped_image_shape[0]
    input_width_f = cropped_image_shape[1]

    if random_rotation is not None and random_rotation is not False:
        if isinstance(random_rotation, bool) and random_rotation:
            random_rotation = tf.convert_to_tensor(
                [-math.pi, math.pi], dtype=tf.float32, name='random_theta')
        if isinstance(random_rotation, float):
            random_rotation = tf.convert_to_tensor(
                [-random_rotation, random_rotation], dtype=tf.float32, name='random_theta')
        print('random rotation: ' + str(random_rotation))
        theta = tf.random_uniform([1], minval=random_rotation[0], maxval=random_rotation[1], seed=seed, dtype=tf.float32)
        grasp_center_rotation_theta += theta
    transforms += [tf.contrib.image.angles_to_projective_transforms(
                         grasp_center_rotation_theta, input_height_f, input_width_f)]
    transform = tf.contrib.image.compose_transforms(*transforms)
    # TODO(ahundt) rename features random_* to a more general name, and make the same change in random_crop.py
    features = {
        # TODO(ahundt) validate negative theta vs negative theta for grasp pose consistency
        'random_rotation': -grasp_center_rotation_theta,
        'random_translation_offset': crop_offset,
        'random_projection_transform': transform
    }

    return transform, features


def grasp_success_yx_3(grasp_success=None, cy=None, cx=None, features=None):
    if features is not None:
        combined = [K.cast(features['bbox/grasp_success'], 'float32'),
                    features['bbox/cy'], features['bbox/cx']]
    else:
        combined = [K.cast(grasp_success, 'float32'), cy, cx]
    return K.concatenate(combined)


def parse_and_preprocess(
        examples_serialized, is_training=True, crop_shape=None, output_shape=None,
        crop_to=None, random_translation=None, random_rotation=None,
        bbox_padded_count=None,
        preprocessing_mode='tf', seed=None, verbose=1):
    """ Parse an example and perform image preprocessing.

    Right now see the code below for the specific feature strings available.

    crop_shape: The shape to which images should be cropped as an intermediate step, this
        affects how much images can be shifted when random crop is used during training.
    output_shape: The shape to which images should be resized after cropping,
        this is also the final output shape.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
    preprocessing_mode: string for the type of channel preprocessing,
       see keras'
       [preprocess_input()](https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py) for details.


    # Returns

        features


      width is space between the gripper plates
     height is range of possible gripper positions along the line
    """
    if crop_shape is None:
        crop_shape = (FLAGS.crop_height, FLAGS.crop_width, 3)
    if output_shape is None:
        output_shape = (FLAGS.resize_height, FLAGS.resize_width)
        output_shape = K.constant(output_shape, 'int32')
    if bbox_padded_count is None:
        bbox_padded_count = FLAGS.bbox_padded_count
    if FLAGS.redundant_images:
        feature = parse_example_proto_redundant_images(examples_serialized)
    else:
        if bbox_padded_count is not None and bbox_padded_count > 0:
            # TODO(ahundt) get this check from the csv file rather than FLAGS
            feature = parse_example_proto_redundant_images(
                examples_serialized,
                bbox_padded_count=bbox_padded_count)
        else:
            feature = parse_example_proto(examples_serialized)

    if random_translation is None:
        random_translation = FLAGS.random_translation
    if random_rotation is None:
        random_rotation = FLAGS.random_rotation
    if crop_to is None:
        crop_to = FLAGS.crop_to
    if verbose > 0:
        feature['image/encoded'] = tf.Print(feature['image/encoded'], [], 'parse_and_preprocess called')
    # TODO(ahundt) clean up, use grasp_dataset.py as reference, possibly refactor to reuse the code
    sensor_image_dimensions = [FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
    image_buffer = feature['image/encoded']
    image = tf.image.decode_png(image_buffer, channels=FLAGS.sensor_color_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, sensor_image_dimensions)
    feature['image/decoded'] = image
    input_image_shape = tf.shape(image)

    # It is critically important if any image augmentation
    # modifies the coordinate system of the image,
    # such as a random crop, rotation etc, the
    # gripper coordinate features must be updated accordingly
    # and consistently, with visualization to ensure it isn't
    # backwards. An example is +theta rotation vs -theta rotation.
    grasp_center_coordinate = K.stack([feature['bbox/cy'], feature['bbox/cx']], axis=-1)
    print('grasp_center_coordinate: ' + str(grasp_center_coordinate))
    grasp_center_rotation_theta = feature['bbox/theta']
    feature['bbox/sin2_theta'] = tf.sin(feature['bbox/theta'] * 2.0)
    feature['bbox/cos2_theta'] = tf.cos(feature['bbox/theta'] * 2.0)
    feature['sin_cos_height_width_4'] = sin_cos_height_width_4(features=feature)
    feature['sin2_cos2_height_width_4'] = sin2_cos2_height_width_4(features=feature)
    # width is space between the gripper plates
    # height is range of possible gripper positions along the line
    feature['sin_cos_width_3'] = sin_cos_width_3(features=feature)
    feature['sin2_cos2_width_3'] = sin2_cos2_width_3(features=feature)
    random_translation_box = K.cast(feature['sin_cos_height_width_4'][-2:] // 2, 'int32')

    # perform image augmentation with projective transforms
    # TODO(ahundt) add scaling and use that change to augment width (gripper openness) param
    image, preprocessed_grasp_center_coordinate, grasp_center_rotation_theta, random_features = projective_image_augmentation(
        image, is_training, crop_to, crop_shape, random_translation_box, grasp_center_coordinate, grasp_center_rotation_theta, output_shape,
        random_rotation=random_rotation, random_translation=random_translation,
        verbose=verbose)

    if random_features is not None:
        feature.update(random_features)

    feature['image/transformed'] = image

    # perform color augmentation and scaling
    image = preprocess_image(
        image, is_training=is_training, fast_mode=False,
        lower=0.5, upper=1.5, hue_max_delta=0.2,
        brightness_max_delta=32. / 255., mode=preprocessing_mode)

    # generate all the preprocessed features for training
    feature['image/preprocessed'] = image
    feature['image/preprocessed/height'] = K.shape(image)[0]
    feature['image/preprocessed/width'] = K.shape(image)[1]
    feature['image/preprocessed/max_side'] = K.max(K.shape(image))

    feature['bbox/preprocessed/cy'] = preprocessed_grasp_center_coordinate[:, 0]
    feature['bbox/preprocessed/cx'] = preprocessed_grasp_center_coordinate[:, 1]
    # normalized center y, x
    cyn = preprocessed_grasp_center_coordinate[:, 0] / K.cast(feature['image/preprocessed/height'], 'float32')
    cxn = preprocessed_grasp_center_coordinate[:, 1] / K.cast(feature['image/preprocessed/width'], 'float32')
    feature['bbox/preprocessed/cy_cx_normalized_2'] = K.stack([cyn, cxn], axis=-1)
    feature['bbox/preprocessed/theta'] = grasp_center_rotation_theta
    feature['bbox/preprocessed/sin_cos_2'] = K.stack(
        [K.sin(grasp_center_rotation_theta), K.cos(grasp_center_rotation_theta)], axis=-1)
    feature['bbox/preprocessed/sin2_cos2_2'] = K.stack(
        [K.sin(grasp_center_rotation_theta * 2.0), K.cos(grasp_center_rotation_theta * 2.0)], axis=-1)
    # Here we are mapping sin(theta) cos(theta) such that:
    #  - every 180 degrees is a rotation because the grasp plates are symmetrical
    #  - the values are from 0 to 1 so sigmoid can correctly approximate them
    feature['bbox/preprocessed/norm_sin2_cos2_2'] = K.stack(
        [K.sin(grasp_center_rotation_theta * 2.0) / 2 + 0.5,
         K.cos(grasp_center_rotation_theta * 2.0) / 2 + 0.5], axis=-1)
    random_scale = K.constant(1.0, 'float32')
    if 'random_scale' in feature:
        random_scale = feature['random_scale']

    # preprocessed height and width are divided by the longest side in the input image
    # plus they are also scaled if augmentation does scaling.
    feature['bbox/preprocessed/height'] = feature['bbox/height'] * random_scale / K.cast(feature['image/preprocessed/max_side'], 'float32')
    feature['bbox/preprocessed/width'] = feature['bbox/width'] * random_scale / K.cast(feature['image/preprocessed/max_side'], 'float32')
    feature['bbox/preprocessed/logarithm_height_width_2'] = K.stack(
        [K.log(feature['bbox/preprocessed/height'] + K.epsilon()),
         K.log(feature['bbox/preprocessed/width'] + K.epsilon())], axis=-1)
    # TODO(ahundt) difference between "redundant_images" and regular proto parsing, figure out how to deal with grasp_success rename properly
    feature['grasp_success'] = feature['bbox/grasp_success']
    multi_grasp_success = K.expand_dims(feature['bbox/grasp_success'])
    grasp_success_coordinate_label = K.concatenate(
        [K.expand_dims(tf.cast(feature['bbox/grasp_success'], tf.float32)), grasp_center_coordinate], axis=-1)
    # make coordinate labels 4d because that's what keras expects
    grasp_success_coordinate_label = K.expand_dims(K.expand_dims(grasp_success_coordinate_label))
    feature['grasp_success_yx_3'] = grasp_success_coordinate_label
    # expand the dimension of width and heights to match the multigrasp data dimensions
    bbox_width = K.expand_dims(feature['bbox/preprocessed/width'])
    bbox_height = K.expand_dims(feature['bbox/preprocessed/height'])
    # preprocessing adds some extra normalization and incorporates changes in the values due to augmentatation.
    feature['preprocessed_sin_cos_width_3'] = K.concatenate([feature['bbox/preprocessed/sin_cos_2'], ], axis=-1)
    feature['preprocessed_norm_sin2_cos2_width_3'] = K.concatenate([feature['bbox/preprocessed/norm_sin2_cos2_2'], bbox_width], axis=-1)
    feature['preprocessed_norm_sin2_cos2_height_width_4'] = K.concatenate(
        [feature['bbox/preprocessed/norm_sin2_cos2_2'], bbox_height, bbox_width], axis=-1)

    # This feature should be useful for pixelwise predictions
    grasp_success_sin2_cos2_hw_yx_7 = K.concatenate(
        [tf.cast(multi_grasp_success, tf.float32), feature['bbox/preprocessed/norm_sin2_cos2_2'],
         bbox_height, bbox_width, preprocessed_grasp_center_coordinate], axis=-1)
    feature['grasp_success_sin2_cos2_hw_yx_7'] = K.expand_dims(K.expand_dims(grasp_success_sin2_cos2_hw_yx_7))

    # print('>>>>CREATE CXN: ' + str(cxn) + ' CYN: ' + str(cyn))
    # cxn = tf.Print(cxn, [cxn, cyn, feature['bbox/preprocessed/cy_cx_normalized_2']], '>>>> CXN CYN combined ')
    # v = K.constant([0.5], 'float32')
    # v1 = K.constant([0.05], 'float32')
    # v2 = K.constant([100], 'float32')
    multi_bbox_height = K.expand_dims(feature['bbox/preprocessed/height'])
    multi_bbox_width = K.expand_dims(feature['bbox/preprocessed/width'])
    # This feature should be useful for grasp regression
    grasp_success_norm_sin2_cos2_hw_yx_7 = K.concatenate(
        [K.cast(multi_grasp_success, 'float32'), feature['bbox/preprocessed/norm_sin2_cos2_2'],
         multi_bbox_height, multi_bbox_width, feature['bbox/preprocessed/cy_cx_normalized_2']])
    feature['grasp_success_norm_sin2_cos2_hw_yx_7'] = grasp_success_norm_sin2_cos2_hw_yx_7
    feature['norm_sin2_cos2_hw_yx_6'] = grasp_success_norm_sin2_cos2_hw_yx_7[1:]
    feature['grasp_success_norm_sin2_cos2_hw_5'] = grasp_success_norm_sin2_cos2_hw_yx_7[:5]
    feature['sin2_cos2_hw_4'] = grasp_success_norm_sin2_cos2_hw_yx_7[1:5]
    preprocessed_norm_sin2_cos2_w_yx_5 = K.concatenate(
        [feature['bbox/preprocessed/norm_sin2_cos2_2'],
         multi_bbox_width, feature['bbox/preprocessed/cy_cx_normalized_2']])
    feature['preprocessed_norm_sin2_cos2_w_yx_5'] = preprocessed_norm_sin2_cos2_w_yx_5
    feature['preprocessed_norm_sin2_cos2_w_3'] = preprocessed_norm_sin2_cos2_w_yx_5[:-2]

    # TODO(ahundt) reenable pixelwise gaussians but with separate success and failure layers and compare performance against segmentation_gaussian_measurement()
    if False:
        feature['grasp_success_2D'] = approximate_gaussian_ground_truth_image(
            image_shape=keras.backend.int_shape(image),
            center=[feature['bbox/cy'], feature['bbox/cx']],
            grasp_theta=feature['bbox/theta'],
            grasp_width=feature['bbox/width'],
            grasp_height=feature['bbox/height'],
            label=feature['grasp_success'])

    if verbose > 0:
        print(feature)

    return feature


def projective_image_augmentation(
        image, is_training, crop_to, crop_shape, random_translation_box,
        grasp_center_coordinate, grasp_center_rotation_theta, output_shape,
        random_rotation=None, random_translation=None, verbose=1):
    """ perform image augmentation with projective transforms
    """
    # TODO(ahundt) add scaling and use that change to augment width (gripper openness) param
    input_image_shape = tf.shape(image)
    central_crop = False
    # random features is a map from strings to
    # random feature variables such as
    # the amount of rotation
    random_features = None
    if not is_training:
        # disable random rotation and translation if we are not training
        random_rotation = False
        random_translation = False

    # Images can only be transformed once for all the gripper coordinates,
    # so choose a random coordinate to use as the dominant coordinate
    # for the image transforms if there is only one coordinate, use that.
    # TODO(ahundt) should this call occur before/outside this function?
    random_center_coordinate = choose_random_center_coordinate(grasp_center_coordinate)

    # by default we won't be doing any rotations based on the grasp box theta
    crop_to_gripper_theta = K.constant(0, 'float32', name='crop_to_gripper_theta')
    translation_in_box = K.constant(0, 'float32', name='translation_in_box')

    # configure the image transform based on the training mode
    if crop_to == 'resize_height_and_resize_width':
        # Note: this option only works well if the crop size is similar to the input size
        if is_training and (random_rotation or random_translation):
            transform, random_features = rcp.random_projection_transform(
                K.shape(image), crop_shape, scale=False, rotation=random_rotation, translation=random_translation)
        else:
            # simply do a central crop then a resize when not training
            # to match the input without changes
            central_crop = True
            transform = None
    elif 'center_on_gripper_grasp_box' in crop_to:
        if 'rotate_upright' in crop_to:
            crop_to_gripper_theta = grasp_center_rotation_theta
            # limit random rotation to 15 degrees to stay within jaccard metrics
            if is_training and random_rotation:
                # set the range limit of the random rotations
                random_rotation = math.pi / 12
            # random_rotation is True, which allows arbitrary random rotation

        if is_training and random_translation:
            # translation in box is the range limit of the random translations
            translation_in_box = random_translation_box
            translation_in_box = tf.minimum(translation_in_box[0], translation_in_box[1])
            # TODO(ahundt) possibly add scale
        else:
            translation_in_box = None
    elif crop_to == 'image_contains_grasp_box_center':
        if is_training and random_translation:
            # translation in box is the range limit of the random translations,
            # translate by up to 1/3 the shorter crop shape size to prevent
            # a grasp box in the corner from being rotated out of the image
            crop_shape_tensor = tf.convert_to_tensor(
                crop_shape[:2], dtype=tf.int32, name='crop_shape_tensor')
            translation_in_box = K.cast(crop_shape_tensor // 3, 'int32')
            translation_in_box = tf.minimum(translation_in_box[0], translation_in_box[1])
            # TODO(ahundt) possibly add scale
        elif not is_training:
            # simply do a central crop then a resize when not training
            # to match the input without changes
            central_crop = True
            transform = None
    else:
        raise ValueError('Unsupported choice of crop_to: %s try resize_height_and_resize_width, '
                         'image_contains_grasp_box_center, center_on_gripper_grasp_box, or '
                         'center_on_gripper_grasp_box_and_rotate_upright' % (crop_to))

    if ('center_on_gripper_grasp_box' in crop_to or
            (crop_to == 'image_contains_grasp_box_center' and is_training)):
        # crop the image to based on a gripper position and orientation
        transform, random_features = crop_to_gripper_transform(
            input_image_shape, random_center_coordinate,
            crop_to_gripper_theta, crop_shape,
            random_translation_max_pixels=translation_in_box,
            random_rotation=random_rotation)
    # verbose = 1

    if verbose > 0:
        image = tf.Print(
            image,
            [grasp_center_coordinate, transform],
            'grasp_center_coordinate before, and transform:')
    image, preprocessed_grasp_center_coordinate = rcp.transform_crop_and_resize_image(
        image, crop_shape=crop_shape, resize_shape=output_shape, central_crop=central_crop,
        transform=transform, coordinate=grasp_center_coordinate)
    if verbose > 0:
        image = tf.Print(
            image,
            [preprocessed_grasp_center_coordinate],
            'preprocessed_grasp_center_coordinate after:')

    if random_features is not None:
        if 'random_rotation' in random_features:
            # TODO(ahundt) validate if we must subtract or add based on the transform
            grasp_center_rotation_theta += random_features['random_rotation']
    return image, preprocessed_grasp_center_coordinate, grasp_center_rotation_theta, random_features


def choose_random_center_coordinate(grasp_center_coordinate):
    """

    Images can only be transformed once for all the gripper coordinates,
    so choose a random coordinate to use as the dominant coordinate
    for the image transforms if there is only one coordinate, use that.

    # Arguments

    grasp_center_coordinate: tensor of dimension nx2 containing grasp center coordinates
    """
    if len(K.int_shape(grasp_center_coordinate)) > 1:
        # can only transform to the location of one grasp coordinate, so choose one at random
        grasp_center_coordinate_shape = K.int_shape(grasp_center_coordinate)
        print('grasp_center_coordinate_shape: ' + str(grasp_center_coordinate_shape))
        maxval = grasp_center_coordinate_shape[0]
        random_coordinate_index = tf.random_uniform([1], maxval=maxval, dtype=tf.int32)[0]
        random_center_coordinate = grasp_center_coordinate[random_coordinate_index, :]
    else:
        # there is only one center coordinate
        random_center_coordinate = grasp_center_coordinate
    return random_center_coordinate


def filter_features(feature_map, label_features_to_extract, data_features_to_extract, verbose=0):
    """ Strip out all features that aren't needed to reduce processing time.
    """
    if verbose:
        feature_map['image/preprocessed'] = tf.Print(feature_map['image/preprocessed'], [], 'filter_features called')
    simplified_feature = {}
    for feature_name in data_features_to_extract:
        simplified_feature[feature_name] = feature_map[feature_name]
    for feature_name in label_features_to_extract:
        simplified_feature[feature_name] = feature_map[feature_name]
    return simplified_feature


def filter_grasp_success_only(x, verbose=0):
    """ Only traverse data where grasp_success is true
    """
    should_filter = tf.equal(x['grasp_success'][0], K.constant(1, 'int64'))
    if verbose:
        should_filter = tf.Print(should_filter, [should_filter, x['grasp_success']], 'filter_fn x should filter, grasp_success ')
    return should_filter


def yield_record(
        tfrecord_filenames, label_features_to_extract=None, data_features_to_extract=None,
        parse_example_proto_fn=parse_and_preprocess, batch_size=32,
        device='/cpu:0', steps=None, buffer_size=int(1e6),
        shuffle=True, shuffle_buffer_size=100, num_parallel_calls=None,
        apply_filter=False, filter_fn=filter_grasp_success_only,
        dynamic_pad=False, **kwargs):
    """ TFRecord data python generator.

    # Arguments

        steps: integer indicating number of times to repeat the dataset.
            None means unlimited repeats. This is not the same as a keras step!
        parse_example_proto_fn: A function which takes a single example
            from the tfrecord as input and returns a dictionary from
            strings to feature tensors as output.
            See `parse_and_preprocess()` for an example.
        label_features_to_extract: return only specific
            feature strings as the label values, default of
            None returns all features. Specifying
            features reduces overhead and works with
            yield_record() for keras generator compatibility.
        data_features_to_extract: return only specific
            feature strings as the input values,
            default of None returns all features. Specifying
            features reduces overhead and works with
            yield_record() for keras generator compatibility.
        is_training: performs additional data augmentation when true.
        success_only: filters out any grasps that aren't labeled as successful grasps.
        filter_fn: A function which takes the feature tensor dict as input and returns
            a tensor boolean. Used to filter out examples that should be skipped.
            See `filter_grasp_success_only()` for an example.
        dynamic_pad: dynamically pad data which is of variable length, for example
            when one image has a variable number of grasp labels.
            see https://www.tensorflow.org/api_docs/python/tf/train/batch
        kwargs: Any additional parameters you specify will be passed directly to
            the parse_example_proto_fn.
    """
    if num_parallel_calls is None:
        num_parallel_calls = FLAGS.num_readers

    if shuffle and isinstance(tfrecord_filenames, list):
        random.shuffle(tfrecord_filenames)
    # based_on https://github.com/visipedia/tfrecords/blob/master/iterate_tfrecords.py
    # with tf.device(device):
    # sess = keras.backend.get_session()
    # tf_graph = tf.get_default_graph()

    with tf.Session() as sess:

        dataset = tf.data.TFRecordDataset(
            tfrecord_filenames, buffer_size=buffer_size)
        # Repeat the input indefinitely.
        dataset = dataset.repeat(count=steps)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)

        # call the parse_example_proto_fn with the is_training flag set.
        def parse_fn_with_kwargs(example):
            return parse_example_proto_fn(
                examples_serialized=example,
                **kwargs)
        dataset = dataset.map(
            map_func=parse_fn_with_kwargs,
            num_parallel_calls=num_parallel_calls)

        if apply_filter:
            # success_only mode skipps all grasps labeled a failure.
            dataset = dataset.filter(filter_fn)

        # Here we extract only the features required for this run
        if data_features_to_extract is not None:
            def get_simplified_features(feature_map):
                return filter_features(feature_map,
                                       label_features_to_extract,
                                       data_features_to_extract)
            dataset = dataset.map(
                map_func=get_simplified_features,
                num_parallel_calls=num_parallel_calls)
        if dynamic_pad:
            dataset = dataset.padded_batch(batch_size=batch_size)
        else:
            dataset = dataset.batch(batch_size=batch_size)  # Parse the record into tensors.
        dataset = dataset.prefetch(batch_size * 5)
        # tensor_iterator = dataset.make_initializable_iterator()
        tensor_iterator = dataset.make_one_shot_iterator()
        #     # Construct a Reader to read examples from the .tfrecords file
        #     reader = tf.TFRecordReader()
        #     _, serialized_example = reader.read(filename_queue)

        #     features = decode_serialized_example(serialized_example, features_to_extract)

        # coord = tf.train.Coordinator()
        # sess = K.get_session()
        # tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
        # tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            # while not coord.should_stop():
            # features = sess.run(tensor_iterator.initializer)
            # If `Iterator.get_next()` is being called inside a training loop,
            # which will cause gradual slowdown and eventual resource exhaustion.
            # If this is the case, restructure your code to call `next_element = iterator.get_next()
            # once outside the loop, and use `next_element` inside the loop.
            next_element = tensor_iterator.get_next()
            while True:
                # print('start iteration')
                features = sess.run(next_element)
                # print('run complete')

                # extract the features in a specific order if
                # features_to_extract is specified,
                # useful for yielding to a generator
                # like with keras.model.fit().
                # outputs should be the outputs of the
                # dataset API, consisting of two lists:
                # [[inputs], [labels]]
                # "labels" is called "targets" by keras
                #
                # See documentation for keras.model.fit_generator()
                # https://keras.io/models/model/
                #
                # print('start list_processing')
                if data_features_to_extract is not None and label_features_to_extract is not None:
                    outputs = ([features[feature_name] for feature_name in data_features_to_extract],
                               [features[feature_name] for feature_name in label_features_to_extract])
                # print('done list proc, about to yield')
                    yield outputs
                else:
                    yield features
                #

        except tf.errors.OutOfRangeError as e:
            if steps is not None:
                raise e
            else:
                pass


def print_feature(feature_map, feature_name):
    """ Print the contents of a feature map
    """
    print(feature_name + ': ' + str(feature_map[feature_name]))


def main(argv):
    # tf.enable_eager_execution()
    batch_size = 1
    is_training = True
    glob_path = os.path.join(FLAGS.data_dir, FLAGS.tfrecord_filename_base + '-imagewise-*' + '.tfrecord')
    tfrecord_files = glob.glob(glob_path)
    print('tfrecord_files: ' + str(tfrecord_files))

    crop_to = 'image_contains_grasp_box_center'
    # crop_to = 'resize_height_and_resize_width'
    # crop_to = 'center_on_gripper_grasp_box_and_rotate_upright'

    # success_only flag must be false in multigrasp case
    success_only = False
    shuffle = False
    # FLAGS.random_translation = False
    # FLAGS.random_rotation = False
    if 'multigrasp' in glob_path:
        FLAGS.redundant_images = False
    elif 'singlegrasp' in glob_path:
        FLAGS.redundant_images = True

    redundant_images = FLAGS.redundant_images

    for example_dict in tqdm(yield_record(
            tfrecord_files, is_training=is_training,
            batch_size=batch_size, apply_filter=success_only,
            shuffle=shuffle, crop_to=crop_to)):
        # loop through one example
        if redundant_images:
            grasp_visualization.visualize_redundant_images_example(
                example_dict, showTextBox=True, blocking=True)
        else:
            img = np.squeeze(example_dict['image/decoded'])
            grasp_visualization.visualize_example(img, example_dict)


if __name__ == '__main__':
    tf.app.run(main=main)
