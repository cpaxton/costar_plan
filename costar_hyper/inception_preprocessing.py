# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#
# Source:
# https://github.com/tensorflow/models/blob/cbb624791068590026f50f01d370d9328fe8ebf2/research/slim/preprocessing/inception_preprocessing.py
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
import PIL.Image as Image

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None,
                  lower=0.75, upper=1.25, hue_max_delta=0.1,
                  brightness_max_delta=16. / 255.):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Note that in the imagenet training code lower is 1.25 and upper is upper,
    and applies to contrast and saturation, hue_max_delta is hue_max_delta, and
    brightness max delta is 32./255., we are modifying these to smaller values
    with grasp attempts because they tend to affect visibility of the gripper,
    often causing it to become nearly all dark pixels.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
            else:
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def blend_images_np(image, image2, alpha=0.5):
    """Draws image2 on an image.
    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      image2: a uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)
    Raises:
      ValueError: On incorrect data type for image or image2s.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if image2.dtype != np.uint8:
        raise ValueError('`image2` not of type np.uint8')
    if image.shape[:2] != image2.shape:
        raise ValueError('The image has spatial dimensions %s but the image2 has '
                         'dimensions %s' % (image.shape[:2], image2.shape))
    pil_image = Image.fromarray(image)
    pil_image2 = Image.fromarray(image2)

    pil_image = Image.blend(pil_image, pil_image2, alpha)
    np.copyto(image, np.array(pil_image.convert('RGB')))
    return image


def blend_images(image, image2, should_blend=0, alpha=0.5, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Note that in the imagenet training code lower is 1.25 and upper is upper,
    and applies to contrast and saturation, hue_max_delta is hue_max_delta, and
    brightness max delta is 32./255., we are modifying these to smaller values
    with grasp attempts because they tend to affect visibility of the gripper,
    often causing it to become nearly all dark pixels.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'blend_images', [image, image2]):
        if should_blend == 0:
            return image
        else:
            image = tf.py_func(blend_images_np, [image, image2, alpha])
        return image


def preprocess_input_0_1(image, mode='tf', data_format=None):
    """ Assumes images with range [0, 1]

    Shifts the channel value ranges for direct input into a neural network.

      image: A floating point image with range [0, 1]
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
    """
    if mode == 'tf':
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
    else:
        # shift back to 0-255
        image *= 255.0
        image = keras.applications.imagenet_utils.preprocess_input(
            image, mode=mode, data_format=data_format)
    return image


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def preprocess_for_train(image,
                         fast_mode=True,
                         lower=0.75,
                         upper=1.25,
                         hue_max_delta=0.1,
                         brightness_max_delta=16. / 255.,
                         scope=None,
                         add_image_summaries=True,
                         mode='tf', data_format=None,
                         image2=None,
                         blend_alpha=0.5,
                         skip_blend_every_n=10):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations (i.e.
        bi-cubic resizing, random_hue or random_contrast).
      scope: Optional scope for name_scope.
      add_image_summaries: Enable image summaries.
      image2: An optional second image tensor to blend in to the main image.
      blend_alpha: The alpha transparency channel in range [0, 1] for image2.
        Only applies if image2 is None.
      skip_blend_every_n: Blending should not be performed randomly every n
        examples. Only applies if image2 is None.

    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """
    with tf.name_scope(scope, 'distort_image', [image]):
        # blend images with a random selector

        # Randomly distort the colors. There are 4 ways to do it.
        if image2 is not None:
            image = apply_with_random_selector(
                image, lambda x, x2, should_blend: blend_images(image=x,
                                                                image2=x2,
                                                                should_blend=should_blend,
                                                                alpha=blend_alpha),
                num_cases=skip_blend_every_n)

        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if add_image_summaries:
            tf.summary.image('cropped_resized_image',
                             tf.expand_dims(image, 0))

        # Randomly distort the colors. There are 4 ways to do it.
        image = apply_with_random_selector(
            image,
            lambda x, ordering:
                distort_color(
                    image=x, color_ordering=ordering, fast_mode=fast_mode,
                    scope=scope, lower=lower, upper=upper, hue_max_delta=hue_max_delta,
                    brightness_max_delta=brightness_max_delta),
            num_cases=4)

        if add_image_summaries:
            tf.summary.image('final_distorted_image',
                             tf.expand_dims(image, 0))

        image = preprocess_input_0_1(image, mode=mode, data_format=data_format)
        return image


def preprocess_for_eval(image, height=None, width=None,
                        central_fraction=None, scope=None,
                        mode='tf', data_format=None):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
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
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])

        image = preprocess_input_0_1(image, mode=mode, data_format=data_format)
        return image


def preprocess_image(image, height=None, width=None,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     lower=0.75,
                     upper=1.25,
                     hue_max_delta=0.1,
                     brightness_max_delta=16. / 255.,
                     add_image_summaries=True,
                     mode='tf', data_format=None,
                     blend_alpha=0.5,
                     skip_blend_every_n=10,
                     image2=None):
    """Pre-process one image for training or evaluation.

    Args:
      image: 3-D Tensor [height, width, channels] with the image. If dtype is
        tf.float32 then the range should be [0, 1], otherwise it would converted
        to tf.float32 assuming that the range is [0, MAX], where MAX is largest
        positive representable number for int(8/16/32) data type (see
        `tf.image.convert_image_dtype` for details).
      height: integer, image expected height.
      width: integer, image expected width.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations.
      add_image_summaries: Enable image summaries.
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

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    if is_training:
        return preprocess_for_train(image, fast_mode,
                                    add_image_summaries=add_image_summaries,
                                    mode=mode, data_format=data_format,
                                    lower=lower,
                                    upper=upper,
                                    hue_max_delta=hue_max_delta,
                                    brightness_max_delta=brightness_max_delta,
                                    image2=image2)
    else:
        return preprocess_for_eval(image, height, width, mode=mode, data_format=data_format)
