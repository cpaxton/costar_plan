import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.layers.core import RepeatVector
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.merge import Concatenate
from keras.layers.merge import _Merge
from keras.models import Model
from keras.layers import Lambda
from keras.layers import Reshape
from keras.applications.imagenet_utils import _obtain_input_shape

import keras_contrib
from keras_contrib.applications.densenet import DenseNetFCN
from keras_contrib.applications.densenet import DenseNet
from keras_contrib.applications.densenet import DenseNetImageNet121

from keras.engine import Layer


def tile_vector_as_image_channels(vector_op, image_shape):
    """

    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.
    """
    ivs = K.shape(vector_op)
    vector_op = K.reshape(vector_op, [ivs[0], 1, 1, ivs[1]])
    vector_op = K.tile(vector_op, K.stack([1, image_shape[1], image_shape[2], 1]))
    return vector_op


def grasp_model_pretrained(clear_view_image_op,
                           current_time_image_op,
                           input_vector_op,
                           input_image_shape=None,
                           input_vector_op_shape=None,
                           growth_rate=12,
                           reduction=0.75,
                           dense_blocks=4,
                           include_top=True,
                           dropout_rate=0.0,
                           train_densenet=False):
    """export CUDA_VISIBLE_DEVICES="1" && python grasp_train.py --random_crop=1 --batch_size=1 --grasp_model grasp_model_pretrained --resize_width=320 --resize_height=256
    """
    if input_vector_op_shape is None:
        input_vector_op_shape = [K.shape(input_vector_op)[0], 5]
        input_vector_op = K.reshape(input_vector_op, input_vector_op_shape)
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]
    
    print('input_image_shape:', input_image_shape)
    print('shape(input_image_shape:', input_image_shape)

    clear_view_model = DenseNetImageNet121(
        input_shape=input_image_shape,
        input_tensor=clear_view_image_op,
        include_top=False)

    current_time_model = DenseNetImageNet121(
        input_shape=input_image_shape,
        input_tensor=current_time_image_op,
        include_top=False)

    if not train_densenet:
        for layer in clear_view_model.layers:
            layer.trainable = False
        for layer in current_time_model.layers:
            layer.trainable = False

    print('input_vector_op before tile: ', input_vector_op)
    input_vector_op = tile_vector_as_image_channels(
        input_vector_op,
        K.shape(clear_view_model.outputs[0]))
    print('input_vector_op after tile: ', input_vector_op)
    print('clear_view_model.outputs: ', clear_view_model.outputs)
    print('current_time_model.outputs: ', current_time_model.outputs)
    combined_input_data = tf.concat([clear_view_model.outputs[0], input_vector_op, current_time_model.outputs[0]], -1)

    print('combined_input_data.get_shape().as_list():', combined_input_data.get_shape().as_list())
    combined_input_shape = K.shape(clear_view_model.outputs[0]).get_shape().as_list()
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[-1]
    model_name = 'densenet'
    if model_name == 'dense':
        final_nb_layer = 4
        nb_filter = combined_input_shape[-1]
        weight_decay = 1e-4
        # The last dense_block does not have a transition_block
        x, nb_filter = keras_contrib.applications.densenet.__dense_block(
            combined_input_data, final_nb_layer, nb_filter, growth_rate, 
            bottleneck=True, dropout_rate=dropout_rate, weight_decay=weight_decay)

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
    if model_name == 'densenet':
        model = DenseNet(input_shape=combined_input_shape[1:],
                         include_top=include_top,
                         input_tensor=combined_input_data,
                         activation='sigmoid',
                         classes=1,
                         nb_filter=int(combined_input_shape[-1]*2),
                         growth_rate=growth_rate,
                         reduction=reduction,
                         nb_dense_block=dense_blocks,
                         dropout_rate=dropout_rate,
                         nb_layers_per_block=[6, 12, 24, 16],
                         subsample_initial_block=False,
                         weight_decay=1e-4,
                         pooling=None,
                         bottleneck=True)
    return model


def grasp_model(clear_view_image_op,
                current_time_image_op,
                input_vector_op,
                input_image_shape=None,
                input_vector_op_shape=None,
                growth_rate=12,
                reduction=0.5,
                dense_blocks=4,
                include_top=True,
                dropout_rate=0.0):
    if input_vector_op_shape is None:
        input_vector_op_shape = [5]
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]
    print('input_vector_op pre tile: ', input_vector_op)

    input_vector_op = tile_vector_as_image_channels(input_vector_op, K.shape(clear_view_image_op))

    combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    combined_input_shape = input_image_shape
    # add up the total number of channels
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    # initial number of filters should be
    # the number of input channels times the growth rate
    # nb_filters = combined_input_shape[-1] * growth_rate
    print('combined_input_shape: ', combined_input_shape)
    # print('nb_filters: ', nb_filters)
    print('combined_input_data: ', combined_input_data)
    print('clear_view_image_op: ', clear_view_image_op)
    print('current_time_image_op: ', current_time_image_op)
    print('input_vector_op: ', input_vector_op)
    model = DenseNet(input_shape=combined_input_shape,
                     include_top=include_top,
                     input_tensor=combined_input_data,
                     activation='sigmoid',
                     classes=1,
                     weights=None,
                     #  nb_filter=nb_filters,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     nb_dense_block=dense_blocks,
                     dropout_rate=dropout_rate)
    return model


def grasp_model_segmentation(clear_view_image_op=None,
                             current_time_image_op=None,
                             input_vector_op=None,
                             input_image_shape=None,
                             input_vector_op_shape=None,
                             growth_rate=12,
                             reduction=0.5,
                             dense_blocks=4,
                             dropout_rate=0.0):
    if input_vector_op_shape is None:
        input_vector_op_shape = [5]
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]

    if input_vector_op is not None:
        ims = tf.shape(clear_view_image_op)
        ivs = tf.shape(input_vector_op)
        input_vector_op = tf.reshape(input_vector_op, [1, 1, 1, ivs[0]])
        input_vector_op = tf.tile(input_vector_op, tf.stack([ims[0], ims[1], ims[2], ivs[0]]))

    combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    combined_input_shape = input_image_shape
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    model = DenseNetFCN(input_shape=combined_input_shape,
                        include_top='global_average_pooling',
                        input_tensor=combined_input_data,
                        activation='sigmoid',
                        growth_rate=growth_rate,
                        reduction=reduction,
                        nb_dense_block=dense_blocks)
    return model
