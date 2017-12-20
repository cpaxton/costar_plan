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
from keras_contrib.applications.resnet import ResNet

from keras.engine import Layer


def tile_vector_as_image_channels(vector_op, image_shape):
    """
    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.

    # Params

      vector_op: A tensor vector to tile.
      image_shape: A list of integers [width, height] with the desired dimensions.
    """
    with K.name_scope('tile_vector_as_image_channels'):
        ivs = K.int_shape(vector_op)
        # reshape the vector into a single pixel
        vector_pixel_shape = [ivs[0], 1, 1, ivs[1]]
        vector_op = K.reshape(vector_op, vector_pixel_shape)
        # tile the pixel into a full image
        tile_dimensions = [1, image_shape[1], image_shape[2], 1]
        vector_op = K.tile(vector_op, tile_dimensions)
        if K.backend() is 'tensorflow':
            output_shape = [ivs[0], image_shape[1], image_shape[2], ivs[1]]
            vector_op.set_shape(output_shape)
        return vector_op


def concat_images_with_tiled_vector(images, vector):
    """Combine a set of images with a vector, tiling the vector at each pixel in the images and concatenating on the channel axis.

    # Params

        images: list of images with the same dimensions
        vector: vector to tile on each image. If you have
            more than one vector, simply concatenate them
            all before calling this function.

    # Returns

    """
    with K.name_scope('concat_images_with_tiled_vector'):
        if not isinstance(images, list):
            images = [images]
        image_shape = K.int_shape(images[0])
        tiled_vector = tile_vector_as_image_channels(vector, image_shape)
        images.append(tiled_vector)
        combined = K.concatenate(images)

        return combined


def concat_images_with_tiled_vector_layer(images, vector, image_shape=None, vector_shape=None):
    """Tile a vector as if it were channels onto every pixel of an image.

    This version is designed to be used as layers within a Keras model.

    # Params
       images: a list of images to combine, must have equal dimensions
       vector: the 1D vector to tile onto every pixel
       image_shape: Tuple with 3 entries defining the shape (batch, height, width)
           images should be expected to have, do not specify the number
           of batches.
       vector_shape: Tuple with 3 entries defining the shape (batch, height, width)
           images should be expected to have, do not specify the number
           of batches.
    """
    with K.name_scope('concat_images_with_tiled_vector_layer'):
        if not isinstance(images, list):
            images = [images]
        if vector_shape is None:
            # check if K.shape, K.int_shape, or vector.get_shape().as_list()[1:] is better
            # https://github.com/fchollet/keras/issues/5211
            vector_shape = K.int_shape(vector)[1:]
        if image_shape is None:
            # check if K.shape, K.int_shape, or image.get_shape().as_list()[1:] is better
            # https://github.com/fchollet/keras/issues/5211
            image_shape = K.int_shape(images[0])[1:]
        vector = Reshape([1, 1, vector_shape[-1]])(vector)
        tile_shape = (int(1), int(image_shape[0]), int(image_shape[1]), int(1))
        tiled_vector = Lambda(lambda x: K.tile(x, tile_shape))(vector)
        x = Concatenate(axis=-1)([] + images + [tiled_vector])
    return x


def grasp_model_resnet(clear_view_image_op,
                       current_time_image_op,
                       input_vector_op,
                       input_image_shape=None,
                       input_vector_op_shape=None,
                       include_top=True,
                       dropout_rate=0.0,
                       initial_filters=96,
                       initial_kernel_size=(3, 3),
                       activation='sigmoid',
                       repetitions=None):
    if repetitions is None:
        repetitions = [1, 1, 1, 1]
    combined_input_data = concat_images_with_tiled_vector([clear_view_image_op, current_time_image_op], input_vector_op)
    combined_input_shape = K.int_shape(combined_input_data)
    # the input shape should be a tuple of 3 values
    # if the batch size is present, strip it out
    # for call to ResNet constructor.
    if len(combined_input_shape) == 4:
        combined_input_shape = combined_input_shape[1:]
    model = ResNet(input_shape=combined_input_shape,
                   classes=1,
                   block='bottleneck',
                   repetitions=repetitions,
                   include_top=include_top,
                   input_tensor=combined_input_data,
                   activation='sigmoid',
                   initial_filters=initial_filters,
                   initial_kernel_size=initial_kernel_size,
                   initial_pooling=None,
                   final_pooling=None,
                   dropout=dropout_rate)
    return model


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
        input_vector_op_shape = K.int_shape(input_vector_op)
    if input_image_shape is None:
        input_image_shape = K.int_shape(clear_view_image_op)

    print('input_image_shape:', input_image_shape)

    clear_view_model = ResNet50(
        input_shape=input_image_shape,
        input_tensor=clear_view_image_op,
        include_top=False)

    current_time_model = ResNet50(
        input_shape=input_image_shape,
        input_tensor=current_time_image_op,
        include_top=False)

    if not train_densenet:
        for layer in clear_view_model.layers:
            layer.trainable = False
        for layer in current_time_model.layers:
            layer.trainable = False

    clear_view_unpooled_layer = clear_view_model.layers[-2].get_output_at(0)
    unpooled_shape = clear_view_unpooled_layer.get_shape().as_list()
    print('clear_view_unpooled_layer: ', clear_view_unpooled_layer)
    print('unpooled_shape: ', unpooled_shape)
    current_time_unpooled = current_time_model.layers[-2].get_output_at(0)
    print('input_vector_op before tile: ', input_vector_op)
    input_vector_op = tile_vector_as_image_channels(
        input_vector_op,
        unpooled_shape
        )
    print('input_vector_op after tile: ', input_vector_op)
    print('clear_view_model.outputs: ', clear_view_model.outputs)
    print('current_time_model.outputs: ', current_time_model.outputs)
    combined_input_data = tf.concat([clear_view_unpooled_layer, input_vector_op, current_time_unpooled], -1)

    print('combined_input_data.get_shape().as_list():', combined_input_data.get_shape().as_list())
    combined_input_shape = combined_input_data.get_shape().as_list()
    combined_input_shape[-1] = unpooled_shape[-1] * 2 + input_vector_op_shape[-1]
    model_name = 'resnet'
    if model_name == 'dense':
        final_nb_layer = 4
        nb_filter = combined_input_shape[-1]
        weight_decay = 1e-4
        # The last dense_block does not have a transition_block
        x, nb_filter = keras_contrib.applications.densenet.__dense_block(
            combined_input_data, final_nb_layer, nb_filter, growth_rate,
            bottleneck=True, dropout_rate=dropout_rate, weight_decay=weight_decay)

        concat_axis = -1
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
    elif model_name == 'densenet':
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
    elif model_name == 'resnet':
        print('combined_input_shape: ', combined_input_shape)
        print('combined_input_data: ', combined_input_data)
        model = ResNet(input_shape=combined_input_shape[1:],
                       classes=1,
                       block='bottleneck',
                       repetitions=[1, 1, 1, 1],
                       include_top=include_top,
                       input_tensor=combined_input_data,
                       activation='sigmoid',
                       initial_filters=96,
                       initial_kernel_size=(3, 3),
                       pooling=None,
                       dropout=dropout_rate)

    return model


def grasp_model(clear_view_image_op,
                current_time_image_op,
                input_vector_op,
                input_image_shape=None,
                input_vector_op_shape=None,
                depth=40,
                growth_rate=36,
                reduction=0.5,
                dense_blocks=3,
                include_top=True,
                dropout_rate=0.0):
    if input_vector_op_shape is None:
        input_vector_op_shape = input_vector_op.get_shape().as_list()
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
                     depth=depth,
                     activation='sigmoid',
                     classes=1,
                     weights=None,
                     #  nb_filter=nb_filters,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     nb_dense_block=dense_blocks,
                     dropout_rate=dropout_rate,
                     bottleneck=True)
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
        input_vector_op_shape = input_vector_op.get_shape().as_list()
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


def grasp_model_levine_2016(clear_view_image_op,
                            current_time_image_op,
                            input_vector_op,
                            input_image_shape=None,
                            input_vector_op_shape=None,
                            dropout_rate=None,
                            pooling='max'):
    """Model designed to match prior work.

    Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection.

    Original paper input dimensions:
    img_rows, img_cols, img_channels = 472, 472, 3  # 6 or 3
    """
    img_shape = K.int_shape(clear_view_image_op)[1:]
    inputImg1 = Input(shape=img_shape, tensor=clear_view_image_op)
    inputImg2 = Input(shape=img_shape, tensor=current_time_image_op)
    combImg = Concatenate(-1)([inputImg1, inputImg2])
    # img Conv 1
    imgConv = Conv2D(64, kernel_size=(6, 6),
                     activation='relu',
                     strides=(2, 2),
                     padding='same')(combImg)

    if pooling is 'max':
        # img maxPool
        imgConv = MaxPooling2D(pool_size=(3, 3))(imgConv)

    # img Conv 2 - 7
    for i in range(6):
        imgConv = Conv2D(64, (5, 5), padding='same', activation='relu')(imgConv)

    if pooling is 'max':
        # img maxPool 2
        imgConv = MaxPooling2D(pool_size=(3, 3))(imgConv)

    # motor Data
    vector_shape = K.int_shape(input_vector_op)[1:]
    motorData = Input(shape=vector_shape, tensor=input_vector_op)

    # motor full conn
    motorConv = Dense(64, activation='relu')(motorData)

    # tile and concat the data
    combinedData = concat_images_with_tiled_vector_layer(imgConv, motorConv)
    print('Combined', combinedData)

    # combined conv 8
    combConv = Conv2D(64, (3, 3), activation='relu', padding='same')(combinedData)

    # combined conv 9 - 13
    for i in range(3):
        combConv = Conv2D(64, (3, 3), activation='relu', padding='same')(combConv)

    if pooling is 'max':
        # combined maxPool
        combConv = MaxPooling2D(pool_size=(2, 2))(combConv)

    # combined conv 14 - 16
    for i in range(3):
        combConv = Conv2D(64, (3, 3), activation='relu', padding='same')(combConv)

    # Extra Global Average Pooling allows more flexible input dimensions
    # but only use if necessary.
    feature_shape = K.int_shape(combConv)
    if (feature_shape[1] > 1 or feature_shape[2] > 1):
        combConv = GlobalAveragePooling2D()(combConv)
    # combined full connected layers
    combConv = Dense(64, activation='relu')(combConv)
    combConv = Dense(64, activation='relu')(combConv)
    # get down to a single prediction
    combConv = Dense(1, activation='sigmoid')(combConv)

    model = Model(inputs=[inputImg1, inputImg2, motorData], outputs=combConv)
    return model
