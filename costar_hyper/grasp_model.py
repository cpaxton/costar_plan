import tensorflow as tf

import keras
# https://github.com/aurora95/Keras-FCN
# TODO(ahundt) move keras_fcn directly into this repository, into keras-contrib, or make a proper installer
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
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
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import RepeatVector
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.merge import Concatenate
from keras.layers.merge import Add
from keras.models import Model
from keras.layers import Lambda
from keras.layers import Reshape
from keras_applications.imagenet_utils import _obtain_input_shape
import keras_applications

import keras_contrib
from keras_contrib.applications.densenet import DenseNetFCN
from keras_contrib.applications.densenet import DenseNet
from keras_contrib.applications.densenet import DenseNetImageNet121
from keras_contrib.applications.resnet import ResNet
from keras_contrib.layers.normalization import GroupNormalization
import keras_contrib.applications.fully_convolutional_networks as fcn
import keras_contrib.applications.densenet as densenet
import keras_tqdm

from keras.engine import Layer
import coord_conv


def tile_vector_as_image_channels(vector_op, image_shape):
    """
    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.

    # Params

      vector_op: A tensor vector to tile.
      image_shape: A list of integers [width, height] with the desired dimensions.
    """
    with K.name_scope('tile_vector_as_image_channels'):
        ivs = K.shape(vector_op)
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
       vector: the 1D vector to tile onto every pixel.
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
            # TODO(ahundt) ensure shape works in both google brain/cornell dataset input tensor and keras Input() aka numpy array cases
            vector_shape = K.int_shape(vector)[1:]
        if image_shape is None:
            # check if K.shape, K.int_shape, or image.get_shape().as_list()[1:] is better
            # https://github.com/fchollet/keras/issues/5211
            # TODO(ahundt) ensure shape works in both google brain/cornell dataset input tensor and keras Input() aka numpy array cases
            image_shape = K.int_shape(images[0])[1:]
        vector = Reshape([1, 1, vector_shape[-1]])(vector)
        tile_shape = (int(1), int(image_shape[0]), int(image_shape[1]), int(1))
        tiled_vector = Lambda(lambda x: K.tile(x, tile_shape))(vector)
        x = Concatenate(axis=-1)([] + images + [tiled_vector])
    return x


def add_images_with_tiled_vector_layer(images, vector, image_shape=None, vector_shape=None):
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
    with K.name_scope('add_images_with_tiled_vector_layer'):
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
        x = Add()([] + images + [tiled_vector])
    return x


def create_tree_roots(inputs=None, input_shapes=None, make_layer_fn=None, trainable=True):
    """ Create Inputs then independent sequences of layers

    create_tree_roots works like roots of a tree taking inputs, and applying layers
    up towards the base where model or layer data might be combined.

    # Arguments

        inputs: A single input tensor or list of input tensors.
            Optional if input_shapes is specified.
        input_shapes: A shape or list of shapes.
            Optional if inputs is specified.
        make_layer_fn: A function which will create either a Model
            or the layers which make up each root.
            If set to None, then the return value
            root_inputs will be equal to root_logits.

    # Returns

        [root_inputs, root_logits]

        root_inputs: A list of results for each call to Input().
        root_logits: A list of results for each call to make_branch_fn().
    """
    branch_inputs = []
    branch_logits = None
    if inputs is not None:
        if not isinstance(inputs, list):
            inputs = [inputs]

        branch_logits = []
        for vector in inputs:
            v = Input(tensor=vector)
            branch_inputs += [v]
            if make_layer_fn is not None:
                v = make_layer_fn(v)
            branch_logits += [v]

    elif input_shapes is not None:
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]

        branch_logits = []
        for shape in input_shapes:
            v = Input(shape=shape)
            branch_inputs += [v]
            if make_layer_fn is not None:
                v = make_layer_fn(v)
            branch_logits += [v]

    if not trainable:
        for logit in branch_logits:
            if isinstance(logit, Model):
                for layer in logit.layers:
                    layer.trainable = False
            else:
                raise ValueError(
                    'Only set trainable=False Keras for Models, '
                    'layers and lists of layers can be done later '
                    'when the Model has been created. '
                    'Got Type: ' + str(type(logit)))

    return branch_inputs, branch_logits


def classifier_block(input_tensor, include_top=True, top='classification',
                     classes=1, activation='sigmoid',
                     input_shape=None, final_pooling=None, name='', verbose=1):
    """ Performs the final Activation for the classification of a given problem.

    # Arguments

        include_top: Whether to include the fully-connected
            layer at the top of the network. Also maps to require_flatten
            option in `keras.applications.imagenet_utils._obtain_input_shape()`.
    """
    x = input_tensor
    if include_top and top == 'classification':
        if verbose:
            print("    classification of x: " + str(x))
        x = Dense(units=classes, activation=activation,
                  kernel_initializer="he_normal", name=name + 'fc' + str(classes))(x)

    elif include_top and top == 'segmentation':
        if verbose > 0:
            print("    segmentation of x: " + str(x))
        x = Conv2D(classes, (1, 1), activation='linear', padding='same')(x)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        x = Reshape((row * col, classes))(x)
        x = Activation(activation)(x)
        x = Reshape((row, col, classes))(x)
    elif include_top and top == 'quaternion':
        x = Dense(units=classes, activation='linear',
                  kernel_initializer="he_normal", name=name + 'fc' + str(classes))(x)
        # normalize the output so we have a unit quaternion
        x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    elif final_pooling == 'avg':
        if verbose:
            print("    GlobalAveragePooling2D")
        x = GlobalAveragePooling2D()(x)

    elif final_pooling == 'max':
        if verbose:
            print("    GlobalMaxPooling2D")
        x = GlobalMaxPooling2D()(x)
    else:
        raise ValueError('grasp_model.py::classifier_block() unsupported top: ' + str(top))
    return x


def top_block(x, output_image_shape=None, top='classification', dropout_rate=0.0, include_top=True,
              classes=1, activation='sigmoid', final_pooling=None,
              filters=64, dense_layers=0, hidden_activation='relu', name='', verbose=0):
    """ Perform final convolutions for decision making, then apply the classification block.

        The top block adds the final "decision making" layers
        and the classifier block according to the problem type.
        Dense layers for single prediction problems, and
        Conv2D layers for pixel-wise prediction problems.

    # Arguments

        x: The input features which are expected to contain rows, columns and channels.
        output_image_shape: The expected shape for the final output of the top_block.
            Typically this will match the shape of x. However, if you have a segmentation
            problem output_image_shape can be larger, and upsampling will be applied.
        top: The type of problem you are attempting to solve,
            'classification' or 'segmentation'.
        include_top: Whether to include the fully-connected
            layer at the top of the network. Also maps to require_flatten
            option in `keras.applications.imagenet_utils._obtain_input_shape()`.
        dense_layers: Number of additional dense layers before the final dense layer.
            The final dense layer defines number of output classes is created.
        hidden_activation: Activation to be used in internal top block layers,
            before the final output activation.
    """
    print('top block top: ' + str(top))
    # Extra Global Average Pooling allows more flexible input dimensions
    # but only use if necessary.
    if top == 'classification' or top == 'quaternion':
        feature_shape = K.int_shape(x)
        if len(feature_shape) == 4:
            x = GlobalMaxPooling2D()(x)
            # x = Flatten()(x)

        for i in range(dense_layers):
            # combined full connected layers
            if dropout_rate is not None:
                x = Dropout(dropout_rate)(x)

            x = Dense(filters, activation=hidden_activation)(x)

        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)

    elif top == 'segmentation':

        for i in range(dense_layers):
            if dropout_rate is not None:
                x = Dropout(dropout_rate)(x)

            x = Conv2D(filters, (1, 1), activation=hidden_activation, padding='same')(x)

        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)

        # if the image was made smaller to save space,
        # upsample before calculating the final output
        if K.image_data_format() == 'channels_first':
            batch, channel, row, col = 0, 1, 2, 3
        else:
            batch, row, col, channel = 0, 1, 2, 3

        comb_conv_shape = K.int_shape(x)
        iidim = (output_image_shape[row-1], output_image_shape[col-1])
        ccdim = (comb_conv_shape[row], comb_conv_shape[col])
        if iidim != ccdim:
            x = UpSampling2D(size=(iidim[0]/ccdim[0], iidim[1]/ccdim[1]))(x)

    # calculate the final classification output
    x = classifier_block(x, include_top=include_top, top=top, classes=classes,
                         activation=activation, input_shape=output_image_shape,
                         final_pooling=final_pooling, name=name, verbose=verbose)
    print('top_block x: ' + str(x))
    return x


def hypertree_model(
        images=None, vectors=None,
        image_shapes=None, vector_shapes=None,
        dropout_rate=None,
        activation='sigmoid',
        final_pooling=None,
        include_top=True,
        top='segmentation',
        top_block_filters=64,
        classes=1,
        output_shape=None,
        create_image_tree_roots_fn=None,
        create_vector_tree_roots_fn=None,
        create_tree_trunk_fn=None,
        top_block_dense_layers=0,
        top_block_hidden_activation='relu',
        verbose=0):

    if images is None and image_shapes is None:
        raise ValueError('There must be at least one entry in the images parameter '
                         'or the image_shapes parameter.')

    if output_shape is None:
        if images is not None:
            output_shape = keras.backend.int_shape(images[0])[1:]

        elif image_shapes is not None:
            output_shape = image_shapes[0]

    image_inputs, image_logits = create_tree_roots(
        images, image_shapes, make_layer_fn=create_image_tree_roots_fn)

    vector_inputs, vector_logits = create_tree_roots(
        vectors, vector_shapes, make_layer_fn=create_vector_tree_roots_fn)

    if vector_logits is None and isinstance(image_logits, list):
        # combine image inputs
        if len(image_logits) > 1:
            x = Concatenate(axis=-1)(image_logits)
        else:
            [x] = image_logits
    else:
        # combine vector inputs
        v = vector_logits
        if len(vector_logits) > 1:
            v = Concatenate(axis=-1)(v)
        elif isinstance(v, list):
            [v] = v
        else:
            raise ValueError(
                'Unknown configuration of '
                'input vectors, you will need '
                'to look at the code and see what '
                'went wrong with v: ' + str(v))

        if v is not None:
            # combine image and vector inputs
            x = concat_images_with_tiled_vector_layer(image_logits, v)

    if create_tree_trunk_fn is not None:
        x = create_tree_trunk_fn(x)

    if not isinstance(x, list):
        x = [x]

    # handle multiple outputs for networks like NASNet
    xs = []
    name = ''
    for i, xi in enumerate(x):
        if len(x) > 1:
            # multiple separate outputs need multiple names
            name = str(i)
        # The top block adds the final "decision making" layers
        # and the classifier block according to the problem type.
        # Dense layers for single prediction problems, and
        # Conv2D layers for pixel-wise prediction problems.
        xi = top_block(
            xi, output_shape, top, dropout_rate,
            include_top, classes, activation,
            final_pooling, top_block_filters,
            dense_layers=top_block_dense_layers,
            hidden_activation=top_block_hidden_activation,
            name=name, verbose=verbose)
        xs += [xi]

    if len(xs) == 1:
        [x] = xs
    else:
        x = xs

    # Make a list of all inputs into the model
    # Each of these should be a list or the empty list [].
    inputs = image_inputs + vector_inputs

    print('hypertree_model x: ' + str(x))
    # create the model
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def choose_normalization(x, normalization, name=None):
    """ Choose the type of normalization to use such as batchnorm or groupnorm.

    x: the input tensor
    normalization: One of None, 'none', 'batch_norm', or 'group_norm'
    name: A name to give the layer.
    """
    if normalization is not None and normalization is not 'none':
        if normalization == 'batch_norm':
            # x = BatchNormalization()(x)
            # TODO(ahundt) using nasnet default BatchNorm options, revert if this causes problems
            x = BatchNormalization(axis=-1, momentum=0.9997, epsilon=1e-3)(x)
        elif normalization == 'group_norm':
            x = GroupNormalization()(x)
    return x


def choose_hypertree_model(
        images=None, vectors=None,
        image_shapes=None, vector_shapes=None,
        dropout_rate=0.25,
        vector_dense_filters=256,
        dilation_rate=2,
        activation='sigmoid',
        final_pooling=None,
        include_top=True,
        top='classification',
        top_block_filters=64,
        top_block_dense_layers=0,
        classes=1,
        output_shape=None,
        trainable=False,
        verbose=0,
        image_model_name='vgg',
        vector_model_name='dense',
        trunk_layers=4,
        trunk_filters=128,
        trunk_model_name='dense',
        vector_branch_num_layers=3,
        image_model_weights='shared',
        use_auxiliary_branch=True,
        weights='imagenet',
        trunk_hidden_activation='relu',
        vector_hidden_activation='linear',
        top_block_hidden_activation='relu',
        trunk_normalization='none',
        coordinate_data=None,
        hidden_activation=None,
        vector_normalization='batch_norm'):
    """ Construct a variety of possible models with a tree shape based on hyperparameters.

    # Arguments

        dropout_rate: a dropout rate of None will disable dropout.
        top_block_filters: the number of filters for the two final fully connected layers,
            before a prediction is made based on the number of classes.
        image_model_weights: How should the image model weights be stored for each image?
            Options are 'shared' and 'separate'.
        trunk_filters: the initial number of filters for the concatenated network trunk.
            Setting the parameters to None or 0 will use the number of cannels in the
            input data provided.
        trunk_normalization: options are 'batch_norm', 'group_norm', and None or 'none'
        trunk_hidden_activation:
            Deprecated, use hidden_activation instead.
            The activation to use for hidden layers in configurations that support it
            the vgg trunk is the only option at the time of writing.
        vector_normalization: options are 'batch_norm', 'group_norm', and None or 'none'
        vector_hidden_activation: Deprecated, use hidden_activation instead.
            the activation to use for hidden layers in configurations that support it
            the vgg trunk is the only option at the time of writing.
        top_block_hidden_activation: Deprecated, use hidden_activation instead.
        hidden_activation: override vector, trunk, and top_block hidden activation,
            setting them all to this value. Example values include 'relu', 'elu', and 'linear'.

    # Notes

    Best result for classification

    2018-02-23-09-35-21

        - 0.25 dropout
        - hyperopt_logs_cornell/2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success
          {"vector_dense_filters": 256, "vector_branch_num_layers": 0, "trunk_filters": 128,
          "image_model_name": "vgg", "vector_model_name": "dense", "preprocessing_mode": "tf",
          "trainable": true, "top_block_filters": 64, "learning_rate": 0.02, "trunk_layers": 4}

    Best 1 epoch run with image_preprocessed_sin_cos_height_3 and 0.25 dropout, 2018-02-19:
        - val_binary_accuracy 0.9115646390282378
        - val_loss 0.26308334284290974
        {"vector_dense_filters": 64, "vector_branch_num_layers": 3, "trainable": false,
         "image_model_name": "vgg", "vector_model_name": "dense", "learning_rate": 0.03413896253431821, "trunk_filters": 256,
         "top_block_filters": 128, "trunk_layers": 4, "feature_combo_name": "image_preprocessed_sin_cos_height_3"}

    Best 1 epoch run with only gripper openness parameter, 2018-02-17:
        - val_binary_accuracy 0.9134199238
        - val_loss 0.2269693456

        {"vector_dense_filters": 64, "vector_branch_num_layers": 2, "trainable": true,
         "image_model_name": "vgg", "vector_model_name": "dense_block", "learning_rate": 0.005838979061490798,
         "trunk_filters": 128, "dropout_rate": 0.0, "top_block_filters": 64, "trunk_layers": 4, "feature_combo_name":
         "image_preprocessed_height_1"}
    Current best 1 epoch run as of 2018-02-16:
        - note there is a bit of ambiguity so until I know I'll have case 0 and case 1.
            - two models were in that run and didn't have hyperparam records yet.
            - The real result is probably case 1, since the files are saved each run,
              so the data will be for the latest run.
        - 2018-02-15-22-00-12_-vgg_dense_model-dataset_cornell_grasping-grasp_success2018-02-15-22-00-12_-vgg_dense_model-dataset_cornell_grasping-grasp_success
        - input
            - height_width_sin_cos_4
        - vgg16 model
        - val_binary_accuracy
            - 0.9202226425
        - lr
            - 0.06953994
        - vector dense layers
            - 4 in case 0 with 64 channels
            - 1 in case 1 with 64 channels
        - dense block trunk case 1
            - 5 conv blocks
            - growth rate 48
            - 576 input channels
            - 816 output channels
        - dense layers before fc1, case 1
            - 64 output channels

    """
    if trainable is None:
        trainable = False

    if hidden_activation is not None:
        trunk_hidden_activation = hidden_activation
        vector_hidden_activation = hidden_activation
        top_block_hidden_activation = hidden_activation

    if top == 'segmentation':
        name_prefix = 'dilated_'
    else:
        name_prefix = 'single_'
        dilation_rate = 1
    with K.name_scope(name_prefix + 'hypertree') as scope:

        # input_image_tensor = None
        # get the shape of images
        # currently assumes all images have the same shape
        if image_shapes is None and isinstance(images[0], tf.Tensor):
            image_input_shape = K.int_shape(images[0])
        elif isinstance(image_shapes[0], tf.Tensor):
            image_input_shape = K.int_shape(image_shapes[0])
        elif image_shapes is None:
            raise ValueError(
                'image_shapes is None and could not be determined'
                'automatically. Try specifying it again or correcting it.'
                'The images param was also: ' + str(images)
            )
        else:
            image_input_shape = image_shapes[0]
        if coordinate_data is not None and coordinate_data == 'coord_conv_img':
            # coord_conv_img case currently adds 2 additional channels.
            image_input_shape = (image_input_shape[0], image_input_shape[1], image_input_shape[2] + 2)
            if weights is not None and weights == 'imagenet':
                print('hypertree coordinate_data configuration coord_conv_img is not compatible with imagenet weights, setting weights to None')
                weights = None
        print('hypertree image_input_shape: ' + str(image_input_shape))
        print('hypertree images: ' + str(images))
        print('hypertree classes: ' + str(classes))
        if trunk_filters == 0:
            trunk_filters = None

        if image_input_shape is not None and len(image_input_shape) == 4:
            # cut off batch size
            image_input_shape = image_input_shape[1:]

        if image_model_weights not in ['shared', 'separate']:
            raise ValueError('Unsupported image_model_weights: ' +
                             str(image_model_weights) +
                             'Options are shared and separate.')

        print('hypertree image_input_shape with batch stripped: ' + str(image_input_shape))
        # VGG16 weights are shared and not trainable
        if top == 'segmentation':
            if image_model_name == 'vgg':
                if image_model_weights == 'shared':
                    image_model = fcn.AtrousFCN_Vgg16_16s(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, upsample=False)
                elif image_model_weights == 'separate':
                    image_model = fcn.AtrousFCN_Vgg16_16s
            elif image_model_name == 'resnet':
                if image_model_weights == 'shared':
                    image_model = fcn.AtrousFCN_Resnet50_16s(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, upsample=False)
                elif image_model_weights == 'separate':
                    image_model = fcn.AtrousFCN_Resnet50_16s
            else:
                raise ValueError('Unsupported segmentation model name: ' +
                                 str(image_model_name) + 'options are vgg and resnet.')
        else:

            if image_model_name == 'vgg':
                if image_model_weights == 'shared':
                    image_model = keras.applications.vgg16.VGG16(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, weights=weights)
                elif image_model_weights == 'separate':
                    image_model = keras.applications.vgg16.VGG16
            elif image_model_name == 'vgg19':
                if image_model_weights == 'shared':
                    image_model = keras.applications.vgg19.VGG19(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, weights=weights)
                elif image_model_weights == 'separate':
                    image_model = keras.applications.vgg19.VGG19
            elif image_model_name == 'nasnet_large':
                if image_model_weights == 'shared':
                    image_model = NASNetLarge(
                        input_shape=image_input_shape, include_top=False, pooling=None,
                        classes=classes, weights=weights
                    )
                elif image_model_weights == 'separate':
                    image_model = NASNetLarge
                else:
                    raise ValueError('Unsupported image_model_name')

                # TODO(ahundt) switch to keras_contrib model below when keras_contrib is updated with correct weights https://github.com/keras-team/keras/pull/10209.
                # please note that with nasnet_large, no pooling,
                # and an aux network the two outputs will be different
                # dimensions! Therefore, we need to add our own pooling
                # for the aux network.
                # TODO(ahundt) just max pooling in NASNetLarge for now, but need to figure out pooling for the segmentation case.
                # image_model = keras_contrib.applications.nasnet.NASNetLarge(
                #     input_shape=image_input_shape, include_top=False, pooling=None,
                #     classes=classes, use_auxiliary_branch=use_auxiliary_branch,
                #     weights=weights
                # )
            elif image_model_name == 'nasnet_mobile':
                image_model = keras.applications.nasnet.NASNetMobile(
                    input_shape=image_input_shape, include_top=False,
                    classes=classes, pooling=False, weights=weights
                )
            elif image_model_name == 'inception_resnet_v2':
                if image_model_weights == 'shared':
                    image_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, weights=weights)
                elif image_model_weights == 'separate':
                    image_model = keras.applications.inception_resnet_v2.InceptionResNetV2
                else:
                    raise ValueError('Unsupported image_model_name')
            elif image_model_name == 'mobilenet_v2':
                if image_model_weights == 'shared':
                    image_model = MobileNetV2(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, weights=weights)
                elif image_model_weights == 'separate':
                    image_model = MobileNetV2
                else:
                    raise ValueError('Unsupported image_model_name')
            elif image_model_name == 'resnet':
                # resnet model is special because we need to
                # skip the average pooling part.
                if image_model_weights == 'shared':
                    resnet_model = keras.applications.resnet50.ResNet50(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, weights=weights)
                elif image_model_weights == 'separate':
                    image_model = keras.applications.resnet50.ResNet50
                if not trainable:
                    for layer in resnet_model.layers:
                        layer.trainable = False
                # get the layer before the global average pooling
                # TODO(ahundt) this may need to be changed due to recent resnet restructuring in keras
                image_model = resnet_model.layers[-2]
            elif image_model_name == 'densenet':
                if image_model_weights == 'shared':
                    image_model = keras.applications.densenet.DenseNet169(
                        input_shape=image_input_shape, include_top=False,
                        classes=classes, weights=weights)
                elif image_model_weights == 'separate':
                    image_model = keras.applications.densenet.DenseNet169
                else:
                    raise ValueError('Unsupported image_model_name')
            elif image_model_name is None or image_model_name == 'none':
                if image_model_weights == 'shared':
                    x = Input(shape=image_input_shape)
                    image_model = Model(x, x)
                elif image_model_weights == 'separate':
                    def identity_model(input_shape=image_input_shape, weights=None, classes=None,
                                       input_tensor=None):
                        """ Identity Model is an empty model that returns the input.
                        """
                        if input_tensor is None:
                            x = Input(shape=input_shape)
                        else:
                            x = Input(tensor=input_tensor)
                        return Model(x, x)

                    image_model = identity_model
                else:
                    raise ValueError('Unsupported image_model_name')
            else:
                raise ValueError('Unsupported image_model_name')

        set_trainable_layers(trainable, image_model)

        class ImageModelCarrier():
            # Create a temporary scope for the list
            # compatible with python 2.7 and 3.5
            # note this may not work as expected if multiple
            # instances of choose_hypertree_model
            # are created at once.
            image_models = []
            image_model_num = 0

        def create_image_model(tensor, coordinate_data=coordinate_data):
            """ Image classifier weights are shared or separate.

            This function helps set up the weights.
            """
            if coordinate_data is not None and coordinate_data == 'coord_conv_img':
                print('Hypertree applying coord_conv.CoordinateChannel2D before creating the image model to ' + str(tensor))
                tensor = coord_conv.CoordinateChannel2D()(tensor)
            if image_model_weights == 'shared':
                ImageModelCarrier.image_models += [image_model]
                return image_model(tensor)
            elif image_model_weights == 'separate':
                imodel = image_model(
                    input_tensor=tensor,
                    include_top=False,
                    classes=classes)
                # TODO(ahundt) Can't have duplicate layer names in a network, figure out a solution
                # It is needed for when two of a model are used in a network
                # see https://stackoverflow.com/questions/43452441/keras-all-layer-names-should-be-unique
                # and https://github.com/keras-team/keras/issues/7412
                if ImageModelCarrier.image_model_num > 0:
                    for layer in imodel.layers:
                        layer.name += str(ImageModelCarrier.image_model_num)
                if not trainable:
                    for layer in imodel.layers:
                        layer.trainable = False
                ImageModelCarrier.image_models += [imodel]
                ImageModelCarrier.image_model_num += 1
                print('Hypertree create_image_model called for tensor: ' + str(tensor))
                return imodel.outputs[0]

        def vector_branch_dense(
                tensor, vector_dense_filters=vector_dense_filters,
                num_layers=vector_branch_num_layers,
                model_name=vector_model_name,
                vector_normalization=vector_normalization):
            """ Vector branches that simply contain a single dense layer.
            """
            x = tensor
            # create the chosen layers starting with the vector input
            # accepting num_layers == 0 is done so hyperparam search is simpler
            if num_layers is None or num_layers == 0:
                return x
            elif model_name == 'dense':
                x = Dense(vector_dense_filters, activation=vector_hidden_activation)(x)
                # Important! some old models saved to disk
                # are invalidated by the BatchNorm and Dropout
                # lines below, comment them if you really need to go back
                x = choose_normalization(x, vector_normalization)
                x = Dropout(dropout_rate)(x)
                if num_layers > 1:
                    for i in range(num_layers - 1):
                        x = Dense(vector_dense_filters, activation=vector_hidden_activation)(x)
            elif model_name == 'dense_block':
                densenet.__dense_block(
                    x, nb_layers=num_layers,
                    nb_filter=vector_dense_filters,
                    growth_rate=48, dropout_rate=dropout_rate,
                    dims=0)
            else:
                raise ValueError('vector_branch_dense called with '
                                 'unsupported model name %s, options '
                                 'are dense and dense_block.' % model_name)
            print('Hypertree vector_branch_dense completed for tensor: ' + str(tensor))
            return x

        def create_tree_trunk(tensor, filters=trunk_filters, num_layers=trunk_layers, coordinate_data=coordinate_data):
            """
                filters: the initial number of filters for the concatenated network trunk.
                    Setting the parameters to None or 0 will use the number of cannels in the
                    input data provided.
            """
            if (coordinate_data is not None and
                    coordinate_data is not 'none' and
                    coordinate_data is 'coord_conv_trunk'):
                tensor = coord_conv.CoordinateChannel2D()(tensor)

            x = tensor

            if filters is None or filters == 0:
                channels = K.int_shape(tensor)[-1]
            else:
                channels = filters

            # create the chosen layers starting with the combined image and vector input
            # accepting num_layers == 0 is done so hyperparam search is simpler
            if num_layers is None or num_layers == 0:
                return x
            elif num_layers is not None:
                if trunk_model_name == 'dense_block' or trunk_model_name == 'dense':
                    # unfortunately, dense above really means dense_block
                    # but some past hyperopt logs have dense so we need to keep it
                    #
                    # growth rate is 48 due to the "wider convolutional network" papers
                    # and comments by the densenet authors in favor of this param choice.
                    # see https://github.com/liuzhuang13/DenseNet
                    x, num_filters = densenet.__dense_block(
                        x, nb_layers=trunk_layers, nb_filter=channels,
                        growth_rate=48, dropout_rate=dropout_rate)
                elif trunk_model_name == 'resnet_conv_identity_block':
                    stage = 'trunk'
                    x = fcn.conv_block(3, [filters, filters, filters * 4], stage, '_' + str(0))(x)
                    if num_layers > 1:
                        for l in range(num_layers - 1):
                            x = fcn.identity_block(3, [filters, filters, filters * 4], stage, '_' + str(l + 1))(x)
                elif trunk_model_name == 'vgg_conv_block':
                    # Vgg "Block 6"
                    name = 'trunk'
                    weight_decay = 0.
                    for l in range(num_layers):
                        x = Conv2D(filters, (3, 3), activation=trunk_hidden_activation, padding='same',
                                   name=name + 'block6_conv%d' % l, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
                        x = choose_normalization(x, trunk_normalization)
                elif trunk_model_name == 'nasnet_normal_a_cell':

                    x = Conv2D(filters, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               use_bias=False,
                               name='trunk_conv1',
                               kernel_initializer='he_normal')(x)

                    x = choose_normalization(x, trunk_normalization)
                    p = x
                    for l in range(num_layers):
                        block_id = 'trunk_' + str(l)
                        # _normal_a_cell call changed due to creation of keras-applications repository
                        # https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py
                        x, p = keras_applications.nasnet._normal_a_cell(x, p, filters, block_id=block_id)
                else:
                    raise ValueError('Unsupported trunk_model_name ' + str(trunk_model_name) +
                                     ' options are dense, resnet, vgg, nasnet')

            return x

        model = hypertree_model(
            images=images, vectors=vectors,
            image_shapes=image_shapes, vector_shapes=vector_shapes,
            dropout_rate=dropout_rate,
            create_image_tree_roots_fn=create_image_model,
            create_vector_tree_roots_fn=vector_branch_dense,
            create_tree_trunk_fn=create_tree_trunk,
            activation=activation,
            final_pooling=final_pooling,
            include_top=include_top,
            top=top,
            top_block_filters=top_block_filters,
            top_block_hidden_activation=top_block_hidden_activation,
            classes=classes,
            output_shape=output_shape,
            verbose=verbose
        )
    print('hypertree model complete')
    return model


def set_trainable_layers(trainable, image_model):
    """ Set the trainable layers in a model.

    trainable: Either a boolean to set all layers or a
       floating point value from 0 to 1 indicating the proportion of
       layer depths to make trainable. In other words with 0.5 all
       layers past the halfway point in terms of depth will be trainable.
    image_model: The model to configure
    """
    if ((not isinstance(trainable, bool) or not trainable)
            and getattr(image_model, 'layers', None) is not None):
        # enable portion of network depending on the depth
        if not trainable:
            for layer in image_model.layers:
                layer.trainable = False
        else:
            # Set all layers past a certain depth to trainable
            # using a fractional scale
            num_depths = len(image_model.layers_by_depth)
            num_untrainable_depths = np.round((1.0 - trainable) * num_depths)
            should_train = False
            for i, layers in enumerate(image_model.layers):
                if i > num_untrainable_depths:
                    should_train = True
                for layer in layers:
                    layer.trainable = should_train


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
                   activation=activation,
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
                           growth_rate=36,
                           reduction=0.5,
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

    clear_view_model = NASNetMobile(
        input_shape=input_image_shape[1:],
        input_tensor=clear_view_image_op,
        include_top=False)

    current_time_model = NASNetMobile(
        input_shape=input_image_shape[1:],
        input_tensor=current_time_image_op,
        include_top=False)

    if not train_densenet:
        for layer in clear_view_model.layers:
            layer.trainable = False
        for layer in current_time_model.layers:
            layer.trainable = False

    # clear_view_unpooled_layer = clear_view_model.layers[-2].get_output_at(0)
    clear_view_unpooled_layer = clear_view_model.outputs[0]
    unpooled_shape = K.int_shape(clear_view_unpooled_layer)
    print('clear_view_unpooled_layer: ', clear_view_unpooled_layer)
    print('unpooled_shape: ', unpooled_shape)
    # current_time_unpooled = current_time_model.layers[-2].get_output_at(0)
    current_time_unpooled = current_time_model.outputs[0]
    print('clear_view_model.outputs: ', clear_view_model.outputs)
    print('current_time_model.outputs: ', current_time_model.outputs)
    unpooled_shape = [input_vector_op_shape[0], unpooled_shape[1], unpooled_shape[2], unpooled_shape[3]]
    clear_view_unpooled_layer = clear_view_unpooled_layer.set_shape(unpooled_shape)
    current_time_unpooled = current_time_unpooled.set_shape(unpooled_shape)

    combined_input_data = concat_images_with_tiled_vector_layer([clear_view_unpooled_layer, current_time_unpooled], input_vector_op,
                                                                image_shape=unpooled_shape)

    print('combined_input_data.get_shape().as_list():', combined_input_data.get_shape().as_list())
    combined_input_shape = K.int_shape(combined_input_data)
    model_name = 'dense'
    if model_name == 'dense':
        final_nb_layer = 4
        nb_filter = combined_input_shape[-1]
        weight_decay = 1e-4
        # The last dense_block does not have a transition_block
        x, nb_filter = keras_contrib.applications.densenet.__dense_block(
            combined_input_data, final_nb_layer, nb_filter, growth_rate,
            bottleneck=True, dropout_rate=dropout_rate, weight_decay=weight_decay)

        concat_axis = -1
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='final_bn')(x)
        x = Activation('relu')(x)
        x = GlobalMaxPooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[clear_view_image_op, current_time_image_op, input_vector_op], outputs=[x])
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


def grasp_model_densenet(
        clear_view_image_op,
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
                             dropout_rate=0.0,
                             activation='sigmoid',
                             classes=1,
                             early_transition=True):
    if input_vector_op_shape is None:
        input_vector_op_shape = input_vector_op.get_shape().as_list()
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]

    if input_vector_op is not None:
        combined_input_data = concat_images_with_tiled_vector([clear_view_image_op, current_time_image_op], input_vector_op)
        combined_input_shape = K.int_shape(combined_input_data)

    # the input shape should be a tuple of 3 values
    # if the batch size is present, strip it out
    # for call to ResNet constructor.
    if len(combined_input_shape) == 4:
        combined_input_shape = combined_input_shape[1:]
    model = DenseNetFCN(input_shape=combined_input_shape,
                        include_top='global_average_pooling',
                        input_tensor=combined_input_data,
                        activation=activation,
                        growth_rate=growth_rate,
                        reduction=reduction,
                        nb_dense_block=dense_blocks,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        early_transition=True)
    return model


def grasp_model_levine_2016(
        clear_view_image_op,
        current_time_image_op,
        input_vector_op,
        input_image_shape=None,
        input_vector_op_shape=None,
        dropout_rate=None,
        strides_initial_conv=(2, 2),
        dilation_rate_initial_conv=1,
        pooling='max',
        dilation_rate=1,
        final_pooling='max',
        include_top=True,
        activation='sigmoid',
        top='classification',
        classes=1,
        verbose=0,
        name='grasp_model_levine_2016'):
    """Model designed to match prior work.

    Learning Hand-Eye Coordination for Robotic Grasping with
    Deep Learning and Large-Scale Data Collection.

    Original paper input dimensions:
    img_rows, img_cols, img_channels = 472, 472, 3  # 6 or 3
    """
    with K.name_scope(name) as scope:
        if input_image_shape is None:
            input_image_shape = K.int_shape(clear_view_image_op)[1:]
        if len(input_image_shape) == 4:
            input_image_shape = input_image_shape[1:]

        if activation not in ['softmax', 'sigmoid', None]:
            raise ValueError('activation must be one of "softmax" or "sigmoid"'
                             ' or None, but is: ' + str(activation))

        if activation == 'sigmoid' and classes != 1:
            raise ValueError('sigmoid activation can only be used when classes = 1')

        # Determine proper input shape
        input_image_shape = _obtain_input_shape(input_image_shape,
                                                default_size=32,
                                                min_size=8,
                                                data_format=K.image_data_format(),
                                                require_flatten=include_top)

        clear_view_image_input = Input(shape=input_image_shape,
                                       tensor=clear_view_image_op,
                                       name='clear_view_image_input')
        current_time_image_input = Input(shape=input_image_shape,
                                         tensor=current_time_image_op,
                                         name='current_time_image_input')

        conv_counter = 1
        maxpool_counter = 1
        dense_counter = 1
        dropout_counter = 1

        # img1 Conv 1
        clear_view_img_conv = Conv2D(64, kernel_size=(6, 6),
                                     activation='relu',
                                     strides=strides_initial_conv,
                                     dilation_rate=dilation_rate_initial_conv,
                                     padding='same',
                                     name='conv' + str(conv_counter))(clear_view_image_input)
        conv_counter += 1

        # img2 Conv 1
        current_time_img_conv = Conv2D(64, kernel_size=(6, 6),
                                       activation='relu',
                                       strides=strides_initial_conv,
                                       dilation_rate=dilation_rate_initial_conv,
                                       padding='same',
                                       name='conv' + str(conv_counter))(current_time_image_input)
        conv_counter += 1
        if verbose > 0:
            print('conv2 shape:' + str(K.int_shape(current_time_img_conv)))
        if pooling == 'max':
            # img maxPool
            maxpool_name_a = 'maxpool' + str(maxpool_counter)
            clear_view_img_conv = MaxPooling2D(pool_size=(3, 3),
                                               name=maxpool_name_a)(clear_view_img_conv)
            maxpool_counter += 1
            maxpool_name_b = 'maxpool' + str(maxpool_counter)
            current_time_img_conv = MaxPooling2D(pool_size=(3, 3),
                                                 name=maxpool_name_b)(current_time_img_conv)
            maxpool_counter += 1
            if verbose > 0:
                print(maxpool_name_a, maxpool_name_b)

        x = Add()([clear_view_img_conv, current_time_img_conv])

        # img Conv 2
        x = Conv2D(64, (5, 5), padding='same', activation='relu',
                   dilation_rate=dilation_rate,
                   name='conv' + str(conv_counter))(x)
        conv_counter += 1

        # img Conv 3 - 8
        for i in range(6):
            x = Conv2D(64, (5, 5), padding='same', activation='relu',
                       name='conv' + str(conv_counter))(x)
            conv_counter += 1

        if verbose > 0:
            print('pre max pool shape:' + str(K.int_shape(x)))
        if pooling == 'max':
            # img maxPool 2
            maxpool_name = 'maxpool' + str(maxpool_counter)
            x = MaxPooling2D(pool_size=(3, 3), name=maxpool_name)(x)
            if verbose > 0:
                print(maxpool_name + ' shape:' + str(K.int_shape(x)))
            maxpool_counter += 1

        if verbose > 0:
            print('post max pool shape:' + str(K.int_shape(x)))
        if input_vector_op is not None or input_vector_op is not None:
            # Handle input command data
            vector_shape = K.int_shape(input_vector_op)[1:]
            motorData = Input(shape=vector_shape, tensor=input_vector_op, name='motion_command_vector_input')

            # motor full conn
            motorConv = Dense(64, activation='relu',
                              name='dense' + str(dense_counter))(motorData)
            dense_counter += 1

            # tile and concat the data
            x = add_images_with_tiled_vector_layer(x, motorConv)

            if dropout_rate is not None:
                x = Dropout(dropout_rate, name='dropout' + str(dropout_counter))(x)
                dropout_counter += 1

        # combined conv 8
        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate,
                   name='conv' + str(conv_counter))(x)
        conv_counter += 1

        #TODO(dingyu95) check if this is the right number of convs, update later and make a model
        # combined conv 10 - 12
        for i in range(2):
            x = Conv2D(64, (3, 3), activation='relu', padding='same',
                       name='conv' + str(conv_counter))(x)
            conv_counter += 1

        # combined conv 13
        x = Conv2D(64, (5, 5), padding='same', activation='relu',
                   name='conv' + str(conv_counter))(x)
        conv_counter += 1

        if verbose > 0:
            print('pre max pool shape:' + str(K.int_shape(x)))
        if pooling == 'max':
            # combined maxPool
            maxpool_name = 'maxpool' + str(maxpool_counter)
            x = MaxPooling2D(pool_size=(2, 2), name=maxpool_name)(x)
            if verbose > 0:
                print(maxpool_name + ' shape:' + str(K.int_shape(x)))
            maxpool_counter += 1

        if verbose > 0:
            print('post max pool shape:' + str(K.int_shape(x)))
        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate,
                   name='conv' + str(conv_counter))(x)
        conv_counter += 1
        # combined conv 14 - 16
        for i in range(2):
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv' + str(conv_counter))(x)
            conv_counter += 1

        # The top block adds the final "decision making" layers
        # and the classifier block according to the problem type.
        # Dense layers for single prediction problems, and
        # Conv2D layers for pixel-wise prediction problems.
        x = top_block(x, input_image_shape, top, dropout_rate,
                      include_top, classes, activation,
                      final_pooling, verbose=verbose)

        # make a list of all inputs into the model
        inputs = [clear_view_image_input, current_time_image_input]
        if input_vector_op is not None or input_vector_op is not None:
            inputs = inputs + [motorData]

        # create the model
        model = Model(inputs=inputs, outputs=x)
        return model


def grasp_model_levine_2016_segmentation(
        clear_view_image_op,
        current_time_image_op,
        input_vector_op,
        input_image_shape=None,
        input_vector_op_shape=None,
        dropout_rate=None,
        strides_initial_conv=(2, 2),
        dilation_rate_initial_conv=1,
        pooling=None,
        dilation_rate=2,
        final_pooling=None,
        require_flatten=True,
        activation='sigmoid',
        include_top=True,
        top='segmentation',
        classes=1,
        verbose=0):
    """ The levine 2016 model adapted for segmentation.
    #TODO(ahundt) require_flatten is formerly include_top from keras, change to something like include_classifier_block
    Based on the prior work:

        Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection.
    """
    print('grasp_model_levine_2016_segmentation')
    return grasp_model_levine_2016(
        clear_view_image_op=clear_view_image_op,
        current_time_image_op=current_time_image_op,
        input_vector_op=input_vector_op,
        input_image_shape=input_image_shape,
        input_vector_op_shape=input_vector_op_shape,
        dropout_rate=dropout_rate,
        strides_initial_conv=strides_initial_conv,
        dilation_rate_initial_conv=dilation_rate_initial_conv,
        pooling=pooling,
        dilation_rate=dilation_rate,
        final_pooling=final_pooling,
        activation=activation,
        top=top,
        classes=classes,
        verbose=verbose)
