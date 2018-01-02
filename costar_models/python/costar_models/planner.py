from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np
import tensorflow as tf

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers import Lambda
from keras.layers.merge import Add, Multiply
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam

'''
PLANNER MODEL TOOLS
-------------------

This file contains models for performing hierarchical planner operations.


Returns for all tools:
--------
out: an output tensor
'''

def AddConv2D(x, filters, kernel, stride, dropout_rate, padding="same",
        discriminator=False, momentum=0.9):
    '''
    Helper for creating networks. This one will add a convolutional block.

    Parameters:
    -----------
    x: input tensor
    filters: num conv filters
    kernel: kernel size to use
    stride: stride to use
    dropout_rate: amount of dropout to apply

    Returns:
    --------
    x: output tensor
    '''
    x = Conv2D(filters,
            kernel_size=kernel,
            strides=(stride,stride),
            padding=padding)(x)
    x = BatchNormalization(momentum=momentum)(x)
    if discriminator:
        x = LeakyReLU(alpha=0.2)(x)
    else:
        x = Activation("relu")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def AddConv2DTranspose(x, filters, kernel, stride, dropout_rate,
        padding="same", momentum=0.9):
    '''
    Helper for creating networks. This one will add a convolutional block.

    Parameters:
    -----------
    x: input tensor
    filters: num conv filters
    kernel: kernel size to use
    stride: stride to use
    dropout_rate: amount of dropout to apply

    Returns:
    --------
    x: output tensor
    '''
    x = Conv2DTranspose(filters,
            kernel_size=kernel,
            strides=(stride,stride),
            padding=padding)(x)
    x = BatchNormalization(momentum=momentum)(x)
    discriminator = False
    if discriminator:
        x = LeakyReLU(alpha=0.2)(x)
    else:
        x = Activation("relu")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def AddDense(x, size, activation, dropout_rate, output=False, momentum=0.9):
    '''
    Add a single dense block with batchnorm and activation.

    Parameters:
    -----------
    x: input tensor
    size: number of dense neurons
    activation: activation fn to use
    dropout_rate: dropout to use after activation

    Returns:
    --------
    x: output tensor
    '''
    x = Dense(size)(x)
    if not output:
        x = BatchNormalization(momentum=momentum)(x)
    if activation == "lrelu":
        x = LeakyReLU(alpha=0.2)(x)
    else:
        x = Activation(activation)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def CombinePose(pose_in, dim=64):
    robot = Dense(dim, activation="relu")(pose_in)
    return robot

def CombinePoseAndOption(pose_in, option_in, dim=64):
    robot = Concatenate(axis=-1)([pose_in, option_in])
    robot = Dense(dim, activation="relu")(robot)
    return robot

def CombineArmAndGripper(arm_in, gripper_in, dim=64):
    robot = Concatenate(axis=-1)([arm_in, gripper_in])
    robot = Dense(dim, activation="relu")(robot)
    return robot

def CombineArmAndGripperAndOption(arm_in, gripper_in, option_in, dim=64):
    robot = Concatenate(axis=-1)([arm_in, gripper_in, option_in])
    robot = Dense(dim, activation="relu")(robot)
    return robot

def GetArmGripperEncoder(arm_size, gripper_size, dim=64):
    arm_in = Input((arm_size,))
    gripper_in = Input((gripper_size,))
    

def TileOnto(x,z,zlen,xsize):
    z = Reshape([1,1,zlen])(z)
    tile_shape = (int(1), int(xsize[0]), int(xsize[1]), 1)
    z = Lambda(lambda x: K.tile(x, tile_shape))(z)
    x = Concatenate(axis=-1)([x,z])
    return x



def TileArmAndGripper(x, arm_in, gripper_in, tile_width, tile_height,
        option=None, option_in=None,
        time_distributed=None, dim=64):
    arm_size = int(arm_in.shape[-1])
    gripper_size = int(gripper_in.shape[-1])

    # handle error: options and grippers
    if option is None and option_in is not None \
        or option is not None and option_in is None:
            raise RuntimeError('must provide both #opts and input')

    # generate options and tile things together
    if option is None:
        robot = CombineArmAndGripper(arm_in, gripper_in, dim=dim)
        #reshape_size = arm_size+gripper_size
        reshape_size = dim
    else:
        robot = CombineArmAndGripperAndOption(arm_in, 
                                              gripper_in,
                                              option_in,
                                              dim=dim)
        reshape_size = dim
        #reshape_size = arm_size+gripper_size+option

    # time distributed or not
    robot0 = robot
    if time_distributed is not None and time_distributed > 0:
        tile_shape = (1, 1, tile_width, tile_height, 1)
        robot = Reshape([time_distributed, 1, 1, reshape_size])(robot)
    else:
        tile_shape = (1, tile_width, tile_height, 1)
        robot = Reshape([1, 1, reshape_size])(robot)

    # finally perform the actual tiling
    robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
    x = Concatenate(axis=-1)([x,robot])

    return x, robot0

def TilePose(x, pose_in, tile_width, tile_height,
        option=None, option_in=None,
        time_distributed=None, dim=64):
    pose_size = int(pose_in.shape[-1])
    

    # handle error: options and grippers
    if option is None and option_in is not None \
        or option is not None and option_in is None:
            raise RuntimeError('must provide both #opts and input')

    # generate options and tile things together
    if option is None:
        robot = CombinePose(pose_in, dim=dim)
        #reshape_size = arm_size+gripper_size
        reshape_size = dim
    else:
        robot = CombinePoseAndOption(pose_in, option_in, dim=dim)
        reshape_size = dim
        #reshape_size = arm_size+gripper_size+option

    # time distributed or not
    if time_distributed is not None and time_distributed > 0:
        tile_shape = (1, 1, tile_width, tile_height, 1)
        robot = Reshape([time_distributed, 1, 1, reshape_size])(robot)
    else:
        tile_shape = (1, tile_width, tile_height, 1)
        robot = Reshape([1, 1, reshape_size])(robot)

    # finally perform the actual tiling
    robot0 = robot
    robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
    x = Concatenate(axis=-1)([x,robot])

    return x, robot

def GetImageEncoder(img_shape, dim, dropout_rate,
        filters, dropout=True, leaky=True,
        dense=True, flatten=True,
        layers=2,
        kernel_size=[3,3],
        time_distributed=0,):

    if time_distributed <= 0:
        ApplyTD = lambda x: x
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2
        height = img_shape[0]
        width = img_shape[1]
        channels = img_shape[2]
    else:
        ApplyTD = lambda x: TimeDistributed(x)
        height4 = img_shape[1]/4
        width4 = img_shape[2]/4
        height2 = img_shape[1]/2
        width2 = img_shape[2]/2
        height = img_shape[1]
        width = img_shape[2]
        channels = img_shape[3]

    samples = Input(shape=img_shape)

    '''
    Convolutions for an image, terminating in a dense layer of size dim.
    '''

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    x = samples

    x = ApplyTD(Conv2D(filters,
                kernel_size=kernel_size, 
                strides=(1, 1),
                padding='same'))(x)
    x = ApplyTD(relu())(x)
    if dropout:
        x = ApplyTD(Dropout(dropout_rate))(x)

    for i in range(layers):

        x = ApplyTD(Conv2D(filters,
                   kernel_size=kernel_size, 
                   strides=(2, 2),
                   padding='same'))(x)
        x = ApplyTD(relu())(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)

    if flatten or dense:
        x = ApplyTD(Flatten())(x)
    if dense:
        x = ApplyTD(Dense(dim))(x)
        x = ApplyTD(relu())(x)

    return [samples], x

def SliceImageHypotheses(image_shape, num_hypotheses, x):
    '''
    Slice images. When we sample a set of images, we want to maintain the
    spatial organization inherent in the inputs. This is used to split one
    output into many different hypotheses.

    Here, we assume x is an input tensor of shape:
        (w,h,c) = image_shape
        x.shape == (w,h,c*num_hypotheses)

    For reference when debugging:
        # SLICING EXAMPLE:
        import keras.backend as K
        t = K.ones((12, 3))
        t1 = t[:, :1] + 1
        t2 = t[:, 1:] - 1
        t3 = K.concatenate([t1, t2])
        print(K.eval(t3))

    Parameters:
    -----------
    image_shape: (width,height,channels)
    num_hypotheses: number of images being created
    x: tensor of shape (width,height,num_hypotheses*channels)
    '''

    size = 1.
    for dim in image_shape:
        size *= dim
    y = []
    for i in range(num_hypotheses):
        xi = x[:,:,:,(3*i):(3*(i+1))]
        xi = K.expand_dims(xi,1)
        y.append(xi)
    return K.concatenate(y,axis=1)


def GetImageDecoder(dim, img_shape,
        dropout_rate, filters, kernel_size=[3,3], dropout=True, leaky=True,
        dense_rep_size=None,
        batchnorm=True,dense=True, num_hypotheses=None, tform_filters=None,
        original=None, upsampling=None,
        resnet_blocks=False,
        skips=False,
        stride2_layers=2, stride1_layers=1,
        stride2_layers_no_skip=0):

    '''
    Initial decoder: just based on getting images out of the world state
    created via the encoder.
    '''

    if tform_filters is None:
        tform_filters = filters

    if resnet_blocks:
        raise RuntimeError('RESNET: this option has been removed.')

    height = int(img_shape[0]/(2**stride2_layers))
    width = int(img_shape[1]/(2**stride2_layers))
    nchannels = img_shape[2]

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    if not dense:
        z = Input((height,width,tform_filters),name="input_image")
        x = z
        #z = Input((int(width*height*tform_filters),),name="input_image")
        #x = Reshape((height,width,tform_filters))(z)
    else:
        print ("dens rep size ", dense_rep_size)
        z = Input((int(dense_rep_size),),name="input_latent")
        x = Dense(int(height*width*3),name="dense_input_size")(z)
        if batchnorm:
            x = BatchNormalization()(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        x = Reshape((height,width,3))(x)
    skip_inputs = []

    height = height * 2
    width = width * 2
    for i in range(stride2_layers):

        if skips and i >= stride2_layers_no_skip:
            skip_in = Input((width/2,height/2,filters))
            x = Concatenate(axis=-1)([x, skip_in])
            skip_inputs.append(skip_in)

        # Upsampling.
        # Alternatives to Conv2D transpose for generation; this is because
        # conv2d transpose is known to result in artifacts, and we want to
        # avoid those when learning our nice decoder.
        if upsampling == "bilinear":
            x = Conv2D(filters,
                       kernel_size=kernel_size, 
                       strides=(1, 1),
                       padding='same')(x)

            x = Lambda(lambda x: tf.image.resize_bilinear(x,
                [height, width]),
                name="bilinear%dx%d"%(height,width))(x)
        elif upsampling == "upsampling":
            x = UpSampling2D(size=(2,2))(x)
            x = Conv2D(filters,
                       kernel_size=kernel_size, 
                       strides=(1, 1),
                       padding='same')(x)
        else:
            x = Conv2DTranspose(filters,
                       kernel_size=kernel_size, 
                       strides=(2, 2),
                       padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

        height *= 2
        width *= 2
 
    if skips:
        skip_in = Input((img_shape[0],img_shape[1],filters))
        x = Concatenate(axis=-1)([x,skip_in])
        #x = Add()([x, skip_in])
        skip_inputs.append(skip_in)

    for i in range(stride1_layers):
        x = Conv2D(filters, # + num_labels
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    if num_hypotheses is not None:
        x = Conv2D(num_hypotheses*nchannels, (1, 1), padding='same')(x)
        x = Lambda(lambda x: SliceImages(img_shape,num_hypotheses,x))(x)
    else:
        x = Conv2D(nchannels, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)
    print ("z ", [z], "skip inputs ", skip_inputs)
    ins = [z] + skip_inputs

    return ins, x


def GetImagePoseDecoder(dim, img_shape,
        dropout_rate, filters, dense_size, kernel_size=[3,3], dropout=True, leaky=True,
        dense_rep_size=None,
        batchnorm=True,dense=True, num_hypotheses=None, tform_filters=None,
        upsampling=None,
        original=None, num_options=64, pose_size=6,
        resnet_blocks=False, skips=None,
        stride2_layers=2, stride1_layers=1,
        stride2_layers_no_skip=0):
    '''
    Decode image and gripper setup.

    Parameters:
    -----------
    dim: dimensionality of hidden representation
    img_shape: shape of hidden image representation
    '''

    height = int(img_shape[0]/(2**stride2_layers))
    width = int(img_shape[1]/(2**stride2_layers))
    rep, dec = GetImageDecoder(dim,
                        img_shape,
                        dense_rep_size=dense_rep_size,
                        dropout_rate=dropout_rate,
                        kernel_size=kernel_size,
                        filters=filters,
                        stride2_layers=stride2_layers,
                        stride1_layers=stride1_layers,
                        stride2_layers_no_skip=stride2_layers_no_skip,
                        tform_filters=tform_filters,
                        dropout=dropout,
                        upsampling=upsampling,
                        leaky=leaky,
                        dense=dense,
                        skips=skips,
                        original=original,
                        resnet_blocks=resnet_blocks,
                        batchnorm=batchnorm,)

    if tform_filters is None:
        tform_filters = filters

    # =====================================================================
    # Decode arm/gripper state.
    # Predict the next joint states and gripper position. We add these back
    # in from the inputs once again, in order to make sure they don't get
    # lost in all the convolution layers above...
    if not dense:
        x = Reshape((height,width,tform_filters))(rep[0])
        x = Flatten()(x)
    else:
        x = rep[0]

    """
    x = Dense(dense_size)(x)
    x = BatchNormalization()(x)
    if leaky:
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation("relu")(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    """

    x1 = DenseHelper(x, 2*dense_size, dropout_rate, 2)
    x2 = DenseHelper(x, 2*dense_size, dropout_rate, 2)

    pose_out_x = Dense(pose_size,name="next_pose")(x1)
    label_out_x = Dense(num_options,name="next_label",activation="softmax")(x2)

    decoder = Model(rep,
                    [dec, pose_out_x, label_out_x],
                    name="decoder")

    return decoder

def DenseHelper(x, dense_size, dropout_rate, repeat):
    '''
    Add a repeated number of dense layers of the same size.
    '''
    for _ in range(repeat):
        AddDense(x, dense_size, "relu", dropout_rate)
    return x

def GetArmGripperDecoder(dim, img_shape,
        dropout_rate, filters, dense_size, kernel_size=[3,3], dropout=True, leaky=True,
        batchnorm=True,dense=True, num_hypotheses=None, tform_filters=None,
        upsampling=None,
        dense_rep_size=128,
        original=None, num_options=64, arm_size=7, gripper_size=1,
        resnet_blocks=False, skips=None,
        stride2_layers=2, stride1_layers=1):
    '''
    Create a version of the decoder that just estimates the robot's arm and
    gripper state, plus the label of the resulting action.
    '''

    if tform_filters is None:
        tform_filters = filters

    # =====================================================================
    # Decode arm/gripper state.
    # Predict the next joint states and gripper position. We add these back
    # in from the inputs once again, in order to make sure they don't get
    # lost in all the convolution layers above...
    if not dense:
        height = int(img_shape[0]/(2**stride2_layers))
        width = int(img_shape[1]/(2**stride2_layers))
        rep = Input((height,width,tform_filters))
        x = Flatten()(rep)
    else:
        rep = Input((dim,))
        x = rep

    """
    x = Dense(dense_size)(x)
    x = BatchNormalization()(x)
    if leaky:
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation("relu")(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    """

    x1 = DenseHelper(x, 2*dense_size, dropout_rate, 2)
    x2 = DenseHelper(x, 2*dense_size, dropout_rate, 2)

    arm_out_x = Dense(arm_size, name="next_arm", activation="linear")(x1)
    gripper_out_x = Dense(gripper_size,
            activation="sigmoid",
            name="next_gripper_flat")(x1)
    label_out_x = Dense(num_options,name="next_label",activation="softmax")(x2)

    decoder = Model(rep,
                    [arm_out_x, gripper_out_x, label_out_x],
                    name="decoder")
    return decoder

def GetImageArmGripperDecoder(dim, img_shape,
        dropout_rate, filters, dense_size, kernel_size=[3,3], dropout=True, leaky=True,
        dense_rep_size=None,
        batchnorm=True,dense=True, num_hypotheses=None, tform_filters=None,
        upsampling=None,
        original=None, num_options=64, arm_size=7, gripper_size=1,
        resnet_blocks=False, skips=None,
        stride2_layers=2, stride1_layers=1,
        stride2_layers_no_skip=0):
    '''
    Decode image and gripper setup.

    Parameters:
    -----------
    dim: dimensionality of hidden representation
    img_shape: shape of hidden image representation
    '''

    height = int(img_shape[0]/(2**stride2_layers))
    width = int(img_shape[1]/(2**stride2_layers))
    rep, dec = GetImageDecoder(dim,
                        img_shape,
                        dense_rep_size=dense_rep_size,
                        dropout_rate=dropout_rate,
                        kernel_size=kernel_size,
                        filters=filters,
                        stride2_layers=stride2_layers,
                        stride1_layers=stride1_layers,
                        stride2_layers_no_skip=stride2_layers_no_skip,
                        tform_filters=tform_filters,
                        dropout=dropout,
                        upsampling=upsampling,
                        leaky=leaky,
                        dense=dense,
                        skips=skips,
                        original=original,
                        resnet_blocks=resnet_blocks,
                        batchnorm=batchnorm,)

    if tform_filters is None:
        tform_filters = filters

    # =====================================================================
    # Decode arm/gripper state.
    # Predict the next joint states and gripper position. We add these back
    # in from the inputs once again, in order to make sure they don't get
    # lost in all the convolution layers above...
    if not dense:
        x = Reshape((height,width,tform_filters))(rep[0])
        x = Flatten()(x)
    else:
        x = rep[0]

    x1 = DenseHelper(x, 2*dense_size, dropout_rate, 2)
    x2 = DenseHelper(x, 2*dense_size, dropout_rate, 2)

    arm_out_x = Dense(arm_size,name="next_arm")(x1)
    gripper_out_x = Dense(gripper_size,
            name="next_gripper_flat")(x1)
    label_out_x = Dense(num_options,name="next_label",activation="softmax")(x2)

    decoder = Model(rep,
                    [dec, arm_out_x, gripper_out_x, label_out_x],
                    name="decoder")

    return decoder


def GetTransform(rep_size, filters, kernel_size, idx, num_blocks=2, batchnorm=True, 
        leaky=True,
        relu=True,
        dropout_rate=0.,
        dropout=False,
        use_noise=False,
        pred_option_in=None,
        option=None,
        noise_dim=32):
    '''
    Old version of the "transform" block. It assumes the hidden representation
    will be a very small image (say, 8x8x64).

    In general, all our predictor models are set up as:

        h ~ f_{enc}(x)
        h' ~ T(h)
        x ~ f_{dec}

    This is the middle part, where we compute the new hidden world state.
    '''

    #dim = filters
    #xin = Input((rep_size) + (dim,),"features_input")
    xin = Input(rep_size)
    if pred_option_in is not None:
        dim += pred_option_in
    if use_noise:
        zin = Input((noise_dim,))
        x = TileOnto(xin,zin,noise_dim,rep_size)

    if dropout:
        dr = dropout_rate
    else:
        dr = 0.

    x = xin
    for i in range(num_blocks):
        x = AddConv2D(x, filters, kernel_size, 1, dr)
    #x = AddConv2D(x, 64, [1,1], 1, 0.)
    #x = AddConv2D(x, 128, kernel_size, 1, 0.)
    #x = AddConv2D(x, 128, kernel_size, 1, 0.)
    #x = AddConv2DTranspose(x, 64, kernel_size, 1, 0.)
    #x = AddConv2DTranspose(x, filters, [1,1], 1, 0.)

    ins = [xin]
    if use_noise:
        ins += [zin]
    if pred_option_in is not None:
        ins += [oin]
    return Model(ins, x, name="transform%d"%idx)

def GetDenseTransform(dim, input_size, output_size, num_blocks=2, batchnorm=True, 
        idx=0,
        leaky=True,
        relu=True,
        dropout_rate=0.,
        dropout=False,
        use_noise=False,
        option=None,
        use_sampling=False,
        noise_dim=32):
    '''
    This is the suggested way of creating a "transform" -- AKA a mapping
    between the observed hidden world state at an encoding.

    In general, all our predictor models are set up as:

        h ~ f_{enc}(x)
        h' ~ T(h)
        x ~ f_{dec}

    This is the middle part, where we compute the new hidden world state.

    Parameters:
    -----------
    dim: size of the hidden representation
    input_size: 
    leaky: use LReLU instead of normal ReLU
    dropout_rate: amount of dropout to use (not recommended for MHP)
    dropout: use dropout (recommended FALSE for MHP)
    sampler: set up as a "sampler" model
    '''

    xin = Input((input_size,),name="tform%d_hidden_in"%idx)
    x = xin
    extra = []
    extra_concat = []
    if use_noise:
        zin = Input((noise_dim,),name="tform%d_noise_in"%idx)
        extra += [zin]
        extra_concat += [zin]
    if option is not None:
        oin = Input((option,),name="tform%d_option_in"%idx)
        extra += [oin]
        extra_concat += [oin]
        #option_x= OneHot(option)(oin)
    if len(extra) > 0:
        x = Concatenate()([x] + extra)
    for j in range(num_blocks):
        x = Dense(dim,name="dense_%d_%d"%(idx,j))(x)
        if batchnorm:
            x = BatchNormalization(name="normalize_%d_%d"%(idx,j))(x)
        if relu:
            if leaky:
                x = LeakyReLU(0.2,name="lrelu_%d_%d"%(idx,j))(x)
            else:
                x = Activation("relu",name="relu_%d_%d"%(idx,j))(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    # =========================================================================
    # In this block we divide into two separate paths:
    # (a) we deterministically return a hidden world
    # (b) we compute a mean and variance, then draw a sampled hidden world
    # The default path right now is via (a); (b) is experimental.
    if not use_sampling:
        return Model([xin] + extra, x, name="transform%d"%idx)
    else:
        mu = Dense(dim, name="tform%d_mu"%idx)(x)
        sigma = Dense(dim, name="tform%d_sigma"%idx)(x)

        def _sampling(args):
            '''
            Helper function for continuously sampling based on Mu and Sigma
            '''
            mu, sigma = args
            eps = K.random_normal(shape=(K.shape(mu)[0], dim),
                    mean=0.,
                    stddev=1.)
            return mu + K.exp(sigma / 2) * eps

        x = Lambda(_sampling,
                output_shape=(dim,),
                name="tform%d_sample"%idx)([mu, sigma])

        # Note that mu and sigma are both important outputs for computing the
        # KL regularization termin the loss function
        return Model([xin] + extra, [x, mu, sigma], name="transform%d"%idx)


def GetNextOptionAndValue(x, num_options, dense_size, dropout_rate=0.5, option_in=None):
    '''
    Predict some information about an observed/encoded world state

    Parameters:
    -----------
    x: input vector/image of hidden representation
    num_options: number of possible actions to predict
    '''
    if len(x.shape) > 2:
        x = Flatten()(x)
    if option_in is not None:
        option_x = OneHot(num_options)(option_in)
        option_x = Flatten()(option_x)
        x = Concatenate()([x, option_x])

    x1 = DenseHelper(x, int(2*dense_size), dropout_rate, 1)
    x2 = DenseHelper(x, dense_size, dropout_rate, 1)

    next_option_out = Dense(num_options,
            activation="softmax", name="next_label",)(x1)
    value_out = Dense(1, activation="sigmoid", name="value",)(x2)
    return value_out, next_option_out


def GetHypothesisProbability(x, num_hypotheses, num_options, labels,
        filters, kernel_size,
        dropout_rate=0.5):

    '''
    Compute probabilities across a whole set of action hypotheses, to ensure
    that the most likely hypothesis is one that seems reasonable.

    This is interesting because we might actually see multiple different
    hypotheses assigned to the same possible action. So the way it works is
    that we compute p(h) for all hypotheses h, and then construct a matrix of
    size:

        M = N_h x N_a

    with N_h = num hypotheses and N_a = number of actions.
    The "labels" input should contain p(a | h) for all a, so we can compute the
    matrix M as:

        M(h,a) = p(h) p(a | h)

    Then sum across all h to marginalize this out.

    Parameters:
    -----------
    x: the input hidden state representation
    num_hypotheses: N_h, as above
    num_options: N_a, as above
    labels: the input matrix of p(a | h), with size (?, N_h, N_a)
    filters: convolutional filters for downsampling
    kernel_size: kernel size for CNN downsampling
    dropout_rate: dropout rate applied to model
    '''

    #x = Conv2D(filters,
    #        kernel_size=kernel_size, 
    #        strides=(2, 2),
    #        padding='same',
    #        name="p_hypothesis")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = Dropout(dropout_rate)(x)
    #x = Flatten()(x)
    for _ in range(1):
        x = Dense(filters)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(num_hypotheses)(x)
    x = Activation("sigmoid")(x)
    x2 = x

    def make_p_matrix(pred, num_actions):
        x = K.expand_dims(pred,axis=-1)
        x = K.repeat_elements(x, num_actions, axis=-1)
        return x
    x = Lambda(lambda x: make_p_matrix(x, num_options),name="p_mat")(x)
    #labels.trainable = False
    x = Multiply()([x, labels])
    x = Lambda(lambda x: K.sum(x,axis=1),name="sum_p_h")(x)

    return x, x2

def OneHot(size=64):
    return Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32),size))

def AddOptionTiling(x, option_length, option_in, height, width):
    tile_shape = (1, width, height, 1)
    option = Reshape([1,1,option_length])(option_in)
    option = Lambda(lambda x: K.tile(x, tile_shape))(option)
    x = Concatenate(
            axis=-1,
            name="add_option_%dx%d"%(width,height),
        )([x, option])
    return x

def GetActor(enc0, enc_h, supervisor, label_out, num_hypotheses, *args, **kwargs):
    '''
    Set up an actor according to the probability distribution over decent next
    states.
    '''
    p_o = K.expand_dims(supervisor, axis=1)
    p_o = K.repeat_elements(p_o, num_hypotheses, axis=1)

    # Compute the probability of a high-level label under our distribution
    p_oh = K.sum(label_out, axis=1) / num_hypotheses
