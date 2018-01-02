from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Multiply
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from .abstract import *
from .callbacks import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *
from .sampler2 import *


class ConditionalSampler2(PredictionSampler2):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.
    '''

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(ConditionalSampler2, self).__init__(*args, **kwargs)
        self.PredictorCb = ImageCb

    def _makePredictor(self, features):
        # =====================================================================
        # Create many different image decoders
        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []
        
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img_in, arm_in, gripper_in, label_in]

        encoder = self._makeImageEncoder(img_shape)
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
        except Exception as e:
            pass

        if self.skip_connections:
            decoder = self._makeImageDecoder(self.hidden_shape,self.skip_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            pass

        sencoder = self._makeStateEncoder(arm_size, gripper_size, False)
        sdecoder = self._makeStateDecoder(arm_size, gripper_size,
                self.tform_filters)

        # =====================================================================
        # Load the arm and gripper representation

        # =====================================================================
        # combine these models together with state information and label
        # information
        hidden_encoder = self._makeToHidden(img_shape, arm_size, gripper_size, self.rep_size)
        hidden_decoder = self._makeFromHidden()

        try:
            hidden_encoder.load_weights(self._makeName(
                "pretrain_sampler_model",
                "hidden_encoder.h5f"))
            hidden_decoder.load_weights(self._makeName(
                "pretrain_sampler_model",
                "hidden_decoder.h5f"))
            hidden_encoder.trainable = self.retrain
            hidden_decoder.trainable = self.retrain
        except Exception as e:
            pass

        h = hidden_encoder(ins)
        value_out, next_option_out = GetNextOptionAndValue(h,
                                                           self.num_options,
                                                           self.rep_size,
                                                           dropout_rate=0.5,
                                                           option_in=None)

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))
            ins += [z]

        next_option_in = Input((48,), name="next_option_in")
        ins += [next_option_in]

        #y = OneHot(self.num_options)(next_option_in)
        #y = Flatten()(y)
        y = next_option_in
        x = h
        for i in range(self.num_transforms):
            x = TileOnto(x, y, self.num_options, (8,8))
            x = AddConv2D(x, self.tform_filters*2,
                    self.tform_kernel_size,
                    stride=1,
                    dropout_rate=self.tform_dropout_rate)
        x = AddConv2D(x, self.tform_filters,
                self.tform_kernel_size,
                stride=1,
                dropout_rate=self.tform_dropout_rate)
        image_out, arm_out, gripper_out, label_out = hidden_decoder(x)

        # =====================================================================
        # Create models to train
        predictor = Model(ins,
                [image_out, arm_out, gripper_out, label_out, next_option_out,
                    value_out])
        actor = None
        predictor.compile(
                loss=["mae", "mae", "mae", "mae", "categorical_crossentropy",
                      "mae"],
                loss_weights=[1., 1., 0.2, 0.025, 0.1, 0.1],
                optimizer=self.getOptimizer())
        predictor.summary()
        return predictor, predictor, actor, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
            return [I, q, g, oin, z, o1], [o1, v, I_target, q_target, g_target,
                    o1]
        else:
            return [I, q, g, oin, o1], [I_target, q_target, g_target, o1, o1,
                    v,]

