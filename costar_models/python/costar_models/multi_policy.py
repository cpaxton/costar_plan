from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from matplotlib import pyplot as plt

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from .abstract import HierarchicalAgentBasedModel
from .multi_hierarchical import RobotMultiHierarchical
from .preprocess import *
from .robot_multi_models import *
from .split import *

class RobotPolicy(RobotMultiHierarchical):

    '''
    This class learns one policy at a time
    '''

    def __init__(self, taskdef, *args, **kwargs):
        super(RobotPolicy, self).__init__(taskdef, *args, **kwargs)

        if self.option is None:
            raise RuntimeError("Policy model requires an 'option' argument")
        if self.option >= self.null_option:
            raise RuntimeError("Policy model requires an option between 0 and " +str(self.null_option)) 


        # add option to self.name, for saving
        self.name += "_o" + str(self.option)


    def _makeModel(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        Set up all models necessary to create actions
        '''
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                features,
                arm,
                gripper)
        encoder = self._makeImageEncoder(img_shape)
        encoder.summary()
        #decoder = self._makeImageDecoder(self.hidden_shape)
        ''' debug
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            #decoder.load_weights(self._makeName(
            #    "pretrain_image_encoder_model",
            #    #"pretrain_image_gan_model",
            #    "image_decoder.h5f"))
            #decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e
        '''

        # Make end-to-end conditional actor
        self.model = self._makePolicy(
                encoder, features, arm, gripper,
                arm_cmd, gripper_cmd, option=self.option)

    def _getData(self, *args, **kwargs):
        '''
        Filter out the data not relevant to the current option
        '''
        features, targets = self._getAllData(*args, **kwargs)
        [_, _, _, _, label, _, _,] = features
        # find the matches
        idx = label == self.option
        [I, q, g, _, _, _, _,] = [f[idx] for f in features]
        [_, _, _, qa, ga, _] = [t[idx] for t in targets]
        return [I, q, g], [np.squeeze(qa), np.squeeze(ga)]

