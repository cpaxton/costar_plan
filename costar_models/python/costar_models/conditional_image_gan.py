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
from .conditional_image import *
from .pretrain_image_gan import *

class ConditionalImageGan(PretrainImageGan):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.

    This one generates:
      - image
      - arm command
      - gripper command
    '''

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(ConditionalImageGan, self).__init__(*args, **kwargs)
        self.PredictorCb = ImageWithFirstCb
        self.rep_size = 256
        self.num_transforms = 3
        self.do_all = True
        self.transform_model = None
        self.skip_connections = False
 
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
        img0_in = Input(img_shape,name="predictor_img0_in")
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img0_in, img_in]

        if self.skip_connections:
            encoder = self._makeImageEncoder2(img_shape)
        else:
            encoder = self._makeImageEncoder(img_shape)
        try:
            encoder.load_weights(self._makeName(
                #pretrain_image_encoder_model",
                "pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        if self.skip_connections:
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            decoder.load_weights(self._makeName(
                #"pretrain_image_encoder_model",
                "pretrain_image_gan_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))
            ins += [z]

        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
            h = encoder([img_in])
            h0 = encoder(img0_in)

        # next option - used to compute the next image 
        next_option_in = Input((1,), name="next_option_in")
        next_option2_in = Input((1,), name="next_option2_in")
        ins += [next_option_in, next_option2_in]

        y = OneHot(self.num_options)(next_option_in)
        y = Flatten()(y)
        y2 = OneHot(self.num_options)(next_option2_in)
        y2 = Flatten()(y2)
        x = h
        tform = self._makeTransform()
        x = tform([h0,h,y])
        x2 = tform([h0,x,y2])
        image_out = decoder([x])
        image_out2 = decoder([x2])

        # =====================================================================
        # Make the discriminator
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        is_fake = image_discriminator([
            img0_in, img_in,
            next_option_in, 
            next_option2_in,
            image_out,
            image_out2])

        # =====================================================================
        # Create generator model to train
        lfn = self.loss
        predictor = Model(ins,
                [image_out, image_out2])
        predictor.compile(
                loss=[lfn, lfn],
                optimizer=self.getOptimizer())
        self.generator = predictor

        # =====================================================================
        # And adversarial model 
        model = Model(ins, [image_out, image_out2, is_fake])
        model.compile(
                loss=["mae"]*2 + ["binary_crossentropy"],
                loss_weights=[100., 100., 1.],
                optimizer=self.getOptimizer())
        model.summary()
        self.model = model

        return predictor, model, model, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets

        # Create the next image including input image
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 

        # Extract the next goal
        I_target2, o2 = self._getNextGoal(features, targets)
        return [I0, I, o1, o2], [ I_target, I_target2 ]

    def _makeImageDiscriminator(self, img_shape):
        '''
        create image-only encoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        '''
        img0 = Input(img_shape,name="img0_encoder_in")
        img = Input(img_shape,name="img_encoder_in")
        img_goal = Input(img_shape,name="goal_encoder_in")
        img_goal2 = Input(img_shape,name="goal2_encoder_in")
        option = Input((1,),name="disc_options")
        option2 = Input((1,),name="disc2_options")
        ins = [img0, img, option, option2, img_goal, img_goal2]
        dr = self.dropout_rate
        dr = 0
        x = AddConv2D(img, 64, [5,5], 1, dr, "same", lrelu=True, bn=False)
        x0 = AddConv2D(img0, 64, [5,5], 1, dr, "same", lrelu=True, bn=False)
        x = Add()([x, x0])
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=True)

        # -------------------------------------------------------------
        y = OneHot(self.num_options)(option)
        y = AddDense(y, 64, "lrelu", dr)
        x = TileOnto(x, y, 64, (32,32), add=True)
        x = AddConv2D(x, 64, [5,5], 1, dr, "same", lrelu=True)

        xg = AddConv2D(img_goal, 64, [5,5], 2, dr, "same", lrelu=True, bn=False)
        x = Add()([x, xg])

        # -------------------------------------------------------------
        y = OneHot(self.num_options)(option2)
        y = AddDense(y, 64, "lrelu", dr)
        x = TileOnto(x, y, 64, (32,32), add=True)
        x = AddConv2D(x, 64, [5,5], 1, dr, "same", lrelu=True)

        xg2 = AddConv2D(img_goal2, 64, [5,5], 2, dr, "same", lrelu=True, bn=False)
        x = Add()([x, xg2])

        x = AddConv2D(x, 64, [5,5], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=True)
        #x = AddConv2D(x, 128, [5,5], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 256, [5,5], 2, dr, "same", lrelu=True)
        #x = AddConv2D(x, 256, [5,5], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 1, [5,5], 1, 0., "same", activation="sigmoid")

        #x = MaxPooling2D(pool_size=(8,8))(x)
        x = AveragePooling2D(pool_size=(8,8))(x)
        x = Flatten()(x)
        discrim = Model(ins, x, name="image_discriminator")
        self.lr *= 2.
        discrim.compile(loss="binary_crossentropy", loss_weights=[1.],
                optimizer=self.getOptimizer())
        self.lr *= 0.5
        self.image_discriminator = discrim
	discrim.summary()
        return discrim


