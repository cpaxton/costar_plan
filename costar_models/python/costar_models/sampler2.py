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
from .multi_sampler import *

class PredictionSampler2(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(PredictionSampler2, self).__init__(taskdef, *args, **kwargs)
        self.rep_size = None
        self.rep_channels = 8
        self.tform_filters = 32
        self.num_hypotheses = 4
        self.dense_representation = False
        self.num_transforms = 3
        self.tform_kernel_size  = [7,7]
        self.hidden_shape = (8,8,self.tform_filters)
        self.always_same_transform = False
        #self.PredictorCb = ImageCb

    def _makeToHidden(self, img_shape, arm_size, gripper_size, rep_size):
        '''
        Aggregate data and use it to compute a single hidden representation
        that we can use to update and store the world state

        Parameters:
        -----------
        img_shape: shape of input image data, e.g. (64,64,3)
        arm_size: shape of the arm data, e.g. 6
        gripper_size: shape of gripper data, e.g. 1
        '''
        img_in = Input(img_shape,name="predictor_img_in")
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        label_in = Input((1,))
        ins = [img_in, arm_in, gripper_in, label_in]

        if self.skip_connections:
            img_rep, skip_rep = self.image_encoder(img_in)
        else:
            #img_rep = self.image_encoder(img_in)
            img_rep = self.image_encoder(img_in)
        state_rep = self.state_encoder([arm_in, gripper_in, label_in])
        # Compress the size of the network
        x = TileOnto(img_rep, state_rep, 64, [8,8])
        x = AddConv2D(x, 128, [3,3], 1, self.dropout_rate, "same", False)
        # Projection down to the right size
        x = AddConv2D(x, self.tform_filters, [1,1], 1, 0.,
                "same", False)
        #x = Flatten()(x)
        self.rep_size = int(8 * 8 * self.tform_filters)
        self.hidden_size = (8, 8, self.tform_filters)

        if self.skip_connections:
            model = Model(ins, [x, skip_rep], name="encode_hidden_state")
        else:
            model = Model(ins, x, name="encoder")
        model.compile(loss="mae", optimizer=self.getOptimizer())
        #model.summary()
        self.hidden_encoder = model
        return model

    def _makeFromHidden(self):
        '''
        Create the "Decoder" half of the AE

        Parameters:
        -----------
        size: number of dimensions in the hidden representation
        disc: whether or not this should be set up as a new discriminator.
        '''
        ih, iw, ic = self.hidden_size
        h = Input((ih, iw, self.tform_filters),name="from_hidden_input")

        # ---------------------------------
        x = h
        dr = 0.
        x = AddConv2D(x, 128, [1,1], 1,
                0., "same", False)
        x_img = AddConv2D(x, self.encoder_channels, [5,5], 1,
                dr, "same", False)
        x_arm = AddConv2D(x, self.tform_filters, [5,5], 1,
                dr, "same", False)
        if self.skip_connections:
            skip_in = Input(self.skip_shape, name="skip_input_hd")
            ins = [x_img, skip_in]
            hidden_decoder_ins = [h, skip_in]
        else:
            ins = x_img
            hidden_decoder_ins = h

        img = self.image_decoder(ins)
        arm, gripper, label = self.state_decoder(x_arm)
        model = Model(hidden_decoder_ins, [img, arm, gripper, label],
                name="decoder")
        self.hidden_decoder = model
        return model

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

        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))

        if self.always_same_transform:
            transform = self._getTransform(0)
        for i in range(self.num_hypotheses):
            if not self.always_same_transform:
                transform = self._getTransform(i)

            if i == 0:
                transform.summary()

            if self.use_noise:
                zi = Lambda(lambda x: x[:,i], name="slice_z%d"%i)(z)
                x = transform([h, zi])
            else:
                x = transform([h])

            if self.skip_connections:
                img_x, arm_x, gripper_x, label_x = hidden_decoder([x, skip_rep])
            else:
                img_x, arm_x, gripper_x, label_x = hidden_decoder(x)

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            Flatten(name="flatten_img_%d"%i)(img_x),
                            arm_x,
                            gripper_x,
                            label_x])
            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)
            arm_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="arm_hypothesis_%d"%i)(arm_x)
            gripper_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="gripper_hypothesis_%d"%i)(gripper_x)
            label_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="label_hypothesis_%d"%i)(label_x)
            train_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="flattened_hypothesis_%d"%i)(train_x)

            image_outs.append(img_x)
            arm_outs.append(arm_x)
            gripper_outs.append(gripper_x)
            label_outs.append(label_x)
            train_outs.append(train_x)

        image_out = Concatenate(axis=1)(image_outs)
        arm_out = Concatenate(axis=1)(arm_outs)
        gripper_out = Concatenate(axis=1)(gripper_outs)
        label_out = Concatenate(axis=1)(label_outs)
        train_out = Concatenate(axis=1,name="all_train_outs")(train_outs)

        # =====================================================================
        # Create models to train
        if self.use_noise:
            ins += [z]
        predictor = Model(ins,
                [image_out, arm_out, gripper_out, label_out, next_option_out,
                    value_out])
        actor = None
        losses = [MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size, self.num_options],
                        weights=[1., 0.5, 0.1, 0.025],
                        loss=["mae","mae","mae","categorical_crossentropy"],
                        #stats=stats,
                        avg_weight=0.1),]
        if self.success_only and False:
            outs = [train_out, next_option_out]
            losses += ["binary_crossentropy"]
            loss_weights = [1.0, 0.]
        else:
            outs = [train_out, next_option_out, value_out]
            loss_weights = [1.0, 0.01, 0.01]
            losses += ["categorical_crossentropy", "binary_crossentropy"]
        train_predictor = Model(ins, outs)
        train_predictor.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=self.getOptimizer())
        train_predictor.summary()
        predictor.compile(
                loss=["mae", "mae", "mae", "mae", "categorical_crossentropy",
                      "mae"],
                optimizer=self.getOptimizer())

        return predictor, train_predictor, actor, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
            return [I, q, g, oin, z], [tt, o1, v]
        else:
            return [I, q, g, oin], [tt, o1, v]

