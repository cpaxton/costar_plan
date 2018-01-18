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
from .preprocess import *
from .robot_multi_models import *
from .split import *

class RobotMultiHierarchical(HierarchicalAgentBasedModel):

    '''
    This is the "divide and conquer"-style classifier for training a multilevel
    model. We use our supervised action labels to learn a superviser that will
    classify which action we should be performing from any particular frame,
    and then separately we learn a model of what we should be doing at each
    frame.

    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Similarly to everything else -- we need a taskdef here.

        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiHierarchical, self).__init__(taskdef, *args, **kwargs)

        self.img_col_dim = 512
        self.img_num_filters = 64
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = self.img_col_dim
        self.pose_col_dim = 64
        self.num_options = 48
        self.null_option = 37
        self.supervisor = None
        self.actor = None

        # Feature presets
        self.arm_cmd_size = 6
        self.gripper_cmd_size = 1

        self.use_spatial_softmax = False

    def _makeModel(self, features, arm, gripper, _, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        Set up all models necessary to create actions
        '''
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                features,
                arm,
                gripper)
        encoder = self._makeImageEncoder(img_shape)
        decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        # Make end-to-end conditional actor
        actor = self._makeConditionalActor(features, arm, gripper, arm_cmd,
                gripper_cmd, *args, **kwargs)
        self.model = actor

    def _makeSimpleActor(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        This creates a "dumb" actor model based on a set of features.
        '''
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        arm_cmd_size = arm_cmd.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        ins, x, skips = GetEncoder(
                img_shape,
                [arm_size, gripper_size],
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                pose_col_dim=self.pose_col_dim,
                discriminator=False,
                kernel_size=[3,3],
                tile=True,
                batchnorm=self.use_batchnorm,
                pre_tiling_layers=1,
                post_tiling_layers=3,
                stride1_post_tiling_layers=1)

        arm_out = Dense(arm_cmd_size, name="arm")(x)
        gripper_out = Dense(gripper_size, name="gripper")(x)

        if self.model is not None:
            raise RuntimeError('overwriting old model!')

        model = Model(ins, [arm_out, gripper_out])
        optimizer = self.getOptimizer()
        model.compile(loss=self.loss, optimizer=optimizer)
        return model

    def _makeConditionalActor(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        This creates a "dumb" actor model based on a set of features.
        '''
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        arm_cmd_size = arm_cmd.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1
        
        new = True
        if not new:
            ins, x, skips = GetEncoder(
                    img_shape,
                    [arm_size, gripper_size],
                    self.img_col_dim,
                    self.dropout_rate,
                    self.img_num_filters,
                    pose_col_dim=self.pose_col_dim,
                    discriminator=False,
                    kernel_size=[3,3],
                    tile=True,
                    batchnorm=self.use_batchnorm,
                    pre_tiling_layers=1,
                    post_tiling_layers=3,
                    stride1_post_tiling_layers=1,
                    option=self.num_options,
                    )
        else:
            img_in = Input(img_shape, name="ca_img_in")
            x = img_in
            x = AddConv2D(x, 64, [5,5], 2, self.dropout_rate, "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 2, self.dropout_rate, "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 1, 0., "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 1, 0., "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 1, 0., "valid", bn=self.use_batchnorm)

            arm_in = Input((arm_size,),name="ca_arm_in")
            gripper_in = Input((gripper_size,),name="ca_gripper_in")
            y = Concatenate()([arm_in, gripper_in])
            y = AddDense(y, 128, "relu", 0., output=True, constraint=3)
            x = TileOnto(x, y, 128, (8,8), add=True)

            cmd_in = Input((1,), name="option_cmd_in")
            cmd = OneHot(self.num_options)(cmd_in)
            cmd = AddDense(cmd, 128, "relu", 0., output=True, constraint=3)
            x = TileOnto(x, cmd, 128, (8,8), add=True)
            x = AddConv2D(x, 64, [3,3], 1, self.dropout_rate, "valid",
                    bn=self.use_batchnorm)
            #x = BatchNormalization()(x)
            x = Flatten()(x)
            x = AddDense(x, 512, "relu", 0.,
                    constraint=3,
                    output=True)
            x = Dropout(self.dropout_rate)(x)
            x = AddDense(x, 512, "relu", 0.,
                    constraint=3,
                    output=True)



        arm_out = Dense(arm_cmd_size, name="arm")(x)
        gripper_out = Dense(gripper_size, name="gripper")(x)

        if self.model is not None:
            raise RuntimeError('overwriting old model!')

        model = Model(ins, [arm_out, gripper_out])
        optimizer = self.getOptimizer()
        model.compile(loss=self.loss, optimizer=optimizer)
        return model


    def _makeAll(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        images = features
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1

        ins, x, skips = GetEncoder(
                img_shape,
                [arm_size, gripper_size],
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                pose_col_dim=self.pose_col_dim,
                kernel_size=[3,3],
                tile=True,
                pre_tiling_layers=1,
                post_tiling_layers=3,
                stride1_post_tiling_layers=1,
                discriminator=False,
                dense=False,
                option=self.num_options,
                flatten=False,
                )

        # =====================================================================
        # SUPERVISOR
        # Predict the next option -- does not depend on option
        for _ in range(2):
            # Repeat twice to scale down to a very small size -- this will help
            # a little with the final image layers
            x = Conv2D(int(self.img_num_filters),
                    kernel_size=[5, 5], 
                    strides=(2, 2),
                    padding='same')(x)
            x = Dropout(self.dropout_rate)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(self.num_options, activation="softmax",name="next_option")(x)

        supervisor = Model(ins, label_out, name="supervisor")
        actor = self._makeConditionalActor(features, arm, gripper, arm_cmd,
                gripper_cmd, *args, **kwargs)

        supervisor.summary()
        print("make model setup")
        print(ins, actor.inputs)
        #model_ins = Input(name="img_in")

    def _makePolicy(self, encoder, features, arm, gripper,
            arm_cmd, gripper_cmd, option):
        '''
        Create a single policy corresponding to option 

        Parameters:
        -----------
        option: index of the policy to create
        '''
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        arm_cmd_size = arm_cmd.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        h = encoder(img_in)
        x = Flatten()(h)
        h0 = encoder(img0_in)
        x0 = Flatten()(h0)
        arm = Input((arm_size,), name="ee_in")
        gripper = Input((gripper_size,), name="gripper_in")
        ins = Concatenate()([x0, x, arm, gripper])

        x = AddDense(x, 512, "lrelu", dr, output=True, bn=False)

        gripper_out = Dense(gripper_size, name="gripper" + str(option))(x)
        arm_out = Dense(arm_cmd_size, name="arm" + str(option))(x)

        model = Model(ins, [arm_out, gripper_out])
        model.compile(loss=self.loss, optimizer=self.getOptimizer())
        model.summary()
        return model

    def plotInfo(self, features, targets, axes):
        # debugging: plot every 5th image from the dataset
        subset = [f[range(0,25,5)] for f in features]
        data = self.predictor.predict(subset)
        for j in range(5):
            jj = j * 5
            ax = axes[1][j]
            ax.imshow(np.squeeze(data[0][j]))
            ax.axis('off')
            ax = axes[4][j]
            ax.imshow(np.squeeze(data[3][j]))
            ax.axis('off')
            ax = axes[0][j]
            ax.imshow(np.squeeze(features[0][jj]))
            ax.axis('off')
            ax = axes[2][j]
            ax.imshow(np.squeeze(targets[0][jj]))
            ax.axis('off')
            
            q0 = features[1][jj]
            q = data[1][j]
            q1 = targets[1][jj]
            ax = axes[3][j]
            ax.bar(np.arange(6),q0,1./3.,color='b')
            ax.bar(np.arange(6)+1./3.,q,1./3.,color='r')
            ax.bar(np.arange(6)+2./3.,q1,1./3.,color='g')

        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def _getAllData(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, value, *args, **kwargs):
        I = np.array(features) / 255. # normalize the images
        q = np.array(arm)
        g = np.array(gripper) * -1
        qa = np.array(arm_cmd)
        ga = np.array(gripper_cmd) * -1
        oin = prev_label
        I_target = np.array(goal_features) / 255.
        q_target = np.array(goal_arm)
        g_target = np.array(goal_gripper) * -1
        o_target = label

        # Preprocess values
        value_target = np.array(np.array(value) > 1.,dtype=float)
        #if value_target[-1] == 0:
        #    value_target = np.ones_like(value) - np.array(label == label[-1], dtype=float)
        q[:,3:] = q[:,3:] / np.pi
        q_target[:,3:] = np.array(q_target[:,3:]) / np.pi
        qa /= np.pi

        o_target_1h = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        train_target = self._makeTrainTarget(
                I_target,
                q_target,
                g_target,
                o_target_1h)

        return [I, q, g, oin, label, q_target, g_target,], [
                np.expand_dims(train_target, axis=1),
                o_target,
                value_target,
                np.expand_dims(qa, axis=1),
                np.expand_dims(ga, axis=1),
                I_target]

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        return [I, q, g, label], [np.squeeze(qa), np.squeeze(ga)]

    def _makeTrainTarget(self, I_target, q_target, g_target, o_target):
        if I_target is not None:
            length = I_target.shape[0]
            image_shape = I_target.shape[1:]
            image_size = 1
            for dim in image_shape:
                image_size *= dim
            image_size = int(image_size)
            Itrain = np.reshape(I_target,(length, image_size))
            return np.concatenate([Itrain, q_target,g_target,o_target],axis=-1)
        else:
            length = q_target.shape[0]
            return np.concatenate([q_target,g_target,o_target],axis=-1)


    def _loadWeights(self, *args, **kwargs):
        '''
        Load model weights. This is the default load weights function; you may
        need to overload this for specific models.
        '''
        if self.model is not None:
            print("using " + self.name + ".h5f")
            self.model.load_weights(self.name + ".h5f")
            if self.supervisor is not None:
                try:
                    self.supervisor.load_weights(self.name + "_supervisor.h5f")
                except Exception as e:
                    print(e)
            if self.actor is not None:
                try:
                    self.actor.load_weights(self.name + "_actor.h5f")
                except Exception as e:
                    print(e)
        else:
            raise RuntimeError('_loadWeights() failed: model not found.')

    def save(self):
        '''
        Save to a filename determined by the "self.name" field.
        '''
        if self.model is not None:
            print("saving to " + self.name)
            self.model.save_weights(self.name + ".h5f")
            if self.supervisor is not None:
                self.supervisor.save_weights(self.name + "_supervisor.h5f")
            if self.actor is not None:
                self.actor.save_weights(self.name + "_actor.h5f")
        else:
            raise RuntimeError('save() failed: model not found.')

    def trainFromGenerators(self, train_generator, test_generator, data=None, *args, **kwargs):
        #[features, arm, gripper, oi], [arm_cmd, gripper_cmd] = self._getData(**data)
        in_l, out_l = self._getData(**data)

        # concatenate all the arguments for splatting
        args = in_l + out_l + list(args)
        if self.model is None:
            self._makeModel(*args, **kwargs)
        self.model.summary()
        self.model.fit_generator(
                train_generator,
                self.steps_per_epoch,
                epochs=self.epochs,
                validation_steps=self.validation_steps,
                validation_data=test_generator,)

    def _makeImageEncoder(self, img_shape, disc=False):
        '''
        create image-only decoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        disc: is this being created as part of a discriminator network? If so,
              we handle things slightly differently.
        '''
        img = Input(img_shape,name="img_encoder_in")
        bn = not disc and self.use_batchnorm
        #img0 = Input(img_shape,name="img0_encoder_in")
        dr = self.dropout_rate
        x = img
        #x0 = img0
        x = AddConv2D(x, 32, [7,7], 1, 0., "same", lrelu=disc, bn=bn)
        #x0 = AddConv2D(x0, 32, [7,7], 1, dr, "same", lrelu=disc, bn=bn)
        #x = Concatenate(axis=-1)([x,x0])

        x = AddConv2D(x, 32, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
        x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
        x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
        x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=disc, bn=bn)

        if self.use_spatial_softmax and not disc:
            def _ssm(x):
                return spatial_softmax(x)
            self.encoder_channels = 32
            x = AddConv2D(x, self.encoder_channels, [1,1], 1, 0.*dr,
                    "same", lrelu=disc, bn=(not disc))
            x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
            self.hidden_shape = (self.encoder_channels*2,)
            self.hidden_size = 2*self.encoder_channels
            self.hidden_shape = (self.hidden_size,)
        else:
            self.encoder_channels = 8
            x = AddConv2D(x, self.encoder_channels, [1,1], 1, 0.*dr,
                    "same", lrelu=disc, bn=(not disc))
            self.steps_down = 3
            self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
            self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.encoder_channels)

        if not disc:

            image_encoder = Model([img], x, name="Ienc")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_encoder = image_encoder
        else:
            x = Flatten()(x)
            x = AddDense(x, 512, "lrelu", dr, output=True)
            x = AddDense(x, self.num_options, "softmax", 0., output=True)
            image_encoder = Model([img], x, name="Idisc")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_discriminator = image_encoder
        return image_encoder

    def _makeImageDecoder(self, hidden_shape, img_shape=None, copy=False):
        '''
        helper function to construct a decoder that will make images.

        parameters:
        -----------
        img_shape: shape of the image, e.g. (64,64,3)
        '''
        if self.use_spatial_softmax:
            rep = Input((self.hidden_size,),name="decoder_hidden_in")
        else:
            rep = Input(hidden_shape,name="decoder_hidden_in")

        x = rep
        if self.hypothesis_dropout:
            dr = self.decoder_dropout_rate
        else:
            dr = 0.
        
        if self.use_spatial_softmax:
            self.steps_up = 3
            hidden_dim = int(img_shape[0]/(2**self.steps_up))
            (h,w,c) = (hidden_dim,
                       hidden_dim,
                       self.encoder_channels)
            x = AddDense(x, int(h*w*c), "relu", dr)
            x = Reshape((h,w,c))(x)

        #x = AddConv2DTranspose(x, 64, [5,5], 1, dr)
        x = AddConv2DTranspose(x, 128, [1,1], 1, 0.)
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 64, [5,5], 1, 0.)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 1, 0.)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 1, 0.)
        ins = rep
        x = Conv2D(3, kernel_size=[1,1], strides=(1,1),name="convert_to_rgb")(x)
        x = Activation("sigmoid")(x)
        if not copy:
            decoder = Model(ins, x, name="Idec")
            decoder.compile(loss="mae",optimizer=self.getOptimizer())
            self.image_decoder = decoder
        else:
            decoder = Model(ins, x,)
            decoder.compile(loss="mae",optimizer=self.getOptimizer())
        return decoder

    def _makeImageEncoder2(self, img_shape, disc=False):
        '''
        create image-only decoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        disc: is this being created as part of a discriminator network? If so,
              we handle things slightly differently.
        '''
        img = Input(img_shape,name="img_encoder_in")
        img0 = Input(img_shape,name="img0_encoder_in")
        dr = self.dropout_rate
        x = img
        x0 = img0
        x = AddConv2D(x, 32, [7,7], 1, dr, "same", lrelu=disc, bn=(not disc))
        x0 = AddConv2D(x0, 32, [7,7], 1, dr, "same", lrelu=disc, bn=(not disc))
        #x = Add(axis=-1)([x,x0])
        x = Concatenate(axis=-1)([x,x0])
        xi = x

        x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=(not disc))
        x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=(not disc))
        x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=(not disc))
        xa = x
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=(not disc))
        x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=(not disc))
        xb = x
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=disc, bn=(not disc))
        xc = x

        if self.use_spatial_softmax and not disc:
            def _ssm(x):
                return spatial_softmax(x)
            self.encoder_channels = 32
            x = AddConv2D(x, self.encoder_channels, [1,1], 1, 0.*dr,
                    "same", lrelu=disc, bn=(not disc))
            x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
            self.hidden_shape = (self.encoder_channels*2,)
            self.hidden_size = 2*self.encoder_channels
            self.hidden_shape = (self.hidden_size,)
        else:
            self.encoder_channels = 32
            x = AddConv2D(x, self.encoder_channels, [1,1], 1, 0.*dr,
                    "same", lrelu=disc, bn=(not disc))
            self.steps_down = 3
            self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
            self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.encoder_channels)

        if not disc:

            image_encoder = Model([img0, img], [x, xi, xa, xb, xc], name="Ienc")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_encoder = image_encoder
        else:
            x = Flatten()(x)
            x = AddDense(x, 512, "lrelu", dr, output=True)
            x = AddDense(x, self.num_options, "softmax", 0., output=True)
            image_encoder = Model([img], x, name="Idisc")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_discriminator = image_encoder
        return image_encoder

    def _makeImageDecoder2(self, hidden_shape, img_shape=None, skip=False):
        '''
        helper function to construct a decoder that will make images.

        parameters:
        -----------
        img_shape: shape of the image, e.g. (64,64,3)
        '''
        if self.use_spatial_softmax:
            rep = Input((self.hidden_size,),name="decoder_hidden_in")
        else:
            rep = Input(hidden_shape,name="decoder_hidden_in")

        x = rep
        if self.hypothesis_dropout:
            dr = self.decoder_dropout_rate
        else:
            dr = 0.
        
        if self.use_spatial_softmax:
            self.steps_up = 3
            hidden_dim = int(img_shape[0]/(2**self.steps_up))
            (h,w,c) = (hidden_dim,
                       hidden_dim,
                       self.encoder_channels)
            x = AddDense(x, int(h*w*c), "relu", dr)
            x = Reshape((h,w,c))(x)

        s64 = Input((64,64,64),name="skip_64")
        s32 = Input((32,32,64),name="skip_32")
        s16 = Input((16,16,64),name="skip_16")
        s8 = Input((8,8,128),name="skip_8")
        x = AddConv2DTranspose(x, 128, [1,1], 1, 0.*dr)
        x = Add()([x,s8])
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = Add()([x,s16])
        x = AddConv2DTranspose(x, 64, [5,5], 1, 0.)
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = Add()([x,s32])
        x = AddConv2DTranspose(x, 64, [5,5], 1, 0.)
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = Add()([x,s64])
        x = AddConv2DTranspose(x, 64, [5,5], 1, 0.)
        x = Conv2D(3, kernel_size=[1,1], strides=(1,1),name="convert_to_rgb")(x)
        x = Activation("sigmoid")(x)
        ins = [rep, s32, s16, s8]
        decoder = Model(ins, x, name="Idec")
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        self.image_decoder = decoder
        return decoder

    def _sizes(self, images, arm, gripper):
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1
        image_size = 1
        for dim in img_shape:
            image_size *= dim
        image_size = int(image_size)

        return img_shape, image_size, arm_size, gripper_size


