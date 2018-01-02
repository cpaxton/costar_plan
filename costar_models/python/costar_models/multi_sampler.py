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
from .multi_hierarchical import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *

class RobotMultiPredictionSampler(RobotMultiHierarchical):

    '''
    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiPredictionSampler, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1
        self.img_num_filters = 32
        self.tform_filters = 32
        self.tform_kernel_size  = [5,5]
        self.num_hypotheses = 4
        self.validation_split = 0.05
        self.num_options = 48
        self.num_features = 4
        self.null_option = 37

        # For the new model setup
        self.encoder_channels = 64
        self.skip_shape = (64,64,32)

        # Layer and model configuration
        self.extra_layers = 1
        self.use_spatial_softmax = False
        self.dense_representation = True
        if self.use_spatial_softmax and self.dense_representation:
            self.steps_down = 2
            self.steps_down_no_skip = 0
            self.steps_up = 4
            self.steps_up_no_skip = self.steps_up - self.steps_down
            #self.encoder_stride1_steps = 2+1
            self.encoder_stride1_steps = 2
            self.padding="same"
        else:
            self.steps_down = 4
            self.steps_down_no_skip = 3
            self.steps_up = 4
            self.steps_up_no_skip = 3
            self.encoder_stride1_steps = 1
            self.padding = "same"

        self.num_actor_policy_layers = 2
        self.num_generator_layers = 1
        self.num_arm_vars = 6

        # Number of nonlinear transformations to be applied to the hidden state
        # in order to compute a possible next state.
        if self.dense_representation:
            self.num_transforms = 1
        else:
            self.num_transforms = 3

        # Feature presets
        self.arm_cmd_size = 6
        self.gripper_cmd_size = 1

        # Used for classifiers: value and next option
        self.combined_dense_size = 256
        self.value_dense_size = 32

        # Size of the "pose" column containing arm, gripper info
        self.pose_col_dim = 64

        # Size of the hidden representation when using dense hidden
        # repesentations
        self.img_col_dim = int(2 * self.tform_filters)

        self.PredictorCb = PredictorShowImage
        self.hidden_dim = int(64/(2**self.steps_down))
        self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.tform_filters)

        # These are the list of models that we want to learn
        self.train_predictor = None
        self.image_discriminator = None
        self.predictor = None
        self.actor = None
        self.image_encoder = None
        self.image_decoder = None
        self.state_encoder = None
        self.state_decoder = None
        self.hidden_encoder = None
        self.hidden_decoder = None

        # ===================================================================
        # These are hard coded settings -- tweaking them may break a bunch of
        # things.
        # ===================================================================

        # This controls how we use the previous option.
        self.use_prev_option = True
        # Give transforms a prior on the next action
        self.use_next_option = False
        # Train actor model
        self.train_actor = True
        # Use the same transform for everything
        self.always_same_transform = False

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

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)
        
        # =====================================================================
        # Create the encoder and decoder networks -- these are sub-networks
        # that we may find useful in different situations.
        img_in = Input(img_shape,name="predictor_img_in")
        ins, enc, skips = self._makeEncoder(img_shape, arm_size, gripper_size)
        decoder = self._makeDecoder(img_shape, arm_size, gripper_size)

        # ===================================================================
        # Encode history
        if self.use_prev_option:
            img_in, arm_in, gripper_in, option_in = ins
        else:
            img_in, arm_in, gripper_in = ins
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))

        # =====================================================================
        # Decode results -- create all the various outputs we want from this
        # image

        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []

        if self.skip_connections:
            skips.reverse()
        
        if not self.use_prev_option:
            option_in = Input((1,),name="prev_option_in")
            ins += [option_in]
            pv_option_in = option_in
        else:
            pv_option_in = None
            
        #next_option_in = Input((self.num_options,),name="next_option_in")
        #ins += [next_option_in]

        value_out, next_option_out = GetNextOptionAndValue(enc,
                                                           self.num_options,
                                                           self.value_dense_size,
                                                           dropout_rate=self.dropout_rate,
                                                           option_in=pv_option_in)

        # =====================================================================
        # Create many different image decoders
        stats = []
        if self.always_same_transform:
            transform = self._getTransform(0)
        for i in range(self.num_hypotheses):
            if not self.always_same_transform:
                transform = self._getTransform(i)

            if i == 0:
                transform.summary()
            if self.use_noise:
                zi = Lambda(lambda x: x[:,i], name="slice_z%d"%i)(z)
                if self.use_next_option:
                    x = transform([enc, zi, next_option_in])
                else:
                    x = transform([enc, zi])
            else:
                if self.use_next_option:
                    x = transform([enc, next_option_in])
                else:
                    x = transform([enc])
            
            if self.sampling:
                x, mu, sigma = x
                stats.append((mu, sigma))

            # This maps from our latent world state back into observable images.
            if self.skip_connections:
                decoder_inputs = [x] + skips
            else:
                decoder_inputs = [x]

            img_x, arm_x, gripper_x, label_x = decoder(decoder_inputs)

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
                        weights=[0.5, 0.40, 0.05, 0.01],
                        loss=["mae","mae","mae","categorical_crossentropy"],
                        stats=stats,
                        avg_weight=0.05),]
        if self.success_only:
            outs = [train_out, next_option_out]
            losses += ["binary_crossentropy"]
            loss_weights = [0.60, 0.40]
        else:
            outs = [train_out, next_option_out, value_out]
            loss_weights = [0.90, 0.1, 0.0]
            losses += ["categorical_crossentropy", "binary_crossentropy"]

        train_predictor = Model(ins, outs)

        # =====================================================================
        # Create models to train
        train_predictor.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=self.getOptimizer())
        predictor.compile(loss=[
                MhpLoss(self.num_hypotheses,avg_weight=0.,loss="mae"),
                MhpLoss(self.num_hypotheses,avg_weight=0.,loss="mae"),
                MhpLoss(self.num_hypotheses,avg_weight=0.,loss="mae"),
                MhpLoss(self.num_hypotheses,avg_weight=0.,loss="categorical_crossentropy"),
                "categorical_crossentropy",
                "mae"],
        optimizer=self.getOptimizer())
        train_predictor.summary()

        return predictor, train_predictor, actor, ins, enc

    def _getTransform(self,i=0):
        transform_dropout = False
        use_options_again = self.use_next_option
        transform_batchnorm = True
        transform_relu = True
        if use_options_again:
            options = self.num_options
        else:
            options = None
        if self.dense_representation:
            transform = GetDenseTransform(
                    dim=self.img_col_dim,
                    input_size=self.img_col_dim,
                    output_size=self.img_col_dim,
                    idx=i,
                    batchnorm=transform_batchnorm,
                    dropout=transform_dropout,
                    dropout_rate=self.dropout_rate,
                    leaky=True,
                    num_blocks=self.num_transforms,
                    use_sampling=self.sampling,
                    relu=transform_relu,
                    option=options,
                    use_noise=self.use_noise,
                    noise_dim=self.noise_dim,)
        else:
            transform_kernel_size = self.tform_kernel_size
            transform = GetTransform(
                    rep_size=(self.hidden_dim, self.hidden_dim,
                        self.tform_filters),
                    filters=self.tform_filters,
                    kernel_size=transform_kernel_size,
                    idx=i,
                    batchnorm=True,
                    dropout=transform_dropout,
                    dropout_rate=self.dropout_rate,
                    leaky=True,
                    num_blocks=self.num_transforms,
                    relu=True,
                    option=options,
                    use_noise=self.use_noise,
                    noise_dim=self.noise_dim,)
        return transform

    def _makeModel(self, features, arm, gripper, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        features, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.train_predictor, self.actor, ins, hidden = \
            self._makePredictor(
                (features, arm, gripper))

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

    def _getAllData(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, value, *args, **kwargs):
        I = features / 255. # normalize the images
        q = arm
        g = gripper * -1
        qa = arm_cmd
        ga = gripper_cmd * -1
        oin = prev_label
        I_target = goal_features / 255.
        q_target = goal_arm
        g_target = goal_gripper * -1
        o_target = label

        # Preprocess values
        value_target = np.array(value > 1.,dtype=float)
        q[:,3:] = q[:,3:] / np.pi
        q_target[:,3:] = q_target[:,3:] / np.pi
        qa /= np.pi

        o_target = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        train_target = self._makeTrainTarget(
                I_target,
                q_target,
                g_target,
                o_target)

        return [I, q, g, oin, q_target, g_target,], [
                np.expand_dims(train_target, axis=1),
                o_target,
                value_target,
                np.expand_dims(qa, axis=1),
                np.expand_dims(ga, axis=1),
                I_target]


    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        tt, o1, v, qa, ga, I = targets
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
            if self.success_only:
                return features[:self.num_features] + [z], [tt, o1]
            else:
                return features[:self.num_features] + [z], [tt, o1, v]
        else:
            if self.success_only:
                return features[:self.num_features], [tt, o1]
            else:
                return features[:self.num_features], [tt, o1, v]

    def trainFromGenerators(self, train_generator, test_generator, data=None):
        '''
        Train tool from generator

        Parameters:
        -----------
        train_generator: produces training examples
        test_generator: produces test examples
        data: some extra data used for debugging (should be validation data)
        '''
        if data is not None:
            features, targets = self._getAllData(**data)
        else:
            raise RuntimeError('predictor model sets sizes based on'
                               'sample data; must be provided')
        # ===================================================================
        # Use sample data to compile the model and set everything else up.
        # Check to make sure data makes sense before running the model.

        [I, q, g, oprev, q_target, g_target,] = features
        [I_target2, o_target, value_target, qa, ga, I_target0] = targets

        if self.predictor is None:
            self._makeModel(I, q, g, qa, ga)

        # Compute helpful variables
        image_shape = I.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        arm_size = q.shape[-1]
        gripper_size = g.shape[-1]

        train_size = image_size + arm_size + gripper_size + self.num_options
        assert gripper_size == 1
        # NOTE: arm size is one bigger when we have quaternions
        #assert train_size == 12295 + self.num_options
        #assert train_size == 12296 + self.num_options
        assert train_size == (64*64*3) + self.num_arm_vars + 1 + self.num_options

        # ===================================================================
        # Create the callbacks and actually run the training loop.
        modelCheckpointCb = ModelCheckpoint(
            filepath=self.name+"_predictor_weights.h5f",
            verbose=1,
            save_best_only=True # does not work without validation wts
        )
        logCb = LogCallback(self.name,self.model_directory)
        cbf, cbt = self._getData(**data)
        imageCb = self.PredictorCb(
            self.predictor,
            name=self.name_prefix,
            features=cbf,
            targets=cbt,
            model_directory=self.model_directory,
            num_hypotheses=self.num_hypotheses,
            verbose=True,
            use_noise=self.use_noise,
            noise_dim=self.noise_dim,
            min_idx=0,
            max_idx=70,
            step=10,)
        callbacks=[modelCheckpointCb, logCb, imageCb]
        self._fit(train_generator, test_generator, callbacks)

    def _fit(self, train_generator, test_generator, callbacks):
        self.train_predictor.fit_generator(
            train_generator,
            self.steps_per_epoch,
            epochs=self.epochs,
            validation_steps=self.validation_steps,
            validation_data=test_generator,
            callbacks=callbacks)


    def save(self):
        '''
        Save to a filename determined by the "self.name" field.
        '''
        if self.predictor is not None:
            print("----------------------------")
            print("Saving to " + self.name + "_{predictor, actor, image_decoder}")
            self.train_predictor.save_weights(self.name + "_train_predictor.h5f")
            if self.actor is not None:
                self.predictor.save_weights(self.name + "_predictor.h5f")
                self.actor.save_weights(self.name + "_actor.h5f")
            if self.image_decoder is not None:
                self.image_decoder.save_weights(self.name +
                "_image_decoder.h5f")
            if self.image_encoder is not None:
                self.image_encoder.save_weights(self.name + 
                "_image_encoder.h5f")
            if self.state_encoder is not None:
                self.state_encoder.save_weights(self.name +
                "_state_encoder.h5f")
            if self.state_decoder is not None:
                self.state_decoder.save_weights(self.name + 
                "_state_decoder.h5f")
            if self.hidden_encoder is not None:
                self.hidden_encoder.save_weights(self.name + 
                "_hidden_encoder.h5f")
            if self.hidden_decoder is not None:
                self.hidden_decoder.save_weights(self.name + 
                "_hidden_decoder.h5f")
        else:
            raise RuntimeError('save() failed: model not found.')

    def _loadWeights(self, *args, **kwargs):
        '''
        Load model weights. This is the default load weights function; you may
        need to overload this for specific models.
        '''
        if self.predictor is not None:
            print("----------------------------")
            print("using " + self.name + " to load")
            try:
                self.actor.load_weights(self.name + "_actor.h5f")
                #self.predictor.load_weights(self.name + "_predictor.h5f")
            except Exception as e:
                print("Could not load actor:", e)
            self.train_predictor.load_weights(self.name + "_train_predictor.h5f")
            if self.image_decoder is not None:
                self.image_decoder.load_weights(self.name +
                "_image_decoder.h5f")
            if self.image_encoder is not None:
                self.image_encoder.load_weights(self.name +
                "_image_encoder.h5f")
            if self.state_decoder is not None:
                self.state_decoder.load_weights(self.name +
                "_state_decoder.h5f")
            if self.state_encoder is not None:
                self.state_encoder.load_weights(self.name +
                "_state_encoder.h5f")
        else:
            raise RuntimeError('_loadWeights() failed: model not found.')

    def predict(self, world):
        '''
        Evaluation for a feature-predictor model. This has two steps:
          - predict a set of features associated with the current world state
          - predict the expected reward based on each of those features
          - choose the best one to execute
        '''
        features = world.initial_features
        test_features = []
        next_option_idx = 0
        for f in features:
            f2 = np.expand_dims(f,axis=0)
            tile_shape = [self.batch_size,] + [1]*len(f.shape)
            f2 = np.tile(f2,tile_shape)
            test_features.append(f2)
            next_option_idx += 1

        if self.use_prev_option:
            # previous options
            #prev_option = self._makeOption1h(self.prev_option)
            #tile_shape = [self.batch_size,1]
            #prev_option = np.tile(prev_option, tile_shape)
            if self.prev_option is None:
                prev = self.null_option
            else:
                prev = self.prev_option
            prev_option = np.ones((self.batch_size,1)) * prev
            test_features.append(prev_option)
            next_option_idx += 1
            
        if self.use_next_option:
            # don't include anything from the next options...
            next_opt = np.zeros((self.batch_size,self.num_options))
            next_opt[0,34] = 1
            test_features.append(next_opt)

        if self.use_noise:
            z = np.random.random((
                self.batch_size,
                self.num_hypotheses,
                self.noise_dim))
            test_features.append(z)

        data, arms, grippers, label, probs, v = self.predictor.predict(test_features)
        if self.use_next_option:
            next_probs = np.zeros_like(probs)
            for i in range(self.batch_size):
                p = np.cumsum(probs[i]) / np.sum(probs[i])
                r = np.random.random()
                opt = np.argmax(r < p)
                print (i, r, p, opt)
                next_probs[i,opt] = 1   
            test_features[next_option_idx] = probs
            data, arms, grippers, label, probs, v = self.predictor.predict(test_features)

        for i in range(self.batch_size):
            a = np.argmax(probs[i])
            print ("action = ",
                    a,
                    np.max(probs[i]),
                    self.taskdef.name(a))

        idx = np.random.randint(self.num_hypotheses)

        fig = plt.figure()
        for i in range(self.num_hypotheses):
            print ("label = ", np.argmax(label[0,i]),np.max(label[0,i]))
            for j in range(self.batch_size):
                idx = (i*self.batch_size) + j + 1
                plt.subplot(self.num_hypotheses,self.batch_size,idx)
                plt.imshow(data[j,i])
                print("arms = ", arms[j,i])
        plt.show()

        i = np.random.randint(self.num_hypotheses)
        j = np.random.randint(self.batch_size)
        self.prev_option = np.argmax(label[j,i])
        print ("choosing ", j, i, "=",
                self.prev_option,
                self.taskdef.name(self.prev_option))

        # Return the chosen goal pose
        return arms[j,i], grippers[j,i]

    def _makeActorPolicy(self):
        '''
        Helper function: creates a model for the "actor" policy that will
        generate the controls to move towards a particular end effector pose.
        The job of this policy should be pretty simple.

        The actor policy is trained separately from the predictor/sampler
        policies, but using the same underlying representation.
        '''
        enc = Input((self.img_col_dim,))
        arm_goal = Input((self.num_arm_vars,),name="actor_arm_goal_in")
        gripper_goal = Input((1,),name="actor_gripper_goal_in")
        y = enc
        if not self.dense_representation:
            raise RuntimeError('Not yet supported!')
            y = Conv2D(int(self.img_num_filters/4),
                    kernel_size=[5,5], 
                    strides=(2, 2),
                    padding='same')(y)
            y = Dropout(self.dropout_rate)(y)
            y = LeakyReLU(0.2)(y)
            y = BatchNormalization(momentum=0.9)(y)
            y = Flatten()(y)
        else:
            y = Concatenate()([y, arm_goal, gripper_goal])
            for _ in range(self.num_actor_policy_layers):
                y = Dense(self.combined_dense_size)(y)
                y = BatchNormalization(momentum=0.9)(y)
                y = LeakyReLU(0.2)(y)
                y = Dropout(self.dropout_rate)(y)
        arm_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="arm_action")(
                Dense(self.arm_cmd_size)(y))
        gripper_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="gripper_action")(
                Dense(self.gripper_cmd_size)(y))
        actor = Model([enc, arm_goal, gripper_goal], [arm_cmd_out,
            gripper_cmd_out], name="actor")
        return actor

    def _makeGenerator(self, img_shape, kernel_size, skips=None):
        rep, dec = GetImageDecoder(self.img_col_dim,
                        img_shape,
                        dropout_rate=self.dropout_rate,
                        kernel_size=kernel_size,
                        filters=self.img_num_filters,
                        stride2_layers=self.steps_down,
                        stride1_layers=self.extra_layers,
                        tform_filters=self.tform_filters,
                        dropout=self.hypothesis_dropout,
                        upsampling=self.upsampling_method,
                        dense=self.dense_representation,
                        dense_rep_size=self.img_col_dim,
                        leaky=True,
                        skips=skips,
                        original=None,
                        resnet_blocks=self.residual,
                        batchnorm=True,)
        decoder = Model(rep, dec)
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        hidden = Input((self.img_col_dim,), name="generator_hidden_in")
        arm_goal = Input((self.num_arm_vars,),name="generator_arm_goal_in")
        gripper_goal = Input((1,),name="generator_gripper_goal_in")

        y = Concatenate()([hidden, arm_goal, gripper_goal])
        for _ in range(self.num_generator_layers):
            y = Dense(self.img_col_dim)(y)
            y = BatchNormalization(momentum=0.9)(y)
            y = LeakyReLU(0.2)(y)
            y = Dropout(self.dropout_rate)(y)
    
        skip_ins = []
        if self.skip_connections:
            for skip in skips:
                shape = [int(i) for i in skip.shape[1:]]
                skip_ins.append(Input(shape))
            img_out = decoder([y] + skip_ins)
        else:
            img_out = decoder([y])

        generator = Model(
                [hidden, arm_goal, gripper_goal] + skip_ins,
                img_out,
                name="generator")
        return generator

    def _makeEncoder(self, img_shape, arm_size, gripper_size):
        '''
        Make the encoder, update our model's associated "encoder" network.
        Unlike with the decoder, this one actually returns a parsed list of
        outputs -- because in this case we may not want to explicitly break up
        the outputs ourselves every time as we do for the multiple hypotheses.

        The job of the encoder is to reduce the dimensionality of a single
        image. Integrating this information from multiple images is a different
        matter entirely.

        Parameters:
        -----------
        img_shape: shape of the input images
        arm_size: size of the arm's end effector representation
        gripper_size: number of variables representing our gripper state

        Returns:
        --------
        ins: the input variables we need to fill (for the predictor model, not
             the encoder block)
        enc: encoded state
        skips: list of skip connections
        '''
        if self.use_prev_option:
            enc_options = self.num_options
        else:
            enc_options = None
        ins, enc, skips = GetEncoder(img_shape,
                [arm_size, gripper_size],
                self.img_col_dim,
                dropout_rate=self.dropout_rate,
                filters=self.img_num_filters,
                leaky=False,
                dropout=True,
                padding=self.padding,
                pre_tiling_layers=self.extra_layers,
                post_tiling_layers=self.steps_down,
                post_tiling_layers_no_skip=self.steps_down_no_skip,
                stride1_post_tiling_layers=self.encoder_stride1_steps,
                pose_col_dim=self.pose_col_dim,
                kernel_size=[5,5],
                kernel_size_stride1=[5,5],
                dense=self.dense_representation,
                batchnorm=True,
                tile=True,
                flatten=(not self.use_spatial_softmax),
                option=enc_options,
                use_spatial_softmax=self.use_spatial_softmax,
                output_filters=self.tform_filters,
                )
        self.encoder = Model(ins, [enc]+skips, name="encoder")
        self.encoder.compile(loss="mae",optimizer=self.getOptimizer())
        self.encoder.summary()
        new_ins = []
        for idx, i in enumerate(ins):
            i2 = Input(
                    [int(d) for d in i.shape[1:]],
                    name="predictor_input_%d"%idx)
            new_ins.append(i2)

        outs = self.encoder(new_ins)
        new_enc = outs[0]
        new_skips = outs[1:]
        return new_ins, new_enc, new_skips

    def _makeDecoder(self, img_shape, arm_size, gripper_size,
            skips=None):
        '''
        Make the decoder network. This one takes in a hidden state (a set of
        keypoints in 2D image space) and from these keypoints computes:
        - an image of the world
        - a robot end effector
        - a robot gripper
        '''
        decoder = GetImageArmGripperDecoder(
                        self.img_num_filters,
                        img_shape,
                        dropout_rate=self.decoder_dropout_rate,
                        dense_size=self.combined_dense_size,
                        kernel_size=[5,5],
                        filters=self.img_num_filters,
                        stride2_layers=self.steps_up,
                        stride1_layers=self.extra_layers,
                        stride2_layers_no_skip=self.steps_up_no_skip,
                        tform_filters=self.tform_filters,
                        num_options=self.num_options,
                        arm_size=arm_size,
                        gripper_size=gripper_size,
                        dropout=self.hypothesis_dropout,
                        upsampling=self.upsampling_method,
                        leaky=True,
                        dense=self.dense_representation,
                        dense_rep_size=self.img_col_dim,
                        skips=self.skip_connections,
                        resnet_blocks=self.residual,
                        batchnorm=True,)
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        decoder.summary()
        return decoder

    def _makeStateEncoder(self, arm_size, gripper_size, disc=False):
        '''
        Encode arm state.

        Parameters:
        -----------
        arm_size: number of arm input variables
        gripper_size: number of gripper input variables
        disc: is this a discriminator? if so, use leaky relu
        '''
        arm = Input((arm_size,))
        gripper = Input((gripper_size,))
        option = Input((1,))
        if disc:
            activation = "lrelu"
        else:
            activation = "relu"
        y = OneHot(self.num_options)(option)
        y = Flatten()(y)
        x = Concatenate()([arm,gripper,y])

        dr = self.dropout_rate * 0.
        x = AddDense(x, 64, activation, dr)
        
        state_encoder = Model([arm, gripper, option], x,
                name="state_encoder")
        #state_encoder = Model([arm, gripper], x)
        state_encoder.compile(loss="mae", optimizer=self.getOptimizer())
        if not disc:
            self.state_encoder = state_encoder
        return state_encoder

    def _makeStateDecoder(self, arm_size, gripper_size, rep_channels):
        '''
        Compute actions from hidden representation

        Parameters:
        -----------
        arm_size: number of arm output variables to predict
        gripper_size: number of gripper output variables to predict
        '''
        rep_in = Input((8,8,rep_channels,))
        dr = self.decoder_dropout_rate 

        x = rep_in
        x = AddConv2D(x, 64, [3,3], 2, 0., "same", False)
        x = AddConv2D(x, 64, [3,3], 1, 0., "same", False)
        x = Flatten()(x)
        x1 = AddDense(x, 512, "relu", 0.)
        x1 = AddDense(x1, 512, "relu", 0.)
        x2 = AddDense(x, 256, "relu", 0.)
        arm = AddDense(x1, arm_size, "linear", 0., output=True)
        gripper = AddDense(x1, gripper_size, "sigmoid", 0., output=True)
        #y = AddDense(x, 512, "relu", dr, output=True)
        option = AddDense(x2, self.num_options, "softmax", 0., output=True)
        state_decoder = Model(rep_in, [arm, gripper, option],
                name="state_decoder")
        state_decoder.compile(loss="mae", optimizer=self.getOptimizer())
        self.state_decoder = state_decoder
        state_decoder.summary()
        return state_decoder

    def _makeMergeEncoder(self, img_shape, arm_shape, gripper_shape):
        '''
        Take input image and state information and encode them into a single
        hidden representation
        '''
        img_in = Input(img_shape,name="predictor_img_in")
        option_in = Input((1,), name="predictor_option_in")
 

    def _makeMergeDecoder(self, rep_size):
        '''
        Take input state and image information projected into a latent space
        and decode them back into their appropriate output representations
        '''

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
        dr = self.dropout_rate
        x = img
        x = AddConv2D(x, 32, [5,5], 2, dr, "same", disc)
        x = AddConv2D(x, 32, [5,5], 1, 0., "same", disc)
        x = AddConv2D(x, 32, [5,5], 1, 0., "same", disc)
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", disc)
        x = AddConv2D(x, 64, [5,5], 1, 0., "same", disc)
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", disc)
        self.encoder_channels = 8
        x = AddConv2D(x, self.encoder_channels, [1,1], 1, 0.*dr,
                "same", disc)

        if self.use_spatial_softmax and not disc:
            def _ssm(x):
                return spatial_softmax(x)
            x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
            self.hidden_shape = (self.encoder_channels*2,)
            #x = AddDense(x, self.hidden_size, "relu", dr)
            self.hidden_size = 2*self.encoder_channels
            self.hidden_shape = (self.hidden_size,)
        else:
            self.steps_down = 3
            self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
            self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.encoder_channels)

        if not disc:

            if self.skip_connections:
                image_encoder = Model([img], [x, y], name="image_encoder")
            else:
                image_encoder = Model([img], x, name="image_encoder")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_encoder = image_encoder
        else:
            x = Flatten()(x)
            x = AddDense(x, 512, "lrelu", dr, output=True)
            x = AddDense(x, self.num_options, "softmax", 0., output=True)
            image_encoder = Model([img], x, name="image_discriminator")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_discriminator = image_encoder
        return image_encoder

    def _makeImageDecoder(self, hidden_shape, img_shape=None, skip=False):
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

        if skip:
            skip1 = Input((32,32,32),name="decoder_skip_in_1")
            #skip2 = Input((16,16,32),name="decoder_skip_in_2")
            #skip3 = Input((8,8,32),name="decoder_skip_in_3")
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
        x = AddConv2DTranspose(x, 128, [1,1], 1, 0.*dr)
        #x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 64, [5,5], 1, 0.)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 1, 0.)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 1, 0.)
        if self.skip_connections and img_shape is not None:
            x = Concatenate(axis=-1)([x, skip])
            ins = [rep, skip]
        else:
            ins = rep
        x = Conv2D(3, kernel_size=[1,1], strides=(1,1),name="convert_to_rgb")(x)
        x = Activation("sigmoid")(x)
        decoder = Model(ins, x, name="image_decoder")
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        self.image_decoder = decoder
        return decoder

    def _targetsFromTrainTargets(self, train_targets):
        '''
        This helper function splits the train targets into separate fields. It
        is equivalent to the targets used in the training data.

        Parameters:
        -----------
        train_targets: training for multiple hypothesis data

        Returns:
        --------
        list of separated training targets
        '''
        imglen = 64*64*3
        if len(train_targets[0].shape) == 2:
            img = train_targets[0][:,:imglen]
        elif len(train_targets[0].shape) == 3:
            assert train_targets[0].shape[1] == 1
            img = train_targets[0][:,0,:imglen]
        else:
            raise RuntimeError('did not recognize big train target shape; '
                               'are you sure you meant to use this callback'
                               'and not a normal image callback?')
        num = train_targets[0].shape[0]
        img = np.reshape(img, (num,64,64,3))
        arm = np.squeeze(train_targets[0][:,:,imglen:imglen+6])
        gripper = train_targets[0][:,:,imglen+6]
        option = np.squeeze(train_targets[0][:,:,imglen+7:])
        #print (train_targets[0].shape,imglen,imglen+6)
        #print("img",img.shape)
        #print("arm",arm.shape,arm[0])
        #print("gripper",gripper.shape,gripper[0])
        #print("option",option.shape,np.argmax(option[0,0]))
        return [img,arm,gripper,option]

    def _parsePredictorLoss(self, losses):
        (_, img_loss, arm_loss, gripper_loss, label_loss, next_opt_loss,
            val_loss) = losses
        #print("img loss = ", img_loss)
        #print("arm loss = ", arm_loss)
        #print("gripper loss = ", gripper_loss)
        #print("label loss = ", label_loss)
        #print("next_opt loss = ", next_opt_loss)
        return [img_loss, arm_loss, gripper_loss, label_loss]

    def validate(self, *args, **kwargs):
        '''
        Run validation on a given trial.

        Note: this takes in whatever data your model needs to extract
        information for the next task. It's designed to work for any variant of
        the "predictor" model architecture, regardless of the specifics of the
        dataset -- or at least so we hope.

        > For a special case of the multi-predictor model:
          You MUST override the _targetsFromTrainTargets function above.

        Parameters:
        ----------
        None - just args and kwargs passed to _getData.

        Returns:
        --------
        error
        train_loss
        [loss per train target]
        '''
        features, targets = self._getData(*args, **kwargs)
        length = features[0].shape[0]
        prediction_targets = self._targetsFromTrainTargets(targets)
        for i in range(len(prediction_targets)):
                prediction_targets[i] = np.expand_dims(
                        prediction_targets[i],
                        axis=1)
        prediction_targets += [np.zeros((length,self.num_options))]
        prediction_targets += [np.zeros((length,))]
        sums = None
        train_sum = 0
        for i in range(length):
            f = [np.array([f[i]]) for f in features]
            t = [np.array([t[i]]) for t in targets]
            pt = [np.array([pt[i]]) for pt in prediction_targets]
            loss, train_loss, next_loss = self.train_predictor.evaluate(f, t,
                    verbose=0)
            #print ("actual arm = ", kwargs['goal_arm'][0])
            #print ("actual gripper = ", kwargs['goal_gripper'][0])
            #print ("actual prev opt = ", kwargs['label'][0])
            predictor_losses = self.predictor.evaluate(f, pt, verbose=0)
            losses = self._parsePredictorLoss(predictor_losses)
            train_sum += train_loss
            if sums is None:
                sums = np.array(losses)
            else:
                sums += np.array(losses)

        return sums, train_sum, length
