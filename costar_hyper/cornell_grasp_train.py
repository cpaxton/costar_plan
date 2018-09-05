#!/usr/local/bin/python
'''
Training a network on cornell grasping dataset for detecting grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code Based on:
    https://github.com/tnikolla/robot-grasp-detection

'''
import os
import re
import errno
import sys
import json
import csv
import argparse
import os.path
import glob
import datetime
import tensorflow as tf
import numpy as np
import six
import random
from shapely.geometry import Polygon
import cornell_grasp_dataset_reader

from block_stacking_reader import CostarBlockStackingSequence
from block_stacking_reader import block_stacking_generator

import time
from tensorflow.python.platform import flags
# TODO(ahundt) consider removing this dependency
import grasp_visualization

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


from keras import backend as K
import keras
import keras_contrib
import keras_tqdm
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import model_from_json
from grasp_model import concat_images_with_tiled_vector_layer
from grasp_model import top_block
from grasp_model import create_tree_roots
from grasp_model import choose_hypertree_model
from cornell_grasp_dataset_reader import parse_and_preprocess

from callbacks import EvaluateInputGenerator
from callbacks import PrintLogsCallback
from callbacks import FineTuningCallback
from callbacks import SlowModelStopping
from callbacks import InaccurateModelStopping
from keras.utils import OrderedEnqueuer

import grasp_loss
import grasp_metrics
import grasp_utilities


flags.DEFINE_float(
    'learning_rate',
    0.02,
    'Initial learning rate.'
)
flags.DEFINE_float(
    'random_augmentation',
    0.25,
    'Frequency from 0.0 to 1.0 with which random augmentation is performed. Currently for block stacking dataset only.'
)
flags.DEFINE_float(
    'fine_tuning_learning_rate',
    0.001,
    'Initial learning rate, this is the learning rate used if load_weights is passed.'
)
flags.DEFINE_integer(
    'fine_tuning_epochs',
    100,
    'Number of epochs to run trainer with all weights marked as trainable.'
)
flags.DEFINE_integer(
    'epochs',
    200,
    'Number of epochs to run trainer.'
)
flags.DEFINE_integer(
    'batch_size',
    16,
    'Batch size.'
)
flags.DEFINE_string(
    'log_dir',
    './logs_cornell/',
    'Directory for tensorboard, model layout, model weight, csv, and hyperparam files'
)
flags.DEFINE_string(
    'run_name',
    '',
    'A string that will become part of the logged directories and filenames.'
)
flags.DEFINE_integer(
    'num_folds',
    5,
    'Total number of folds, how many times should the data be split between training and validation'
)
flags.DEFINE_integer(
    'num_train',
    8,
    'Number of files used for training in one fold, '
    'must be less than the number of tfrecord files, aka splits.'
)
flags.DEFINE_integer(
    'num_validation',
    2,
    'Number of tfrecord files for validation.'
    'must be less than the number of tfrecord files, aka splits.'
    'This number also automatically determines the number of folds '
    'when running when the pipeline_stage flag includes k_fold.'
)
flags.DEFINE_integer(
    'num_test',
    0,
    'num of fold used for the test dataset'
    'must be less than the number of tfrecord files, aka splits.'
    'This must be 0 when the pipeline_stage flag includes k_fold'
)
flags.DEFINE_string('load_weights', None,
                    """Path to hdf5 file containing model weights to load and continue training.""")
flags.DEFINE_string('load_hyperparams', None,
                    """Load hyperparams from a json file.""")
flags.DEFINE_string('pipeline_stage', 'train_test',
                    """Choose to "train", "test", "train_test", or "train_test_kfold" with the grasp_dataset
                       data for training and grasp_dataset_test for testing.""")
flags.DEFINE_integer(
    'override_train_steps',
    None,
    'TODO(ahundt) REMOVE THIS HACK TO SKIP TRAINING BUT KEEP USING CALLBACKS.'
)
flags.DEFINE_string(
    'split_dataset', 'objectwise',
    """Options are imagewise and objectwise, this is the type of split chosen when the tfrecords were generated.""")
flags.DEFINE_string('tfrecord_filename_base', 'cornell-grasping-dataset', 'base of the filename used for the cornell dataset tfrecords and csv files')
flags.DEFINE_string('costar_filename_base', 'costar_block_stacking_v0.3_success_only',
                    'base of the filename used for the costar block stacking dataset tfrecords and csv files, '
                    'specifying None or empty string will generate a new file list from the files in FLAGS.data_dir')
flags.DEFINE_string(
    'feature_combo', 'image_preprocessed_norm_sin2_cos2_width_3',
    """
    feature_combo: The name for the combination of input features being utilized.
        Options include 'image_preprocessed', image_preprocessed_width_1,
        'image_preprocessed_norm_sin2_cos2_width_3'
        See choose_features_and_metrics() for details.
    """
)
flags.DEFINE_string(
    'problem_type', 'grasp_classification',
    """Choose between different formulations of the grasping task.
    Problem type options are 'segmentation', 'classification',
    'image_center_grasp_regression',
    'grasp_regression' which tries to predict successful grasp bounding boxes,
    'grasp_classification'
    'grasp_segmentation' which tries to classify input grasp parameters at each pixel.
    'pixelwise_grasp_regression' which tries to predict successful grasp bounding boxes at each pixel.

    """
)
flags.DEFINE_boolean(
    'fine_tuning', False,
    """ If true the model will be fine tuned the entire training run.

        This means that any imagenet weights will be made trainable,
        and the learning rate will be set to fine_tuning_learning_rate.
    """)
flags.DEFINE_string(
    'kfold_params', None,
    """ Load the json file containing parameters from a kfold cross validation run.
    """
)
flags.DEFINE_string(
    'dataset_name',
    'cornell_grasping',
    'Configure training run for a specific dataset.'
    ' Options are: cornell_grasping and costar_block_stacking.'
)

flags.DEFINE_string(
    'learning_rate_schedule',
    'reduce_lr_on_plateau',
    """Options are: reduce_lr_on_plateau, triangular, triangular2, exp_range, none.

    For details see the keras callback ReduceLROnPlateau and the
    keras_contrib callback CyclicLR. With triangular, triangular2,
    and exp_range the maximum learning rate
    will be double the input learning rate you specify
    so that the average initial learning rate is as specified..
    """
)

FLAGS = flags.FLAGS


def save_user_flags(save_filename, line_limit=80, verbose=1):
    """ print and save the tf FLAGS

    based on https://github.com/melodyguan/enas
    """
    if verbose > 0:
        print("-" * 80)
    flags_dict = FLAGS.flag_values_dict()

    for flag_name, flag_value in six.iteritems(flags_dict):
        value = "{}".format(getattr(FLAGS, flag_name))
        flags_dict[flag_name] = value
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        if save_filename is not None:
            with open(save_filename, 'w') as fp:
                # save out all flags params so they can be reloaded in the future
                json.dump(flags_dict, fp)
        if verbose > 0:
            print(log_string)


class GraspJaccardEvaluateCallback(keras.callbacks.Callback):
    """ Validate a model which needs custom numpy metrics during training.

    Note that this may have bugs due to issues when multiple tf sessions are created.
    Therefore, this may be deleted in the future.
    #TODO(ahundt) replace when https://github.com/keras-team/keras/pull/9105 is available

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, filenames=None, example_generator=None, steps=None, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(GraspJaccardEvaluateCallback, self).__init__()
        print('filenames: ' + str(filenames))
        print('generator: ' + str(example_generator))
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix
        self.filenames = filenames
        self.example_generator = example_generator

    def on_epoch_end(self, epoch, logs={}):
        # results = self.model.evaluate_generator(self.generator, steps=int(self.num_steps))
        metrics_str = '\n'
        metric_name = self.metrics_prefix + '_grasp_jaccard'
        # all our results come together in this call

        # TODO(ahundt) VAL_ON_TRAIN_TEMP_REMOVEME
        results = evaluate(self.model, example_generator=self.example_generator, val_filenames=self.filenames, visualize=True)
        for name, result in results:
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
        if self.verbose > 0:
            metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


def run_training(
        learning_rate=None,
        batch_size=None,
        num_gpus=1,
        top='classification',
        epochs=None,
        preprocessing_mode=None,
        train_data=None,
        validation_data=None,
        train_filenames=None,
        train_size=None,
        val_filenames=None,
        val_size=None,
        test_filenames=None,
        test_size=None,
        feature_combo_name=None,
        problem_name=None,
        image_model_name='vgg',
        optimizer_name='sgd',
        log_dir=None,
        hyperparams=None,
        load_weights=None,
        pipeline=None,
        run_name=None,
        fine_tuning_learning_rate=None,
        fine_tuning=None,
        fine_tuning_epochs=None,
        loss=None,
        checkpoint=True,
        dataset_name=None,
        should_initialize=False,
        hyperparameters_filename=None,
        **kwargs):
    """

    top: options are 'segmentation' and 'classification'.
    problem_name: options are 'grasp_regression', 'grasp_classification',
        'pixelwise_grasp_regression', 'pixelwise_grasp_classification',
        'image_center_grasp_regression'. Image center grasp regression is
        a pretraining step for pixel
        Make sure this is properly coordinated with 'top' param.
    feature_combo_name: The name for the combination of input features being utilized.
        Options include 'image_preprocessed', image_preprocessed_width_1,
        image_preprocessed_sin2_cos2_width_3
        'grasp_regression', image_center_grasp_regression.
        See choose_features_and_metrics() for details.
    hyperparams: a dictionary of hyperparameter selections made for this training run.
       If provided these values will simply be dumped to a file and
       not utilized in any other way.
    checkpoint: if True, checkpoints will be save, if false they will not.
    should_initialize: Workaround for some combined tf/keras bug (Maybe fixed in tf 1.8?)
       see https://github.com/keras-team/keras/issues/4875#issuecomment-313166165,
       TODO(ahundt) remove should_initialize and the corresponding code below if it has been False for a while without issue.
    hyperparameters_filename: Write a file '*source_hyperparameters_filename.txt' to a txt file with a path to baseline hyperparams
        on which this training run is based. The file will not be loaded, only the filename will be copied for
        purposes of tracing where models were generated from, such as if they are the product of hyperparmeter optimization.
        Specify the actual hyperparams using the argument "hyperparams".
    """
    if epochs is None:
        epochs = FLAGS.epochs
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if learning_rate is None:
        learning_rate = FLAGS.learning_rate
    if log_dir is None:
        log_dir = FLAGS.log_dir
    if load_weights is None:
        load_weights = FLAGS.load_weights
    if pipeline is None:
        pipeline = FLAGS.pipeline_stage
    if problem_name is None:
        problem_name = FLAGS.problem_type
    if run_name is None:
        run_name = FLAGS.run_name
    if fine_tuning_learning_rate is None:
        fine_tuning_learning_rate = FLAGS.fine_tuning_learning_rate
    if fine_tuning is None:
        fine_tuning = FLAGS.fine_tuning
    if fine_tuning_epochs is None:
        fine_tuning_epochs = FLAGS.fine_tuning_epochs
    if feature_combo_name is None:
        feature_combo_name = FLAGS.feature_combo
    if dataset_name is None:
        dataset_name = FLAGS.dataset_name

    if image_model_name == 'nasnet_large':
        # set special dimensions for nasnet
        FLAGS.crop_height = 331
        FLAGS.crop_width = 331
        FLAGS.resize_height = 331
        FLAGS.resize_width = 331
        print('Note: special overrides have been applied '
              'to support nasnet_large.'
              ' crop + resize width/height have been set to 331.')
        #   ' Loss is repeated, and '
    if image_model_name == 'inception_resnet_v2':
        # set special dimensions for inception resnet v2
        # https://github.com/keras-team/keras/blob/master/keras/applications/inception_resnet_v2.py#L194
        FLAGS.crop_height = 299
        FLAGS.crop_width = 299
        FLAGS.resize_height = 299
        FLAGS.resize_width = 299
        print('Note: special overrides have been applied '
              'to support inception_resnet_v2.'
              ' crop + resize width/height have been set to 299.')

    [image_shapes, vector_shapes, data_features, model_name,
     monitor_loss_name, label_features, monitor_metric_name,
     loss, metrics, classes, success_only] = choose_features_and_metrics(feature_combo_name, problem_name, loss=loss)

    if should_initialize:
        keras.backend.get_session().run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # see parse_and_preprocess() for the creation of these features
    model_name = image_model_name + model_name

    # If loading pretrained weights
    # it is very important to preprocess
    # in exactly the same way the model
    # was originally trained
    preprocessing_mode = choose_preprocessing_mode(preprocessing_mode, image_model_name)

    # choose hypertree_model with inputs [image], [sin_theta, cos_theta]
    model = choose_hypertree_model(
        image_shapes=image_shapes,
        vector_shapes=vector_shapes,
        top=top,
        classes=classes,
        image_model_name=image_model_name,
        **kwargs)

    if load_weights:
        if os.path.isfile(load_weights):
            model.load_weights(load_weights)
        else:
            print('Could not load weights {}, '
                  'the file does not exist, '
                  'starting fresh....'.format(load_weights))

    print('Saving weights as ' + monitor_loss_name + ' improves.')
    # TODO(ahundt) add a loss that changes size with how open the gripper is
    # loss = grasp_loss.segmentation_gaussian_measurement

    dataset_names_str = dataset_name
    run_name = grasp_utilities.make_model_description(run_name, model_name, hyperparams, dataset_names_str, label_features[0])
    callbacks = []

    # don't return the whole dictionary of features, only the specific ones we want
    val_all_features = False
    # # # TODO(ahundt) check this more carefully, currently a hack
    # # # Special case for jaccard regression
    # if((feature_combo_name == 'image/preprocessed' or feature_combo_name == 'image_preprocessed') and
    #         problem_name == 'grasp_regression'):
    #     val_all_features = True

    train_data, train_steps, validation_data, validation_steps, test_data, test_steps = load_dataset(
        train_filenames=train_filenames, train_size=train_size,
        val_filenames=val_filenames, val_size=val_size,
        test_filenames=test_filenames, test_size=test_size,
        label_features=label_features, data_features=data_features, batch_size=batch_size,
        train_data=train_data, validation_data=validation_data, preprocessing_mode=preprocessing_mode,
        success_only=success_only, val_batch_size=1, val_all_features=val_all_features, dataset_name=dataset_name
    )

    loss_weights = None
    # if image_model_name == 'nasnet_large':
    #     # TODO(ahundt) switch to keras_contrib NASNet model and enable aux network below when keras_contrib is updated with correct weights https://github.com/keras-team/keras/pull/10209.
    #     # nasnet_large has an auxiliary network,
    #     # so we apply the loss on both
    #     loss = [loss, loss]
    #     loss_weights = [1.0, 0.4]
    #     label_features += label_features

    callbacks, optimizer = choose_optimizer(optimizer_name, learning_rate, callbacks, monitor_loss_name, train_steps=train_steps)

    log_dir = os.path.join(log_dir, run_name)

    print('Writing logs for models, accuracy and tensorboard in ' + log_dir)
    log_dir_run_name = os.path.join(log_dir, run_name)
    grasp_utilities.mkdir_p(log_dir)

    # If this is based on some other past hyperparams configuration,
    # save the original hyperparams path to a file for tracing data pipelines.
    if hyperparameters_filename is not None:
        with open(log_dir_run_name + '_source_hyperparameters_filename.txt', 'w') as fp:
            fp.write(hyperparameters_filename)

    # Save the hyperparams to a json string so it is human readable
    if hyperparams is not None:
        with open(log_dir_run_name + '_hyperparams.json', 'w+') as fp:
            json.dump(hyperparams, fp)

    # Save the current model to a json string so it is human readable
    with open(log_dir_run_name + '_model.json', 'w') as fp:
        fp.write(model.to_json())

    save_user_flags(log_dir_run_name + '_flags.json')

    # Stop when models are extremely slow
    max_batch_time_seconds = 1.0
    if epochs > 10:
        # give extra time if it is a long run because
        # it was probably manually configured
        max_batch_time_seconds *= 2
    callbacks += [SlowModelStopping(max_batch_time_seconds=max_batch_time_seconds)]
    # stop models that make predictions that are close to all true or all false
    # this check works for both classification and sigmoid pose estimation
    callbacks += [InaccurateModelStopping(min_pred=0.01, max_pred=0.99)]
    # TODO(ahundt) some models good at angle are bad at cart & vice-versa, so don't stop models early
    # if 'costar' in dataset_name:
    #     max_cart_error = 1.0
    #     if epochs > 10:
    #         # give extra time if it is a long run because
    #         # it was probably manually configured
    #         max_cart_error *= 2
    #     # stop models that don't at least get within 40 cm after 300 batches.
    #     callbacks += [InaccurateModelStopping(min_pred=0.0, max_pred=max_cart_error, metric='cart_error')]

    if checkpoint:
        checkpoint = keras.callbacks.ModelCheckpoint(
            log_dir_run_name + '-epoch-{epoch:03d}-' +
            monitor_loss_name + '-{' + monitor_loss_name + ':.3f}-' +
            monitor_metric_name + '-{' + monitor_metric_name + ':.3f}.h5',
            save_best_only=True, verbose=1, monitor=monitor_metric_name)

        callbacks = callbacks + [checkpoint]

        # An additional useful param is write_batch_performance:
        #  https://github.com/keras-team/keras/pull/7617
        #  write_batch_performance=True)
        progress_tracker = TensorBoard(log_dir=log_dir, write_graph=True,
                                       write_grads=False, write_images=False,
                                       histogram_freq=0, batch_size=batch_size)
                                       # histogram_freq=0, batch_size=batch_size,
                                       # write_batch_performance=True)
        callbacks = callbacks + [progress_tracker]

    # make sure the TQDM callback is always the final one
    callbacks += [keras_tqdm.TQDMCallback(metric_format="{name}: {value:0.6f}")]

    # TODO(ahundt) enable when https://github.com/keras-team/keras/pull/9105 is resolved
    # callbacks += [FineTuningCallback(epoch=0)]

    # if num_gpus > 1:
    #     model = keras.utils.multi_gpu_model(model, num_gpus)
    # else:
    #     model = model

    # Order matters, so keep the csv logger at the end
    # of the list of callbacks.
    csv_logger = CSVLogger(log_dir_run_name + '.csv')
    callbacks = callbacks + [csv_logger]
    callbacks += [PrintLogsCallback()]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights)

    # # TODO(ahundt) check this more carefully, currently a hack
    # # Special case for jaccard regression
    # if((feature_combo_name == 'image/preprocessed' or feature_combo_name == 'image_preprocessed') and
    #         problem_name == 'grasp_regression'):
    #     # TODO(ahundt) VAL_ON_TRAIN_TEMP_REMOVEME
    #     callbacks = [GraspJaccardEvaluateCallback(example_generator=validation_data, steps=validation_steps)] + callbacks
    #     validation_data = None
    #     validation_steps = None

    # TODO(ahundt) remove hack below or don't directly access flags and initialize it correctly above
    # hack to skip training so we can run
    # val + test steps evaluate without changing the model
    if FLAGS.override_train_steps is not None:
        train_steps = FLAGS.override_train_steps

    # Get the validation dataset in one big numpy array for validation
    # This lets us take advantage of tensorboard visualization
    if 'train' in pipeline:
        if 'test' in pipeline:
            if test_steps == 0 and not hasattr(test_data, '__len__'):
                raise ValueError('Attempting to run test data' + str(test_filenames) +
                                 ' with an invalid number of steps: ' + str(test_steps))
            # we need this callback to be at the beginning of the callbacks list!
            print('test_data function: ' + str(test_data) + ' steps: ' + str(test_steps) +
                  'test filenames: ' + str(test_filenames))
            callbacks = [EvaluateInputGenerator(generator=test_data,
                                                steps=test_steps,
                                                metrics_prefix='test',
                                                verbose=0)] + callbacks

        # print('calling model.fit_generator()')

        # TODO(ahundt) Do hack which resets the session & reloads weights for now... will fix later. (Fixed in 1.8?)
        # Workaround for some combined tf/keras bug
        # see https://github.com/keras-team/keras/issues/4875#issuecomment-313166165
        # keras.backend.manual_variable_initialization(True)
        if should_initialize:
            sess = keras.backend.get_session()
            # init_g = tf.global_variables_initializer()
            # init_l = tf.local_variables_initializer()
            # sess.run(init_g)
            # sess.run(init_l)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        print('check 2 - train_steps: ' + str(train_steps) + ' validation_steps: ' + str(validation_steps) + ' test_steps: ' + str(test_steps))
        # fit the model
        # TODO(ahundt) may need to disable multiprocessing for cornell and enable it for costar stacking
        history = model.fit_generator(
            train_data,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            use_multiprocessing=False,
            workers=20,
            verbose=0)

        #  TODO(ahundt) remove when FineTuningCallback https://github.com/keras-team/keras/pull/9105 is resolved
        if fine_tuning and fine_tuning_epochs is not None and fine_tuning_epochs > 0:
            # do fine tuning stage after initial training
            print('')
            print('')
            print('Initial training complete, beginning fine tuning stage')
            print('------------------------------------------------------')
            _, optimizer = choose_optimizer(optimizer_name, fine_tuning_learning_rate, [], monitor_loss_name)

            for layer in model.layers:
                layer.trainable = True

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics)

            history = model.fit_generator(
                train_data,
                steps_per_epoch=train_steps,
                epochs=epochs + fine_tuning_epochs,
                validation_data=validation_data,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=0,
                initial_epoch=epochs)

    elif 'test' in pipeline:
        if test_steps == 0:
            raise ValueError('Attempting to run test data' + str(test_filenames) +
                             ' with an invalid number of steps: ' + str(test_steps))
        history = model.evaluate_generator(generator=test_data, steps=test_steps)
    else:
        raise ValueError('unknown pipeline configuration ' + pipeline + ' chosen, try '
                         'train, test, train_test, or train_test_kfold')

    if checkpoint:
        # only save the weights if we are also checkpointing
        model.save_weights(log_dir_run_name + '_model_weights.h5')

    print('')
    print('')
    print('This training run is complete')
    print('------------------------------------------------------')
    return history


def get_compiled_model(learning_rate=None,
                       batch_size=None,
                       num_gpus=1,
                       top='classification',
                       epochs=None,
                       preprocessing_mode=None,
                       input_filenames=None,
                       feature_combo_name='image_preprocessed_norm_sin2_cos2_width_3',
                       problem_name=None,
                       image_model_name='vgg',
                       optimizer_name='sgd',
                       log_dir=None,
                       hyperparams=None,
                       load_weights=None,
                       pipeline=None,
                       train_size=None,
                       val_size=None,
                       test_size=None,
                       run_name=None,
                       train_filenames=None,
                       test_filenames=None,
                       val_filenames=None,
                       loss=None,
                       **kwargs):
    """
    Get a compiled model instance and input data.
    input_filenames: path to tfrecord want to load.

    top: options are 'segmentation' and 'classification'.
    problem_name: options are 'grasp_regression', 'grasp_classification',
        'pixelwise_grasp_regression', 'pixelwise_grasp_classification',
        'image_center_grasp_regression'. Image center grasp regression is
        a pretraining step for pixel
        Make sure this is properly coordinated with 'top' param.
    feature_combo_name: The name for the combination of input features being utilized.
        Options include 'image_preprocessed', image_preprocessed_width_1,
        image_preprocessed_sin2_cos2_width_3
        'grasp_regression', image_center_grasp_regression.
        See choose_features_and_metrics() for details.
    hyperparams: a dictionary of hyperparameter selections made for this training run.
       If provided these values will simply be dumped to a file and
       not utilized in any other way.
    """
    if epochs is None:
        epochs = FLAGS.epochs
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if log_dir is None:
        log_dir = FLAGS.log_dir
    if learning_rate is None:
        if load_weights is None:
            learning_rate = FLAGS.learning_rate
        else:
            learning_rate = FLAGS.fine_tuning_learning_rate
    if load_weights is None:
        load_weights = FLAGS.load_weights
    if problem_name is None:
        problem_name = FLAGS.problem_type

    [image_shapes, vector_shapes, data_features, model_name,
     monitor_loss_name, label_features, monitor_metric_name,
     loss, metrics, classes, success_only] = choose_features_and_metrics(feature_combo_name, problem_name)

    # see parse_and_preprocess() for the creation of these features
    model_name = image_model_name + model_name

    # If loading pretrained weights
    # it is very important to preprocess
    # in exactly the same way the model
    # was originally trained
    preprocessing_mode = choose_preprocessing_mode(preprocessing_mode, image_model_name)

    # choose hypertree_model with inputs [image], [sin_theta, cos_theta]
    model = choose_hypertree_model(
        image_shapes=image_shapes,
        vector_shapes=vector_shapes,
        top=top,
        classes=classes,
        image_model_name=image_model_name,
        **kwargs)

    # if num_gpus > 1:
    #     model = keras.utils.multi_gpu_model(model, num_gpus)
    # else:
    #     model = model

    model.load_weights(load_weights)

    model.compile(
        optimizer=optimizer_name,
        loss=loss,
        metrics=metrics)

    return model


def choose_preprocessing_mode(preprocessing_mode, image_model_name):
    """ Choose preprocessing for specific pretrained weights
    it is very important to preprocess
    in exactly the same way the model
    was originally trained
    """
    if preprocessing_mode is None:
        if 'densenet' in image_model_name:
            preprocessing_mode = 'torch'
        elif 'nasnet' in image_model_name:
            preprocessing_mode = 'tf'
        elif 'vgg' in image_model_name:
            preprocessing_mode = 'caffe'
        elif 'inception_resnet' in image_model_name:
            preprocessing_mode = 'tf'
        elif 'resnet' in image_model_name:
            preprocessing_mode = 'caffe'
        else:
            raise ValueError('You need to explicitly set the preprocessing mode to '
                             'torch, tf, or caffe for these weights')
    return preprocessing_mode


def choose_optimizer(optimizer_name, learning_rate, callbacks, monitor_loss_name, train_steps, learning_rate_schedule=None):
    if learning_rate_schedule is None:
        learning_rate_schedule = FLAGS.learning_rate_schedule

    if optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate * 1.0)
        print('sgd initialized with learning rate: ' + str(learning_rate) + ' this might be overridden by callbacks later.')
    elif optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam()
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop()
    else:
        raise ValueError('Unsupported optimizer ' + str(optimizer_name) +
                         'try adam, sgd, or rmsprop.')

    if ((optimizer_name == 'sgd' or optimizer_name == 'rmsprop') and
            learning_rate_schedule is not None and learning_rate_schedule and learning_rate_schedule != 'none'):
        if learning_rate_schedule == 'reduce_lr_on_plateau':
            callbacks = callbacks + [
                # Reduce the learning rate if training plateaus.
                # patience of 13 is a good option for the cornell datasets and costar stack regression
                keras.callbacks.ReduceLROnPlateau(patience=13, verbose=1, factor=0.5, monitor=monitor_loss_name, min_delta=1e-6)
            ]
        else:
            callbacks = callbacks + [
                # In this case the max learning rate is double the specified one,
                # so that the average initial learning rate is as specified.
                keras_contrib.callbacks.CyclicLR(
                    step_size=train_steps * 8, base_lr=1e-6, max_lr=learning_rate * 2,
                    mode=learning_rate_schedule, gamma=0.99998)
            ]
    return callbacks, optimizer


def train_k_fold(split_type=None,
                 tfrecord_filename_base=None, csv_path='-k-fold-stat.csv',
                 log_dir=None, run_name=None, num_validation=None,
                 num_train=None, num_test=None, **kwargs):
    """ Do K_Fold training

    Please be aware that the number of train and validation
    steps changes every time the dataset is converted.
    These values are automatically loaded from a csv file,
    but be certain you do not mix the csv files up or
    overwrite the datasets and csv files separately.


    # Important Note

    Do not change this function without special care because we have
    files saved from past k-fold runs that use specific strings which
    have been written out to json files, and thus will remain the way they are... forever.

    # Arguments

        split_type: str, either 'imagewise' or 'objectwise', should be consistent with
        splits type desired when doing actual splits.
    """
    if log_dir is None:
        log_dir = FLAGS.log_dir
    if run_name is None:
        run_name = FLAGS.run_name
    if split_type is None:
        split_type = FLAGS.split_dataset
    if tfrecord_filename_base is None:
        tfrecord_filename_base = FLAGS.tfrecord_filename_base
    if num_validation is None:
        num_validation = FLAGS.num_validation
    if num_test is None:
        num_test = FLAGS.num_test
    if num_train is None:
        num_train = FLAGS.num_train

    if num_test != 0:
        raise ValueError('k_fold training does not support test data. '
                         'Check the command line flags and set --num_test 0')
    # We will extract the number of training steps from the csv files
    cur_csv_path = os.path.join(FLAGS.data_dir, tfrecord_filename_base + '-' + split_type + csv_path)
    csv_reader = csv.DictReader(open(cur_csv_path, mode='r'))
    unique_image_num = []
    num_splits = None
    for row in csv_reader:
        if num_splits is None:
            num_splits = int(row[' num_splits'])
        unique_image_num.append(int(row[' num_total_grasp']))  # file writer repeat each image for num_grasp

    num_fold, divides_evenly = np.divmod(num_splits, num_validation)
    # If this fails you need to fix the number of folds splits the dataset splits evenly!
    if divides_evenly != 0:
        raise ValueError('You must ensure the num_validation flag divides evenly '
                         'with the number of tfrecord files, aka splits. '
                         'Currently %s files divided by --num_validation %s results'
                         ' in %s folds and but this '
                         'leaves a remainder of %s.' % (str(num_validation),
                                                        str(num_fold),
                                                        str(num_splits),
                                                        str(divides_evenly)))

    val_filenames = []
    train_filenames = []
    train_id = ''
    val_size = 0
    train_size = 0
    kfold_run_name = grasp_utilities.timeStamped(run_name + '-' + split_type + '-kfold')
    log_dir = os.path.join(log_dir, kfold_run_name)
    kfold_param_dicts = {'num_fold': num_fold, 'num_splits': num_splits, 'fold_size': num_train}
    kfold_run_train_param_list = []
    fold_name_list = []
    # create the directory we will log to
    grasp_utilities.mkdir_p(log_dir)

    # 2k files, but k folds, so read two file at a time
    progbar_folds = tqdm(range(num_fold), desc='Preparing kfold')
    for i in progbar_folds:
        # This is a special string,
        # make sure to maintain backwards compatibility
        # if you modify it.
        fold_name = 'fold-' + str(i)
        fold_name_list += [fold_name]

        val_filenames = []
        val_sizes = []
        val_id = ''
        for k in range(num_validation):
            current_file_index = num_validation * i + k
            val_id += str(current_file_index)
            val_filenames += [os.path.join(FLAGS.data_dir, tfrecord_filename_base + '-' + split_type + '-fold-' + str(current_file_index) + '.tfrecord')]
            val_sizes += [unique_image_num[current_file_index]]
        val_size = sum(val_sizes)

        train_filenames = []
        train_sizes = []
        train_id = ''
        for j in range(num_fold):
            if j == i:
                continue
            for k in range(num_validation):
                current_file_index = num_validation * j + k
                train_id += str(current_file_index)
                train_filenames += [os.path.join(FLAGS.data_dir, tfrecord_filename_base + '-' + split_type + '-fold-' + str(current_file_index) + '.tfrecord')]
                train_sizes += [unique_image_num[current_file_index]]
        train_size = sum(train_sizes)

        save_splits_weights = run_name + '-' + fold_name + '-' + split_type + '-train-on-' + train_id + '-val-on-' + val_id
        progbar_folds.write('Preparing fold ' + str(i) + ' train dataset splits: ' + train_id + ',   val dataset splits: ' + val_id)
        training_run_params = dict(
            train_filenames=train_filenames, val_filenames=val_filenames, pipeline='train_val',
            train_size=train_size, val_size=val_size,
            log_dir=log_dir, run_name=save_splits_weights,
            **kwargs)
        kfold_param_dicts[fold_name + '-val-ids'] = val_id
        kfold_param_dicts[fold_name + '-train-ids'] = train_id
        kfold_param_dicts[fold_name] = training_run_params
        kfold_run_train_param_list += [training_run_params]
        train_id = ''
        train_size = 0

    # save all folds to disk so we can recover exactly what happened
    json_params_path = os.path.join(log_dir, kfold_run_name + '_params.json')
    with open(json_params_path, 'w') as fp:
        # save out all kfold params so they can be reloaded in the future
        json.dump(kfold_param_dicts, fp)

    json_histories_path = os.path.join(log_dir, kfold_run_name + '_histories.json')
    run_histories = {}
    history_dicts = {}
    progbar_fold_name_list = tqdm(fold_name_list, desc='Training k_fold')
    for i, (params, fold_name) in enumerate(zip(kfold_run_train_param_list, progbar_fold_name_list)):
        progbar_fold_name_list.write('\n------------------------------------------\n'
                                     'Training fold ' + str(i) + ' of ' + str(len(fold_name_list)) + '\n'
                                     '\n------------------------------------------\n')
        # this is a history object, which contains
        # a .history member and a .epochs member
        history = run_training(**params)
        run_histories[fold_name] = history
        history_dicts[fold_name] = history.history
        progbar_fold_name_list.update()
        # save the histories so far, overwriting past updates
        with open(json_histories_path, 'w') as fp:
            # save out all kfold params so they can be reloaded in the future
            json.dump(history_dicts, fp, cls=grasp_utilities.NumpyEncoder)

    # find the k-fold average and save it out to a json file
    # Warning: this file will massively underestimate scores for jaccard distance metrics!
    json_summary_path = os.path.join(log_dir, kfold_run_name + '_summary.json')
    grasp_utilities.multi_run_histories_summary(run_histories, json_summary_path)

    return run_histories


def choose_features_and_metrics(feature_combo_name, problem_name, image_shapes=None, loss=None):
    """ Choose the features to load from the dataset and losses to use during training
    """
    if image_shapes is None:
        image_shapes = [(FLAGS.resize_height, FLAGS.resize_width, 3)]

    # most cases are 0 to 1 outputs
    classes = 1
    # most cases consider both grasp successes and grasp failures
    success_only = False

    # Configure the input data dimensions
    # TODO(ahundt) get input dimensions automatically, based on configured params
    if feature_combo_name == 'image/preprocessed' or feature_combo_name == 'image_preprocessed':
        data_features = ['image/preprocessed']
        vector_shapes = None
    elif feature_combo_name == 'image_preprocessed_norm_sin2_cos2_width_3':
        # recommended for pixelwise classification and image center grasp regression
        data_features = ['image/preprocessed', 'preprocessed_norm_sin2_cos2_w_3']
        vector_shapes = [(3,)]
    elif feature_combo_name == 'image_preprocessed_norm_sin2_cos2_height_width_4':
        # recommended for pixelwise regression, don't use for classification!
        # An exception is ok if you are classifying the results of regression.
        # TODO(ahundt) add losses configured for pixelwise regression
        data_features = ['image/preprocessed', 'preprocessed_norm_sin2_cos2_height_width_4']
        vector_shapes = [(4,)]
    elif feature_combo_name == 'image_preprocessed_norm_sin2_cos2_w_yx_5':
        # recommended for classification of single predictions
        data_features = ['image/preprocessed', 'preprocessed_norm_sin2_cos2_w_yx_5']
        vector_shapes = [(5,)]
    elif feature_combo_name == 'image_preprocessed_norm_sin2_cos2_width_3':
        data_features = ['image/preprocessed', 'preprocessed_norm_sin2_cos2_width_3']
        vector_shapes = [(3,)]
    elif feature_combo_name == 'image_preprocessed_width_1':
        data_features = ['image/preprocessed', 'bbox/preprocessed/width']
        vector_shapes = [(1,)]
    elif feature_combo_name == 'preprocessed':
        data_features = ['image/preprocessed', 'bbox/preprocessed/cy_cx_normalized_2',
                         'bbox/preprocessed/sin_cos_2', 'bbox/preprocessed/logarithm_height_width_2']
        vector_shapes = [(2,), (2,), (2,)]
    elif feature_combo_name == 'raw':
        data_features = ['image/decoded', 'sin_cos_height_width_4']
        vector_shapes = [(4,)]
    else:
        raise ValueError('Selected feature ' + str(feature_combo_name) + ' does not exist. '
                         'feature selection options are image_preprocessed_sin_cos_height_width_4, '
                         'image_preprocessed_sin_cos_height_3, image_preprocessed_height_1,'
                         'preprocessed, and raw')

    # Configure the chosen problem type such as
    # classifying proposed grasps,
    # predicting a single grasp, or
    # predicting a grasp at each pixel.
    if problem_name == 'segmentation' or problem_name == 'grasp_segmentation' or problem_name == 'pixelwise_grasp_classification':
        label_features = ['grasp_success_yx_3']
        monitor_loss_name = 'segmentation_gaussian_binary_crossentropy'
        monitor_metric_name = 'val_segmentation_single_pixel_binary_accuracy'
        loss = grasp_loss.segmentation_gaussian_binary_crossentropy
        metrics = [grasp_loss.segmentation_single_pixel_binary_accuracy, grasp_loss.mean_pred]
        model_name = '_dilated_model'
    elif problem_name == 'classification' or problem_name == 'grasp_classification':
        label_features = ['grasp_success']
        monitor_metric_name = 'val_binary_accuracy'
        monitor_loss_name = 'val_loss'
        metrics = ['binary_accuracy', grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.binary_crossentropy
        model_name = '_classifier_model'
    elif problem_name == 'grasp_regression':
        # predicting a single grasp proposal
        success_only = True
        label_features = ['norm_sin2_cos2_hw_yx_6']
        monitor_metric_name = 'val_grasp_jaccard'
        monitor_loss_name = 'val_loss'
        if loss is None:
            loss = keras.losses.mean_squared_error
        metrics = [grasp_metrics.grasp_jaccard, keras.losses.mean_squared_error, grasp_loss.mean_pred, grasp_loss.mean_true]
        model_name = '_regression_model'
        classes = 6
    elif problem_name == 'image_center_grasp_regression':
        # predicting a single grasp proposal at the image center
        label_features = ['grasp_success_norm_sin2_cos2_hw_5']
        monitor_metric_name = 'grasp_jaccard'
        monitor_loss_name = 'val_loss'
        metrics = [grasp_metrics.grasp_jaccard, keras.losses.mean_squared_error, grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.mean_squared_error
        model_name = '_center_regression_model'
        classes = 5
    elif problem_name == 'pixelwise_grasp_regression':
        raise NotImplementedError
    elif problem_name == 'semantic_translation_regression':
        # only the translation component of semantic grasp regression
        # this is the regression case with the costar block stacking dataset
        # classes = 8
        classes = 3
        # TODO(ahundt) enable grasp_metrics.grasp_accuracy_xyz_aaxyz_nsc metric
        metrics = [grasp_metrics.grasp_acc, grasp_metrics.cart_error, 'mse', 'mae', grasp_metrics.grasp_acc_5mm_7_5deg,
                   grasp_metrics.grasp_acc_1cm_15deg, grasp_metrics.grasp_acc_2cm_30deg,
                   grasp_metrics.grasp_acc_4cm_60deg, grasp_metrics.grasp_acc_8cm_120deg,
                   grasp_metrics.grasp_acc_16cm_240deg, grasp_metrics.grasp_acc_32cm_360deg,
                   grasp_metrics.grasp_acc_256cm_360deg, grasp_metrics.grasp_acc_512cm_360deg]
        #  , grasp_loss.mean_pred, grasp_loss.mean_true]
        # monitor_metric_name = 'val_grasp_acc'
        monitor_metric_name = 'val_cart_error'
        # this is the length of the state vector defined in block_stacking_reader.py
        # label with translation and orientation
        # vector_shapes = [(49,)]
        vector_shapes = [(44,)]
        # data with translation and orientation
        # data_features = ['image/preprocessed', 'current_xyz_aaxyz_nsc_8']
        # translation only
        data_features = ['image/preprocessed', 'current_xyz_3']
        # label with translation and orientation
        # label_features = ['grasp_goal_xyz_aaxyz_nsc_8']
        # label with translation only
        label_features = ['grasp_goal_xyz_3']
        monitor_loss_name = 'val_loss'
        shape = (FLAGS.resize_height, FLAGS.resize_width, 3)
        image_shapes = [shape, shape]
        # loss = keras.losses.mean_absolute_error
        # loss = keras.losses.mean_squared_error
        loss = keras.losses.msle
        model_name = '_semantic_translation_regression_model'
    elif problem_name == 'semantic_rotation_regression':
        # only the rotation component of semantic grasp regression
        # this is the regression case with the costar block stacking dataset
        # classes = 8
        classes = 5
        # TODO(ahundt) enable grasp_metrics.grasp_accuracy_xyz_aaxyz_nsc metric
        metrics = [grasp_metrics.grasp_acc, grasp_metrics.angle_error, 'mse', 'mae', grasp_metrics.grasp_acc_5mm_7_5deg,
                   grasp_metrics.grasp_acc_1cm_15deg, grasp_metrics.grasp_acc_2cm_30deg,
                   grasp_metrics.grasp_acc_4cm_60deg, grasp_metrics.grasp_acc_8cm_120deg,
                   grasp_metrics.grasp_acc_16cm_240deg, grasp_metrics.grasp_acc_32cm_360deg,
                   grasp_metrics.grasp_acc_256cm_360deg, grasp_metrics.grasp_acc_512cm_360deg]
        #  , grasp_loss.mean_pred, grasp_loss.mean_true]
        # monitor_metric_name = 'val_grasp_acc'
        monitor_metric_name = 'val_angle_error'
        # this is the length of the state vector defined in block_stacking_reader.py
        # label with translation and orientation
        vector_shapes = [(49,)]
        # data with translation and orientation
        data_features = ['image/preprocessed', 'current_xyz_aaxyz_nsc_8']
        # label with translation and orientation
        # label_features = ['grasp_goal_xyz_aaxyz_nsc_8']
        label_features = ['grasp_goal_aaxyz_nsc_5']
        monitor_loss_name = 'val_loss'
        shape = (FLAGS.resize_height, FLAGS.resize_width, 3)
        image_shapes = [shape, shape]
        # loss = keras.losses.mean_absolute_error
        # loss = keras.losses.mean_squared_error
        loss = keras.losses.msle
        model_name = '_semantic_rotation_regression_model'
    elif problem_name == 'semantic_grasp_regression':
        # this is the regression case with the costar block stacking dataset
        classes = 8
        # TODO(ahundt) enable grasp_metrics.grasp_accuracy_xyz_aaxyz_nsc metric
        metrics = [grasp_metrics.grasp_acc, grasp_metrics.cart_error, grasp_metrics.angle_error, 'mse', 'mae', grasp_metrics.grasp_acc_5mm_7_5deg,
                   grasp_metrics.grasp_acc_1cm_15deg, grasp_metrics.grasp_acc_2cm_30deg,
                   grasp_metrics.grasp_acc_4cm_60deg, grasp_metrics.grasp_acc_8cm_120deg,
                   grasp_metrics.grasp_acc_16cm_240deg, grasp_metrics.grasp_acc_32cm_360deg,
                   grasp_metrics.grasp_acc_256cm_360deg, grasp_metrics.grasp_acc_512cm_360deg]
        #  , grasp_loss.mean_pred, grasp_loss.mean_true]
        monitor_metric_name = 'val_grasp_acc'
        # this is the length of the state vector defined in block_stacking_reader.py
        # label with translation and orientation
        vector_shapes = [(49,)]
        # data with translation and orientation
        data_features = ['image/preprocessed', 'current_xyz_aaxyz_nsc_8']
        # label with translation and orientation
        label_features = ['grasp_goal_xyz_aaxyz_nsc_8']
        monitor_loss_name = 'val_loss'
        shape = (FLAGS.resize_height, FLAGS.resize_width, 3)
        image_shapes = [shape, shape]
        # loss = keras.losses.mean_absolute_error
        # loss = keras.losses.mean_squared_error
        loss = keras.losses.msle
        model_name = '_semantic_grasp_regression_model'
    elif problem_name == 'semantic_grasp_classification':
        # this is the classification case with the costar block stacking dataset
        # TODO(ahundt) enable grasp_metrics.grasp_accuracy_xyz_aaxyz_nsc metric
        metrics = ['binary_accuracy', grasp_loss.mean_pred, grasp_loss.mean_true]
        monitor_metric_name = 'val_binary_accuracy'
        data_features = ['image/preprocessed', 'proposed_goal_xyz_aaxyz_nsc_8']
        label_features = ['grasp_success']
        # this is the length of the state vector defined in block_stacking_reader.py
        vector_shapes = [(57,)]
        # TODO(ahundt) should grasp_success be renamed action_success?
        monitor_loss_name = 'val_loss'
        shape = (FLAGS.resize_height, FLAGS.resize_width, 3)
        image_shapes = [shape, shape]
        loss = keras.losses.binary_crossentropy
        model_name = '_semantic_grasp_classification_model'
    else:
        raise ValueError('Selected problem_name ' + str(problem_name) + ' does not exist. '
                         'feature selection options are segmentation and classification, '
                         'image_center_grasp_regression, grasp_regression, grasp_classification'
                         'grasp_segmentation')

    if('classification' in problem_name and 'height' in feature_combo_name):
        print(
            """
            # WARNING: DO NOT use height as an input for classification tasks,
            # except to demonstrate the problems described below!
            #
            # The "height" parameter indicates the length of the graspable region,
            # which is highly correlated with the ground truth grasp_success
            # and would not be useful on a real robot.
            # Note that these parameters are used for classification
            # results in several previous papers but they also include
            # regression results and are thus OK overall.
            #
            # Instead, we suggest using image_preprocessed_sin2_cos2_width_3,
            # width is a proposed gripper openness which should be OK.
            """)

    return image_shapes, vector_shapes, data_features, model_name, monitor_loss_name, label_features, monitor_metric_name, loss, metrics, classes, success_only


def load_dataset(
        label_features=None, data_features=None,
        train_filenames=None, train_size=0,
        val_filenames=None, val_size=0,
        test_filenames=None, test_size=0,
        batch_size=None,
        val_batch_size=1, test_batch_size=1,
        train_data=None, validation_data=None, test_data=None,
        in_memory_validation=False,
        preprocessing_mode='tf', success_only=False,
        val_all_features=False, dataset_name='cornell_grasping'):
    """ Load the cornell grasping dataset from the file if it isn't already available.

    # Arguments

    success_only: only traverse successful grasps
    preprocessing_mode: 'tf', 'caffe', or 'torch' preprocessing.
        See keras/applications/imagenet_utils.py for details.
    val_all_features: Instead of getting the specific feature strings, the whole dictionary will be returned.

    """
    print('dataset name: ' + dataset_name)
    if dataset_name == 'cornell_grasping':

        if train_filenames is None and val_filenames is None and test_filenames is None:
            # train/val/test filenames are generated from the CSV file if they aren't provided
                train_filenames, train_size, val_filenames, val_size, test_filenames, test_size = load_dataset_sizes_from_csv()
        train_steps, val_steps, test_steps = steps_per_epoch(
            train_batch=batch_size, val_batch=val_batch_size, test_batch=test_batch_size,
            samples_train=train_size, samples_val=val_size, samples_test=test_size)

        if in_memory_validation:
            val_batch_size = val_size

        if validation_data is None and val_filenames is not None:
            if val_all_features:
                # Workaround for special evaluation call needed for jaccard regression.
                # All features will be returned in a dictionary in this mode
                val_label_features = None
                val_data_features = None
            else:
                val_label_features = label_features
                val_data_features = data_features
            validation_data = cornell_grasp_dataset_reader.yield_record(
                val_filenames, val_label_features, val_data_features,
                batch_size=val_batch_size,
                parse_example_proto_fn=parse_and_preprocess,
                preprocessing_mode=preprocessing_mode,
                apply_filter=success_only,
                is_training=False)

        if in_memory_validation:
            print('loading validation data directly into memory, if you run out set in_memory_validation to False')
            validation_data = next(validation_data)

        if train_data is None and train_filenames is not None:
            train_data = cornell_grasp_dataset_reader.yield_record(
                train_filenames, label_features, data_features,
                batch_size=batch_size,
                parse_example_proto_fn=parse_and_preprocess,
                preprocessing_mode=preprocessing_mode,
                apply_filter=success_only,
                is_training=True)

        if test_data is None and test_filenames is not None:
            test_data = cornell_grasp_dataset_reader.yield_record(
                test_filenames, label_features, data_features,
                batch_size=test_batch_size,
                parse_example_proto_fn=parse_and_preprocess,
                preprocessing_mode=preprocessing_mode,
                apply_filter=success_only,
                is_training=False)

                # val_filenames, batch_size=1, is_training=False,
                # shuffle=False, steps=1,
    elif dataset_name == 'costar_block_stacking':
        if FLAGS.costar_filename_base is None or not FLAGS.costar_filename_base:
            # Generate a new train/test/val split
            if 'cornell' in FLAGS.data_dir:
                # If the user hasn't specified a dir and it is the cornell default,
                # switch to the costar block stacking dataset default
                if 'grasp_success' in label_features or 'action_success' in label_features:
                    # classification case
                    FLAGS.data_dir = '~/.keras/datasets/costar_block_stacking_dataset_v0.3/*.h5f'
                else:
                    # regression case
                    FLAGS.data_dir = '~/.keras/datasets/costar_block_stacking_dataset_v0.3/*success.h5f'
                print('cornell_grasp_train.py: Overriding FLAGS.data_dir with: ' + FLAGS.data_dir)
            # temporarily hardcoded initialization
            # file_names = glob.glob(os.path.expanduser("~/JHU/LAB/Projects/costar_block_stacking_dataset_v0.3/*success.h5f"))
            file_names = glob.glob(os.path.expanduser(FLAGS.data_dir))
            np.random.seed(0)
            print("------------------------------------------------")
            np.random.shuffle(file_names)
            val_test_size = 128
            # TODO(ahundt) actually reach all the images in one epoch, modify CostarBlockStackingSequence
            # videos are at 10hz and there are about 25 seconds of video in each:
            # estimated_time_steps_per_example = 250
            # TODO(ahundt) remove/parameterize lowered number of images visited per example (done temporarily for hyperopt):
            # Only visit 5 images in val/test datasets so it doesn't take an unreasonable amount of time & for historical reasons.

            test_data = file_names[:val_test_size]
            with open('test.txt', mode='w') as myfile:
                myfile.write('\n'.join(test_data))
            validation_data = file_names[val_test_size:val_test_size*2]
            with open('val.txt', mode='w') as myfile:
                myfile.write('\n'.join(validation_data))
            train_data = file_names[val_test_size*2:]
            with open('train.txt', mode='w') as myfile:
                myfile.write('\n'.join(train_data))
        else:
            if 'cornell' in FLAGS.data_dir:
                # If the user hasn't specified a dir and it is the cornell default,
                # switch to the costar block stacking dataset default
                FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.3/')
            # TODO(ahundt) make the data dir user configurable again for costar_block stacking
            FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.3/')
            data_dir = FLAGS.data_dir
            costar_filename_base = FLAGS.costar_filename_base

            test_data_filename = os.path.join(data_dir, costar_filename_base + '_test_files.txt')
            test_data = np.genfromtxt(test_data_filename, dtype='str', delimiter=', ')

            validation_data_filename = os.path.join(data_dir, costar_filename_base + '_val_files.txt')
            validation_data = np.genfromtxt(validation_data_filename, dtype='str', delimiter=', ')

            train_data_filename = os.path.join(data_dir, costar_filename_base + '_train_files.txt')
            train_data = np.genfromtxt(train_data_filename, dtype='str', delimiter=', ')

        # We are multiplying by batch size as a hacky workaround because we want the sizing reduction
        # from steps_per_epoch to not be affected by the batch size.
        estimated_time_steps_per_example = 8 * batch_size
        # train_data = file_names[:5]
        # test_data = file_names[5:10]
        # validation_data = file_names[10:15]
        # print(train_data)
        # TODO(ahundt) use cornell & google dataset data augmentation / preprocessing for block stacking.
        random_augmentation = FLAGS.random_augmentation
        if random_augmentation == 0.0:
            random_augmentation = None

        output_shape = (FLAGS.resize_height, FLAGS.resize_width, 3)
        train_data = CostarBlockStackingSequence(
            train_data, batch_size=batch_size, is_training=True, shuffle=True, output_shape=output_shape,
            data_features_to_extract=data_features, label_features_to_extract=label_features,
            estimated_time_steps_per_example=estimated_time_steps_per_example,
            random_augmentation=random_augmentation)
        validation_data = CostarBlockStackingSequence(
            validation_data, batch_size=batch_size, is_training=False, output_shape=output_shape,
            data_features_to_extract=data_features, label_features_to_extract=label_features,
            estimated_time_steps_per_example=estimated_time_steps_per_example)
        test_data = CostarBlockStackingSequence(
            test_data, batch_size=batch_size, is_training=False, output_shape=output_shape,
            data_features_to_extract=data_features, label_features_to_extract=label_features,
            estimated_time_steps_per_example=estimated_time_steps_per_example)
        train_size = len(train_data) * train_data.get_estimated_time_steps_per_example()
        val_size = len(validation_data) * validation_data.get_estimated_time_steps_per_example()
        test_size = len(test_data) * test_data.get_estimated_time_steps_per_example()
        print('check 1 - train size: ' + str(train_size) + ' val_size: ' + str(val_size) + ' test size: ' + str(test_size))
        # train_data = block_stacking_generator(train_data)
        # test_data = block_stacking_generator(test_data)
        # validation_data = block_stacking_generator(validation_data)
        # validation_data = None
        # train_size = 5

        train_steps, val_steps, test_steps = steps_per_epoch(
            train_batch=batch_size, val_batch=batch_size, test_batch=batch_size,
            samples_train=train_size, samples_val=val_size, samples_test=test_size)

        print('check 1.5 - train_steps: ' + str(train_steps) + ' val_steps: ' + str(val_steps) + ' test_steps: ' + str(test_steps))
        # print("--------", train_steps, val_steps, test_steps)
        # enqueuer = OrderedEnqueuer(
        #             train_data,
        #             use_multiprocessing=False,
        #             shuffle=True)
        # enqueuer.start(workers=1, max_queue_size=1)
        # generator = iter(enqueuer.get())
        # print("-------------------")
        # generator_ouput = next(generator)
        # print("-------------------op")
        # x,y = generator_ouput
        # print(x.shape)
        # print(y.shape)
        # exit()

        # val_steps = None
    else:
        raise ValueError('Unsupported dataset_name ' + str(dataset_name) +
                         ' Options are: cornell_grasping and costar_block_stacking')
    return train_data, train_steps, validation_data, val_steps, test_data, test_steps


def model_predict_k_fold(
        kfold_params=None,
        verbose=0,
        data_features=None,
        prediction_name='norm_sin2_cos2_hw_yx_6',
        metric_name='grasp_jaccard',
        unique_score_category='image/filename',
        metric_fn=grasp_metrics.grasp_jaccard_batch):
    """ Load past runs and make predictions with the model and data.

    Currently only supports evaluating jaccard scores.

    # Arguments

        model: compiled model instance.
        input_data: generator instance.
        kfold_params: a path to a json file containing parameters from a previous k_fold cross validation run



    # Important Note

    Do not change this function without special care because we have
    files saved from past k-fold runs that use specific strings which
    have been written out to json files, and thus will remain the way they are... forever.

    """

    print(""" Predicting on kfold results.
    Double check that you've actually got the best model checkpoint.
    We currently take the checkpoint based on the highest val_acc score
    in the .h5 filename
    """)
    if data_features is None:
        data_features = ['image/preprocessed']
    kfold_params_dir = None

    if kfold_params is not None and isinstance(kfold_params, str):
        kfold_params_dir = os.path.dirname(kfold_params)
        with open(kfold_params, mode='r') as kfold_json_file:
            kfold_param_dicts = json.load(kfold_json_file)
    else:
        raise ValueError('kfold_params can only be a path to a json file generated by train_k_fold() at this time.')

    num_fold = kfold_param_dicts['num_fold']

    # print('WEIGHTS: \n', model.get_weights())

    # TODO(ahundt) low priority: automatically choose feature and metric strings
    # choose_features_and_metrics(feature_combo_name, problem_name)

    metric_fold_averages = np.zeros((num_fold))
    loss_fold_averages = np.zeros((num_fold))
    with tqdm(range(num_fold), desc='kfold prediction', ncols=240) as progbar_folds:
        for i in progbar_folds:
            # This is a special string,
            # make sure to maintain backwards compatibility
            # if you modify it. See train_k_fold().
            fold_name = 'fold-' + str(i)

            # load all the settings from a past run
            training_run_params = kfold_param_dicts[fold_name]

            val_filenames = training_run_params['val_filenames']
            log_dir = training_run_params['log_dir']
            # we prefix every fold with a timestamp, therefore we assume:
            #   lexicographic order == time order == fold order
            # '200_epoch_real_run' is for backwards compatibility before
            # the fold nums were put into each fold's log_dir and run_name.
            directory_listing = os.listdir(log_dir)
            fold_log_dir = []
            for name in directory_listing:
                name = os.path.join(log_dir, name)
                if os.path.isdir(name):
                    if '200_epoch_real_run' in name or fold_name in name:
                        fold_log_dir += [name]

            if len(fold_log_dir) > 1:
                # more backwards compatibility tricks
                fold_log_dir = fold_log_dir[i]
            else:
                # this should work in most cases excluding the first k_fold run log
                [fold_log_dir] = fold_log_dir

            # Now we have to load the best model
            # '200_epoch_real_run' is for backwards compatibility before
            # the fold nums were put into each fold's log_dir and run_name.
            fold_checkpoint_file = grasp_utilities.find_best_weights(fold_log_dir, fold_name, verbose, progbar_folds)

            progbar_folds.write('Fold ' + str(i) + ' Loading checkpoint: ' + str(fold_checkpoint_file))

            # load the model
            model = get_compiled_model(load_weights=fold_checkpoint_file, **training_run_params)

            # TODO(ahundt) low-medium priority: save iou scores
            # metric_name = 'intersection_over_union'
            if 'preprocessing_mode' in training_run_params:
                preprocessing_mode = training_run_params['preprocessing_mode']
            else:
                preprocessing_mode = 'tf'

            # go over every data entry
            result = evaluate(
                model, val_filenames=val_filenames, data_features=data_features,
                prediction_name=prediction_name, metric_fn=metric_fn,
                progbar_folds=progbar_folds, unique_score_category=unique_score_category,
                metric_name=metric_name, should_initialize=True, load_weights=fold_checkpoint_file,
                fold_num=i, fold_name=fold_name)
            progbar_folds.update()

            # [(metric_name, fold_average), (loss_name, loss_average)]
            metric_fold_averages[i] = result[0][1]
            metric_name = result[0][0]
            if len(result) > 1:
                loss_fold_averages[i] = loss_average[1][1]
                loss_name = result[0][0]
            # TODO(ahundt) low-medium priority: save out all best scores and averages

        metric_overall_average = np.average(metric_fold_averages)
        loss_overall_average = np.average(loss_fold_averages)
        final_result = ('---------------------------------------------\n'
                        '    overall average metric ' + str(metric_name) + ' score for all folds: ' + str(metric_overall_average) +
                        '    averages for each fold: ' + str(metric_fold_averages))

        if len(result) > 1:
            final_result += ('    overall average loss ' + str(loss_name) + ' score for all folds: ' + str(loss_overall_average) +
                             '    averages for each fold: ' + str(loss_fold_averages))

        final_result += '\n---------------------------------------------\n'
        progbar_folds.write(final_result)
        with open(os.path.join(log_dir, 'summary_results.txt'), 'w') as summary_results:
            summary_results.write(final_result)


def evaluate(
        model, example_generator=None, val_filenames=None, data_features=None, prediction_name='norm_sin2_cos2_hw_yx_6',
        metric_fn=grasp_metrics.grasp_jaccard_batch,
        progbar_folds=sys.stdout, unique_score_category='image/filename', metric_name='grasp_jaccard',
        steps=None, visualize=False,
        preprocessing_mode='tf', apply_filter=True, loss_fn=None, loss_name='loss',
        should_initialize=False, load_weights=None, fold_num=None, fold_name='', verbose=0):
    """ Evaluate how well a model performs at grasp regression.

        This is specialized for running grasp regression right now,
        so check the defaults if you want to use it for something else.
    """
    if data_features is None:
        data_features = ['image/preprocessed']

    if example_generator is not None:
        input_data = example_generator
    elif val_filenames is not None:
        # Load the validation data and traverse it exactly once
        input_data = cornell_grasp_dataset_reader.yield_record(
            val_filenames, batch_size=1, is_training=False,
            shuffle=False, steps=1, apply_filter=apply_filter,
            preprocessing_mode=preprocessing_mode)
    else:
        raise ValueError('Must provide example generator or val_filenames.')

    if load_weights is not None:
            model.load_weights(load_weights)

    losses = []
    # go over every data entry
    best_results_for_each_image = {}
    sess = keras.backend.get_session()

    try:
        for i, example_dict in enumerate(tqdm(input_data, desc='Evaluating', total=steps)):
            # sess = K.get_session()
            # init_g = tf.global_variables_initializer()
            # init_l = tf.local_variables_initializer()
            # sess.run(init_g)
            # sess.run(init_l)
            # TODO(ahundt) Do insane hack which resets the session & reloads weights for now... will fix later
            if should_initialize:
                    # tensorflow setup to make sure all variables are initialized
                    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                    sess.run(init_op)
                    should_initialize = False

                    if load_weights is not None:
                            model.load_weights(load_weights)

            # todo get
            predict_input = [example_dict[feature_name] for feature_name in data_features]
            ground_truth = example_dict[prediction_name]
            if visualize:
                import matplotlib
                matplotlib.pyplot.imshow((np.squeeze(predict_input) / 2.0) + 0.5)
            result = model.predict_on_batch(predict_input)
            if verbose > 0:
                progbar_folds.write('\nground_truth: ' + str(ground_truth))
                progbar_folds.write('\nresult: ' + str(result))

            score = metric_fn(ground_truth, result)
            if loss_fn is not None:
                loss = loss_fn(ground_truth, result)
                losses += [loss]

            if verbose > 0:
                progbar_folds.write('\nscore: ' + str(score))

            image_filename = example_dict[unique_score_category][0]

            # TODO(ahundt) make this a flag
            if visualize:
                # TODO(ahundt) account for other setups
                if len(np.squeeze(result)) == 1:
                    predictions = None
                else:
                    predictions = np.squeeze(result)
                viz_filename = None
                if load_weights is not None:
                    # remove .h5 add the image number, and save the file
                    viz_filename = load_weights[:-3] + '_' + str(i) + '.jpg'
                grasp_visualization.visualize_redundant_images_example(example_dict, predictions=predictions, save_filename=viz_filename, show=False)

            score = np.squeeze(score)
            # save this score and prediction if there is no score yet
            # or this score is a new best
            if(len(best_results_for_each_image) == 0 or
                    image_filename not in best_results_for_each_image or
                    prediction_name not in best_results_for_each_image[image_filename] or
                    score > best_results_for_each_image[image_filename][metric_name]):
                # We have a new best score!
                best_score = {
                    prediction_name: result,
                    metric_name: score}
                if image_filename in best_results_for_each_image:
                    # old_best = best_results_for_each_image.pop(image_filename)
                    if verbose > 0:
                        progbar_folds.write('\nreplacing score ' +
                                            str(best_results_for_each_image[image_filename][metric_name]) +
                                            ' \nwith score ' + str(score) + ' \nin ' + image_filename + '\n')
                best_results_for_each_image[image_filename] = best_score
    except tf.errors.OutOfRangeError as e:
        # finished going through the dataset once
        pass

    best_scores = np.zeros([len(best_results_for_each_image)])
    for j, (filename, best_score) in enumerate(six.iteritems(best_results_for_each_image)):
        best_scores[j] = best_score[metric_name]
        # TODO(ahundt) low priority: calculate other stats like stddev?
        continue

    # it is a lot faster to do the losses at the end,
    # if it takes too much memory, move it.
    losses = np.array(sess.run(losses))
    loss_average = np.average(losses)
    fold_average = np.average(best_scores)

    result = [(metric_name, fold_average)]
    progbar_folds.write('---------------------------------------------')
    progbar_folds.write('Completed fold ' + str(fold_num) + ' name ' + str(fold_name) +
                        ' with average ' + str(metric_name) + ' metric score: ' + str(fold_average))
    if loss_fn is not None:
        progbar_folds.write(' average loss: ' + str(loss_average))
        result += [(loss_name, loss_average)]
    progbar_folds.write('---------------------------------------------')
    return result


def load_dataset_sizes_from_csv(
        train_splits=None, val_splits=None, test_splits=None, split_type=None,
        csv_path='-k-fold-stat.csv', data_dir=None, tfrecord_filename_base=None):
    """ Load csv specifying the number of steps to train, validate, and test each fold during k-fold training.

    The csv files are created by cornell_grasp_dataset_writer.py.
    Please be aware that the number of train and validation steps changes every time the dataset is converted.
    These values are automatically loaded from a csv file, but be certain you do not mix the csv files up or
    overwrite the datasets and csv files separately.
    """
    if data_dir is None:
        data_dir = FLAGS.data_dir

    if split_type is None:
        split_type = FLAGS.split_dataset

    if tfrecord_filename_base is None:
        tfrecord_filename_base = FLAGS.tfrecord_filename_base
    # must be sure that train_splits + val_splits + test_filenames = flags.num_splits
    cur_csv_path = os.path.join(data_dir, tfrecord_filename_base + '-' + split_type + csv_path)
    with open(cur_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        unique_image_num = []
        for row in csv_reader:
            unique_image_num.append(int(row[' num_total_grasp']))

        if train_splits is None:
            train_splits = FLAGS.num_train
        if val_splits is None:
            val_splits = FLAGS.num_validation
        if test_splits is None:
            test_splits = FLAGS.num_test

        train_filenames = []
        val_filenames = []
        test_filenames = []

        train_size = 0
        val_size = 0
        test_size = 0
        for i in range(train_splits):
            train_size += unique_image_num[i]
            train_filenames += [os.path.join(
                                    data_dir,
                                    tfrecord_filename_base + '-' + split_type + '-fold-' + str(i) + '.tfrecord')]

        for i in range(train_splits, train_splits + val_splits):
            val_size += unique_image_num[i]
            val_filenames += [os.path.join(
                                    data_dir,
                                    tfrecord_filename_base + '-' + split_type + '-fold-' + str(i) + '.tfrecord')]

        for i in range(train_splits + val_splits, train_splits + val_splits + test_splits):
            test_size += unique_image_num[i]
            test_filenames += [os.path.join(
                                    data_dir,
                                    tfrecord_filename_base + '-' + split_type + '-fold-' + str(i) + '.tfrecord')]

        return train_filenames, train_size, val_filenames, val_size, test_filenames, test_size


def steps_per_epoch(train_batch=None, samples_train=None,
                    val_batch=None, samples_val=None,
                    test_batch=None, samples_test=None):
    """Determine the number of steps per epoch for a given number of samples and batch size.

    Also ensures val and test divides evenly for reproducible results.
    """

    returns = []
    steps_train = None
    steps_val = None
    steps_test = None
    if samples_train is not None and train_batch is not None:
        # for training, just do a little more than once through the dataset if needed
        steps_train = int(np.ceil(float(samples_train) / float(train_batch)))
    if samples_val is not None and val_batch is not None:
        steps_in_val_dataset, divides_evenly = np.divmod(samples_val, val_batch)
        if divides_evenly != 0:
            raise ValueError('You need to fix the validation batch size ' + str(val_batch) +
                             ' so it divides the number of samples ' + str(samples_val) + ' evenly. '
                             'In a worst case you can simply choose a batch size of 1.')
        steps_val = steps_in_val_dataset
    if samples_test is not None and test_batch is not None:
        steps_in_test_dataset, divides_evenly = np.divmod(samples_test, test_batch)
        if divides_evenly != 0:
            raise ValueError('You need to fix the test batch size ' + str(val_batch) +
                             ' so it divides the number of samples ' + str(samples_val) + ' evenly. '
                             'In a worst case you can simply choose a batch size of 1.')
        steps_test = steps_in_test_dataset

    return steps_train, steps_val, steps_test


def main(_):

    tf.enable_eager_execution()
    hyperparams = grasp_utilities.load_hyperparams_json(
        FLAGS.load_hyperparams, FLAGS.fine_tuning, FLAGS.fine_tuning_learning_rate)
    if 'k_fold' in FLAGS.pipeline_stage:
        train_k_fold(hyperparams=hyperparams, **hyperparams)
    else:
        run_training(hyperparams=hyperparams, **hyperparams)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
