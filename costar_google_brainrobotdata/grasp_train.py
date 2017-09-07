import os
import datetime
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from tensorflow.python.platform import flags

import grasp_dataset
import grasp_model

flags.DEFINE_string('learning_rate_decay_algorithm', 'power_decay',
                    """Determines the algorithm by which learning rate decays,
                       options are power_decay, exp_decay, adam and progressive_drops""")
flags.DEFINE_string('grasp_model', 'grasp_model_single',
                    """Choose the model definition to run, options are grasp_model and grasp_model_segmentation""")
flags.DEFINE_string('save_weights', 'grasp_model_weights',
                    """Save a file with the trained model weights.""")
flags.DEFINE_string('load_weights', 'grasp_model_weights.h5',
                    """Load and continue training the specified file containing model weights.""")
flags.DEFINE_integer('epochs', 20,
                     """Epochs of training""")
flags.DEFINE_string('grasp_dataset_eval', '097',
                    """Filter the subset of 1TB Grasp datasets to evaluate.
                    None by default. 'all' will run all datasets in data_dir.
                    '052' and '057' will download the small starter datasets.
                    '102' will download the main dataset with 102 features,
                    around 110 GB and 38k grasp attempts.
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_string('pipeline_stage', 'train_eval',
                    """Choose to "train", "eval", or "train_eval" with the grasp_dataset
                       data for training and grasp_dataset_eval for evaluation.""")
flags.DEFINE_float('learning_rate_scheduler_power_decay_rate', 0.9,
                   """Determines how fast the learning rate drops at each epoch.
                      lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)""")
flags.DEFINE_float('grasp_learning_rate', 0.1,
                   """Determines the initial learning rate""")
flags.DEFINE_integer('eval_batch_size', 1, 'batch size per compute device')
flags.DEFINE_integer('densenet_growth_rate', 12,
                     """DenseNet and DenseNetFCN parameter growth rate""")
flags.DEFINE_integer('densenet_dense_blocks', 4,
                     """The number of dense blocks in the model.""")
flags.DEFINE_float('densenet_reduction', 0.5,
                   """DenseNet and DenseNetFCN reduction aka compression parameter.""")
flags.DEFINE_float('densenet_reduction_after_pretrained', 0.5,
                   """DenseNet and DenseNetFCN reduction aka compression parameter,
                      applied to the second DenseNet component after pretrained imagenet models.""")
flags.DEFINE_float('dropout_rate', 0.2,
                   """Dropout rate for the model during training.""")
flags.DEFINE_string('eval_results_file', 'grasp_model_eval.txt',
                    """Save a file with results of model evaluation.""")
flags.DEFINE_string('device', '/gpu:0',
                    """Save a file with results of model evaluation.""")
flags.DEFINE_bool('tf_allow_memory_growth', True,
                  """False if memory usage will be allocated all in advance
                     or True if it should grow as needed. Allocating all in
                     advance may reduce fragmentation.""")

flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class GraspTrain(object):

    def train(self, dataset=FLAGS.grasp_dataset,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              load_weights=FLAGS.load_weights,
              save_weights=FLAGS.save_weights,
              make_model_fn=grasp_model.grasp_model,
              imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction,
              grasp_sequence_max_time_steps=FLAGS.grasp_sequence_max_time_steps,
              random_crop=FLAGS.random_crop,
              resize=FLAGS.resize,
              resize_height=FLAGS.resize_height,
              resize_width=FLAGS.resize_width,
              learning_rate_decay_algorithm=FLAGS.learning_rate_decay_algorithm,
              learning_rate=FLAGS.grasp_learning_rate,
              learning_power_decay_rate=FLAGS.learning_rate_scheduler_power_decay_rate,
              dropout_rate=FLAGS.dropout_rate,
              model_name=FLAGS.grasp_model):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_steps: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # list of dictionaries the length of batch_size
        (pregrasp_op_batch, grasp_step_op_batch,
         simplified_grasp_command_op_batch,
         example_batch_size,
         grasp_success_op_batch,
         num_samples) = data.single_pose_training_tensors(batch_size=batch_size,
                                                          imagenet_mean_subtraction=imagenet_mean_subtraction,
                                                          random_crop=random_crop,
                                                          resize=resize,
                                                          grasp_sequence_max_time_steps=grasp_sequence_max_time_steps)

        if resize:
            input_image_shape = [resize_height, resize_width, 3]
        else:
            input_image_shape = [512, 640, 3]

        ########################################################
        # End tensor configuration, begin model configuration and training

        weights_name = timeStamped(save_weights + '-' + model_name)

        # ###############learning rate scheduler####################
        # source: https://github.com/aurora95/Keras-FCN/blob/master/train.py
        def lr_scheduler(epoch, learning_rate=learning_rate,
                         mode=learning_rate_decay_algorithm,
                         epochs=epochs,
                         learning_power_decay_rate=learning_power_decay_rate):
            '''if lr_dict.has_key(epoch):
                lr = lr_dict[epoch]
                print 'lr: %f' % lr'''

            if mode is 'power_decay':
                # original lr scheduler
                lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)
            if mode is 'exp_decay':
                # exponential decay
                lr = (float(learning_rate) ** float(learning_power_decay_rate)) ** float(epoch+1)
            # adam default lr
            if mode is 'adam':
                lr = 0.001

            if mode is 'progressive_drops':
                # drops as progression proceeds, good for sgd
                if epoch > 0.9 * epochs:
                    lr = 0.0001
                elif epoch > 0.75 * epochs:
                    lr = 0.001
                elif epoch > 0.5 * epochs:
                    lr = 0.01
                else:
                    lr = 0.1

            print('lr: %f' % lr)
            return lr
        scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
        early_stopper = EarlyStopping(monitor='acc', min_delta=0.001, patience=10)
        csv_logger = CSVLogger(weights_name + '.csv')
        checkpoint = keras.callbacks.ModelCheckpoint(weights_name + '-epoch-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}.h5',
                                                     save_best_only=True, verbose=1, monitor='acc')

        callbacks = [scheduler, early_stopper, csv_logger, checkpoint]

        # Will need to try more things later.
        # Nadam parameter choice reference:
        # https://github.com/tensorflow/tensorflow/pull/9175#issuecomment-295395355

        # 2017-08-28 afternoon trying NADAM with higher learning rate
        # optimizer = keras.optimizers.Nadam(lr=0.03, beta_1=0.825, beta_2=0.99685)

        # 2017-08-28 trying SGD
        optimizer = keras.optimizers.SGD(lr=learning_rate)

        # 2017-08-27
        # Tried NADAM for a while with the settings below, only improved for first 2 epochs.
        # optimizer = keras.optimizers.Nadam(lr=0.004, beta_1=0.825, beta_2=0.99685)

        # create the model
        model = make_model_fn(
            pregrasp_op_batch,
            grasp_step_op_batch,
            simplified_grasp_command_op_batch,
            input_image_shape=input_image_shape,
            dropout_rate=dropout_rate)

        if(load_weights):
            if os.path.isfile(load_weights):
                model.load_weights(load_weights)
            else:
                print('Could not load weights {}, '
                      'the file does not exist, '
                      'starting fresh....'.format(load_weights))

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[grasp_success_op_batch])

        model.summary()

        # make sure we visit every image once
        steps_per_epoch = int(np.ceil(float(num_samples)/float(batch_size)))

        try:
            model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        finally:
            # always try to save weights
            final_weights_name = weights_name + '-final.h5'
            model.save_weights(final_weights_name)
            return final_weights_name

    def eval(self, dataset=FLAGS.grasp_dataset_eval,
             batch_size=FLAGS.eval_batch_size,
             load_weights=FLAGS.load_weights,
             save_weights=FLAGS.save_weights,
             make_model_fn=grasp_model.grasp_model,
             imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction,
             grasp_sequence_max_time_steps=FLAGS.grasp_sequence_max_time_steps,
             resize=FLAGS.resize,
             resize_height=FLAGS.resize_height,
             resize_width=FLAGS.resize_width,
             eval_results_file=FLAGS.eval_results_file,
             model_name=FLAGS.grasp_model):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_steps: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # list of dictionaries the length of batch_size
        (pregrasp_op_batch, grasp_step_op_batch,
         simplified_grasp_command_op_batch,
         example_batch_size,
         grasp_success_op_batch,
         num_samples) = data.single_pose_training_tensors(batch_size=batch_size,
                                                          imagenet_mean_subtraction=imagenet_mean_subtraction,
                                                          random_crop=False,
                                                          resize=resize,
                                                          grasp_sequence_max_time_steps=grasp_sequence_max_time_steps)

        if resize:
            input_image_shape = [resize_height, resize_width, 3]
        else:
            input_image_shape = [512, 640, 3]

        ########################################################
        # End tensor configuration, begin model configuration and training
        csv_logger = CSVLogger(load_weights + '_eval.csv')

        # create the model
        model = make_model_fn(
            pregrasp_op_batch,
            grasp_step_op_batch,
            simplified_grasp_command_op_batch,
            input_image_shape=input_image_shape,
            dropout_rate=0.0)

        if(load_weights):
            if os.path.isfile(load_weights):
                model.load_weights(load_weights)
            else:
                raise ValueError('Could not load weights {}, '
                                 'the file does not exist.'.format(load_weights))

        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[grasp_success_op_batch])

        model.summary()

        steps = float(num_samples)/float(batch_size)

        if not steps.is_integer():
            raise ValueError('The number of samples was not exactly divisible by the batch size!'
                             'For correct, reproducible evaluation your number of samples must be exactly'
                             'divisible by the batch size. Right now the batch size cannot be changed for'
                             'the last sample, so in a worst case choose a batch size of 1. Not ideal,'
                             'but manageable. num_samples: {} batch_size: {}'.format(num_samples, batch_size))

        try:
            loss, acc = model.evaluate(None, None, steps=int(steps))
            results_str = '\nevaluation results loss: ' + str(loss) + ' accuracy: ' + str(acc) + ' dataset: ' + dataset
            print(results_str)
            weights_name_str = load_weights + '_evaluation_dataset_{}_loss_{:.3f}_acc_{:.3f}'.format(dataset, loss, acc)
            weights_name_str = weights_name_str.replace('.h5', '') + '.h5'
            with open(eval_results_file, 'w') as results_file:
                results_file.write(results_str + '\n')
            if save_weights:
                model.save_weights(weights_name_str)
                print('\n saved weights with evaluation result to ' + weights_name_str)

        except KeyboardInterrupt, e:
            print('Evaluation canceled at user request, '
                  'any results are incomplete for this run.')


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    with K.get_session() as sess:
        load_weights = FLAGS.load_weights
        if FLAGS.grasp_model == 'grasp_model_pretrained':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_pretrained(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction_after_pretrained,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_single':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_segmentation':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_segmentation(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    *a, **kw)
        else:
            raise ValueError('unknown model selected: {}'.format(FLAGS.grasp_model))

        gt = GraspTrain()

        if 'train' in FLAGS.pipeline_stage:
            print('Training ' + FLAGS.grasp_model)
            load_weights = gt.train(make_model_fn=make_model_fn,
                                    load_weights=load_weights,
                                    model_name=FLAGS.grasp_model)
        if 'eval' in FLAGS.pipeline_stage:
            print('Evaluating ' + FLAGS.grasp_model + ' on weights ' + load_weights)
            # evaluate using weights that were just computed, if available
            gt.eval(make_model_fn=make_model_fn,
                    load_weights=load_weights,
                    model_name=FLAGS.grasp_model)


if __name__ == '__main__':
    FLAGS._parse_flags()
    main()
