import os
import sys
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
from keras.callbacks import TensorBoard

from tensorflow.python.platform import flags

import grasp_dataset
import grasp_model

from keras_tqdm import TQDMCallback  # Keras tqdm progress bars https://github.com/bstriner/keras-tqdm

flags.DEFINE_string('learning_rate_decay_algorithm', 'power_decay',
                    """Determines the algorithm by which learning rate decays,
                       options are power_decay, exp_decay, adam and progressive_drops.
                       Only applies with optimizer flag is SGD""")
flags.DEFINE_string('grasp_model', 'grasp_model_single',
                    """Choose the model definition to run, options are grasp_model and grasp_model_segmentation""")
flags.DEFINE_string('save_weights', 'grasp_model_weights',
                    """Save a file with the trained model weights.""")
flags.DEFINE_string('load_weights', 'grasp_model_weights.h5',
                    """Load and continue training the specified file containing model weights.""")
flags.DEFINE_integer('epochs', 100,
                     """Epochs of training""")
flags.DEFINE_string('grasp_datasets_train', '062_b,063,072_a,082_b,102',
                    """Filter multiple subsets of 1TB Grasp datasets to train.
                    Comma separated list 062_b,063,072_a,082_b,102 by default,
                    totaling 513,491 grasp attempts.
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_string('grasp_datasets_batch_algorithm', 'proportional',
                    """Use default batch if constant,
                    'constant' training on multiple datasets reads `batch_size`
                        elements from each dataset at each training step when this parameter.
                    'proportional' each dataset's batch size will be individually
                        set to int(batch_size*single_batch/max_batch_size)
                        so smaller datasets are run more slowly than larger datasets.""")
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
flags.DEFINE_float('learning_rate_scheduler_power_decay_rate', 1.5,
                   """Determines how fast the learning rate drops at each epoch.
                      lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)
                      Training from scratch within an initial learning rate of 0.1 might find a
                         power decay value of 2 to be useful.
                      Fine tuning with an initial learning rate of 0.001 may consder 1.5 power decay.""")
flags.DEFINE_float('grasp_learning_rate', 0.001,
                   """Determines the initial learning rate""")
flags.DEFINE_integer('eval_batch_size', 1, 'batch size per compute device')
flags.DEFINE_integer('densenet_growth_rate', 12,
                     """DenseNet and DenseNetFCN parameter growth rate""")
flags.DEFINE_integer('densenet_depth', 40,
                     """DenseNet total number of layers, aka depth""")
flags.DEFINE_integer('densenet_dense_blocks', 3,
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
flags.DEFINE_string('learning_rate_scheduler', None,
                    """Options are None and learning_rate_scheduler,
                       turning this on activates the scheduler which follows
                       a power decay path for the learning rate over time.
                       This is most useful with SGD, currently disabled with Adam.""")
flags.DEFINE_string('optimizer', "Adam", "Options are Adam and SGD.")
flags.DEFINE_string('progress_tracker', 'tensorboard',
                    """Utility to follow training progress, options are tensorboard and None.""")

flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class GraspTrain(object):

    def train(self, dataset=FLAGS.grasp_datasets_train,
              grasp_datasets_batch_algorithm=FLAGS.grasp_datasets_batch_algorithm,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              load_weights=FLAGS.load_weights,
              save_weights=FLAGS.save_weights,
              make_model_fn=grasp_model.grasp_model,
              imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction,
              grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
              grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
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

            grasp_sequence_max_time_step: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
        """
        if (grasp_datasets_batch_algorithm != 'constant' and grasp_datasets_batch_algorithm != 'proportional'):
            raise ValueError('grasp_datasets_batch_algorithm string value must be either constant or proportional.')
        datasets = dataset.split(',')
        max_num_samples = 0
        grasp_datasets = []
        pregrasp_op_batch = []
        grasp_step_op_batch = []
        # simplified_network_grasp_command_op
        simplified_grasp_command_op_batch = []
        grasp_success_op_batch = []

        # Aggregate multiple datasets into training tensors
        # Note that one limitation of this setup is that we will
        # iterate over samples according to the largest dataset,
        # which means we see smaller datasets more than once in
        # a single epoch. Try not to aggregate a very small dataset
        # with a very large one!
        dataset_batch_sizes = []
        grasp_dataset_list = []
        for single_dataset in datasets:
            data = grasp_dataset.GraspDataset(dataset=single_dataset)
            grasp_dataset_list.append(data)
            dataset_batch_sizes.append(data.get_features()[1])

        max_batch_size = max(dataset_batch_sizes)
        # Not sure why any thing assigned to max_batch_size, it can pass
        for single_dataset, single_batch in zip(grasp_dataset_list, dataset_batch_sizes):
            proportional_batch_size = batch_size
            if(grasp_datasets_batch_algorithm == 'proportional'):
                proportional_batch_size = int(batch_size * single_batch / max_batch_size)
            data = single_dataset
            # list of dictionaries the length of batch_size
            (pregrasp_op, grasp_step_op,
             simplified_grasp_command_op,
             grasp_success_op,
             num_samples) = data.get_training_tensors(batch_size=proportional_batch_size,
                                                      imagenet_mean_subtraction=imagenet_mean_subtraction,
                                                      random_crop=random_crop,
                                                      resize=resize,
                                                      grasp_sequence_min_time_step=grasp_sequence_min_time_step,
                                                      grasp_sequence_max_time_step=grasp_sequence_max_time_step)
            max_num_samples = max(num_samples, max_num_samples)
            pregrasp_op_batch.append(pregrasp_op)
            grasp_step_op_batch.append(grasp_step_op)
            simplified_grasp_command_op_batch.append(simplified_grasp_command_op)
            grasp_success_op_batch.append(grasp_success_op)

        pregrasp_op_batch = tf.concat(pregrasp_op_batch, 0)
        grasp_step_op_batch = tf.concat(grasp_step_op_batch, 0)
        simplified_grasp_command_op_batch = tf.concat(simplified_grasp_command_op_batch, 0)
        print('grasp_success_op_batch before concat: ', grasp_success_op_batch)
        grasp_success_op_batch = tf.concat(grasp_success_op_batch, 0)
        print('grasp_success_op_batch after concat: ', grasp_success_op_batch)

        if resize:
            input_image_shape = [resize_height, resize_width, 3]
        else:
            input_image_shape = [512, 640, 3]

        ########################################################
        # End tensor configuration, begin model configuration and training

        weights_name = timeStamped(save_weights + '-' + model_name)

        # ###############learning rate scheduler####################
        # source: https://github.com/aurora95/Keras-FCN/blob/master/train.py
        # some quick lines to see what a power_decay schedule would do at each epoch:
        # import numpy as np
        # epochs = 100
        # learning_rate = 0.1
        # learning_power_decay_rate = 2
        # print([learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate) for epoch in np.arange(epochs)])

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

        callbacks = []
        scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
        early_stopper = EarlyStopping(monitor='acc', min_delta=0.001, patience=10)
        csv_logger = CSVLogger(weights_name + '.csv')
        checkpoint = keras.callbacks.ModelCheckpoint(weights_name + '-epoch-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}.h5',
                                                     save_best_only=True, verbose=1, monitor='acc')
        progress_bar = TQDMCallback()

        callbacks = callbacks + [early_stopper, csv_logger, checkpoint, progress_bar]

        if FLAGS.progress_tracker is 'tensorboard':
            progress_tracker = TensorBoard(log_dir='./' + weights_name, write_graph=True,
                                           write_grads=True, write_images=True,
                                           embeddings_freq=10, embeddings_layer_names=None, embeddings_metadata=None)
        # Will need to try more things later.
        # Nadam parameter choice reference:
        # https://github.com/tensorflow/tensorflow/pull/9175#issuecomment-295395355

        # 2017-08-28 afternoon trying NADAM with higher learning rate
        # optimizer = keras.optimizers.Nadam(lr=0.03, beta_1=0.825, beta_2=0.99685)

        # 2017-08-28 trying SGD
        # 2017-12-18 SGD worked very well and has been the primary training optimizer from 2017-09 to 2017-12
        if FLAGS.optimizer is 'SGD':
            optimizer = keras.optimizers.SGD(lr=learning_rate)
            callbacks = callbacks + [scheduler]

        # 2017-08-27 Tried NADAM for a while with the settings below, only improved for first 2 epochs.
        # optimizer = keras.optimizers.Nadam(lr=0.004, beta_1=0.825, beta_2=0.99685)

        # 2017-12-18
        # Try ADAM with AMSGrad
        if FLAGS.optimizer is 'Adam':
            optimizer = keras.optimizers.Adam(amsgrad=True)

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
        steps_per_epoch = int(np.ceil(float(max_num_samples)/float(batch_size)))

        try:
            model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
            final_weights_name = weights_name + '-final.h5'
            model.save_weights(final_weights_name)
        except (Exception, KeyboardInterrupt) as e:
            # always try to save weights
            final_weights_name = weights_name + '-autosaved-on-exception.h5'
            model.save_weights(final_weights_name)
            raise e
        return final_weights_name

    def eval(self, dataset=FLAGS.grasp_dataset_eval,
             batch_size=FLAGS.eval_batch_size,
             load_weights=FLAGS.load_weights,
             save_weights=FLAGS.save_weights,
             make_model_fn=grasp_model.grasp_model,
             imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction,
             grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
             grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
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

            grasp_sequence_max_time_step: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.

        # Returns

           weights_name_str or None if a new weights file was not saved.
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # list of dictionaries the length of batch_size
        (pregrasp_op_batch, grasp_step_op_batch,
         simplified_grasp_command_op_batch,
         grasp_success_op_batch,
         num_samples) = data.get_training_tensors(batch_size=batch_size,
                                                  imagenet_mean_subtraction=imagenet_mean_subtraction,
                                                  random_crop=False,
                                                  resize=resize,
                                                  grasp_sequence_min_time_step=grasp_sequence_min_time_step,
                                                  grasp_sequence_max_time_step=grasp_sequence_max_time_step)

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

        except KeyboardInterrupt as e:
            print('Evaluation canceled at user request, '
                  'any results are incomplete for this run.')
            return None

        return weights_name_str


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    with K.get_session() as sess:
        # Launch the training script for the particular model specified on the command line
        # or via the default flag value
        load_weights = FLAGS.load_weights
        if FLAGS.grasp_model == 'grasp_model_resnet':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_resnet(
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_pretrained':
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
                    depth=FLAGS.densenet_depth,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_segmentation':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_segmentation(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_levine_2016':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_levine_2016(
                    *a, **kw)
        else:
            available_functions = globals()
            if FLAGS.grasp_model in available_functions:
                make_model_fn = available_functions[FLAGS.grasp_model]
            else:
                raise ValueError('unknown model selected: {}'.format(FLAGS.grasp_model))

        gt = GraspTrain()

        # train the model
        if 'train' in FLAGS.pipeline_stage:
            print('Training ' + FLAGS.grasp_model)
            load_weights = gt.train(make_model_fn=make_model_fn,
                                    load_weights=load_weights,
                                    model_name=FLAGS.grasp_model)
        # evaluate the model
        if 'eval' in FLAGS.pipeline_stage:
            print('Evaluating ' + FLAGS.grasp_model + ' on weights ' + load_weights)
            # evaluate using weights that were just computed, if available
            gt.eval(make_model_fn=make_model_fn,
                    load_weights=load_weights,
                    model_name=FLAGS.grasp_model)
        return None


if __name__ == '__main__':
    FLAGS._parse_flags()
    main()
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
