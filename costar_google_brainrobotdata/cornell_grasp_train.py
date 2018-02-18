#!/usr/local/bin/python
'''
Training a network on cornell grasping dataset for detecting grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code Based on:
    https://github.com/tnikolla/robot-grasp-detection

'''
import os
import errno
import sys
import json
import argparse
import os.path
import glob
import datetime
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
import cornell_grasp_dataset_reader
import time
from tensorflow.python.platform import flags

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
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from grasp_model import concat_images_with_tiled_vector_layer
from grasp_model import top_block
from grasp_model import create_tree_roots
from grasp_model import hypertree_model
from cornell_grasp_dataset_reader import parse_and_preprocess
# https://github.com/aurora95/Keras-FCN
# TODO(ahundt) move keras_fcn directly into this repository, into keras-contrib, or make a proper installer
import keras_contrib.applications.fully_convolutional_networks as fcn
import keras_contrib.applications.densenet as densenet
import keras_tqdm

import grasp_loss as grasp_loss


flags.DEFINE_float(
    'learning_rate',
    0.00584,
    'Initial learning rate.'
)
flags.DEFINE_integer(
    'epochs',
    5,
    'Number of epochs to run trainer.'
)
flags.DEFINE_integer(
    'batch_size',
    8,
    'Batch size.'
)
flags.DEFINE_string(
    'log_dir',
    './logs_cornell/',
    'Directory for tensorboard, model layout, model weight, csv, and hyperparam files'
)
flags.DEFINE_string(
    'model_path',
    '/tmp/tf/model.ckpt',
    'Variables for the model.'
)
flags.DEFINE_string(
    'train_or_validation',
    'validation',
    'Train or evaluate the dataset'
)
flags.DEFINE_string(
    'run_name',
    '',
    'A string that will become part of the logged directories and filenames.'
)

FLAGS = flags.FLAGS

# TODO(ahundt) put these utility functions in utils somewhere


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command

    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def choose_hypertree_model(
        images=None, vectors=None,
        image_shapes=None, vector_shapes=None,
        dropout_rate=0.25,
        vector_dense_filters=64,
        dilation_rate=2,
        activation='sigmoid',
        final_pooling=None,
        include_top=True,
        top='classification',
        top_block_filters=64,
        classes=1,
        output_shape=None,
        trainable=False,
        verbose=0,
        image_model_name='vgg',
        vector_model_name='dense_block',
        trunk_layers=5,
        trunk_filters=None,
        vector_branch_num_layers=5):
    """ Construct a variety of possible models with a tree shape based on hyperparameters.

    # Arguments

        dropout_rate: a dropout rate of None will disable dropout
        top_block_filters: the number of filters for the two final fully connected layers,
            before a prediction is made based on the number of classes.

    # Notes

    Best 1 epoch run with only gripper openness parameter, 2018-02-17:
        - val_binary_accuracy 0.9134199238
        - val_loss 0.2269693456

        {"vector_dense_filters": 64, "vector_branch_num_layers": 2, "trainable": true,
         "image_model_name": "vgg", "vector_model_name": "dense_block", "learning_rate": 0.005838979061490798,
         "trunk_filters": 128, "dropout_rate": 0.0, "top_block_filters": 64, "trunk_layers": 4, "feature_combo_name":
         "image_preprocessed_height_1"}
    Current best 1 epoch run as of 2018-02-16:
        - note there is a bit of ambiguity so until I know I'll have case 0 and case 1.
            - two models were in that run and didn't have hyperparam records yet.
            - The real result is probably case 1, since the files are saved each run,
              so the data will be for the latest run.
        - 2018-02-15-22-00-12_-vgg_dense_model-dataset_cornell_grasping-grasp_success2018-02-15-22-00-12_-vgg_dense_model-dataset_cornell_grasping-grasp_success
        - input
            - height_width_sin_cos_4
        - vgg16 model
        - val_binary_accuracy
            - 0.9202226425
        - lr
            - 0.06953994
        - vector dense layers
            - 4 in case 0 with 64 channels
            - 1 in case 1 with 64 channels
        - dense block trunk case 1
            - 5 conv blocks
            - growth rate 48
            - 576 input channels
            - 816 output channels
        - dense layers before fc1, case 1
            - 64 output channels

    """
    if trainable is None:
        trainable = False

    # TODO(ahundt) deal with model names being too long due to https://github.com/keras-team/keras/issues/5253
    if top == 'segmentation':
        name_prefix = 'dilated_'
    else:
        name_prefix = 'single_'
        dilation_rate = 1
    with K.name_scope(name_prefix + 'hypertree') as scope:
        # VGG16 weights are shared and not trainable
        if top == 'segmentation':
            image_model = fcn.AtrousFCN_Vgg16_16s(
                input_shape=image_shapes[0], include_top=False,
                classes=classes, upsample=False)
        else:
            if image_model_name == 'vgg':
                image_model = keras.applications.vgg16.VGG16(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes)
            elif image_model_name == 'nasnet_large':
                image_model = keras.applications.nasnet.NASNetLarge(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes
                )
            elif image_model_name == 'nasnet_mobile':
                image_model = keras.applications.nasnet.NASNetMobile(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes
                )
            elif image_model_name == 'resnet':
                # resnet model is special because we need to
                # skip the average pooling part.
                resnet_model = keras.applications.resnet50.ResNet50(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes)
                if not trainable:
                    for layer in resnet_model.layers:
                        layer.trainable = False
                # get the layer before the global average pooling
                image_model = resnet_model.layers[-2]
            elif image_model_name == 'densenet':
                image_model = keras.applications.densenet.DenseNet169(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes)
            else:
                raise ValueError('Unsupported image_model_name')

        if not trainable and getattr(image_model, 'layers', None) is not None:
            for layer in image_model.layers:
                layer.trainable = False

        def create_image_model(tensor):
            """ Image classifier weights are shared.
            """
            return image_model(tensor)

        def vector_branch_dense(
                tensor, vector_dense_filters=vector_dense_filters,
                num_layers=vector_branch_num_layers,
                model_name=vector_model_name):
            """ Vector branches that simply contain a single dense layer.
            """
            x = tensor
            # create the chosen layers starting with the vector input
            # accepting num_layers == 0 is done so hyperparam search is simpler
            if num_layers is None or num_layers == 0:
                return x
            elif model_name == 'dense':
                for i in range(num_layers):
                    x = Dense(vector_dense_filters)(x)
            elif model_name == 'dense_block':
                keras.backend.expand_dims
                densenet.__dense_block(
                    x, nb_layers=num_layers,
                    nb_filter=vector_dense_filters,
                    growth_rate=48, dropout_rate=dropout_rate,
                    dims=0)
            return x

        def create_tree_trunk(tensor, filters=trunk_filters, num_layers=trunk_layers):
            x = tensor
            if filters is None:
                channels = K.int_shape(tensor)[-1]
            else:
                channels = filters

            # create the chosen layers starting with the combined image and vector input
            # accepting num_layers == 0 is done so hyperparam search is simpler
            if num_layers is None or num_layers == 0:
                return x
            elif num_layers is not None:
                x, num_filters = densenet.__dense_block(
                    x, nb_layers=trunk_layers, nb_filter=channels,
                    growth_rate=48, dropout_rate=dropout_rate)

            return x

        model = hypertree_model(
            images=images, vectors=vectors,
            image_shapes=image_shapes, vector_shapes=vector_shapes,
            dropout_rate=dropout_rate,
            create_image_tree_roots_fn=create_image_model,
            create_vector_tree_roots_fn=vector_branch_dense,
            create_tree_trunk_fn=create_tree_trunk,
            activation=activation,
            final_pooling=final_pooling,
            include_top=include_top,
            top=top,
            top_block_filters=top_block_filters,
            classes=classes,
            output_shape=output_shape,
            verbose=verbose
        )
    return model


class PrintLogsCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('\nlogs:', logs)


def run_training(
        learning_rate=None,
        batch_size=None,
        num_gpus=1,
        top='classification',
        epochs=None,
        preprocessing_mode=None,
        save_model=True,
        train_file=None,
        validation_file=None,
        train_data=None,
        validation_data=None,
        feature_combo_name='image_preprocessed_sin_cos_height_3',
        image_model_name='vgg',
        log_dir=None,
        hyperparams=None,
        **kwargs):
    """

    top: options are 'segmentation' and 'classification'.
    hyperparams: a dictionary of hyperparameter selections made for this training run.
       If provided these values will simply be dumped to a file and not utilized in any other way.
    """
    if epochs is None:
        epochs = FLAGS.epochs
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if train_file is None:
        train_file = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
    if validation_file is None:
        validation_file = os.path.join(FLAGS.data_dir, FLAGS.evaluate_filename)
    if learning_rate is None:
        learning_rate = FLAGS.learning_rate
    if log_dir is None:
        log_dir = FLAGS.log_dir

    [image_shapes, vector_shapes, data_features, model_name,
     monitor_loss_name, label_features, monitor_metric_name,
     loss, metrics] = choose_features_and_metrics(feature_combo_name, top)

    # see parse_and_preprocess() for the creation of these features
    model_name = image_model_name + model_name

    # If loading pretrained weights
    # it is very important to preprocess
    # in exactly the same way the model
    # was originally trained
    if preprocessing_mode is None:
        if 'densenet' in image_model_name:
            preprocessing_mode = 'torch'
        elif 'nasnet' in image_model_name:
            preprocessing_mode = 'tf'
        elif 'vgg' in image_model_name or 'resnet' in image_model_name:
            preprocessing_mode = 'caffe'
        else:
            raise ValueError('You need to explicitly set the preprocessing mode to '
                             'torch, tf, or caffe for these weights')

    # choose hypertree_model with inputs [image], [sin_theta, cos_theta]
    model = choose_hypertree_model(
        image_shapes=image_shapes,
        vector_shapes=vector_shapes,
        top=top,
        image_model_name=image_model_name,
        **kwargs)

    print(monitor_loss_name)
    # TODO(ahundt) add a loss that changes size with how open the gripper is
    # loss = grasp_loss.segmentation_gaussian_measurement

    save_weights = ''
    dataset_names_str = 'cornell_grasping'
    run_name = timeStamped(save_weights + '-' + model_name + '-dataset_' + dataset_names_str + '-' + label_features[0])
    callbacks = []

    optimizer = keras.optimizers.SGD(learning_rate * 1.0)
    callbacks = callbacks + [
        # Reduce the learning rate if training plateaus.
        keras.callbacks.ReduceLROnPlateau(patience=12, verbose=1, factor=0.5, monitor=monitor_loss_name)
    ]

    log_dir = os.path.join(log_dir, run_name)
    log_dir_run_name = os.path.join(log_dir, run_name)
    csv_logger = CSVLogger(log_dir_run_name + run_name + '.csv')
    callbacks = callbacks + [csv_logger]
    callbacks += [PrintLogsCallback()]
    print('Writing logs for models, accuracy and tensorboard in ' + log_dir)
    mkdir_p(log_dir)

    # Save the hyperparams to a json string so it is human readable
    if hyperparams is not None:
        with open(log_dir_run_name + run_name + '_hyperparams.json', 'w') as fp:
            json.dump(hyperparams, fp)

    # Save the current model to a json string so it is human readable
    with open(log_dir_run_name + run_name + '_model.json', 'w') as fp:
        fp.write(model.to_json())

    checkpoint = keras.callbacks.ModelCheckpoint(log_dir_run_name + run_name + '-epoch-{epoch:03d}-' +
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
    callbacks += [keras_tqdm.TQDMCallback()]

    # TODO(ahundt) WARNING: THE NUMBER OF TRAIN/VAL STEPS VARIES EVERY TIME THE DATASET IS CONVERTED, AUTOMATE SETTING THOSE NUMBERS
    samples_in_val_dataset, steps_per_epoch_train, steps_in_val_dataset, val_batch_size = epoch_params(batch_size)

    if num_gpus > 1:
        parallel_model = keras.utils.multi_gpu_model(model, num_gpus)
    else:
        parallel_model = model

    parallel_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    # i = 0
    # for batch in tqdm(
    #     cornell_grasp_dataset_reader.yield_record(
    #         train_file, label_features, data_features,
    #         batch_size=batch_size,
    #         parse_example_proto_fn=choose_parse_example_proto_fn()))):
    #     i += 1

    # Get the validation dataset in one big numpy array for validation
    # This lets us take advantage of tensorboard visualization
    train_data, validation_data = load_dataset(
        validation_file, label_features, data_features,
        samples_in_val_dataset, train_file, batch_size,
        val_batch_size, train_data=train_data, validation_data=validation_data)

    # print('calling model.fit_generator()')
    history = parallel_model.fit_generator(
        train_data,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=steps_in_val_dataset,
        callbacks=callbacks,
        verbose=0)

    model.save_weights(log_dir_run_name + run_name + '_model.h5')
    return history


def choose_features_and_metrics(feature_combo_name, top):
    """ Choose the features to load from the dataset and losses to use during training
    """
    # TODO(ahundt) get input dimensions automatically, based on configured params
    if feature_combo_name == 'image_preprocessed_sin_cos_height_width_4':
        data_features = ['image/preprocessed', 'sin_cos_height_width_4']
        image_shapes = [(FLAGS.resize_height, FLAGS.resize_width, 3)]
        vector_shapes = [(4,)]
    elif feature_combo_name == 'image_preprocessed_sin_cos_height_3':
        data_features = ['image/preprocessed', 'sin_cos_height_3']
        image_shapes = [(FLAGS.resize_height, FLAGS.resize_width, 3)]
        vector_shapes = [(3,)]
    elif feature_combo_name == 'image_preprocessed_height_1':
        data_features = ['image/preprocessed', 'bbox/height']
        image_shapes = [(FLAGS.resize_height, FLAGS.resize_width, 3)]
        vector_shapes = [(1,)]
    elif feature_combo_name == 'preprocessed':
        data_features = ['image/preprocessed', 'bbox/preprocessed/cy_cx_normalized_2',
                         'bbox/preprocessed/sin_cos_2', 'bbox/preprocessed/logarithm_height_width_2']
        image_shapes = [(FLAGS.resize_height, FLAGS.resize_width, 3)]
        vector_shapes = [(2,), (2,), (2,)]
    elif feature_combo_name == 'raw':
        data_features = ['image/decoded', 'sin_cos_height_width_4']
        image_shapes = [(FLAGS.sensor_image_height, FLAGS.sensor_image_width, 3)]
        vector_shapes = [(4,)]
    else:
        raise ValueError('Selected feature ' + str(feature_combo_name) + ' does not exist. '
                         'feature selection options are image_preprocessed_sin_cos_height_width_4, '
                         'image_preprocessed_sin_cos_height_3, image_preprocessed_height_1,'
                         'preprocessed, and raw')

    if top == 'segmentation':
        label_features = ['grasp_success_yx_3']
        monitor_loss_name = 'segmentation_gaussian_binary_crossentropy'
        monitor_metric_name = 'val_segmentation_single_pixel_binary_accuracy'
        loss = grasp_loss.segmentation_gaussian_binary_crossentropy
        metrics = [grasp_loss.segmentation_single_pixel_binary_accuracy, grasp_loss.mean_pred]
        model_name = '_dilated_model'
    elif top == 'classification':
        label_features = ['grasp_success']
        monitor_metric_name = 'val_binary_accuracy'
        monitor_loss_name = 'val_loss'
        metrics = ['binary_accuracy', grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.binary_crossentropy
        model_name = '_dense_model'
    else:
        raise ValueError('Selected top ' + str(top) + ' does not exist. '
                         'feature selection options are segmentation and classification')
    return image_shapes, vector_shapes, data_features, model_name, monitor_loss_name, label_features, monitor_metric_name, loss, metrics


def load_dataset(validation_file, label_features, data_features, samples_in_val_dataset, train_file, batch_size, val_batch_size,
                 train_data=None, validation_data=None, in_memory_validation=False):
    """ Load the cornell grasping dataset from the file if it isn't already available.
    """

    if in_memory_validation:
        val_batch_size = samples_in_val_dataset

    if validation_data is None:
        validation_data = cornell_grasp_dataset_reader.yield_record(
            validation_file, label_features, data_features,
            is_training=False, batch_size=val_batch_size,
            parse_example_proto_fn=parse_and_preprocess)

    if in_memory_validation:
        print('loading validation data directly into memory, if you run out set in_memory_validation to False')
        validation_data = next(validation_data)

    if train_data is None:
        train_data = cornell_grasp_dataset_reader.yield_record(
            train_file, label_features, data_features,
            batch_size=batch_size,
            parse_example_proto_fn=parse_and_preprocess)
    return train_data, validation_data


def epoch_params(batch_size):
    """ Determine the number of steps to train and validate

    TODO(ahundt) WARNING: THE NUMBER OF TRAIN/VAL STEPS VARIES EVERY TIME THE DATASET IS CONVERTED, AUTOMATE SETTING THOSE NUMBERS
    """
    samples_in_training_dataset = 6402
    samples_in_val_dataset = 1617
    val_batch_size = 11
    steps_in_val_dataset, divides_evenly = np.divmod(samples_in_val_dataset, val_batch_size)
    assert divides_evenly == 0
    steps_per_epoch_train = np.ceil(float(samples_in_training_dataset) / float(batch_size))
    return samples_in_val_dataset, steps_per_epoch_train, steps_in_val_dataset, val_batch_size


def bboxes_to_grasps(bboxes):
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    box = tf.unstack(bboxes, axis=1)
    x = (box[0] + (box[4] - box[0])/2) * 0.35
    y = (box[1] + (box[5] - box[1])/2) * 0.47
    tan = (box[3] -box[1]) / (box[2] -box[0]) *0.47/0.35
    h = tf.sqrt(tf.pow((box[2] -box[0])*0.35, 2) + tf.pow((box[3] -box[1])*0.47, 2))
    w = tf.sqrt(tf.pow((box[6] -box[0])*0.35, 2) + tf.pow((box[7] -box[1])*0.47, 2))
    return x, y, tan, h, w


def grasp_to_bbox(x, y, tan, h, w):
    theta = tf.atan(tan)
    edge1 = (x -w/2*tf.cos(theta) +h/2*tf.sin(theta), y -w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge2 = (x +w/2*tf.cos(theta) +h/2*tf.sin(theta), y +w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge3 = (x +w/2*tf.cos(theta) -h/2*tf.sin(theta), y +w/2*tf.sin(theta) +h/2*tf.cos(theta))
    edge4 = (x -w/2*tf.cos(theta) -h/2*tf.sin(theta), y -w/2*tf.sin(theta) +h/2*tf.cos(theta))
    return [edge1, edge2, edge3, edge4]


def old_run_training():
    print(FLAGS.train_or_validation)
    if FLAGS.train_or_validation == 'train':
        print('distorted_inputs')
        data_files_ = TRAIN_FILE
        features = cornell_grasp_dataset_reader.distorted_inputs(
                  [data_files_], FLAGS.epochs, batch_size=FLAGS.batch_size)
    else:
        print('inputs')
        data_files_ = VALIDATE_FILE
        features = cornell_grasp_dataset_reader.inputs([data_files_])

    # loss, x_hat, tan_hat, h_hat, w_hat, y_hat = old_loss(tan, x, y, h, w)
    train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = keras.backend.get_session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #save/restore model
    d={}
    l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2']
    for i in l:
        d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    dg={}
    lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in lg:
        dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    saver = tf.train.Saver(d)
    saver_g = tf.train.Saver(dg)
    #saver.restore(sess, "/root/grasp/grasp-detection/models/imagenet/m2/m2.ckpt")
    saver_g.restore(sess, FLAGS.model_path)
    try:
        count = 0
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train
            if FLAGS.train_or_validation == 'train':
                _, loss_value, x_value, x_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, x, x_hat, tan, tan_hat, h, h_hat, w, w_hat])
                duration = time.time() - start_batch
                if step % 100 == 0:
                    print('Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n')%(step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration)
                if step % 1000 == 0:
                    saver_g.save(sess, FLAGS.model_path)
            else:
                bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat)
                bbox_value, bbox_model, tan_value, tan_model = sess.run([bboxes, bbox_hat, tan, tan_hat])
                bbox_value = np.reshape(bbox_value, -1)
                bbox_value = [(bbox_value[0]*0.35,bbox_value[1]*0.47),(bbox_value[2]*0.35,bbox_value[3]*0.47),(bbox_value[4]*0.35,bbox_value[5]*0.47),(bbox_value[6]*0.35,bbox_value[7]*0.47)]
                p1 = Polygon(bbox_value)
                p2 = Polygon(bbox_model)
                iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                angle_diff = np.abs(np.arctan(tan_model)*180/np.pi -np.arctan(tan_value)*180/np.pi)
                duration = time.time() -start_batch
                if angle_diff < 30. and iou >= 0.25:
                    count+=1
                    print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
            step +=1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.epochs, step, (time.time()-start_time)/60))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def old_loss(tan, x, y, h, w):
    from grasp_inf import inference
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference(images), axis=1) # list
    # tangent of 85 degree is 11
    tan_hat_confined = tf.minimum(11., tf.maximum(-11., tan_hat))
    tan_confined = tf.minimum(11., tf.maximum(-11., tan))
    # Loss function
    gamma = tf.constant(10.)
    loss = tf.reduce_sum(tf.pow(x_hat -x, 2) +tf.pow(y_hat -y, 2) + gamma*tf.pow(tan_hat_confined - tan_confined, 2) +tf.pow(h_hat -h, 2) +tf.pow(w_hat -w, 2))
    return loss, x_hat, tan_hat, h_hat, w_hat, y_hat


def main(_):
    run_training()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
