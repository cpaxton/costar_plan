'''
Training a network on cornell grasping dataset for regression of grasping positions.
This will go through sorted hyperopt results from best to worst and train each model,
ranking the best final results.

In other words, this network tries to predict a grasp pose from an input image, current pose, and action vector.

Example run:

    export CUDA_VISIBLE_DEVICES="2" && python2 costar_block_stacking_train_ranked_regression.py --log_dir hyperopt_logs_costar_block_stacking_train_ranked_regression

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import os
import traceback
import tensorflow as tf
import pandas
import json
import keras
from tensorflow.python.platform import flags

import grasp_utilities
import cornell_grasp_train

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


flags.DEFINE_string(
    'rank_csv',
    'hyperopt_rank.csv',
    """Sorted csv ranking models on which to perform full runs after hyperparameter optimization.

    See cornell_hyperopt.py to perform hyperparameter optimization,
    and then hyperopt_rank.py to generate the ranking csv file.
    The file is expected to be in the directory specified by the log_dir flag.

    Example file path:
        hyperopt_logs_costar_grasp_regression/hyperopt_rank.csv
        hyperopt_logs_costar_translation_regression/hyperopt_rank.csv
        hyperopt_logs_costar_block_stacking_train_ranked_regression/hyperopt_rank.csv
    """
)

flags.DEFINE_boolean(
    'filter_epoch',
    False,
    'Filter results, dropping everything except a single specific epoch specified by --epoch'
)

flags.DEFINE_integer(
    'epoch',
    0,
    'Results should only belong to this epoch if --filter_epoch=True'
)

flags.DEFINE_integer(
    'max_epoch',
    40,
    'Results should only belong to this epoch or lower, not enabled by default.'
)

flags.DEFINE_integer(
    'skip_models',
    0,
    'number of models to skip before starting to train, useful for resuming past runs'
)


FLAGS = flags.FLAGS


def main(_):
    use_best_model = True
    # epoch to filter, or None if we should just take the best performing value ever
    filter_epoch = FLAGS.filter_epoch
    problem_type = 'semantic_translation_regression'
    # problem_type = 'semantic_rotation_regression'
    # problem_type = 'semantic_grasp_regression'
    feature_combo = 'image_preprocessed'
    # Override some default flags for this configuration
    # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
    FLAGS.problem_type = problem_type
    FLAGS.feature_combo = feature_combo
    FLAGS.crop_to = 'image_contains_grasp_box_center'
    load_weights = None
    FLAGS.batch_size = 32
    optimizer_name = 'sgd'
    # FLAGS.crop_height = 480
    # FLAGS.crop_width = 640
    # FLAGS.resize_height = 480
    # FLAGS.resize_width = 640
    # print('Note: special overrides have been applied '
    #       'for an experiment. '
    #       'crop + resize width/height have been set to 640x480.')
    # FLAGS.log_dir = r'C:/Users/Varun/JHU/LAB/Projects/costar_plan/costar_google_brainrobotdata/hyperparams/'
    # FLAGS.data_dir = r'C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.4/*success.h5f'
    skip_models = FLAGS.skip_models

    # We generate a summary of the best algorithms as the program runs,
    # so here we configure the summary metrics for the problem type.
    if problem_type == 'semantic_grasp_regression':
        histories_metrics = ['val_cart_error', 'val_angle_error', 'val_grasp_acc']
        histories_summary_metrics = ['min', 'min', 'max']
    elif problem_type == 'semantic_rotation_regression':
        histories_metrics = ['val_angle_error', 'val_grasp_acc']
        histories_summary_metrics = ['min', 'max']
    elif problem_type == 'semantic_translation_regression':
        histories_metrics = ['val_cart_error', 'val_grasp_acc']
        histories_summary_metrics = ['min', 'max']
    else:
        raise ValueError(
            'costar_block_stacking_train_ranked_regression.py::main(): '
            'unsupported problem_type ' + str(problem_type))

    # FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.4/*success.h5f')
    FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.4/')
    FLAGS.fine_tuning_epochs = 0
    # final training run:
    # FLAGS.epochs = 600
    FLAGS.epochs = 200
    # FLAGS.random_augmentation = 0.25
    # evaluating top few models run:
    # FLAGS.epochs = 10
    # FLAGS.epochs = 40
    FLAGS.random_augmentation = None
    print('Regression Training on costar block stacking is about to begin. '
          'It overrides some command line parameters including '
          'training on mae loss so to change them '
          'you will need to modify costar_block_stacking_train_ranked_regression.py directly.')

    dataset_name = 'costar_block_stacking'

    json_histories_path = os.path.join(FLAGS.log_dir, FLAGS.run_name + '_' + problem_type + '_histories.json')
    json_histories_summary_path = os.path.join(FLAGS.log_dir, FLAGS.run_name + '_' + problem_type + '_histories_summary.json')
    run_histories = {}
    history_dicts = {}
    sort_by = None

    csv_file = os.path.join(os.path.expanduser(FLAGS.log_dir), FLAGS.rank_csv)

    # load the hyperparameter optimization ranking csv file created by hyperopt_rank.py
    dataframe = pandas.read_csv(csv_file, index_col=None, header=0)
    if problem_type == 'semantic_rotation_regression':
        # sort by val_angle_error from low to high
        dataframe = dataframe.sort_values('val_angle_error', ascending=True)
        dataframe = dataframe.sort_values('val_grasp_acc', ascending=False)
        sort_by = 'val_grasp_acc'
        # DISABLE RANDOM AUGMENTATION FOR ROTATION
        FLAGS.random_augmentation = None
    elif problem_type == 'semantic_translation_regression':
        # sort by cart_error from low to high
        # sort_by = 'cart_error'
        # dataframe = dataframe.sort_values(sort_by, ascending=True)
        # # sort by val_cart_error from low to high
        sort_by = 'val_cart_error'
        dataframe = dataframe.sort_values(sort_by, ascending=True)
        # # sort by grasp accuracy within 4 cm and 60 degrees
        # sort_by = 'val_grasp_acc_4cm_60deg'
        # dataframe = dataframe.sort_values(sort_by, ascending=False)
        # sort_by = 'val_grasp_acc'
        # dataframe = dataframe.sort_values(sort_by, ascending=False)
    elif problem_type == 'semantic_grasp_regression':
        dataframe = dataframe.sort_values('val_grasp_acc', ascending=False)
        sort_by = 'val_grasp_acc'
    else:
        raise ValueError('costar_block_stacking_train_ranked_regression.py: '
                         'unsupported problem type: ' + str(problem_type))

    # don't give really long runs an unfair advantage
    if FLAGS.max_epoch is not None:
        dataframe = dataframe.loc[dataframe['epoch'] <= FLAGS.max_epoch]
    # filter only the specified epoch so we don't redo longer runs
    if filter_epoch is not None and filter_epoch is True:
        dataframe = dataframe.loc[dataframe['epoch'] == FLAGS.epoch]
        # TODO(ahundt) we are really looking for "is this a hyperopt result?" not "checkpoint"
        # hyperopt search results don't have checkpoints, but full training runs do
        dataframe = dataframe.loc[dataframe['checkpoint'] == False]

    # loop over the ranked models
    row_progress = tqdm(dataframe.iterrows(), ncols=240)
    i = -1
    for index, row in row_progress:
        i += 1
        if i < skip_models:
            # we designated this model as one to skip, so continue on
            continue
        history = None
        hyperparameters_filename = row['hyperparameters_filename']

        hyperparams = grasp_utilities.load_hyperparams_json(
            hyperparameters_filename, FLAGS.fine_tuning, FLAGS.learning_rate,
            feature_combo_name=feature_combo)

        row_progress.write('-' * 80)
        row_progress.write('Training with hyperparams at index ' + str(index) + ' from: ' + str(hyperparameters_filename) + '\n\n' + str(hyperparams))
        if sort_by is not None:
            row_progress.write('Sorting by: ' + str(sort_by) + ', the value in the rank_csv is: ' + str(row[sort_by]))
        row_progress.write('-' * 80)
        hyperparams['loss'] = 'msle'
        # save weights at checkpoints as the model's performance improves
        hyperparams['checkpoint'] = True
        hyperparams['learning_rate'] = 1.0
        hyperparams['batch_size'] = FLAGS.batch_size

        if i > 0:
            # only load weights for the first entry
            # TODO(ahundt) allow automated loading of weights from past runs
            load_weights = None
            FLAGS.load_weights = None

        try:
            history = cornell_grasp_train.run_training(
                problem_name=problem_type,
                # feature_combo_name=feature_combo,
                hyperparams=hyperparams,
                dataset_name=dataset_name,
                optimizer_name=optimizer_name,
                load_weights=load_weights,
                hyperparameters_filename=hyperparameters_filename,
                **hyperparams)

            run_histories[hyperparameters_filename] = history
            history_dicts[hyperparameters_filename] = history.history
            # save the histories so far, overwriting past updates
            with open(json_histories_path, 'w') as fp:
                # save out all kfold params so they can be reloaded in the future
                json.dump(history_dicts, fp, cls=grasp_utilities.NumpyEncoder)

            # generate the summary results
            results = grasp_utilities.multi_run_histories_summary(
                run_histories,
                metrics=histories_metrics,
                multi_history_metrics=histories_summary_metrics,
                save_filename=json_histories_summary_path,
                description_prefix=problem_type,
                results_prefix='ranked_results',
            )
        except tf.errors.ResourceExhaustedError as exception:
            print('Hyperparams caused algorithm to run out of resources, '
                  'will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you might set more memory sensitive hyperparam ranges,'
                  'or add constraints to your hyperparam search so it does not choose'
                  'huge values for all the parameters at once'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
        except (ValueError, tf.errors.FailedPreconditionError, tf.errors.OpError) as exception:
            print('Hyperparams encountered a model that failed with an invalid combination of values, '
                  'we will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you will need to debug your model w.r.t. '
                  'the current hyperparam choice.'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
        except KeyboardInterrupt as e:
            print('Evaluation of this model canceled based on a user request. '
                  'We will continue to next stage and return infinity loss for the canceled model.')
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb

        # TODO(ahundt) consider shutting down dataset generators and clearing the session when there is an exception
        # https://github.com/tensorflow/tensorflow/issues/4735#issuecomment-363748412
        keras.backend.clear_session()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
