import tensorflow as tf
from keras import backend as K
from tensorflow.python.platform import flags

import hyperopt
import cornell_grasp_train
import cornell_grasp_dataset_reader
import grasp_utilities

FLAGS = flags.FLAGS


def cornell_hyperoptions(problem_type, param_to_optimize):
    """ Set some hyperparams based on the problem type and parameter to optimize

    Deal with some string renames and modes in which the hyperparameter
    optimization algorithm can run such as grasp classification vs regression.
    """
    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    batch_size = FLAGS.batch_size
    if problem_type == 'classification' or problem_type == 'grasp_classification':
        FLAGS.problem_type = problem_type
        # note: this version will have some correlation with success,
        # but it will be OK to use to classify the output of regression
        # feature_combo_name = 'image_preprocessed_norm_sin2_cos2_height_width_4'

        # recommended for
        # - pixelwise classification
        # - classification of images centered and rotated to grasp proposals
        feature_combo_name = 'image_preprocessed_norm_sin2_cos2_width_3'

        # Another simpler option with less input data:
        # feature_combo_name = 'image_preprocessed_width_1'
        FLAGS.crop_to = 'center_on_gripper_grasp_box_and_rotate_upright'
        if param_to_optimize == 'val_acc':
            param_to_optimize = 'val_binary_accuracy'
        min_top_block_filter_multiplier = 6
        FLAGS.crop_height = 224
        FLAGS.crop_width = 224
    elif problem_type in ['grasp_regression', 'regression', 'semantic_grasp_regression',
                          'semantic_translation_regression', 'semantic_rotation_regression']:
        feature_combo_name = 'image_preprocessed'
        # Override some default flags for this configuration
        # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
        FLAGS.problem_type = problem_type
        FLAGS.feature_combo = feature_combo_name
        # only meaningful on the cornell and google dataset readers
        FLAGS.crop_to = 'image_contains_grasp_box_center'
        if param_to_optimize == 'val_acc':
            param_to_optimize = 'val_grasp_jaccard'
        min_top_block_filter_multiplier = 8
        FLAGS.crop_height = 331
        FLAGS.crop_width = 331
    return feature_combo_name, min_top_block_filter_multiplier, batch_size, param_to_optimize


def main(_):

    # prevent errors from being printed for hours when memory runs out
    # https://github.com/tensorflow/tensorflow/issues/20998
    # commented because this approach doesn't work...
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.report_tensor_allocations_upon_oom = False
    # # config.inter_op_parallelism_threads = 40
    # tf_session = tf.Session(config=config)
    # K.set_session(tf_session)
    # Edit these flags to choose your configuration:
    # FLAGS.problem_type = 'classification'
    # FLAGS.dataset_name = 'cornell_grasping'
    FLAGS.dataset_name = 'costar_block_stacking'
    # FLAGS.problem_type = 'semantic_grasp_regression'

    # ----------------------------------------------------
    # When ranking rotation use the following settings:
    # FLAGS.log_dir = 'hyperopt_logs_costar_block_stacking_train_ranked_regression'
    # FLAGS.problem_type = 'semantic_rotation_regression'
    # ----------------------------------------------------
    # When ranking translation use the following settings
    FLAGS.log_dir = 'hyperopt_logs_costar_translation_regression'
    FLAGS.problem_type = 'semantic_translation_regression'
    # ----------------------------------------------------
    FLAGS.batch_size = 16
    FLAGS.num_validation = 1
    FLAGS.num_test = 1
    FLAGS.epochs = 1
    FLAGS.fine_tuning_epochs = 0
    run_name = FLAGS.run_name
    log_dir = FLAGS.log_dir
    run_name = grasp_utilities.timeStamped(run_name)
    run_training_fn = cornell_grasp_train.run_training
    problem_type = FLAGS.problem_type
    param_to_optimize = 'loss'
    # TODO(ahundt) re-enable seed once memory leak issue below is fixed
    # seed = 44
    seed = None
    # TODO(ahundt) costar generator has a memory leak! only do 100 samples as a quick fix. Large values can be used for the cornell dataset without issue.
    # initial_num_samples = 4000
    initial_num_samples = 100
    maximum_hyperopt_steps = 10
    # enable random learning rates, if enabled,
    # this will be the primary motivator for good/bad
    # performance, so once you find a good setting
    # lock it to find a good model
    learning_rate_enabled = False

    # checkpoint is a special parameter to not save hdf5 files because training runs
    # are very quick (~1min) and checkpoint files are very large (~100MB)
    # which is forwarded to cornell_grasp_train.py run_training() function.
    checkpoint = False

    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    feature_combo_name, min_top_block_filter_multiplier, batch_size, param_to_optimize = cornell_hyperoptions(problem_type, param_to_optimize)

    print('Overriding some flags, edit cornell_hyperopt.py directly to change them.' +
          ' num_validation: ' + str(FLAGS.num_validation) +
          ' num_test: ' + str(FLAGS.num_test) +
          ' epochs: ' + str(FLAGS.epochs) +
          ' fine_tuning_epochs: ' + str(FLAGS.fine_tuning_epochs) +
          ' problem_type:' + str(FLAGS.problem_type) +
          ' crop (height, width): ({}, {})'.format(FLAGS.crop_height, FLAGS.crop_width))
    best_hyperparams = hyperopt.optimize(
        run_training_fn=run_training_fn,
        feature_combo_name=feature_combo_name,
        problem_type=FLAGS.problem_type,
        run_name=run_name,
        log_dir=log_dir,
        min_top_block_filter_multiplier=min_top_block_filter_multiplier,
        batch_size=batch_size,
        param_to_optimize=param_to_optimize,
        initial_num_samples=initial_num_samples,
        maximum_hyperopt_steps=maximum_hyperopt_steps,
        learning_rate_enabled=learning_rate_enabled,
        seed=seed,
        checkpoint=checkpoint)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
