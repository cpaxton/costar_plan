import tensorflow as tf
from tensorflow.python.platform import flags

import hyperopt
import grasp_train
import grasp_utilities

FLAGS = flags.FLAGS


def main(_):
    """ Hyperparameter optimization of grasp classification with the google brain robot grasping dataset.
    """
    problem_type = 'grasp_classification'
    FLAGS.grasp_success_label = 'move_to_grasp/time_ordered/grasp_success'
    # FLAGS.grasp_sequence_motion_command_feature = 'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_sin_cos_5'
    # FLAGS.grasp_sequence_motion_command_feature = 'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3'
    FLAGS.grasp_sequence_motion_command_feature = 'move_to_grasp/time_ordered/reached_pose/transforms/all_transforms'
    FLAGS.loss = 'binary_crossentropy'
    FLAGS.metric = 'binary_accuracy'
    FLAGS.epochs = 1
    FLAGS.fine_tuning_epochs = 0
    FLAGS.grasp_dataset_validation = '092'
    FLAGS.grasp_dataset_test = '097'
    # TODO(ahundt) also make sure run_hyperopt grasp_dataset is 102, or correct bug in FLAGS
    FLAGS.grasp_dataset = '102'
    FLAGS.grasp_datasets_train = FLAGS.grasp_dataset
    FLAGS.log_dir = './hyperopt_logs_google_brain_classification/'
    print('Overriding some flags, edit google_grasp_hyperopt.py directly to see and change them.' +
          ' epochs: ' + str(FLAGS.epochs) +
          ' fine_tuning_epochs: ' + str(FLAGS.fine_tuning_epochs))
    run_name = FLAGS.run_name
    run_name += '-' + FLAGS.grasp_sequence_motion_command_feature.split('/')[-1]
    log_dir = FLAGS.log_dir
    run_name = grasp_utilities.timeStamped(run_name)
    run_training_fn = grasp_train.run_hyperopt
    param_to_optimize = 'val_acc'
    batch_size = 4
    min_top_block_filter_multiplier = 6
    feature_combo_name = None
    seed = 7

    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    hyperopt.optimize(
        run_training_fn=run_training_fn,
        feature_combo_name=feature_combo_name,
        problem_type=problem_type,
        run_name=run_name,
        log_dir=log_dir,
        min_top_block_filter_multiplier=min_top_block_filter_multiplier,
        batch_size=batch_size,
        param_to_optimize=param_to_optimize,
        seed=seed)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
