'''
Training a network on cornell grasping dataset for classification of grasp command proposals.

In other words, this network tries to predict a grasp rectangle from an input image.

The rectangle width is how open a gripper is,
the height is the length of the line along which grasps will be successful.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import tensorflow as tf
import grasp_utilities
import cornell_grasp_train
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(_):
    problem_type = 'grasp_classification'
    # recommended for
    # - pixelwise classification
    # - classification of images centered and rotated to grasp proposals
    feature_combo = 'image_preprocessed_norm_sin2_cos2_width_3'

    # Warning: this version will have some correlation with due to the height parameter!
    # However, it will be OK to use to classify the output of regression
    # feature_combo = 'image_preprocessed_norm_sin2_cos2_height_width_4'

    # Recommended for single prediction if images are merely center cropped
    # and not rotated or translated to the proposed grasp positions:
    # feature_combo = 'image_preprocessed_norm_sin2_cos2_w_yx_5'

    # recommended for
    # - classification of images centered and rotated to grasp proposals
    # feature_combo = 'image_preprocessed_width_1'

    # NOT RECOMMENDED due to height being correlated with grasp_success:
    # feature_combo = 'image_preprocessed_sin2_cos2_height_width_4'
    # Override some default flags for this configuration
    # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
    FLAGS.crop_height = 224
    FLAGS.crop_width = 224
    FLAGS.problem_type = problem_type
    FLAGS.feature_combo = feature_combo
    FLAGS.crop_to = 'center_on_gripper_grasp_box_and_rotate_upright'
    if FLAGS.load_hyperparams is None:
        # FLAGS.load_hyperparams = ('~/datasets/logs/hyperopt_logs_cornell/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # About 0.95 k-fold accuracy on preprocessed_norm_sin2_cos2_w_3:
        # FLAGS.load_hyperparams = ('hyperparams/classification/2018-03-03-07-14-59_'
        #                           '-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # (not very good) Best result from classification hyperopt run ending 2018-03-16:
        # FLAGS.load_hyperparams = ('~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/hyperparams/classification/'
        #                           '2018-03-14-00-40-09_-vgg19_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # (not very good) Best NON VGG result from classification hyperopt run ending 2018-03-16:
        # FLAGS.load_hyperparams = ('~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/hyperparams/classification/'
        #                           '2018-03-16-06-15-01_-densenet_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # Quite good kfold, best hyperparams from 2018-04 2000 model hyperopt run TODO(ahundt) add details from kfold run
        # FLAGS.load_hyperparams = ('hyperparams/classification/2018-04-08-21-04-19__optimized_hyperparams.json')
        # about to run: best hyperparams with trainable set to False from 2018-04 2000 model hyperopt run
        # FLAGS.load_hyperparams = ('hyperparams/classification/2018-04-07-15-31-34_vgg_classifier_model-_img_vgg_vec_dense_block_trunk_dense_block-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # One of best results from hyperopt run ending 2018-05-08
        # FLAGS.load_hyperparams = ('hyperparams/classification/2018-05-03-17-02-01_inception_resnet_v2_classifier_model-'
        #                           '_img_inception_resnet_v2_vec_dense_trunk_vgg_conv_block-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # Best hyperopt result without "height" (length along which gripper can slide, not gripper openness) parameter from hyperopt run ending 2018-06-02
        FLAGS.load_hyperparams = ('hyperparams/classification/2018-05-26-01-22-00_inception_resnet_v2_classifier_model-'
                                  '_img_inception_resnet_v2_vec_dense_trunk_vgg_conv_block-dataset_cornell_grasping-grasp_success_hyperparams.json')
    FLAGS.epochs = 80
    FLAGS.fine_tuning_epochs = 0
    # 8 training folds
    FLAGS.num_train = 8
    # 1 validation fold, this is automatically
    # changed below for the k_fold case
    FLAGS.num_validation = 1
    FLAGS.num_test = 1

    # This only means that the first part of the run is not fine tuning
    # The later fine_tuning_epochs will work correctly.
    FLAGS.fine_tuning = False
    print('Classification Training on grasp_success is about to begin. '
          'This mode overrides some command line parameters so to change them '
          'you will need to modify cornell_grasp_train_classification.py directly.')

    hyperparams = grasp_utilities.load_hyperparams_json(
        FLAGS.load_hyperparams, fine_tuning=FLAGS.fine_tuning,
        learning_rate=FLAGS.learning_rate,
        feature_combo_name=feature_combo)

    # keep feature_combo_name from causing errors if it is present or not present
    if feature_combo not in hyperparams:
        hyperparams['feature_combo_name'] = feature_combo

    if 'k_fold' in FLAGS.pipeline_stage:
        FLAGS.num_validation = 2
        FLAGS.num_test = 0
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type,
            # feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='objectwise',
            **hyperparams)
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type,
            # feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='imagewise',
            **hyperparams)
    else:
        cornell_grasp_train.run_training(
            problem_name=problem_type,
            # feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            **hyperparams)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
