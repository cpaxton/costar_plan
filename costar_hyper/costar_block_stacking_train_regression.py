'''
Training a network on cornell grasping dataset for regression of grasping positions.

In other words, this network tries to predict a grasp rectangle from an input image.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import os
import tensorflow as tf
import grasp_utilities
import cornell_grasp_train
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(_):
    use_best_model = True
    load_best_weights = False
    # problem_type = 'semantic_translation_regression'
    problem_type = 'semantic_rotation_regression'
    # problem_type = 'semantic_grasp_regression'
    # problem_type2 = 'semantic_grasp_regression'
    problem_type2 = None
    feature_combo = 'image_preprocessed'
    # Override some default flags for this configuration
    # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
    FLAGS.problem_type = problem_type
    FLAGS.feature_combo = feature_combo
    FLAGS.crop_to = 'image_contains_grasp_box_center'
    load_weights = None
    if FLAGS.load_hyperparams is None:
        # Results from classification hyperparameter run
        # FLAGS.load_hyperparams = ('/home/ahundt/datasets/logs/hyperopt_logs_cornell/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # Results from first regression hyperparameter run
        # FLAGS.load_hyperparams = ('/home/ahundt/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-01-15-12-20_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-01-15-12-20_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Just try out NasNet directly without hyperopt (it didn't work well on 2017-03-04)
        # FLAGS.load_hyperparams = ('/home/ahundt/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/nasnet_large.json')

        # decent, but didn't run kfold 2018-03-05, + 2018-03-07 trying with mae
        # FLAGS.load_hyperparams = ('/home/ahundt/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-03-16-33-06_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-03-16-33-06_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Old best first epoch on hyperopt run 2018-03-06:
        # FLAGS.load_hyperparams = ('/home/ahundt/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-06-00-20-24_-vgg19_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-06-00-20-24_-vgg19_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Current best performance with mae on val + test 2018-03-07, haven't tried on kfold yet 2018-03-06
        # FLAGS.load_hyperparams = ('/home/ahundt/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-05-23-05-07_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-05-23-05-07_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Best first and last epoch on hyperopt run 2018-03-08
        # FLAGS.load_hyperparams = ('/home/ahundt/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-07-18-36-17_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-07-18-36-17_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')
        # FLAGS.load_hyperparams = (r'C:/Users/Varun/JHU/LAB/Projects/costar_plan/costar_google_brainrobotdata/hyperparams/regression/2018-03-01-15-12-20_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # FLAGS.load_hyperparams = ('hyperparams/regression/'
        #                           '2018-03-05-23-05-07_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')
        # 2018-06-28 hyperopt 324 model results on actual stacking dataset, may not be a great run... .25 m error. Rerun?
        # FLAGS.load_hyperparams = ('hyperparams/semantic_grasp_regression/2018-06-25-03-45-46_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json')

        # 2018-06-29 hyperopt best of 65 models on val_cart_error in hyperopt, the vgg model above reliably does better.
        # 1 epoch results: val angle error (rad): 0.215976331639 val cart error (m): 0.126106130658
        # 120 epoch results: val cart error 0.25... much worse than the original 1 epoch.
        # FLAGS.load_hyperparams = ('hyperparams/semantic_grasp_regression/2018-06-28-21-16-47_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json')

        # 2018-6-30 hyperop 2nd best of 120 models on val_cart error in hyperopt
        # FLAGS.load_hyperparams = ('hyperopt_logs_costar_grasp_regression/2018-06-28-15-45-13_inception_resnet_v2_semantic_grasp_regression_model-_img_inception_resnet_v2_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-06-28-15-45-13_inception_resnet_v2_semantic_grasp_regression_model-_img_inception_resnet_v2_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json')

        # 2018-07-26 #1 COORDCONV applied to initial input image, 2nd best best performing overall of about 700 hyperopt results (ABOUT TO BE RUN)
        # FLAGS.load_hyperparams = ('hyperopt_logs_costar_grasp_regression/2018-07-26-19-40-57_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-26-19-40-57_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json')

        # 2018-07-07 best performing non-vgg model with 15% val and test grasp accracy for translations with semantic_translation_regression case.
        # FLAGS.load_hyperparams = 'hyperparams/semantic_grasp_regression/2018-07-06-22-34-31_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'

        if (problem_type == 'semantic_translation_regression' or problem_type == 'semantic_grasp_regression') and use_best_model:
            # 2018-08-12 CURRENT SECOND BEST MODEL FOR TRANSLATION, 22% translation only validation accuracy
            # FLAGS.load_hyperparams = 'hyperparams/semantic_grasp_regression/2018-07-06-22-34-31_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'
            # 2018-07-08 BEST of ~400 or maybe the one 2017-07-07?, 30% translation only validation accuracy, 17.5% combined translation + rotation validation accuracy.
            # BEST MODEL Results in: ./logs_cornell/2018-07-09-09-08-15_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3
            FLAGS.load_hyperparams = 'hyperparams/semantic_grasp_regression/2018-07-07-15-05-32_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'
            # weights for TRANSLATION PROBLEM ONLY:
            if problem_type == 'semantic_translation_regression' and load_best_weights:
                # TODO(ahundt) 2018-08-06 Re-enable best weights, currently disabling due to change in input vector which now includes angle. See commits 785eaf4a4501f2e6532ed59b0972d7b1aaa5784e, b78c3e558567c4b3388a99786549d23a2a1e060c, and 19f274c0d77deff533175692046e424165b821df
                load_weights = None
                # use these weights if xyz is input but not axis/angle data
                load_weights = './logs_cornell/2018-07-31-21-40-50_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-07-31-21-40-50_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-018-val_loss-0.000-val_grasp_acc-0.300.h5'
                # load_weights = './logs_cornell/2018-07-30-21-47-16_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-07-30-21-47-16_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-016-val_loss-0.000-val_grasp_acc-0.273.h5'
                # load_weights = './logs_cornell/2018-07-09-09-08-15_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-07-09-09-08-15_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-115-val_loss-0.000-val_grasp_acc-0.258.h5'
                # use these weights for both xyz and axis angle input data
                # TODO(ahundt) are these the wrong weights? "ValueError: You are trying to load a weight file containing 13 layers into a model with 11 layers."
                # load_weights = './logs_cornell/2018-08-09-11-26-03_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-08-09-11-26-03_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-003-val_loss-0.000-val_grasp_acc-0.160.h5'
                # weights below are trained with data augmentation, weights 2018-07-31-21-40-50 above are actual best so far for translation as of 2018-08-12
                # load_weights = 'logs_cornell/2018-08-12-22-45-00_train_200_epochs-nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-08-12-22-45-00_train_200_epochs-nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-198-val_loss-0.000-val_grasp_acc-0.158.h5'
            # weights for TRANSLATION + ROTATION PROBLEM:
            if problem_type == 'semantic_grasp_regression' and load_best_weights:
                # same hyperparams as above, just also the same folder as the weights below it
                # FLAGS.load_hyperparams = './logs_cornell/2018-07-11-14-01-56_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-11-14-01-56_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'
                load_weights = './logs_cornell/2018-07-11-14-01-56_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-11-14-01-56_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8-epoch-018-val_loss-0.027-val_grasp_acc-0.175.h5'

        # 0.6 val_grasp_acc, hyperopt results placed this as #1 of 700 models for val_grasp_acc in 1 epoch
        # FLAGS.load_hyperparams = 'hyperopt_logs_costar_grasp_regression/2018-07-22-07-15-09_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_block_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-22-07-15-09_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_block_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'

        # not so great, was top nasnset_mobile model of 700 models. (0.1 or 0.7 accuracy after 120 epochs, not that great, slow runtime)
        # FLAGS.load_hyperparams = 'hyperopt_logs_costar_grasp_regression/2018-07-22-16-59-27_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-22-16-59-27_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'

        # 2018-07-23 best nasnet image model hyperopt, #6 result of 730 models (ABOUT TO BE RUN)
        # FLAGS.load_hyperparams = 'hyperopt_logs_costar_grasp_regression/2018-07-21-18-59-02_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-21-18-59-02_nasnet_mobile_semantic_grasp_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'

        # 2018-07-30 maybe worth a try, vgg model which did very well on rotation, #4 of 729, and fairly good on both val_grasp_acc and val_cart_error
        # FLAGS.load_hyperparams = 'hyperopt_logs_costar_grasp_regression/2018-07-24-07-43-45_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_block_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8/2018-07-24-07-43-45_vgg_semantic_grasp_regression_model-_img_vgg_vec_dense_block_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_aaxyz_nsc_8_hyperparams.json'

        if problem_type == 'semantic_rotation_regression' and use_best_model:
            # 2018-08-12 EXCELLENT ROTATION MODEL #5 of 730 models for rotation 58% val accuracy for angles within 15 degrees.
            FLAGS.load_hyperparams = 'hyperparams/semantic_rotation_regression/2018-08-09-03-05-18_train_200_epochs-vgg_semantic_rotation_regression_model-_img_vgg_vec_dense_block_trunk_nasnet_normal_a_cell-dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5_hyperparams.json'
            if load_best_weights:
                load_weights = 'hyperopt_logs_costar_block_stacking_train_ranked_regression/2018-08-10-06-55-09_train_200_epochs-vgg_semantic_rotation_regression_model-_img_vgg_vec_dense_block_trunk_nasnet_normal_a_cell-dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5/2018-08-10-06-55-09_train_200_epochs-vgg_semantic_rotation_regression_model-_img_vgg_vec_dense_block_trunk_nasnet_normal_a_cell-dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-041-val_loss-0.007-val_grasp_acc-0.581.h5'
            FLAGS.random_augmentation = None

    FLAGS.epochs = 600
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
    # FLAGS.data_dir = r'C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.3/*success.h5f'

    FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.3/*success.h5f')
    FLAGS.fine_tuning_epochs = 40
    print('Regression Training on costar block stacking is about to begin. '
          'It overrides some command line parameters including '
          'training on mae loss so to change them '
          'you will need to modify cornell_grasp_train_regression.py directly.')

    dataset_name = 'costar_block_stacking'

    print('-' * 80)
    print('Training with hyperparams from: ' + str(FLAGS.load_hyperparams))
    learning_rate = FLAGS.learning_rate

    hyperparams = grasp_utilities.load_hyperparams_json(
        FLAGS.load_hyperparams, FLAGS.fine_tuning, learning_rate,
        feature_combo_name=feature_combo)

    print('n' + str(hyperparams))
    print('-' * 80)

    # TODO: remove loss if it doesn't work or make me the default in the other files if it works really well
    hyperparams['loss'] = 'msle'
    # save weights at checkpoints as the model's performance improves
    hyperparams['checkpoint'] = True
    hyperparams['batch_size'] = FLAGS.batch_size
    # temporary 0 learning rate for eval!
    # learning_rate = 0
    learning_rate = 1.0
    if load_weights is not None:
        FLAGS.load_weights = load_weights
        # For training from scratch
        # learning_rate = 1.0
        # For resuming translation + rotation model training
        # learning_rate = 1e-2
        # For resuming translation model training
        # learning_rate = 1e-3
    # hyperparams['trainable'] = True

    # override all learning rate settings to make sure
    # it is consistent with any modifications made above
    hyperparams['learning_rate'] = learning_rate
    FLAGS.learning_rate = learning_rate
    FLAGS.fine_tuning_learning_rate = learning_rate

    if 'k_fold' in FLAGS.pipeline_stage:
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='objectwise',
            dataset_name=dataset_name,
            **hyperparams)
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='imagewise',
            dataset_name=dataset_name,
            **hyperparams)
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type2,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='objectwise',
            dataset_name=dataset_name,
            **hyperparams)
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type2,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='imagewise',
            dataset_name=dataset_name,
            **hyperparams)
    else:
        print('\n---------------------\ntraining problem type: ' + str(problem_type) + '\n---------------------')
        cornell_grasp_train.run_training(
            problem_name=problem_type,
            # feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            dataset_name=dataset_name,
            optimizer_name=optimizer_name,
            load_weights=load_weights,
            hyperparameters_filename=FLAGS.load_hyperparams,
            **hyperparams)
        if problem_type2 is not None:
            print('\n---------------------\ntraining problem type2: ' + str(problem_type2) + '\n---------------------')
            cornell_grasp_train.run_training(
                problem_name=problem_type2,
                # feature_combo_name=feature_combo,
                hyperparams=hyperparams,
                dataset_name=dataset_name,
                optimizer_name=optimizer_name,
                hyperparameters_filename=FLAGS.load_hyperparams,
                **hyperparams)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
