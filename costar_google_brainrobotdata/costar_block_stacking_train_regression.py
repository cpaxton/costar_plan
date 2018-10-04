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
    load_best_weights = True
    # a bit hacky pseudo-eval on training data
    eval_on_training_data = False
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
    # uncomment when running on combined block only + block and plush datasets
    # FLAGS.costar_filename_base = 'costar_combined_block_plush_stacking_v0.4_success_only'
    FLAGS.costar_filename_base = 'costar_block_stacking_v0.4_success_only'
    load_weights = None
    if FLAGS.load_hyperparams is None:
        # Results from classification hyperparameter run
        # FLAGS.load_hyperparams = ('~/datasets/logs/hyperopt_logs_cornell/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')

        # Results from first regression hyperparameter run
        # FLAGS.load_hyperparams = ('~/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-01-15-12-20_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-01-15-12-20_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Just try out NasNet directly without hyperopt (it didn't work well on 2017-03-04)
        # FLAGS.load_hyperparams = ('~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/nasnet_large.json')

        # decent, but didn't run kfold 2018-03-05, + 2018-03-07 trying with mae
        # FLAGS.load_hyperparams = ('~/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-03-16-33-06_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-03-16-33-06_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Old best first epoch on hyperopt run 2018-03-06:
        # FLAGS.load_hyperparams = ('~/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-06-00-20-24_-vgg19_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-06-00-20-24_-vgg19_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Current best performance with mae on val + test 2018-03-07, haven't tried on kfold yet 2018-03-06
        # FLAGS.load_hyperparams = ('~/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
        #                           '2018-03-05-23-05-07_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6/'
        #                           '2018-03-05-23-05-07_-vgg_regression_model-dataset_cornell_grasping-norm_sin2_cos2_hw_yx_6_hyperparams.json')

        # Best first and last epoch on hyperopt run 2018-03-08
        # FLAGS.load_hyperparams = ('~/.keras/datasets/logs/hyperopt_logs_cornell_regression/'
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
                # load_weights = None
                # use these weights if xyz is input but not axis/angle data
                load_weights = 'logs_cornell/2018-09-26-21-31-00_train_0.4_gripper_center-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3/2018-09-26-21-31-00_train_0.4_gripper_center-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-341-val_loss-0.000-val_cart_error-0.023.h5'
                FLAGS.initial_epoch = 245 + 146 + 341 # from continuing runs below
                load_weights = # 'logs_cornell/2018-09-26-21-31-00_train_0.4_gripper_center-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3/2018-09-26-21-31-00_train_0.4_gripper_center-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-245-val_loss-0.000-val_cart_error-0.024.h5'
 #                FLAGS.initial_epoch = 391 # 245+146 from two continuing runs below
                # load_weights = './logs_cornell/2018-09-25-18-29-20_train_0.4_gripper_center-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3/2018-09-25-18-29-20_train_0.4_gripper_center-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-146-val_loss-0.000-val_cart_error-0.026.h5'
                # load_weights = './logs_cornell/2018-09-15-21-10-39_train_no_augmentation-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-003-val_loss-0.000-val_cart_error-0.031.h5'
                # load_weights = './logs_cornell/2018-08-29-22-47-40_train_v0.3-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3/2018-08-29-22-47-40_train_v0.3-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-230-val_loss-0.000-val_grasp_acc-0.289.h5'
                # in the past weights were disabled due to change in input vector to include angle. See commits 785eaf4a4501f2e6532ed59b0972d7b1aaa5784e, b78c3e558567c4b3388a99786549d23a2a1e060c, and 19f274c0d77deff533175692046e424165b821df
                # weights from 2018-07-31-21-40-50 are technically the best for val_grasp_acc (1cm), but newer weights to better with respect to other metrics like val_cart_error and other grasp acc distance metrics
                # load_weights = './logs_cornell/2018-07-31-21-40-50_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-07-31-21-40-50_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-018-val_loss-0.000-val_grasp_acc-0.300.h5'
                # load_weights = './logs_cornell/2018-07-30-21-47-16_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-07-30-21-47-16_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-016-val_loss-0.000-val_grasp_acc-0.273.h5'
                # load_weights = './logs_cornell/2018-07-09-09-08-15_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3/2018-07-09-09-08-15_nasnet_mobile_semantic_translation_regression_model-_img_nasnet_mobile_vec_dense_trunk_vgg_conv_block-dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-115-val_loss-0.000-val_grasp_acc-0.258.h5'
                # use these weights for both xyz and axis angle input data
                # Be careful if loading the weights below, the correct vector input data and backwards compatibility code must be in place to avoid: 
                # "ValueError: You are trying to load a weight file containing 13 layers into a model with 11 layers."
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
            # 2018-08-12 BEST ROTATION MODEL (#5 of 730 models in hyperopt, but #1 after long term training) for rotation 58% val accuracy for angles within 15 degrees.
            FLAGS.load_hyperparams = 'hyperparams/semantic_rotation_regression/2018-08-09-03-05-18_train_200_epochs-vgg_semantic_rotation_regression_model-_img_vgg_vec_dense_block_trunk_nasnet_normal_a_cell-dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5_hyperparams.json'
            if load_best_weights:
                # TODO(ahundt) 2018-09-25 the next line is not actually the current best weights, but we are training with a new configuration so that's what we will load for now
                load_weights = 'logs_cornell/2018-09-25-23-51-23_train_0.4_gripper_center_rot-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5/2018-09-25-23-51-23_train_0.4_gripper_center_rot-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-080-val_loss-0.003-val_angle_error-0.550.h5'
                FLAGS.initial_epoch = 144
                # load_weights = 'hyperopt_logs_costar_rotation_regression/2018-09-04-20-17-25_train_v0.4_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5/2018-09-04-20-17-25_train_v0.4_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-412-val_loss-0.002-val_angle_error-0.279.h5'
                # FLAGS.initial_epoch = 413
                # load_weights = 'hyperopt_logs_costar_rotation_regression/2018-08-31-20-35-15_train_v0.4_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5/2018-08-31-20-35-15_train_v0.4_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-237-val_loss-0.002-val_angle_error-0.281.h5'
                # FLAGS.initial_epoch = 238
                # load_weights = 'hyperopt_logs_costar_block_stacking_train_ranked_regression/2018-08-10-06-55-09_train_200_epochs-vgg_semantic_rotation_regression_model-_img_vgg_vec_dense_block_trunk_nasnet_normal_a_cell-dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5/2018-08-10-06-55-09_train_200_epochs-vgg_semantic_rotation_regression_model-_img_vgg_vec_dense_block_trunk_nasnet_normal_a_cell-dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-041-val_loss-0.007-val_grasp_acc-0.581.h5'
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
    # FLAGS.data_dir = r'C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.4/*success.h5f'

    FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.4/blocks_only/')
    # FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.3/')
    FLAGS.fine_tuning_epochs = 40
    print('Regression Training on costar block stacking is about to begin. '
          'It overrides some command line parameters including '
          'training on mae loss so to change them '
          'you will need to modify cornell_grasp_train_regression.py directly.')

    dataset_name = 'costar_block_stacking'
    FLAGS.dataset_name = dataset_name

    print('-' * 80)
    print('Training with hyperparams from: ' + str(FLAGS.load_hyperparams))
    learning_rate = FLAGS.learning_rate

    hyperparams = grasp_utilities.load_hyperparams_json(
        FLAGS.load_hyperparams, FLAGS.fine_tuning, learning_rate,
        feature_combo_name=feature_combo)

    print('n' + str(hyperparams))
    print('-' * 80)

    # TODO: remove loss if it doesn't work or make me the default in the other files if it works really well
    if 'rotation' in problem_type:
        # rotation does much better with msle over mse
        hyperparams['loss'] = 'msle'
    elif 'translation' in problem_type:
        # translation does slightly better with mse over msle
        hyperparams['loss'] = 'mse'
    elif 'grasp' in problem_type:
        hyperparams['loss'] = 'mse'
    else:
        raise ValueError(
            'costar_block_stacking_train_regression.py update the train config file, '
            'unsupported problem type: ' + problem_type)

    # save weights at checkpoints as the model's performance improves
    hyperparams['checkpoint'] = True
    hyperparams['batch_size'] = FLAGS.batch_size
    # ------------------------------------
    # temporary 0 learning rate for eval!
    if eval_on_training_data:
        print('EVAL on training data (well, a slightly hacky version) with 0 LR 0 dropout trainable False, no learning rate schedule')
        learning_rate = 0.000000000001
        hyperparams['dropout_rate'] = 0.000000000001
        # TODO(ahundt) it seems set_trainable_layers in grasp_model.py has a bug?
        # hyperparams['trainable'] = 0.00000000001
        FLAGS.learning_rate_schedule = 'none'
    else:
        print('manual initial 1.0 learning rate override applied')
        learning_rate = 1.0
    #------------------------------------
    # learning_rate = 1.0
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
