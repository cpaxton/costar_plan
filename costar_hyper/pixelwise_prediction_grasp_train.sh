export CUDA_VISIBLE_DEVICES="0" && python2 grasp_train.py --batch_size=7 --epochs 300 --save_weights delta_depth_sin_cos_3 --grasp_model grasp_model_levine_2016_segmentation --optimizer SGD --loss segmentation_single_pixel_binary_crossentropy --grasp_sequence_motion_command_feature ''
# Other options to consider:
# --loss segmentation_gaussian_binary_crossentropy
# --loss segmentation_single_pixel_binary_crossentropy
# --grasp_sequence_motion_command_feature 'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3'