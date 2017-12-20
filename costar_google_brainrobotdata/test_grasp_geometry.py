
import numpy as np
import pytest
from grasp_geometry import grasp_dataset_to_transforms_and_features


def test_grasp_dataset_to_transforms_and_features():

    def evaluate_grasp_dataset_to_transforms_and_features(depth_image, intrinsics, camera_to_base, base_to_end_current, base_to_end_final):
        [current_base_T_camera_vec_quat_7_array,
         eectf_vec_quat_7_array,
         camera_T_endeffector_current_vec_quat_7_array,
         camera_T_depth_pixel_current_vec_quat_7_array,
         camera_T_endeffector_final_vec_quat_7_array,
         camera_T_depth_pixel_final_vec_quat_7_array,
         depth_pixel_T_endeffector_current_vec_quat_7_array,
         image_coordinate_current,
         depth_pixel_T_endeffector_final_vec_quat_7_array,
         image_coordinate_final,
         sin_cos_2,
         vec_sin_cos_5,
         delta_depth_sin_cos_3,
         delta_depth_quat_5] = grasp_dataset_to_transforms_and_features(
                                     depth_image,
                                     intrinsics,
                                     camera_to_base,
                                     base_to_end_current,
                                     base_to_end_final)

        assert np.allclose(delta_depth_sin_cos_3[0], np.array([1]))

    depth_image = np.ones([10, 20, 1])
    intrinsics = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [5., 10., 1.]])
    camera_to_base = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
    base_to_end_current = np.array([0., 0., 1., 0., 0., 0., 1.])
    base_to_end_final = np.array([1., 1., 2., 0., 0., 0., 1.])

    evaluate_grasp_dataset_to_transforms_and_features(depth_image, intrinsics, camera_to_base, base_to_end_current, base_to_end_final)

    # test if the end effector is outside the frame
    base_to_end_current = np.array([20., 1., 1., 0., 0., 0., 1.])
    evaluate_grasp_dataset_to_transforms_and_features(depth_image, intrinsics, camera_to_base, base_to_end_current, base_to_end_final)


if __name__ == '__main__':
    pytest.main([__file__])