import grasp_metrics
import numpy as np
import sklearn
import sklearn.preprocessing


def test_encoding_angle_difference(theta1_degrees, theta2_degrees, diff, threshold=30.0, verbose=1):
    theta1 = theta1_degrees
    theta1_rad = np.radians(theta1)
    theta1_2 = theta1 * 2.0
    theta1_2_rad = np.radians(theta1_2)
    s2c2_1 = np.array([np.sin(theta1_2_rad), np.cos(theta1_2_rad)])
    # decode all the way back to theta in radians (not 2 * theta)
    t1_2decoded_1 = grasp_metrics.decode_sin2_cos2(grasp_metrics.encode_sin2_cos2(s2c2_1))
    stct_1 = np.array([np.sin(theta1_rad), np.cos(theta1_rad)])
    e1 = grasp_metrics.encode_2theta(theta1_rad)
    e1_2 = grasp_metrics.encode_sin2_cos2(s2c2_1)
    assert np.allclose(e1, e1_2)
    ad1 = grasp_metrics.decode_sin2_cos2(e1)
    if verbose > 0:
        print('ad1: ' + str(ad1) + ' theta1_rad ' + str(theta1_rad) + ' theta1 degrees:' + str(theta1)) + ' ad1 degrees:' + str(np.degrees(ad1))
    # the gripper can rotate +-180 degrees or -360 degrees and it is considered equivalent
    assert (np.allclose(t1_2decoded_1, theta1_rad) or
            np.allclose(t1_2decoded_1, theta1_rad + np.radians(180.0)) or
            np.allclose(t1_2decoded_1, theta1_rad - np.radians(180.0)) or
            np.allclose(t1_2decoded_1, theta1_rad - np.radians(360.0)))
    assert (np.allclose(ad1, theta1_rad) or
            np.allclose(ad1, theta1_rad + np.radians(180.0)) or
            np.allclose(ad1, theta1_rad - np.radians(180.0)) or
            np.allclose(ad1, theta1_rad - np.radians(360.0)))

    theta2 = theta2_degrees
    theta2_rad = np.radians(theta2)
    theta2_2 = theta2 * 2.0
    theta2_2_rad = np.radians(theta2_2)
    s2c2_2 = np.array([np.sin(theta2_2_rad), np.cos(theta2_2_rad)])
    # decode all the way back to theta in radians (not 2 * theta)
    t2_2decoded_2 = grasp_metrics.decode_sin2_cos2(grasp_metrics.encode_sin2_cos2(s2c2_2))
    print('s2c2_2: ' + str(s2c2_2))
    stct_2 = np.array([np.sin(theta2_rad), np.cos(theta2_rad)])
    e2 = grasp_metrics.encode_2theta(theta2_rad)
    e2_2 = grasp_metrics.encode_sin2_cos2(s2c2_2)
    assert np.allclose(e2, e2_2)
    ad2 = grasp_metrics.decode_sin2_cos2(e2)
    if verbose > 0:
        print('ad2: ' + str(ad2) + ' theta2_rad ' + str(theta2_rad) + ' theta2 degrees:' + str(theta2)) + ' ad2 degrees:' + str(np.degrees(ad2))
    # the gripper can rotate +-180 degrees or -360 degrees and it is considered equivalent
    assert (np.allclose(ad2, theta2_rad) or
            np.allclose(ad2, theta2_rad + np.radians(180.0)) or
            np.allclose(ad2, theta2_rad - np.radians(180.0)) or
            np.allclose(ad2, theta2_rad - np.radians(360.0)))
    assert (np.allclose(t2_2decoded_2, theta2_rad) or
            np.allclose(t2_2decoded_2, theta2_rad + np.radians(180.0)) or
            np.allclose(t2_2decoded_2, theta2_rad - np.radians(180.0)) or
            np.allclose(t2_2decoded_2, theta2_rad - np.radians(360.0)))

    if verbose > 0:
        print('theta1: ' + str(theta1) + ' theta2: ' + str(theta2))

    # convert the threshold to radians
    threshold_radians = np.radians(threshold)
    # determine if the angle difference in degrees is within the threshold
    theta_in_threshold = diff <= threshold
    two_theta_in_threshold = 2 * diff <= threshold
    assert grasp_metrics.angle_difference_less_than_threshold(
        stct_2[0], stct_2[1], stct_1[0], stct_1[1], angle_threshold=threshold_radians) == theta_in_threshold
    assert grasp_metrics.angle_difference_less_than_threshold(
        stct_1[0], stct_1[1], stct_2[0], stct_2[1], angle_threshold=threshold_radians) == theta_in_threshold
    assert grasp_metrics.angle_difference_less_than_threshold(
        s2c2_2[0], s2c2_2[1], s2c2_1[0], s2c2_1[1], angle_threshold=threshold_radians) == two_theta_in_threshold
    assert grasp_metrics.angle_difference_less_than_threshold(
        s2c2_1[0], s2c2_1[1], s2c2_2[0], s2c2_2[1], angle_threshold=threshold_radians) == two_theta_in_threshold


def test_add_sub_angles(a, angle_diff_deg, expect_match=False):
    # make sure the input angle diff is always positive
    test_encoding_angle_difference(a, a + angle_diff_deg, angle_diff_deg)
    test_encoding_angle_difference(a, a - angle_diff_deg, angle_diff_deg)


def test_rotation_encoding():
    test_add_sub_angles(1, 64)
    test_add_sub_angles(30, 64)
    test_add_sub_angles(60, 64)
    test_add_sub_angles(90, 64)
    test_add_sub_angles(100, 64)
    test_add_sub_angles(150, 64)
    test_add_sub_angles(180, 64)
    test_add_sub_angles(340, 64)
    test_add_sub_angles(1, 32)
    test_add_sub_angles(90, 32)
    test_add_sub_angles(100, 32)
    test_add_sub_angles(150, 32)
    test_add_sub_angles(180, 32)
    test_add_sub_angles(340, 32)
    test_add_sub_angles(1, 28)
    test_add_sub_angles(90, 28)
    test_add_sub_angles(100, 28)
    test_add_sub_angles(150, 28)
    test_add_sub_angles(180, 28)
    test_add_sub_angles(340, 28)
    test_add_sub_angles(1, 56)
    test_add_sub_angles(90, 56)
    test_add_sub_angles(100, 56)
    test_add_sub_angles(150, 56)
    test_add_sub_angles(180, 56)
    test_add_sub_angles(340, 56)

if __name__ == '__main__':
    test_add_sub_angles(1, 28)
    pytest.main([__file__])