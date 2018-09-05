import tensorflow as tf
import numpy as np
from keras.losses import binary_crossentropy
import grasp_loss
import grasp_geometry


class TestGraspLoss(tf.test.TestCase):
    """ Unit test for functions in grasp_loss.py
    """
    def test_single_pixel_measurement_index(self):
        with self.test_session() as sess:

            def test_different_input(sess, test_shape_height, test_shape_width):
                random_x_true = np.random.randint(0, test_shape_width)
                random_y_true = np.random.randint(0, test_shape_height)
                random_x_false = np.random.randint(0, test_shape_width)
                random_y_false = np.random.randint(0, test_shape_height)
                test_true_np = np.array([[1, random_y_true, random_x_true],
                                        [0, random_y_false, random_x_false]],
                                        dtype=np.float32)
                test_pred_np = np.zeros((2, test_shape_height, test_shape_width, 1), dtype=np.float32)
                test_pred_np[0, random_y_true, random_x_true, 0] = 1.0
                test_pred_np[0, random_y_false, random_x_false, 0] = 0.0
                test_pred_tf = tf.convert_to_tensor(test_pred_np, tf.float32)
                test_true_tf = tf.convert_to_tensor(test_true_np, tf.float32)

                measure_tf_true = grasp_loss.segmentation_single_pixel_binary_crossentropy(test_true_tf, test_pred_np)
                measure_tf_true = sess.run(measure_tf_true)

                direct_call_result = binary_crossentropy(test_true_tf[:, :1], tf.constant([[1.0], [0.0]], tf.float32))
                direct_call_result = sess.run(direct_call_result)

                assert np.allclose(measure_tf_true, np.array([0.0], dtype=np.float32), atol=1e-06)
                assert np.allclose(direct_call_result, measure_tf_true)

            test_different_input(sess, 30, 20)
            test_different_input(sess, 40, 50)
            test_different_input(sess, 25, 30)
            test_different_input(sess, 35, 35)

    def test_gaussian_kernel_2D(self):
        def test_gaussian_input(size, center, sigma):
            with self.test_session() as sess:
                test_gaussian = grasp_geometry.gaussian_kernel_2D(size, center, sigma)
                tensor = grasp_loss.gaussian_kernel_2D(size, center, sigma)
                result = sess.run(tensor)
                if center is not None:
                    assert result[center[0], center[1]] == 1
                assert np.allclose(test_gaussian, result)
        test_gaussian_input(size=(3, 3), center=None, sigma=1)
        test_gaussian_input(size=(3, 3), center=(1, 1), sigma=1)
        test_gaussian_input(size=(3, 3), center=(0, 1), sigma=1)
        test_gaussian_input(size=(4, 3), center=(3, 1), sigma=1)
        test_gaussian_input(size=(3, 4), center=(0, 0), sigma=1)
        test_gaussian_input(size=(5, 5), center=(3, 3), sigma=2)
        test_gaussian_input(size=(5, 5), center=None, sigma=2)

if __name__ == '__main__':
    tf.test.main()
