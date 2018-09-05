import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import median_filter
from grasp_median_filter import grasp_dataset_median_filter


class MedianFilterTest(tf.test.TestCase):
    """Unit test for median filter use py_func
    """
    def compare_pyfunc_median_filter_with_direct_call(self):
        with self.test_session() as sess:
            test_input = np.random.random((10, 9)).astype(np.float32)
            test_kernel = (3, 3)
            test_input_tf = tf.convert_to_tensor(test_input)
            filter_result_tf = grasp_dataset_median_filter(test_input_tf, test_kernel[0], test_kernel[1])
            filter_result_np = median_filter(test_input, test_kernel)
            filter_result_tf = sess.run(filter_result_tf)
            self.assertAllEqual(filter_result_tf, filter_result_np)

    def test_remove_zeros(self):
        with self.test_session() as sess:
            test_input = np.ones((5, 5), dtype=np.float32)
            test_input[2, 2] = 0
            test_kernel = (3, 3)
            test_input_tf = tf.convert_to_tensor(test_input)
            filter_result_tf = grasp_dataset_median_filter(test_input_tf, test_kernel[0], test_kernel[1])
            filter_result_tf = sess.run(filter_result_tf)
            assert np.count_nonzero(filter_result_tf) == 25

if __name__ == '__main__':
    tf.test.main()
