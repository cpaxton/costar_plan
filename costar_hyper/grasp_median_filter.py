import tensorflow as tf
from scipy.ndimage.filters import median_filter


def grasp_dataset_median_filter(input_tensor, filter_height, filter_width):
    """ Median filter of tensor
        input_tensor is 2D tensor tf.float32
        filter_size is a tuple (x, y)
    """
    filter_size = (filter_height, filter_width)
    [filter_result] = tf.py_func(
        median_filter,
        [input_tensor, filter_size],
        [tf.float32], stateful=False,
        name='py_func/grasp_dataset_median_filter')
    filter_result.set_shape(input_tensor.get_shape().as_list())
    filter_result = tf.reshape(filter_result, tf.shape(input_tensor))
    return filter_result
