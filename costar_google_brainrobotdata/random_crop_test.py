import tensorflow as tf
import random_crop_parameters as rcp


class random_crop_test(tf.test.TestCase):
    """Unit test for random_crop
    """
    def testSquare(self):
        with self.test_session() as sess:
            input_size = tf.convert_to_tensor([10, 10])
            cropped_size = tf.convert_to_tensor([2, 4])
            test_input = tf.reshape(tf.range(input_size[0] * input_size[1]), input_size)
            offset = rcp.random_crop_parameters(input_size, cropped_size)
            rcp_crop = rcp.crop_images(test_input, offset, cropped_size)
            tf_crop = tf.slice(test_input, offset, cropped_size)
            rcp_crop, tf_crop = sess.run([rcp_crop, tf_crop])

            self.assertAllEqual(rcp_crop, tf_crop)

if __name__ == '__main__':
    tf.test.main()
