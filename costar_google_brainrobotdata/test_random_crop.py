import tensorflow as tf
import keras.backend as K
import random_crop as rcp
import numpy as np


class random_crop_test(tf.test.TestCase):
    """Unit test for random_crop
    """
    def testRandomCrop(self):
        with self.test_session() as sess:

            def make_crop_tensors(input_size, cropped_size):
                input_size_tf = tf.convert_to_tensor(input_size)
                cropped_size_tf = tf.convert_to_tensor(cropped_size)

                test_input = tf.random_uniform(input_size_tf)
                test_input = tf.reshape(test_input, input_size_tf)

                offset = rcp.random_crop_offset(input_size, cropped_size)
                rcp_crop = rcp.crop_images(test_input, offset, cropped_size)
                tf_crop = tf.slice(test_input, offset, cropped_size)

                return rcp_crop, tf_crop

            def evaluate_crop(session, input_size, cropped_size, count=100, verbose=0):
                rcp_crop_tensor, tf_crop_tensor = make_crop_tensors(input_size, cropped_size)
                for i in range(count):
                    rcp_crop, tf_crop = session.run([rcp_crop_tensor, tf_crop_tensor])
                    if verbose > 0:
                        print('rcp_crop.shape: ' + str(rcp_crop.shape) + ' tf_crop.shape:' +
                              str(tf_crop.shape) + ' cropped_size_tf:' + str(cropped_size))
                    self.assertAllEqual(rcp_crop, tf_crop)
                    self.assertAllEqual(rcp_crop.shape, cropped_size)
                    self.assertAllEqual(tf_crop.shape, cropped_size)

            evaluate_crop(sess, input_size=[4, 5, 3], cropped_size=[2, 4, 3])
            evaluate_crop(sess, input_size=[4, 5, 3], cropped_size=[4, 5, 3])
            evaluate_crop(sess, input_size=[4, 5, 1], cropped_size=[2, 4, 1])
            evaluate_crop(sess, input_size=[4, 5, 1], cropped_size=[4, 5, 1])

    def testIntrinsics(self):
        with self.test_session() as sess:
            intrinsics_np = np.array([[225., 0., 0.], [0., 225., 0.], [125., 127., 1.]])
            intrinsics_tf = tf.convert_to_tensor(intrinsics_np)
            offset_np = np.array([20, 23, 0])
            offset_tf = tf.convert_to_tensor(offset_np)
            cropped_intrinsics = rcp.crop_image_intrinsics(intrinsics_tf, offset_tf)
            intrinsics_tf = sess.run(cropped_intrinsics)

            intrinsics_np[2, 0] -= offset_np[0]
            intrinsics_np[2, 1] -= offset_np[1]

            self.assertAllEqual(intrinsics_np, intrinsics_tf)

    def testCloudPoints(self):
        with self.test_session() as sess:
            intrinsics_np = np.array([[225., 0., 0.], [0., 225., 0.], [125., 127., 1.]])
            intrinsics_tf = tf.convert_to_tensor(intrinsics_np)
            input_size_tf = tf.constant([20, 20, 1])
            cropped_size_tf = tf.constant([5, 4, 1])

            test_input = tf.random_uniform(input_size_tf)
            test_input = tf.reshape(test_input, input_size_tf)

            offset_tf = rcp.random_crop_offset(input_size_tf, cropped_size_tf)
            rcp_crop = rcp.crop_images(test_input, offset_tf, cropped_size_tf)
            cropped_intrinsics = rcp.crop_image_intrinsics(intrinsics_tf, offset_tf)
            cropped_intrinsics_np = sess.run(cropped_intrinsics)

            pointcloud_x = (test_input-intrinsics_np[0][2])/intrinsics_np[0][0]
            pointcloud_y = (test_input-intrinsics_np[1][2])/intrinsics_np[1][1]
            pointcloud_z = test_input

            crop_pointcloud_x = (rcp_crop-cropped_intrinsics_np[0][2])/cropped_intrinsics_np[0][0]
            crop_pointcloud_y = (rcp_crop-cropped_intrinsics_np[1][2])/cropped_intrinsics_np[1][1] 
            crop_pointcloud_z = rcp_crop

            slice_x = tf.slice(pointcloud_x, offset_tf, cropped_size_tf)
            slice_y = tf.slice(pointcloud_y, offset_tf, cropped_size_tf)
            slice_z = tf.slice(pointcloud_z, offset_tf, cropped_size_tf)

            self.assertAllEqual(slice_x, crop_pointcloud_x)
            self.assertAllEqual(slice_y, crop_pointcloud_y)
            self.assertAllEqual(slice_z, crop_pointcloud_z)
            
if __name__ == '__main__':
    tf.test.main()
