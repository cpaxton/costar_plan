import tensorflow as tf
import keras.backend as K
import random_crop as rcp
import numpy as np
from grasp_geometry import depth_image_to_point_cloud

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

    def test_crop_pointcloud(self):
        """ Test pointcloud use random crop of tensor
        """
        with self.test_session() as sess:
            test_input = np.random.rand(30, 20, 1)
            test_input_size_tf = tf.constant([30, 20, 1])
            intrinsics = np.random.rand(3, 3)
            intrinsics_tf = tf.convert_to_tensor(intrinsics)
            crop_size_np = np.array([5, 4, 1])
            cropped_size_tf = tf.constant([5, 4, 1])

            XYZ_np = depth_image_to_point_cloud(test_input, intrinsics)
            depth_tf = tf.convert_to_tensor(test_input)

            offset_tf = rcp.random_crop_offset(test_input_size_tf, cropped_size_tf)
            rcp_crop_tf = rcp.crop_images(depth_tf, offset_tf, cropped_size_tf)
            cropped_intrinsics_tf = rcp.crop_image_intrinsics(intrinsics_tf, offset_tf)

            cropped_intrinsics_np = sess.run(cropped_intrinsics_tf)
            rcp_crop_np = sess.run(rcp_crop_tf)
            offset_np = sess.run(offset_tf)

            crop_XYZ_np = depth_image_to_point_cloud(rcp_crop_np, cropped_intrinsics_np)

            assert np.allclose(np.squeeze(crop_XYZ_np[:, :, 0]),
                np.squeeze(XYZ_np[offset_np[0]:offset_np[0] + crop_size_np[0],
                                    offset_np[1]:offset_np[1] + crop_size_np[1], 0]))

    def test_crop_pointcloud_np(self):
        """ Test pointcloud use random crop np array directly
        """
        with self.test_session() as sess:
            test_input = np.random.rand(30, 20)
            intrinsics = np.random.rand(3, 3)
            offset_np = np.array([3, 4, 1])
            crop_size_np = np.array([5, 4, 1])

            crop_intrinsics_np = np.copy(intrinsics)
            crop_intrinsics_np[2, 0] -= offset_np[0]
            crop_intrinsics_np[2, 1] -= offset_np[1]

            XYZ_np = depth_image_to_point_cloud(test_input, intrinsics)

            crop_input = test_input[offset_np[0]:offset_np[0]+crop_size_np[0],
                                    offset_np[1]:offset_np[1]+crop_size_np[1]]
            crop_XYZ_np = depth_image_to_point_cloud(crop_input, crop_intrinsics_np)
            crop_XYZ_np = crop_XYZ_np[:, :, 0]
            # Choose Axis 0 in pointcloud
            offset_XYZ_np = XYZ_np[offset_np[0]:offset_np[0]+crop_size_np[0],
                                   offset_np[1]:offset_np[1]+crop_size_np[1], 0]

            assert np.allclose(crop_XYZ_np, offset_XYZ_np)

if __name__ == '__main__':
    tf.test.main()
