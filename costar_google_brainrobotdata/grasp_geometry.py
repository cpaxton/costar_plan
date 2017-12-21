# 3D geometry algorithms for calculating deep learning grasp algorithm input parameters.
#
# Copyright 2017 Andrew Hundt 2017.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
try:
    import eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
    import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
except ImportError:
    print('eigen and sva not available, skipping components utilizing 3D geometry algorithms.'
          'To install run the script at'
          'https://github.com/ahundt/robotics_setup/blob/master/robotics_tasks.sh'
          'or follow the instructions at https://github.com/jrl-umi3218/Eigen3ToPython'
          'and https://github.com/jrl-umi3218/SpaceVecAlg and make sure python bindings'
          'are enabled.')
    eigen = None
    sva = None

try:
    from skimage.draw import circle_perimeter_aa  # Image drawing algorithms http://scikit-image.org
    from skimage.draw import set_color  # Image drawing algorithms http://scikit-image.org
except ImportError:
    circle_perimeter_aa = None
    set_color = None


def vector_quaternion_array_to_ptransform(vector_quaternion_array, q_inverse=True, t_inverse=False, pt_inverse=False):
    """Convert a vector and quaternion pose array to a plucker transform.

    # Params

        vector_quaternion_array: A numpy array containing a pose.
        A pose is a 6 degree of freedom rigid transform represented with 7 values:
        [x, y, z, qx, qy, qz, qw] consisting of a
        vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
        This is the data format used by the google brain robot data grasping dataset's
        tfrecord poses.
        eigen Quaterniond is also ordered xyzw.
        q_inverse: Invert the quaternion before it is loaded into the PTransformd, defaults to True.
            With a PTransform the rotation component must be inverted before loading and after extraction.
            See https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22 for a well tested example.
            See https://github.com/jrl-umi3218/Tasks/issues/10 for a detailed discussion leading to this conclusion.
            The default for q_inverse should be True, do not switch the q_inverse default to False
            without careful consideration and testing, though such a change may be appropriate when
            loading into another transform representation in which the rotation component is expected
            to be inverted.

    # Returns

      A plucker transform PTransformd object as defined by the spatial vector algebra library.
      https://github.com/jrl-umi3218/SpaceVecAlg
      https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
    """
    v = eigen.Vector3d(vector_quaternion_array[:3])
    if t_inverse is True:
        v *= -1
    # TODO(ahundt) use following lines after https://github.com/jrl-umi3218/Eigen3ToPython/pull/15 is fixed
    # qa4 = eigen.Vector4d()
    # q = eigen.Quaterniond(qa4)

    # Quaterniond important coefficient ordering details:
    # scalar constructor is Quaterniond(w,x,y,z)
    # vector constructor is Quaterniond(np.array([x,y,z,w]))
    # Quaterniond.coeffs() is [x,y,z,w]
    # https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html
    xyzw = eigen.Vector4d(vector_quaternion_array[3:])
    q = eigen.Quaterniond(xyzw)

    # The ptransform needs the rotation component to inverted before construction.
    # see https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22 for a well tested example
    # see https://github.com/jrl-umi3218/Tasks/issues/10 for a detailed discussion leading to this conclusion
    if q_inverse is True:
        q = q.inverse()
    pt = sva.PTransformd(q,v)
    if pt_inverse is True:
        pt = pt.inv()
    return pt


def ptransform_to_vector_quaternion_array(ptransform, q_inverse=True, dtype=np.float32):
    """Convert a PTransformD into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)

    # Params

    ptransform: The plucker transform from sva.PTransformd to be converted
    q_inverse: Invert the quaternion after it is extracted, defaults to True.
    With a PTransform the rotation component must be inverted before loading and after extraction.
    See https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22 for a well tested example.
    See https://github.com/jrl-umi3218/Tasks/issues/10 for a detailed discussion leading to this conclusion.
    The default for q_inverse should be True, do not switch the q_inverse default to False
    without careful consideration and testing, though such a change may be appropriate when
    loading into another transform representation in which the rotation component is expected
    to be inverted.
    TODO(ahundt) in process of debugging, correct docstring when all issues are resolved.
    """
    rot = ptransform.rotation()
    quaternion = eigen.Quaterniond(rot)
    if q_inverse:
        quaternion = quaternion.inverse()
    translation = ptransform.translation()
    translation = np.array(translation).reshape(3)
    # coeffs are in xyzw order
    q_floats_array = np.array(quaternion.coeffs())
    vec_quat_7 = np.append(translation, q_floats_array)
    return vec_quat_7.astype(dtype)


def matrix_to_vector_quaternion_array(matrix, inverse=False, verbose=0):
    """Convert a 4x4 Rt transformation matrix into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)

    # Params

        matrix: The 4x4 Rt rigid body transformation matrix to convert into a vector quaternion array.
        inverse: Inverts the full matrix before loading into the array.
            Useful when the transformation to be reversed and for testing/debugging purposes.

    # Returns

      np.array([x, y, z, qx, qy, qz, qw])
    """
    rot = eigen.Matrix3d(matrix[:3, :3])
    quaternion = eigen.Quaterniond(rot)
    translation = matrix[:3, 3].transpose()
    if inverse:
        quaternion = quaternion.inverse()
        translation *= -1
    # coeffs are in xyzw order
    q_floats_array = np.array(quaternion.coeffs())
    # q_floats_array = np.array([quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()]).astype(np.float32)
    vec_quat_7 = np.append(translation, q_floats_array)
    if verbose > 0:
        print(vec_quat_7)
    return vec_quat_7


def matrix_to_ptransform(matrix, q_inverse=True, t_inverse=False, pt_inverse=False):
    """Convert a 4x4 homogeneous transformation matrix into an sva.PTransformd plucker ptransform object.
    """
    vq = matrix_to_vector_quaternion_array(matrix)
    pt = vector_quaternion_array_to_ptransform(vq, q_inverse=q_inverse, t_inverse=t_inverse, pt_inverse=pt_inverse)
    return pt


def vector_to_ptransform(XYZ):
    """Convert (x,y,z) translation to sva.Ptransformd.

    Convert an xyz 3 element translation vector to an sva.PTransformd plucker
    ptransform object with no rotation applied. In other words,
    the rotation component will be the identity rotation.
    """
    q = eigen.Quaterniond()
    q.setIdentity()
    v = eigen.Vector3d(XYZ)
    ptransform = sva.PTransformd(q, v)
    return ptransform


def depth_image_to_point_cloud(depth, intrinsics_matrix, dtype=np.float32):
    """Depth images become an XYZ point cloud in the camera frame with shape (depth.shape[0], depth.shape[1], 3).

    Transform a depth image into a point cloud in the camera frame with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    Based on:
    https://github.com/tensorflow/models/blob/master/research/cognitive_mapping_and_planning/src/depth_utils.py
    https://codereview.stackexchange.com/a/84990/10101

    also see grasp_geometry_tf.depth_image_to_point_cloud().

    # Arguments

      depth: is a 2-D ndarray with shape (rows, cols) containing
          32bit floating point depths in meters. The result is a 3-D array with
          shape (rows, cols, 3). Pixels with invalid depth in the input have
          NaN or 0 for the z-coordinate in the result.
      flip_x: 1.0 leaves data as-is, -1.0 flips the data across the x axis
      flip_y: -1.0 leaves data as-is, -1.0 flips the data across the y axis

      intrinsics_matrix: 3x3 matrix for projecting depth values to z values
      in the point cloud frame. http://ksimek.github.io/2013/08/13/intrinsic/
      In this case x0, y0 are at index [2, 0] and [2, 1], respectively.

      transform: 4x4 Rt matrix for rotating and translating the point cloud
    """
    fy = intrinsics_matrix[1, 1]
    fx = intrinsics_matrix[0, 0]
    # center of image y coordinate
    center_y = intrinsics_matrix[2, 1]
    # center of image x coordinate
    center_x = intrinsics_matrix[2, 0]
    depth = np.squeeze(depth)
    y_range, x_range = depth.shape

    y, x = np.meshgrid(np.arange(y_range),
                       np.arange(x_range),
                       indexing='ij')
    assert y.size == x.size and y.size == depth.size
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()

    X = (x - center_x) * depth / fx
    Y = (y - center_y) * depth / fy

    assert X.size == Y.size and X.size == depth.size
    assert X.shape == Y.shape and X.shape == depth.shape

    print('X np: ', X.shape)
    print('Y np: ', Y.shape)
    print('depth np: ', depth.shape)
    XYZ = np.column_stack([X, Y, depth])
    assert XYZ.shape == (y_range * x_range, 3)
    print('XYZ pre reshape np: ', XYZ.shape)
    XYZ = XYZ.reshape((y_range, x_range, 3))

    return XYZ.astype(dtype)


def surface_relative_transform(xyz_image,
                               camera_intrinsics_matrix,
                               camera_T_endeffector,
                               augmentation_rectangle=None):
    """Get (1) the transform from a depth pixel to a gripper pose, and (2) the corresponding image coordinate.

    Includes optional data augmentation (Not yet implemented).

    # Params

    depth_image:
        width x height x depth image in floating point format, depth in meters
    camera_intrinsics_matrix:
        'camera/intrinsics/matrix33' The 3x3 camera intrinsics matrix.
    camera_T_endeffector:
       PTransformd that takes a point in the endeffector frame and transforms
       it to a point in the camera frame.
    augmentation_rectangle:
       A random offset for the selected (dx, dy) pixel index.
       It will randomly select a pixel in a box around the endeffector coordinate.
       Default (1, 1) has no augmentation.

    # Returns

        [depth_pixel_T_endeffector_final_ptrans, image_coordinate]

        depth_pixel_T_endeffector_final_ptrans is an sva.PTransformd pose from a depth pixel coordinate
        assigned the same orientation as the camera to the position and orientation of the gripper.

        The image_coordinate (dx, dy) is the pixel width, height
        coordinate of the transform in the depth image.

        image_coordinate is used to calculate the point cloud point used for the
        surface relative transform and to generate 2D label weights.
    """
    # xyz coordinate of the endeffector in the camera frame
    XYZ, pixel_coordinate_of_endeffector = endeffector_image_coordinate_and_cloud_point(
        xyz_image, camera_intrinsics_matrix, camera_T_endeffector, augmentation_rectangle)

    # Convert the point cloud point to a transform with identity rotation
    camera_T_cloud_point_ptrans = vector_to_ptransform(XYZ)
    # get the depth pixel to endeffector transform, aka surface relative transform
    depth_pixel_T_endeffector_final_ptrans = camera_T_endeffector * camera_T_cloud_point_ptrans.inv()

    # return the transform and the image coordinate used to generate the transform
    return depth_pixel_T_endeffector_final_ptrans, pixel_coordinate_of_endeffector, camera_T_cloud_point_ptrans


def endeffector_image_coordinate(camera_intrinsics_matrix, xyz):
    """ Get the image coordinate of a point in the camera frame

    This is used to find the [y_height, x_width] image coordinate of the
    end effector given an xyz coordinate in the camera frame.

    """

    # get focal length and camera image center from the intrinsics matrix
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    # center of image x coordinate
    center_x = camera_intrinsics_matrix[2, 0]
    # center of image y coordinate
    center_y = camera_intrinsics_matrix[2, 1]

    # Capital X is horizontal point, right in camera image frame
    # Capital Y is vertical point, up in camera image frame
    # Capital Z is depth in camera frame
    X, Y, Z = xyz
    # x is the image coordinate horizontal axis
    x = (X * fx / Z) + center_x
    # y is the image coordinate vertical axis
    y = (Y * fy / Z) + center_y
    return np.array([y, x])


def endeffector_image_coordinate_and_cloud_point(cartesian_image,
                                                 camera_intrinsics_matrix,
                                                 camera_T_endeffector,
                                                 augmentation_rectangle=None):
    """Get the xyz coordinate of the endeffector in the camera frame as well as its image coordinate.

    cartesian_image: input depth image (Z only) or an XYZ image.
        XYZ images should be a numpy ndarray of size [height, width, 3] containing
        floating point cartesian distance values.
        Depth images should be a numpy ndarray of size [height, width, 1] where each floating point value is a Z distance,
        depth images will automatically be converted to an xyz image using the numpy API.

    # Returns

    [XYZ, pixel_coordinate_of_endeffector]

    xyz: the xyz coordinate of the end effector's point cloud point.
    pixel_coordinate_of_endeffector: the [y_width, x_height] coordinate in the depth image frame of the xyz point cloud point.
    """
    # xyz coordinate of the endeffector in the camera frame
    cte_xyz = camera_T_endeffector.translation()
    # transform the end effector coordinate into the depth image coordinate
    pixel_coordinate_of_endeffector = endeffector_image_coordinate(
        camera_intrinsics_matrix, cte_xyz).astype(np.int32)

    if cartesian_image.shape[-1] == 1 or len(cartesian_image.shape) == 2:
        # cartesian_image is a depth image, convert it to a point cloud xyz image
        cartesian_image = depth_image_to_point_cloud(cartesian_image, camera_intrinsics_matrix)

    # TODO(ahundt) Should we go with a different coordinate or skip training on this data when out of bounds?
    pixel_coordinate_of_endeffector[0] = np.clip(pixel_coordinate_of_endeffector[0], 0, cartesian_image.shape[0]-1)
    pixel_coordinate_of_endeffector[1] = np.clip(pixel_coordinate_of_endeffector[1], 0, cartesian_image.shape[1]-1)
    # The calculation of the gripper pose in the
    # image frame is done with the convention:
    # - X is right in the image frame
    # - Y is up in the image frame
    # - Z is depth
    XYZ = cartesian_image[int(pixel_coordinate_of_endeffector[0]), int(pixel_coordinate_of_endeffector[1]), :]
    return XYZ, pixel_coordinate_of_endeffector


def grasp_dataset_rotation_to_theta(rotation, verbose=0):
    """Convert a rotation to an angle theta specifically for brainrobotdata

    From above, a rotation to the right should be a positive theta,
    and a rotation to the left negative theta. The initial pose is with the
    z axis pointing down, the y axis to the right and the x axis forward.

    This format does not allow for arbitrary rotation commands to be defined,
    and originates from the paper and dataset:
    https://sites.google.com/site/brainrobotdata/home/grasping-dataset
    https://arxiv.org/abs/1603.02199

    In the google brain dataset the gripper is only commanded to
    rotate around a single vertical axis,
    so you might clearly visualize it, this also happens to
    approximately match the vector defined by gravity.
    Furthermore, the original paper had the geometry of the
    arm joints on which params could easily be extracted,
    which is not available here. To resolve this discrepancy
    Here we assume that the gripper generally starts off at a
    quaternion orientation of approximately [qx=-1, qy=0, qz=0, qw=0].
    This is equivalent to the angle axis
    representation of [a=np.pi, x=-1, y=0, z=0],
    which I'll name default_rot.

    It is also important to note the ambiguity of the
    angular distance between any current pose
    and the end pose. This angular distance will
    always have a positive value so the network
    could not naturally discriminate between
    turning left and turning right.
    For this reason, we use the angular distance
    from default_rot to define the input angle parameter,
    and if the angle axis x axis component is > 0
    we will use theta for rotation,
    but if the angle axis x axis component is < 0
    we will use -theta.
    """
    aa = eigen.AngleAxisd(rotation)
    theta = aa.angle()
    if aa.axis().z() < 0:
        multiply = 1.0
    else:
        multiply = -1.0
    if verbose > 0:
        print("ANGLE_AXIS_MULTIPLY: ", aa.angle(), np.array(aa.axis()), multiply)
    theta *= multiply
    return theta


def grasp_dataset_ptransform_to_vector_sin_theta_cos_theta(ptransform, dtype=np.float32):
    """Plucker transform to [dx, dy, dz, sin(theta), cos(theta)]

    Convert a PTransform 3D Rigid body transform into a numpy array with 5 total entries,
    including a 3 entry translation vector and 2 entries for
    a single rotation angle theta containing sin(theta), cos(theta).
    These are the x and y coordinates on the unit circle defining the change in
    gripper angle theta, see https://en.wikipedia.org/wiki/Unit_circle.

    This format does not allow for arbitrary rotation commands to be defined,
    and originates from the paper and dataset:
    https://sites.google.com/site/brainrobotdata/home/grasping-dataset
    https://arxiv.org/abs/1603.02199

    In the google brain dataset the gripper is only commanded to
    rotate around a single vertical axis,
    so you might clearly visualize it, this also happens to
    approximately match the vector defined by gravity.
    Furthermore, the original paper had the geometry of the
    arm joints on which params could easily be extracted,
    which is not available here. To resolve this discrepancy
    Here we assume that the gripper generally starts off at a
    quaternion orientation of approximately [qx=-1, qy=0, qz=0, qw=0].
    This is equivalent to the angle axis
    representation of [a=np.pi, x=-1, y=0, z=0],
    which I'll name default_rot.

    It is also important to note the ambiguity of the
    angular distance between any current pose
    and the end pose. This angular distance will
    always have a positive value so the network
    could not naturally discriminate between
    turning left and turning right.
    For this reason, we use the angular distance
    from default_rot to define the input angle parameter,
    and if the angle axis x axis component is > 0
    we will use [sin(theta), cos(theta)] for rotation,
    but if the angle axis x axis component is < 0
    we will use [sin(-theta), cos(-theta)].

    Also note that in the eigen API being used:
    e.AngleAxisd(e.Quaterniond().Identity()) == [a=0.0, x=1, y=0, z=0]

    # Params

    ptransform: the PTransformd to convert

    # Returns

    vector_sin_theta_cos_theta in format [dx, dy, dz, sin(theta), cos(theta)]
    """
    translation = np.squeeze(ptransform.translation())
    theta = grasp_dataset_rotation_to_theta(ptransform.rotation())
    sin_cos_theta = np.array([np.sin(theta), np.cos(theta)])
    vector_sin_theta_cos_theta = np.concatenate([translation, sin_cos_theta])
    vector_sin_theta_cos_theta = vector_sin_theta_cos_theta.astype(dtype)
    return vector_sin_theta_cos_theta


def grasp_dataset_to_ptransform(camera_T_base, base_T_endeffector):
    """Convert brainrobotdata features camera_T_base and base_T_endeffector to camera_T_endeffector and ptransforms.

    This specific function exists because it accepts the raw feature types
    defined in the google brain robot grasping dataset.

    # Params

    camera_T_base: a vector quaternion array
    base_T_endeffector: a 4x4 homogeneous 3D transformation matrix

    # Returns

      PTransformd formatted transforms:
      camera_T_endeffector_ptrans, base_T_endeffector_ptrans, base_T_camera_ptrans
    """
    base_T_endeffector_ptrans = vector_quaternion_array_to_ptransform(base_T_endeffector)
    # In this case camera_T_base is a transform that takes a point in the base
    # frame of reference and transforms it to the camera frame of reference.
    camera_T_base_ptrans = matrix_to_ptransform(camera_T_base)
    base_T_camera = camera_T_base_ptrans.inv()
    # Perform the actual transform calculation
    camera_T_endeffector_ptrans = base_T_endeffector_ptrans * base_T_camera.inv()
    return camera_T_endeffector_ptrans, base_T_endeffector_ptrans, base_T_camera


def current_endeffector_to_final_endeffector_feature(base_T_endeffector_current,
                                                     base_T_endeffector_final,
                                                     feature_type='vec_sin_cos_5',
                                                     dtype=np.float32):
    """Calculate the ptransform between two poses in the same base frame.

    A pose is a 6 degree of freedom rigid transform represented with 7 values:
    vector (x, y, z) and quaternion (x, y, z, w).
    A pose is always annotated with the target and source frames of reference.
    For example, base_T_camera is a transform that takes a point in the camera
    frame of reference and transforms it to the base frame of reference.

    We will be dealing with, for example:
    grasp/4/reached_pose/transforms/base_T_endeffector/vec_quat_7
    grasp/10/commanded_pose/transforms/base_T_endeffector/vec_quat_7

    # Params

    base_T_endeffector_current: A vector quaternion array from a base frame to an end effector frame at the current time step.
    base_T_endeffector_final: A vector quaternion array from a base frame to an end effector frame at the final time step.
    feature_type: String identifying the feature type to return, which should contain one of the following options:
       'vec_quat_7' A numpy array with 7 total entries including a 3 entry translation vector and 4 entry quaternion.
       'vec_sin_cos_5'  A numpy array with 5 total entries [dx, dy, dz, sin(theta), cos(theta)]
                        including a 3 entry translation vector and 2 entries for the angle of the rotation.
                       for a single rotation angle theta containing sin(theta), cos(theta). This format
                       does not allow for arbitrary commands to be defined, and originates from the paper and dataset:
                       https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                       https://arxiv.org/abs/1603.02199

                       see also: grasp_dataset_ptransform_to_vector_sin_theta_cos_theta()
         'sin_cos_2' A numpy array with 2 total entries [sin(theta), cos(theta)]] for the angle of the rotation.

    # Returns

    A numpy array or object of the type specified in the feature_type parameter.

    """
    base_to_current = vector_quaternion_array_to_ptransform(base_T_endeffector_current)
    base_to_end = vector_quaternion_array_to_ptransform(base_T_endeffector_final)
    # endeffector_current_to_endeffector_final is abbreviated eectf
    eectf = base_to_end * base_to_current.inv()

    # we have ptransforms for both data, now get transform from current to commanded
    if 'vec_quat_7' in feature_type:
        eectf = ptransform_to_vector_quaternion_array(eectf)
    elif 'vec_sin_cos_5' in feature_type:
        eectf, _ = grasp_dataset_ptransform_to_vector_sin_theta_cos_theta(eectf)
    elif 'sin_cos_2' in feature_type:
        eectf, _ = grasp_dataset_ptransform_to_vector_sin_theta_cos_theta(eectf)[-2:]
    else:
        raise ValueError('current_endeffector_to_final_endeffector_feature() '
                         'received unsupported feature type: ' + str(feature_type))
    return eectf.astype(dtype)


def grasp_dataset_to_transforms_and_features(
        cartesian_image,
        camera_intrinsics_matrix,
        camera_T_base,
        base_T_endeffector_current,
        base_T_endeffector_final,
        augmentation_rectangle=None,
        dtype=np.float32):
    """Extract transforms and features necessary to train from the grasping dataset.

    This specific function exists because it accepts the raw feature types
    defined in the google brain robot grasping dataset.

    Includes optional data augmentation. TODO(ahundt) augmentation not implemented
    Also gets the surface relative transform from a depth pixel to a
    gripper pose from data in the brain robot data feature formats

    # Arguments

    cartesian_image: input depth image (Z only) or an XYZ image.
        XYZ images should be a numpy ndarray of size [height, width, 3] containing
        floating point cartesian distance values.
        Depth images should be a numpy ndarray of size [height, width, 1] where each floating point value is a Z distance,
        depth images will automatically be converted to an xyz image using the numpy API.
    camera_intrinsics_matrix:
        'camera/intrinsics/matrix33' The 3x3 camera intrinsics matrix.
    camera_T_base:
        'camera/transforms/camera_T_base/matrix44'
        4x4 transformation matrix from the camera center to the robot base.
        camera_T_base is a transform that takes a point in the base
        frame of reference and transforms it to the camera frame of reference.
    base_T_endeffector_current: A vector quaternion array from a base frame to an end effector frame
        vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
        base_T_endeffector is a transform that takes a point in the endeffector
        frame of reference and transforms it to the base frame of reference at the current time step
        of the grasp attempt move_to_grasp sequence.
    base_T_endeffector_final: A vector quaternion array from a base frame to an end effector frame
        containing the proposed destination of the robot at the final time step of the grasp attempt
        move_to_grasp sequence.

    augmentation_rectangle:
       A random offset for the selected (dx, dy) pixel index.
       It will randomly select a pixel in a box around the endeffector coordinate.
       Default (1, 1) has no augmentation.

    # Returns

        [current_base_T_camera_vec_quat_7_array,
            eectf_vec_quat_7_array,
            camera_T_endeffector_current_vec_quat_7_array,
            camera_T_endeffector_final_vec_quat_7_array,
            depth_pixel_T_endeffector_current_vec_quat_7_array,
            image_coordinate_current,
            depth_pixel_T_endeffector_final_vec_quat_7_array,
            image_coordinate_final,
            sin_cos_2,
            vec_sin_cos_5,
            delta_depth_sin_cos_3,
            delta_depth_quat_5]

        Which will be in the np.float32 numpy format (or their tf equivalents).
        Note: endeffector_current_to_endeffector_final is abbreviated eectf.

        end_surface_relative_transform_pose is a numpy array pose [x, y, z, qx, qy, qz, qw],
        relative to the final time step  which contains:
           - vector (x, y, z) for cartesian motion
           - quaternion (qx, qy, qz, qw) for rotation

        The image_coordinate (dx, dy) is the pixel width, height
        coordinate of the transform in the depth image.

        image_coordinate is used to calculate the point cloud point used for the
        surface relative transform and to generate 2D label weights.


        'delta_depth_sin_cos_3' [delta_depth, sin(theta), cos(theta)] where delta_depth depth offset for the gripper
            from the measured surface, alongside a single rotation angle theta containing sin(theta), cos(theta).
            This format does not allow for arbitrary commands to be defined, and the rotation component
            is based on the paper and dataset:
                        https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                        https://arxiv.org/abs/1603.02199
        'delta_depth_quat_5' A numpy array with 5 total entries including depth offset and 4 entry quaternion.
            (Not yet implemented)
        'vec_quat_7' A numpy array with 7 total entries including a 3 entry translation vector and 4 entry quaternion.
            (Not yet implemented)
        'vec_sin_cos_5'  A numpy array with 5 total entries [dx, dy, dz, sin(theta), cos(theta)]
                        including a 3 entry translation vector and 2 entries for the angle of the rotation.
                        for a single rotation angle theta containing sin(theta), cos(theta). This format
                        does not allow for arbitrary commands to be defined, and originates from the paper and dataset:
                        https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                        https://arxiv.org/abs/1603.02199

                        see also: grasp_dataset_ptransform_to_vector_sin_theta_cos_theta()
    """
    # print('in grasp_dataset_to_transforms_and_features 0')
    # Get input transforms relative to the current time step
    (camera_T_endeffector_current_ptrans, base_T_endeffector_current_ptrans,
     current_base_T_camera_ptrans) = grasp_dataset_to_ptransform(camera_T_base, base_T_endeffector_current)
    # get input transforms relative to the final time step when the gripper closes
    (camera_T_endeffector_final_ptrans, base_T_endeffector_final_ptrans,
     final_base_T_camera_ptrans) = grasp_dataset_to_ptransform(camera_T_base, base_T_endeffector_final)

    # endeffector_current_to_endeffector_final is abbreviated eectf.
    # Get current time to end time gripper position.
    # This is the pose transform defined by the gripper position at the following time frames:
    #     current time -> end time
    #
    # This is the same operation as current_endeffector_to_final_endeffector_feature().
    eectf_ptrans = base_T_endeffector_final_ptrans * base_T_endeffector_current_ptrans.inv()
    # print('in grasp_dataset_to_transforms_and_features 1')

    if cartesian_image.shape[-1] == 1 or len(cartesian_image.shape) == 2:
        # cartesian_image is a depth image, convert it to a point cloud xyz image
        cartesian_image = depth_image_to_point_cloud(cartesian_image, camera_intrinsics_matrix)

    # calculate the surface relative transform from the clear view depth to endeffector final position
    depth_pixel_T_endeffector_current_ptrans, image_coordinate_current, camera_T_depth_pixel_current_ptrans = surface_relative_transform(
        cartesian_image,
        camera_intrinsics_matrix,
        camera_T_endeffector_current_ptrans,
        augmentation_rectangle)
    # print('in grasp_dataset_to_transforms_and_features 2')

    # get the delta depth offset
    # TODO(ahundt) verify that z correctly reflects the depth offset
    delta_depth_current = depth_pixel_T_endeffector_current_ptrans.translation().z()
    # print('in grasp_dataset_to_transforms_and_features 3, delta_depth_current: ', delta_depth_current)

    # calculate the surface relative transform from the clear view depth to endeffector final position
    depth_pixel_T_endeffector_final_ptrans, image_coordinate_final, camera_T_depth_pixel_final_ptrans = surface_relative_transform(
        cartesian_image,
        camera_intrinsics_matrix,
        camera_T_endeffector_final_ptrans,
        augmentation_rectangle)

    # get the delta depth offset
    # TODO(ahundt) verify that z correctly reflects the depth offset
    delta_depth_final = np.array(depth_pixel_T_endeffector_final_ptrans.translation().z(), dtype=dtype)
    # print('in grasp_dataset_to_transforms_and_features 4, delta_depth_final: ', delta_depth_final)

    # Get the delta theta parameter, converting Plucker transform to [dx, dy, dz, sin(theta), cos(theta)]
    # Also see grasp_dataset_ptransform_to_vector_sin_theta_cos_theta()
    eectf_translation = np.squeeze(eectf_ptrans.translation())
    eectf_theta = grasp_dataset_rotation_to_theta(eectf_ptrans.rotation())
    # print('in grasp_dataset_to_transforms_and_features 5, eectf_theta', eectf_theta)
    eectf_sin_theta = np.sin(eectf_theta)
    eectf_cos_theta = np.cos(eectf_theta)
    # print('in grasp_dataset_to_transforms_and_features 5 eectf_sin_theta:', eectf_sin_theta, ' eectf_cos_theta', eectf_cos_theta)

    # Convert each transform into vector + quaternion format
    # [x, y, z, qx, qy, qz, qw], which is identical to the 'vec_quat_7' feature type
    current_base_T_camera_vec_quat_7_array = ptransform_to_vector_quaternion_array(current_base_T_camera_ptrans, dtype=dtype)
    eectf_vec_quat_7_array = ptransform_to_vector_quaternion_array(eectf_ptrans, dtype=dtype)
    camera_T_endeffector_current_vec_quat_7_array = ptransform_to_vector_quaternion_array(camera_T_endeffector_current_ptrans, dtype=dtype)
    camera_T_depth_pixel_current_vec_quat_7_array = ptransform_to_vector_quaternion_array(camera_T_depth_pixel_current_ptrans, dtype=dtype)
    camera_T_endeffector_final_vec_quat_7_array = ptransform_to_vector_quaternion_array(camera_T_endeffector_final_ptrans, dtype=dtype)
    camera_T_depth_pixel_final_vec_quat_7_array = ptransform_to_vector_quaternion_array(camera_T_depth_pixel_final_ptrans, dtype=dtype)
    depth_pixel_T_endeffector_current_vec_quat_7_array = ptransform_to_vector_quaternion_array(depth_pixel_T_endeffector_current_ptrans, dtype=dtype)
    depth_pixel_T_endeffector_final_vec_quat_7_array = ptransform_to_vector_quaternion_array(depth_pixel_T_endeffector_final_ptrans, dtype=dtype)
    # print('in grasp_dataset_to_transforms_and_features 6')

    # [x, y] image coordinate of the final gripper position gripper in the camera image
    image_coordinate_current = image_coordinate_current.astype(dtype)
    image_coordinate_final = image_coordinate_final.astype(dtype)
    # print('in grasp_dataset_to_transforms_and_features 7:')

    # [cte_sin_theta, cte_cos_theta]
    sin_cos_2 = np.array([eectf_sin_theta, eectf_cos_theta], dtype=dtype)
    # print('in grasp_dataset_to_transforms_and_features 8 sin_cos_2:', sin_cos_2)
    # [cte_dx, cte_dy, cte_dz, eectf_sin_theta, eectf_cos_theta] vec_sin_cos_5, the levine 2016 'params' feature format.
    vec_sin_cos_5 = np.concatenate([eectf_translation, sin_cos_2]).astype(dtype)
    # print('in grasp_dataset_to_transforms_and_features 9')
    # [delta_depth_final, sin_theta, cos_theta]
    delta_depth_sin_cos_3 = np.concatenate([[delta_depth_final], sin_cos_2]).astype(dtype)
    # print('in grasp_dataset_to_transforms_and_features 10')
    # [delta_depth_final, qx, qy, qz, qw]
    delta_depth_quat_5 = np.concatenate([[delta_depth_final], depth_pixel_T_endeffector_final_vec_quat_7_array[-4:]]).astype(dtype)
    # print('in grasp_dataset_to_transforms_and_features 11')

    return [current_base_T_camera_vec_quat_7_array,
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
            delta_depth_quat_5]


def vector_quaternion_arrays_allclose(vq1, vq2, rtol=1e-6, atol=1e-6, verbose=0):
    """Check if all the entries are close for two vector quaternion numpy arrays.

    Quaterions are a way of representing rigid body 3D rotations that is more
    numerically stable and compact in memory than other methods such as a 3x3
    rotation matrix.

    This special comparison function is needed because for quaternions q == -q.
    Vector Quaternion numpy arrays are expected to be in format
    [x, y, z, qx, qy, qz, qw].

    # Params

    vq1: First vector quaternion array to compare.
    vq2: Second vector quaternion array to compare.
    rtol: relative tolerance.
    atol: absolute tolerance.

    # Returns

    True if the transforms are within the defined tolerance, False otherwise.
    """
    vq1 = np.array(vq1)
    vq2 = np.array(vq2)
    q3 = np.array(vq2[3:])
    q3 *= -1.
    v3 = vq2[:3]
    vq3 = np.array(np.concatenate([v3, q3]))
    comp12 = np.allclose(np.array(vq1), np.array(vq2), rtol=rtol, atol=atol)
    comp13 = np.allclose(np.array(vq1), np.array(vq3), rtol=rtol, atol=atol)
    if verbose > 0:
        print(vq1)
        print(vq2)
        print(vq3)
        print(comp12, comp13)
    return comp12 or comp13


def gaussian_kernel_2D(size=(3, 3), center=(1, 1), sigma=1):
    """Create a 2D gaussian kernel with specified size, center, and sigma.

    Output with the default parameters:

        [[ 0.36787944  0.60653066  0.36787944]
         [ 0.60653066  1.          0.60653066]
         [ 0.36787944  0.60653066  0.36787944]]

    references:

            https://stackoverflow.com/a/43346070/99379
            https://stackoverflow.com/a/32279434/99379

    To normalize:

        g = gaussian_kernel_2d()
        g /= np.sum(g)
    """
    xx, yy = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    kernel = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2. * sigma ** 2))
    return kernel


def draw_circle(image, coordinate, color=None, radius=10):
    """ Draw a circle at the [y_height, x_width] coordinate of an image.

        Requires scikit-image, http://scikit-image.org
        If it is not available this function is a no-op.
    """
    if circle_perimeter_aa is not None and set_color is not None:
        image_shape_len = len(image.shape)
        if image_shape_len == 4:
            batch, y_height, x_width, channels = image.shape
        elif image_shape_len == 3 and image.shape[0] == 1:
            batch, y_height, x_width = image.shape
            channels = 1
        elif image_shape_len == 3 and image.shape[2] == 1:
            y_height, x_width, channels = image.shape
            batch = 1
        elif image_shape_len == 2:
            y_height, x_width = image.shape
            batch = 1
            channels = 1

        if color is None:
            if channels == 1:
                color = [255]
            else:
                color = [0, 255, 255]
        image = np.squeeze(image)
        y, x = np.array(coordinate, dtype=np.int32)
        # please note that skimage uses a funky coordinate system:
        # origin in the top left with coordinates ordered
        # (y, x) where
        # +y is down from the top left corner and
        # +x is right from the top left corner
        rr, cc, aa = circle_perimeter_aa(y, x, radius, shape=image.shape)
        set_color(image, (rr, cc), color, alpha=aa)
        # axs.imshow(np.squeeze(frame))
        if image_shape_len > len(image.shape):
            # return the image to the shape that was provided
            image = np.expand_dims(image, axis=0)

    return image
