"""
Copyright 2017 Andrew Hundt, Philipp Jund, Kieran Wynn 2017.

Based on:
https://github.com/PhilJd/tf-quaternion
https://github.com/KieranWynn/pyquaternion

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This small library implements quaternion operations with tensorflow.
All operations are derivable.

"""
import numpy as np
import tensorflow as tf
import math

def q_matrix(q):
    """ Matrix representation of quaternion xyzw for multiplication purposes.
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see the following links for reference code:
    # http://kieranwynn.github.io/pyquaternion/
    # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
    # https://github.com/moble/quaternion
    # https://github.com/ClementPinard/SfmLearner-Pytorch
    # https://github.com/bistromath/gr-air-modes/blob/master/python/Quaternion.py
    """
    return tf.convert_to_tensor([
        [q[:, 3], -q[:, 0], -q[:, 1], -q[:, 2]],
        [q[:, 0],  q[:, 3], -q[:, 2],  q[:, 1]],
        [q[:, 1],  q[:, 2],  q[:, 3], -q[:, 0]],
        [q[:, 2], -q[:, 1],  q[:, 0],  q[:, 3]]])

def q_bar_matrix(q):
    """Matrix representation of quaternion xyzw for multiplication purposes.
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py and https://github.com/moble/quaternion for reference.
    """
    return tf.convert_to_tensor([
        [q[:, 3], -q[:, 0], -q[:, 1], -q[:, 2]],
        [q[:, 0],  q[:, 3],  q[:, 2], -q[:, 1]],
        [q[:, 1], -q[:, 2],  q[:, 3],  q[:, 0]],
        [q[:, 2],  q[:, 1], -q[:, 0],  q[:, 3]]])

def q_norm(q):
    """ Get the scalar norm length of a batch of quaternion.
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py and https://github.com/moble/quaternion for reference.
    """
    return tf.linalg.norm(q, axis=1)

def q_log(q, epsilon=1e-14):
    """ Logarithm of a batch of quaternions.
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py and https://github.com/moble/quaternion for reference.
    """
    # https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/norm
    v_norm = tf.linalg.norm(q[:, :3], axis=1)
    quat_norm = q_norm(q)
    # assuming not a 0 quaternion
    vec = q[:, :3] / v_norm
    # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L645
    # https://math.stackexchange.com/a/2554/80218
    # TODO(ahundt) for each quaternion, if v_norm < tolerance, the quaternion is real only quaternion should be tf.concatenate(tf.log(quat_norm[:]) tf.zeros(tf.shape(q)[0], 3)
    return K.stack([tf.log(quat_norm[:, 0]), tf.acos(q[:, 0] / tf.max(quat_norm[:, 0], epsilon)) * vec], axis=-1)

def q_exp(q, epsilon=1e-12):
    """
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py.
    """
    # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L622
    # https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/norm
    v_norm = tf.linalg.norm(q[:, :3], axis=1)
    quat_norm = q_norm(q)
    vec = q[:, :3] / tf.max(quat_norm[:, 0], epsilon)
    magnitude = tf.exp(q[:, 0])
    return K.stack([magnitude * tf.cos(v_norm), magnitude * tf.sin(v_norm) * vec[:]])

def q_sym_log_map(q0, q1):
    """
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py.

    # Note
        Information on the symmetrized formulations given in [Source](https://www.researchgate.net/publication/267191489_Riemannian_L_p_Averaging_on_Lie_Group_of_Nonzero_Quaternions
    """
    # collector quaternion format: xyzw
    # pyquaternion format: wxyz
    inv_sqrt_q = (q ** (-0.5))
return Quaternion.log(inv_sqrt_q * p * inv_sqrt_q)

def quaternion_symmetrized_distance(q0, q1):
    """
    Expects batch of quaternions in xyzw format of shape (n, 4).
    Note that this function and ROS quaternions use xyzw order, other code in this file is wxyz.

    TODO(ahundt) WORK IN PROGRESS, not complete or tested, see # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py.
    """
    return K.l2_normalize(q, axis=1)

def quat_scope(func, *args, **kwargs):
    def scoped_func(*args, **kwargs):
        with tf.name_scope("quat_{}".format(func.__name__)):
            return func(*args, **kwargs)
    return scoped_func


@quat_scope
def point_to_quaternion():
    raise NotImplementedError()


@quat_scope
def from_rotation_matrix():
    raise NotImplementedError()


@quat_scope
def multiply(a, b):
    if isinstance(a, Quaternion) and isinstance(b, Quaternion):
        w1, x1, y1, z1 = tf.unstack(a.value())
        w2, x2, y2, z2 = tf.unstack(b.value())
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return Quaternion(tf.stack((w, x, y, z)))

    elif(isinstance(a, DualQuaternion) and isinstance(b, DualQuaternion)):
        real = a.real() * b.real()
        dual = Quaternion((a.real() * b.dual()).coeffs() + (a.dual() * b.real()).coeffs())
        return DualQuaternion(real_wxyz=real, dual_wxyz=dual)
    else:
        raise NotImplementedError("Multiplication is currently only implemented "
                                  "for Quaternion * Quaternion and DualQuaternion * DualQuaternion")


@quat_scope
def divide(a, b):
    if not isinstance(a, Quaternion) and not isinstance(b, Quaternion):
        msg = "Division is currently only implemented " \
              "for Quaternion \ Quaternion"
        raise NotImplementedError(msg)
    w1, x1, y1, z1 = tf.unstack(a.value())
    w2, x2, y2, z2 = tf.unstack(b.value())
    bnorm = b._squared_norm()
    w = (w1*w2 + x1*x2 + y1*y2 + z1*z2) / bnorm,
    x = (-w1*x2 + x1*w2 - y1*z2 + z1*y2) / bnorm,
    y = (-w1*y2 + x1*z2 + y1*w2 - z1*x2) / bnorm,
    z = (-w1*z2 - x1*y2 + y1*x2 + z1*w2) / bnorm
    return Quaternion(tf.stack((w, x, y, z)))


class Quaternion():

    def __init__(self, wxyz=(1.0, 0.0, 0.0, 0.0), dtype=tf.float32):
        """
        Args:
            wxyz: The values for w, x, y, z. Must have shape=[4].
                - `tf.Tensor` or `tf.Variable` of type float16/float32/float64
                - list/tuple/np.array
                - Quaternion
                Defaults to (1.0, 0.0, 0.0, 0.0)
            dtype: The type to create the value tensor. Only considered if
                wxyz is passed as list, tuple or np.array. Otherwise
                the type of the `Tensor` is used. This is to prevent silent
                changes to the gradient type resulting in a different precision
                Allowed types are float16, float32, float64.

        Returns:
            A Quaternion.

        Raises:
            ValueError, if the shape of wxyz is not [4].
            TypeError, either if the `Tensor` wxyz's type is not float
                or if wxyz is not a Tensor/list/tuple etc.
        """
        if isinstance(wxyz, (tf.Tensor, tf.Variable)):
            self._validate_shape(wxyz)
            self._validate_type(wxyz)
            self._q = (wxyz if wxyz.dtype == dtype
                       else tf.cast(wxyz, dtype))
        elif isinstance(wxyz, (np.ndarray, list, tuple)):
            self._validate_shape(wxyz)
            self._q = tf.constant(wxyz, dtype=dtype)
        elif isinstance(wxyz, Quaternion):
            tensor = wxyz.value()
            self._q = (tensor if tensor.dtype == dtype
                       else tf.cast(tensor, dtype))
        else:
            raise TypeError("Can not convert object of type {} to Quaternion"
                            "".format(type(wxyz)))

    @staticmethod
    def zeros(dtype=tf.float32):
        """Get a quaternion where all coefficients are 0.
        """
        return Quaternion(wxyz=(0.0, 0.0, 0.0, 0.0), dtype=dtype)

    @staticmethod
    def identity(dtype=tf.float32):
        """Get the identity unit quaternion where coefficients wxyz are (1,0,0,0).
        """
        return Quaternion(wxyz=(1.0, 0.0, 0.0, 0.0), dtype=dtype)

    def value(self):
        """ Returns a `Tensor` which holds the value of the quaternion. Note
            that this does not return a reference, so you can not alter the
            quaternion through this.
        """
        return self._q

    def imaginary(self):
        """Returns a Tensor containing [x, y, z] the imaginary part of the quaternion.
        """
        return self._q[1:]

    def real(self):
        """Returns a Tensor containing w, the real part of the quaternion.
        """
        return self._q[0]

    def w(self):
        """Returns a Tensor containing w, the real part of the quaternion.
        """
        return self._q[0]

    @property
    def conjugate(self):
        """Quaternion conjugate, encapsulated in a new instance.
        For a unit quaternion, this is the same as the inverse.
        Returns:
            A new Quaternion object clone with its vector part negated
        """
        q = tf.stack([self._q[0], -self._q[1:]])
        return self.__class__(wxyz=q)

    def _q_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return tf.stack([
            [self._q[0], -self._q[1], -self._q[2], -self._q[3]],
            [self._q[1],  self._q[0], -self._q[3],  self._q[2]],
            [self._q[2],  self._q[3],  self._q[0], -self._q[1]],
            [self._q[3], -self._q[2],  self._q[1],  self._q[0]]], axis=1)

    def _q_bar_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return tf.stack([
            [self._q[0], -self._q[1], -self._q[2], -self._q[3]],
            [self._q[1],  self._q[0],  self._q[3], -self._q[2]],
            [self._q[2], -self._q[3],  self._q[0],  self._q[1]],
            [self._q[3],  self._q[2], -self._q[1],  self._q[0]]], axis=1)

    @property
    def rotation_matrix(self):
        """Get the 3x3 rotation matrix equivalent of the quaternion rotation.

        Returns:
            A 3x3 orthogonal rotation matrix as a 3x3 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalize the Quaternion object to a unit quaternion if it is not already one.

        """
        self.normalize()
        product_matrix = tf.tensordot(self._q_matrix(), tf.transpose(tf.conj(self._q_bar_matrix())))
        return product_matrix[1:][:, 1:]

    @property
    def transformation_matrix(self, t=tf.zeros([1, 3])):
        """Get the 4x4 homogeneous transformation matrix equivalent of the quaternion rotation.

           t: Translation component of the transformation matrix, defaults to identity translation `[0, 0, 0]^t`.

        Returns:
            A 4x4 homogeneous transformation matrix as a 4x4 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalize the Quaternion object to a unit quaternion if it is not already one.
        """
        Rt = tf.stack([self.rotation_matrix(), t])
        return tf.stack([Rt, tf.constant([0.0, 0.0, 0.0, 1.0])], axis=1)

    @property
    def yaw_pitch_roll(self):
        """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention

        Returns:
            yaw:    rotation angle around the z-axis in radians, in the range `[-pi, pi]`
            pitch:  rotation angle around the y'-axis in radians, in the range `[-pi/2, -pi/2]`
            roll:   rotation angle around the x''-axis in radians, in the range `[-pi, pi]`

        The resulting rotation_matrix would be R = R_x(roll) R_y(pitch) R_z(yaw)

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalize the Quaternion object to a unit quaternion if it is not already one.
        """

        self.normalize()
        yaw = tf.atan2(2*(self._q[0]*self._q[3] - self._q[1]*self._q[2]),
                       1 - 2*(self._q[2]**2 + self._q[3]**2))
        pitch = tf.asin(2*(self._q[0]*self._q[2] + self._q[3]*self._q[1]))
        roll = tf.atan2(2*(self._q[0]*self._q[1] - self._q[2]*self._q[3]),
                        1 - 2*(self._q[1]**2 + self._q[2]**2))

        return yaw, pitch, roll

    def _wrap_angle(self, theta):
        """Helper method: Wrap any angle to lie between -pi and pi


        """
        result = ((theta + math.pi) % (2 * math.pi)) - math.pi
        # Odd multiples of pi were wrapped to +pi (as opposed to -pi)
        # if result == -math.pi: result = math.pi
        return result

    def get_axis(self, undefined=tf.zeros(3)):
        """Get the axis or vector about which the quaternion rotation occurs

        For a null rotation (a purely real quaternion), the rotation angle will
        always be `0`, but the rotation axis is undefined.
        It is by default assumed to be `[0, 0, 0]`.

        Params:
            undefined: [optional] specify the axis vector that should define a null rotation.
                This is geometrically meaningless, and could be any of an infinite set of vectors,
                but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.

        Returns:
            A Numpy unit 3-vector describing the Quaternion object's axis of rotation.

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalize the Quaternion object to a unit quaternion if it is not already one.
        """
        tolerance = 1e-17
        self.normalize()
        norm = tf.norm(self._q[1:])
        if norm < tolerance:
            # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
            return undefined
        else:
            return self._q[1:] / norm

    @property
    def axis(self):
        return self.get_axis()

    @property
    def angle(self):
        """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis.

        This is guaranteed to be within the range (-pi:pi) with the direction of
        rotation indicated by the sign.

        When a particular rotation describes a 180 degree rotation about an arbitrary
        axis vector `v`, the conversion to axis / angle representation may jump
        discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`,
        each being geometrically equivalent (see Note in documentation).

        Returns:
            A real number in the range (-pi:pi) describing the angle of rotation
                in radians about a Quaternion object's axis of rotation.

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalize the Quaternion object to a unit quaternion if it is not already one.
        """
        self.normalize()
        norm = tf.norm(self._q[1:])
        return self._wrap_angle(2.0 * tf.atan2(norm, self._q[0]))

    @staticmethod
    def _quaternions_to_tensors(quats):
        return [q.value() if isinstance(q, Quaternion) else q for q in quats]

    def __add__(self, b):
        val_a, val_b = Quaternion._quaternions_to_tensors((self, b))
        return Quaternion(val_a + val_b)

    def __sub__(self, b):
        val_a, val_b = Quaternion._quaternions_to_tensors((self, b))
        return Quaternion(val_a - val_b)

    def __mul__(self, b):
        return multiply(self, b)

    def __imul__(self, other):
        if isinstance(other, Quaternion):
            return multiply(self, other)
        else:
            raise NotImplementedError("Quaternion Multiplication not implemented for this type.")

    def __div__(self, b):
        return divide(self, b)

    def __idiv__(self, other):
        if isinstance(other, Quaternion):
            return divide(self, other)
        else:
            raise NotImplementedError("Quaternion Division not implemented for this type.")

    def __repr__(self):
        return "<tfq.Quaternion ({})>".format(self._q.__repr__()[1:-1])

    @quat_scope
    def coeffs(self):
        return self._q

    @quat_scope
    def inverse(self):
        w, x, y, z = tf.unstack(tf.divide(self._q, self._squared_norm()))
        return Quaternion((w, -x, -y, -z))

    @quat_scope
    def normalized(self):
        """Copies this quaternion, normalizes it, then returns the normalized version.
        """
        return self.__class__(tf.divide(self._q, self._abs()))

    @quat_scope
    def normalize(self):
        """Normalizes this quaternion object.
        """
        self._q = tf.divide(self._q, self._abs())

    @quat_scope
    def as_rotation_matrix(self):
        """Calculates the rotation matrix. See
        [http://www.euclideanspace.com/maths/geometry/rotations/
         conversions/quaternionToMatrix/]

        Returns:
            A 3x3 `Tensor`, the rotation matrix

        """
        # helper functions
        def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
            return 1 - 2 * tf.pow(a, 2) - 2 * tf.pow(b, 2)

        def tr_add(a, b, c, d):  # computes triangle entries with addition
            return 2 * a * b + 2 * c * d

        def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
            return 2 * a * b - 2 * c * d

        w, x, y, z = tf.unstack(self.normalize().value())
        return [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
                [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
                [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]

    # Initialise from axis-angle
    @classmethod
    def _from_axis_angle(self, axisAngle):
        """Initialise from axis and angle representation

        Create a Quaternion by specifying the 3-vector rotation axis and rotation
        angle (in radians) from which the quaternion's rotation should be created.

        Params:
            axis: a valid numpy 3-vector
            angle: a real valued angle in radians
        """
        mag_sq = tf.tensordot(axisAngle, axisAngle)
        # Ensure Provided rotation axis has length
        tf.Assert(tf.count_nonzero(mag_sq))
        # size of axis must be rotation around center
        tf.Assert(tf.less_equal(mag_sq, 1.0))
        # Ensure axis is in unit vector form
        theta = mag_sq
        r = tf.cos(theta)
        i = axis * tf.sin(theta)

        return self(r, i[0], i[1], i[2])

    @quat_scope
    @staticmethod
    def random():
        """Generate a random unit quaternion.
        Uniformly distributed across the rotation space
        As per: http://planning.cs.uiuc.edu/node198.html
        https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        """
        r1, r2, r3 = tf.random.random(3)

        q1 = tf.sqrt(1.0 - r1) * (tf.sin(2 * math.pi * r2))
        q2 = tf.sqrt(1.0 - r1) * (tf.cos(2 * math.pi * r2))
        q3 = tf.sqrt(r1) * (tf.sin(2 * math.pi * r3))
        q4 = tf.sqrt(r1) * (tf.cos(2 * math.pi * r3))

        return Quaternion((q1, q2, q3, q4))

    @staticmethod
    def _validate_shape(x):
        msg = "Can't create a quaternion with shape [4] from {} with shape {}."
        if isinstance(x, (list, tuple)) and np.array(x).shape != (4,):
                raise ValueError(msg.format("list/tuple", np.array(x).shape))
        elif isinstance(x, np.ndarray) and x.shape != (4,):
                raise ValueError(msg.format("np.array", x.shape))
        elif (isinstance(x, (tf.Tensor, tf.Variable))
              and x.shape.as_list() != [4]):
                raise ValueError(msg.format("tf.Tensor", x.shape.as_list()))

    @staticmethod
    def _validate_type(x):
        if not x.dtype.is_floating:
            raise TypeError("Quaternion only supports floating point numbers")

    @quat_scope
    def _squared_norm(self):
        return tf.reduce_sum(tf.square(self._q))

    @quat_scope
    def _abs(self):
        return tf.sqrt(tf.reduce_sum(tf.square(self._q)))

    def w(self):
        return self._q[0]


def dual_quat_scope(func, *args, **kwargs):
    def scoped_func(*args, **kwargs):
        with tf.name_scope("dual_quat_{}".format(func.__name__)):
            return func(*args, **kwargs)
    return scoped_func


class DualQuaternion():
    """Dual Quaternions are useful for rigid body transform representations in SE(3).

    Dual quaternions contain two quaternions, the real quaternion and a dual quaternion component.

    # Arguments

        real_wxyz: rotation component
        translation: translation component, 3 element vector or array
        dual_wxyz: direct quaternion representing translation component, translation parameter must be None.
    """
    def __init__(self, real_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0), dual_wxyz=None, dtype=tf.float32):
        if isinstance(real_wxyz, Quaternion):
            self._real = real_wxyz
        else:
            self._real = Quaternion(wxyz=real_wxyz, dtype=dtype)

        if translation is not None:
            self.set_translation(translation=translation)
        else:
            if isinstance(dual_wxyz, Quaternion):
                self._dual = real_wxyz
            else:
                self._dual = Quaternion(wxyz=dual_wxyz, dtype=dtype)

    @staticmethod
    def zeros(dtype=tf.float32):
        return DualQuaternion(real_wxyz=(0.0, 0.0, 0.0, 0.0), dual_wxyz=(0.0, 0.0, 0.0, 0.0), dtype=dtype)

    @staticmethod
    def identity(dtype=tf.float32):
        return DualQuaternion(real_wxyz=(0.0, 0.0, 0.0, 0.0), dual_wxyz=(0.0, 0.0, 0.0, 0.0), dtype=dtype)

    @dual_quat_scope
    def conjugate(self):
        """Dual Quaternion conjugate, encapsulated in a new instance.

        Returns:
            A new DualQuaternion object clone with the conjugate of the real and dual component quaternions.
        """
        return self.__class__(real_wxyz=self._real.conjugate(), dual_wxyz=self._dual.conjugate())

    @dual_quat_scope
    def inverse(self):
        sqrLen0 = self._real._squared_norm()
        sqrLenE = 2.0 * tf.tensordot(self._real.coeffs(), self._dual.coeffs())

        def invert_valid_dual_quaternion():
            invSqrLen0 = 1.0 / sqrLen0
            invSqrLenE = -sqrLenE / (sqrLen0 * sqrLen0)

            conj = self.conjugate()
            conj._real = Quaternion(invSqrLen0 * conj._real.coeffs())
            conj._dual = Quaternion(invSqrLen0 * conj._dual.coeffs() + invSqrLenE * conj._real.coeffs())
            return conj

        inv = tf.cond(sqrLen0 > 0.0, invert_valid_dual_quaternion, lambda: DualQuaternion.zeros())
        return inv

    def real(self):
        """Get the real quaternion component of the dual quaternion.
        """
        return self._real

    def dual(self):
        """Get the dual quaternion component of the dual quaternion.
        """
        return self._dual

    def rotation_matrix(self):
        return self._real.rotation_matrix()

    @dual_quat_scope
    def translation_quaternion(self):
        """Get the translation quaternion associated with the Dual Quaternion.
        """
        return Quaternion(2.0 * (self._dual * self._real.conjugate()).coeffs())

    @dual_quat_scope
    def translation(self):
        """Get the translation tensor [x, y, z] associated with the Dual Quaternion.
        """
        return self.translation_quaternion().coeffs()[1:3]

    @dual_quat_scope
    def set_translation(self, translation=[0.0, 0.0, 0.0]):
        """Set the translation component of this dual quaternion. Also normalizes the real component.

        # Arguments

        translation: A Tensor, list, or numpy array containing 3 elements, x, y, z.
        """
        self._real = self.normalize()
        cq_arr = tf.concat([0], translation)
        conv_q = Quaternion(cq_arr)
        conv_q = conv_q * self._real
        cq_arr = conv_q.coeffs() * 0.5
        self._dual = Quaternion(cq_arr)

    @dual_quat_scope
    def transformation_matrix(self):
        """Get the 4x4 homogeneous transformation matrix equivalent of the dual quaternion.
        """
        translation = self.translation()
        return self._real.transformation_matrix(t=translation[1:3])

    @dual_quat_scope
    def transform_point(self, point):
        dq = DualQuaternion(real_wxyz=Quaternion.identity(), dual_wxyz=(0.0, point[0], point[1], point[2]))
        dq = self * point_dq * self.conjugate()

        # extract the point
        p = tf.reshape(dq.translation(), [1, 3])

        # perform translation
        p += (2.0 * (self._real.w() * self._dual.imaginary() - self._dual.w() * self._real.imaginary() + tf.cross(self._real.imaginary(), self._dual.imaginary())))
        return p

    @dual_quat_scope
    def transform_vector(self, vector):
        dq = DualQuaternion(real_wxyz=Quaternion.identity(), dual_wxyz=(0.0, vector[0], vector[1], vector[2]))
        # perform the transform
        dq = self * point_dq * self.conjugate()

        # extract the vector
        return tf.reshape(dq.translation(), [1, 3])

    def __mul__(self, b):
        return multiply(self, b)

    def __imul__(self, other):
        if isinstance(other, DualQuaternion):
            return multiply(self, other)
        else:
            raise NotImplementedError("Quaternion Multiplication not implemented for this type.")