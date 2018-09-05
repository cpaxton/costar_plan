
import os
import copy
import math
import numpy as np
from tqdm import tqdm

import keras
import tensorflow as tf
from tensorflow.python.platform import flags
from shapely.geometry import Polygon
from pyquaternion import Quaternion
import sklearn

import grasp_utilities

# class Vector:
#     # http://www.mathopenref.com/coordpolygonarea.html
#     # https://stackoverflow.com/a/45268241/99379
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __add__(self, v):
#         if not isinstance(v, Vector):
#             return NotImplemented
#         return Vector(self.x + v.x, self.y + v.y)

#     def __sub__(self, v):
#         if not isinstance(v, Vector):
#             return NotImplemented
#         return Vector(self.x - v.x, self.y - v.y)

#     def cross(self, v):
#         if not isinstance(v, Vector):
#             return NotImplemented
#         return self.x*v.y - self.y*v.x


# class Line:
#     # ax + by + c = 0
#     def __init__(self, v1, v2):
#         self.a = v2.y - v1.y
#         self.b = v1.x - v2.x
#         self.c = v2.cross(v1)

#     def __call__(self, p):
#         return self.a*p.x + self.b*p.y + self.c

#     def intersection(self, other):
#         # http://www.mathopenref.com/coordpolygonarea.html
#         # https://stackoverflow.com/a/45268241/99379
#         # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
#         if not isinstance(other, Line):
#             return NotImplemented
#         w = self.a*other.b - self.b*other.a
#         return Vector(
#             (self.b*other.c - self.c*other.b)/w,
#             (self.c*other.a - self.a*other.c)/w
#         )


# def rectangle_vertices(cx, cy, w, h, theta):
#     # http://www.mathopenref.com/coordpolygonarea.html
#     # https://stackoverflow.com/a/45268241/99379
#     dx = w/2
#     dy = h/2
#     dxcos = dx*cos(theta)
#     dxsin = dx*sin(theta)
#     dycos = dy*cos(theta)
#     dysin = dy*sin(theta)
#     return (
#         Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
#         Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
#         Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
#         Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
#     )

# def intersection_area(r1, r2):
#     # http://www.mathopenref.com/coordpolygonarea.html
#     # https://stackoverflow.com/a/45268241/99379
#     # r1 and r2 are in (center, width, height, rotation) representation
#     # First convert these into a sequence of vertices

#     rect0 = rectangle_vertices(*r1)
#     rect1 = rectangle_vertices(*r2)

#     # Use the vertices of the first rectangle as
#     # starting vertices of the intersection polygon.
#     rect0 = rect0

#     # Loop over the edges of the second rectangle
#     for p, q in zip(rect1, rect1[1:] + rect1[:1]):
#         if len(rect0) <= 2:
#             break # No intersection

#         line = Line(p, q)

#         # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
#         # any point p with line(p) > 0 is on the "outside".

#         # Loop over the edges of the rect0 polygon,
#         # and determine which part is inside and which is outside.
#         new_intersection = []
#         line_values = [line(t) for t in rect0]
#         for s, t, s_value, t_value in zip(
#                 rect0, rect0[1:] + rect0[:1],
#                 line_values, line_values[1:] + line_values[:1]):
#             if s_value <= 0:
#                 new_intersection.append(s)
#             if s_value * t_value < 0:
#                 # Points are on opposite sides.
#                 # Add the intersection of the lines to new_intersection.
#                 intersection_point = line.intersection(Line(s, t))
#                 new_intersection.append(intersection_point)

#         intersection = new_intersection

#     # Calculate area
#     if len(intersection) <= 2:
#         return 0

#     return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
#                      zip(intersection, intersection[1:] + intersection[:1]))


# intersection_area(r0y0, r0x0, r0y1, r0x1, r0y2, r0x2, r0y3, r0x3, r1y0, r1x0, r1y1, r1x1, r1y2, r1x2,  r1y3, r1x3):
def rectangle_points(r0y0, r0x0, r0y1, r0x1, r0y2, r0x2, r0y3, r0x3):
    p0yx = np.array([r0y0, r0x0])
    p1yx = np.array([r0y1, r0x1])
    p2yx = np.array([r0y2, r0x2])
    p3yx = np.array([r0y3, r0x3])
    return [p0yx, p1yx, p2yx, p3yx]


def rectangle_vectors(rp):
    """
    # Arguments

    rp: rectangle points [p0yx, p1yx, p2yx, p3yx]
    """
    v0 = rp[1] - rp[0]
    v1 = rp[2] - rp[1]
    v2 = rp[3] - rp[2]
    v3 = rp[0] - rp[3]

    return [v0, v1, v2, v3]


def rectangle_homogeneous_lines(rv):
    """

    # Arguments

    rv: rectangle vectors [v0yx, v1yx, v2yx, v3yx]


    # Returns

    [r0abc, r1abc, r2abc, r3abc]

    """
    # ax + by + c = 0
    dv = rv[0] - rv[1]
    # TODO(ahundt) make sure cross product doesn't need to be in xy order
    r0abc = K.concatenate([dv[0], dv[1], tf.cross(rv[0], rv[1])])
    dv = rv[1] - rv[2]
    r1abc = K.concatenate([dv[1], dv[2], tf.cross(rv[1], rv[2])])
    dv = rv[2] - rv[3]
    r2abc = K.concatenate([dv[2], dv[3], tf.cross(rv[2], rv[3])])
    dv = rv[3] - rv[0]
    r3abc = K.concatenate([dv[3], dv[0], tf.cross(rv[3], rv[0])])
    return [r0abc, r1abc, r2abc, r3abc]


def homogeneous_line_intersection(hl0abc, hl1abc):
    """ Given two homogenous lines return the intersection point in y,x coordinates
    """
    a0 = hl0abc[0]
    b0 = hl0abc[1]
    c0 = hl0abc[2]
    a1 = hl1abc[0]
    b1 = hl1abc[1]
    c1 = hl1abc[2]
    w = a0 * b1 - b0 * a1
    py = (c0 * a1 - a0 * c1) / w
    px = (b0 * c1 - c0 * b1) / w
    return [py, px]


def line_at_point(l_abc, p_yx):
    """

    # Arguments

    l_abc: a line in homogenous coodinates
    p_yx: a point with y, x coordinates
    """
    return l_abc[0] * p_yx[1] + l_abc[1] * p_yx[0] + l_abc[2]


def intersection_points(rl0, rp1):
    """ Evaluate rectangle lines at another rectangle's points
    """
    lv = [
        line_at_point(rl0[0], rp1[0]),
        line_at_point(rl0[1], rp1[1]),
        line_at_point(rl0[2], rp1[2]),
        line_at_point(rl0[3], rp1[3]),
    ]
    return lv


def rectangle_intersection_polygon(rp0, rl0, rp1, rl1):
    """ Given two homogenous line rectangles, it returns the points for the polygon representing their intersection.

    # Arguments

    rp0: rectangle 0 defined with points
    rl0: rectangle 0 defined with homogeneous lines
    rp1: rectangle 1 defined with points
    rp1: rectangle 1 defined with homogeneous lines

    # Returns

    Intersection polygon consisting of up to 8 points.
    """
    # TODO(ahundt) this function is still set up for eager execution... figure it out as tf calls...
    # http://www.mathopenref.com/coordpolygonarea.html
    # https://stackoverflow.com/a/45268241/99379
    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = []
    for line1 in rl1:
        line_values = [line_at_point(line1, t) for t in rp0]

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the rect0 polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        # points in rp0 rotated around by one
        rp0_rot = grasp_utilities.rotate(rp0)
        line_values_rot = grasp_utilities.rotate(line_values)
        for s, t, s_value, t_value, line0 in zip(
                rp0, rp0_rot, line_values, line_values_rot, rl0):

            if s_value <= 0:
                new_intersection.append(s)

            st_value = s_value * t_value
            intersection_point = homogeneous_line_intersection(line1, line0)
            if st_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                new_intersection.append(intersection_point)

        intersection = new_intersection

    return intersection


def polygon_area_four_points(rp):
    """
    # Arguments

    rp: polygon defined by 4 points in y,x order
    """
    # partial = p0x * p1y - p0y * p1x
    partial0 = rp[0][1] * rp[1][0] - rp[0][0] * rp[1][1]
    partial1 = rp[1][1] * rp[2][0] - rp[1][0] * rp[2][1]
    partial2 = rp[2][1] * rp[3][0] - rp[2][0] * rp[3][1]
    partial3 = rp[3][1] * rp[0][0] - rp[3][0] * rp[0][1]
    full_sum = partial0 + partial1 + partial2 + partial3
    return 0.5 * full_sum


def polygon_area(poly):
    # Calculate area
    if len(poly) <= 2:
        return 0

    poly_rot = poly[1:] + poly[:1]

    return 0.5 * sum(p[1]*q[0] - p[0]*q[1] for p, q in zip(poly, poly_rot))


def rectangle_vertices(h, w, cy, cx, sin_theta=None, cos_theta=None, theta=None):
    """ Get the vertices from a parameterized bounding box.

    y, x ordering where 0,0 is the top left corner.
    This matches matrix indexing.

    # http://www.mathopenref.com/coordpolygonarea.html
    # https://stackoverflow.com/a/45268241/99379
    """
    if theta is not None:
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
    # normalizing because this may be using the output of the neural network,
    # so we turn it into an x y coordinate on the unit circle without changing
    # the vector.
    sin_theta, cos_theta = normalize_sin_theta_cos_theta(sin_theta, cos_theta)

    dx = w/2
    dy = h/2
    dxcos = dx * cos_theta
    dxsin = dx * sin_theta
    dycos = dy * cos_theta
    dysin = dy * sin_theta
    return [
        np.array([cy, cx]) + np.array([-dxsin + -dycos, -dxcos - -dysin]),
        np.array([cy, cx]) + np.array([ dxsin + -dycos,  dxcos - -dysin]),
        np.array([cy, cx]) + np.array([ dxsin +  dycos,  dxcos -  dysin]),
        np.array([cy, cx]) + np.array([-dxsin +  dycos, -dxcos -  dysin])
    ]


def encode_sin2_cos2(sin2_cos2):
    """ Converts values from the range (-1, 1) to the range (0, 1).

    The value passed is already expected to be in the format:
        np.array([np.sin(2 * theta), np.cos(2 * theta)])

    If you have 2 theta and want to encode that see `encode_2theta()`.

    """
    return (sin2_cos2 / 2.0) + 0.5


def encode_sin_cos(sin_cos):
    """ Converts values from the range (-1, 1) to the range (0, 1).

    The value passed is already expected to be in the format:
        np.array([np.sin(theta), np.cos(theta)])

    If you have theta and want to encode that see `encode_theta()`.

    """
    return (sin_cos / 2.0) + 0.5


def encode_2theta(theta):
    """ Encodes theta in radians to handle gripper symmetry in 0 to 1 domain

    # Returns

        [sin(2 * theta), cos(2 * theta)] / 2 + 0.5

    """
    theta2 = theta * 2.0
    return encode_theta(theta2)


def encode_theta(theta):
    """ Encodes theta in radians to asymmetric grippers in 0 to 1 domain

    # Returns

        [sin(theta), cos(theta)] / 2 + 0.5

    """
    norm_sin_cos = encode_sin_cos(np.array([np.sin(theta), np.cos(theta)]))
    return norm_sin_cos


def denorm_sin2_cos2(norm_sin2_cos2):
    """ Undo normalization step of `encode_2theta_np()`


        This converts values from the range (0, 1) to (-1, 1)
        by subtracting 0.5 and multiplying by 2.0.
        This function does not take any steps to ensure
        the input obeys the law:

            sin ** 2 + cos ** 2 == 1

        Since the values may have been generated by a neural network
        it is important to fix this w.r.t. the provided values.

    # Arguments

        norm_sin2_cos2: normalized sin(2*theta) cos(2*theta)

    # Returns

        return actual sin(2*theta) cos(2*theta)
    """
    return (norm_sin2_cos2 - 0.5) * 2.0


def denorm_sin_cos(norm_sin_cos):
    """ Undo normalization step of `encode_theta_np()`


        This converts values from the range (0, 1) to (-1, 1)
        by subtracting 0.5 and multiplying by 2.0.
        This function does not take any steps to ensure
        the input obeys the law:

            sin ** 2 + cos ** 2 == 1

        Since the values may have been generated by a neural network
        it is important to fix this w.r.t. the provided values.

    # Arguments

        norm_sin2_cos2: normalized sin(2*theta) cos(2*theta)

    # Returns

        return actual sin(theta) cos(theta)
    """
    return (norm_sin_cos - 0.5) * 2.0


def decode_sin2_cos2(norm_sin2_cos2):
    """ Decodes the result of encode_2theta() back into an angle theta in radians.
    """
    return decode_sin_cos(norm_sin2_cos2) / 2.0


def decode_sin_cos(norm_sin2_cos2):
    """ Decodes the result of encode_theta() back into an angle theta in radians.
    """
    # rescale and shift from (0, 1) range
    # back to (-1, 1) range
    #
    # note that denorm step is the same for both sin_cos and sin2_cos2
    sin2, cos2 = denorm_sin2_cos2(norm_sin2_cos2)
    # normalize the values so they are on the unit circle
    sin2, cos2 = normalize_sin_theta_cos_theta(sin2, cos2)
    # extract 2x the angle
    theta2 = np.arctan2(sin2, cos2)
    # return the angle
    return theta2


def parse_rectangle_vertices(s2t_c2t_hw_cycx):
    """ Convert a dimensions, angle, grasp center, based rectangle to vertices.

    s2t_c2t_hw_cycx: [sin(2*theta), cos(2*theta), height, width, center x, center y]
    """
    # sin(2*theta), cos(2*theta)
    theta = decode_sin2_cos2(s2t_c2t_hw_cycx[:2])
    rect_vertices = rectangle_vertices(
        s2t_c2t_hw_cycx[2],  # height
        s2t_c2t_hw_cycx[3],  # width
        s2t_c2t_hw_cycx[4],  # center y
        s2t_c2t_hw_cycx[5],  # center x
        theta=theta)
    return rect_vertices


def parse_rectangle_params(s2t_c2t_hw_cycx):
    rect_vertices = parse_rectangle_vertices(s2t_c2t_hw_cycx)
    rect_hlines = rectangle_homogeneous_lines(rect_vertices)
    return rect_vertices, rect_hlines


def intersection_over_union(true_rp, pred_rp, true_rl, pred_rl):
    """ Intersection over union of two oriented rectangles.

    Also known as the jaccard metric.

    # Arguments

        true_rp: oriented rectanle 0 points
        pred_rp: oriented rectangle 1 points
        true_rl: oriented rectangle 0 homogeneous lines
        pred_rl: oriented rectangle 1 homogeneous lines
    """
    true_area = polygon_area_four_points(true_rp)
    pred_area = polygon_area_four_points(pred_rp)
    intersection_polygon = rectangle_intersection_polygon(true_rp, true_rl, pred_rp, pred_rl)
    intersection_area = polygon_area(intersection_polygon)

    iou = intersection_area / (true_area + pred_area - intersection_area)
    return iou


def shapely_intersection_over_union(rect0_points, rect1_points, verbose=0):
    """ Find the intersection over union of two polygons using shapely
    """
    # create and clean the polygons to eliminate any overlapping points
    # https://toblerity.org/shapely/manual.html
    p0 = Polygon(rect0_points).buffer(0)
    p1 = Polygon(rect1_points).buffer(0)
    if p0.is_valid and p1.is_valid:
        intersection_area = p0.intersection(p1).area

        iou = intersection_area / (p0.area + p1.area - intersection_area)
        if verbose > 0:
            print('iou: ' + str(iou))
        return iou
    else:
        # TODO(ahundt) determine and fix the source of invalid polygons.
        print('Warning: shapely_intersection_over_union() encountered an '
              'invalid polygon. We will return an IOU of 0 so execution '
              'might continue, but this bug should be addressed. '
              'p0: ' + str(p0) + ' p1: ' + str(p1))
        return 0.0


def normalize_sin_theta_cos_theta(sin_theta, cos_theta):
    """ Put sin(theta) cos(theta) on the unit circle.

    Output values will be in (-1, 1).
    normalize the prediction but keep the vector direction the same
    """
    arr = sklearn.preprocessing.normalize(np.array([[sin_theta, cos_theta]], dtype=np.float))
    sin_theta = arr[0, 0]
    cos_theta = arr[0, 1]
    return sin_theta, cos_theta


def prediction_vector_has_grasp_success(y_pred):
    has_grasp_success = (y_pred.size == 7)
    return has_grasp_success


def get_prediction_vector_rectangle_start_index(y_pred):
    """ Get the rectangle start index from an encoded prediction vector of length 6 or 7
    """
    has_grasp_success = prediction_vector_has_grasp_success(y_pred)
    # the grasp rectangle start index
    rect_index = 0
    if has_grasp_success:
        rect_index = 1
    return rect_index


def decode_prediction_vector(y_true):
    """ Decode a prediction vector into sin(2 * theta), cos(2 * theta), and 4 vertices
    """
    rect_index = get_prediction_vector_rectangle_start_index(y_true)
    end_angle_index = rect_index + 2
    y_true[rect_index: end_angle_index] = denorm_sin2_cos2(y_true[rect_index:end_angle_index])
    true_y_sin_theta, true_x_cos_theta = y_true[rect_index:end_angle_index]
    true_rp = parse_rectangle_vertices(y_true[rect_index:])
    return true_y_sin_theta, true_x_cos_theta, true_rp


def decode_prediction_vector_theta_center_polygon(y_true):
    """ Decode a prediction vector into theta and four rectangle vertices

    Only supports vector format that includes center information!
    """
    rect_index = get_prediction_vector_rectangle_start_index(y_true)
    end_angle_index = rect_index + 2
    y_true[rect_index: end_angle_index] = denorm_sin2_cos2(y_true[rect_index:end_angle_index])
    true_y_sin_theta, true_x_cos_theta = y_true[rect_index:end_angle_index]
    true_rp = parse_rectangle_vertices(y_true[rect_index:])
    true_y_sin_theta, true_x_cos_theta = normalize_sin_theta_cos_theta(true_y_sin_theta, true_x_cos_theta)
    # right now it is 2 theta, so get theta
    theta = np.arctan2(true_y_sin_theta, true_x_cos_theta) / 2.0
    # center should be last two entries y, x order
    center = y_true[-2:]
    return theta, center, true_rp


def angle_difference_less_than_threshold(
        true_y_sin_theta, true_x_cos_theta,
        pred_y_sin_theta, pred_x_cos_theta,
        angle_threshold=np.radians(60.0),
        verbose=0):
    """ Returns true if the angle difference is less than the threshold, false otherwise.

    Recall that angle differences are around a circle, so the shortest angular difference
    may be in +theta or the -theta direction with wrapping around the boundaries.

    Note that the angle threshold is set to 60 because we are working with 2*theta.
    TODO(ahundt) double check the implications of this.

    # Arguments
        angle_threshold: The maximum absolute angular difference permitted.
    """
    # print('ad0 ' + str(true_y_sin_theta) + ' cos: ' + str(true_x_cos_theta))
    # normalize the prediction but keep the vector direction the same
    true_y_sin_theta, true_x_cos_theta = normalize_sin_theta_cos_theta(true_y_sin_theta, true_x_cos_theta)
    # print('ad1')
    true_angle = np.arctan2(true_y_sin_theta, true_x_cos_theta)
    # print('ad2')
    # normalize the prediction but keep the vector direction the same
    pred_y_sin_theta, pred_x_cos_theta = normalize_sin_theta_cos_theta(pred_y_sin_theta, pred_x_cos_theta)
    pred_angle = np.arctan2(pred_y_sin_theta, pred_x_cos_theta)
    # print('pred angle: ' + str(pred_angle) + ' true angle: ' + str(true_angle))
    true_pred_diff = true_angle - pred_angle
    # we would have just done this directly at the start if the angle_multiplier wasn't needed
    angle_difference = np.arctan2(np.sin(true_pred_diff), np.cos(true_pred_diff))
    # print('angle_difference: ' + str(angle_difference) + ' deg: ' + str(np.degrees(angle_difference)))
    is_within_angle_threshold = np.abs(angle_difference) <= angle_threshold
    if verbose > 0:
        print(' angle_difference_less_than_threshold(): ' +
              ' angle_difference: ' + str(int(np.degrees(angle_difference))) +
              ' threshold: ' + str(int(np.degrees(angle_threshold))) +
              ' is_within_angle_threshold: ' + str(is_within_angle_threshold) +
              ' true_angle: ' + str(np.degrees(true_angle)) +
              ' pred_angle: ' + str(np.degrees(pred_angle)) +
              ' units: degrees ')
    return is_within_angle_threshold


def jaccard_score(y_true, y_pred, angle_threshold=np.radians(60.0), iou_threshold=0.25, verbose=0):
    """ Scoring for regression
    Note that the angle threshold is set to 60 because we are working with 2*theta.
    TODO(ahundt) double check the implications of this.

    # Arguments

        Feature formats accepted:

        grasp_success_norm_sin2_cos2_hw_yx_7:
            [grasp_success, sin_2theta, cos2_theta, height, width, center_y, center_x]
            [            0,         1,         2,      3,     4,        5,        6]

        norm_sin2_cos2_hw_yx_6:
            [sin_2theta, cos2_theta, height, width, center_y, center_x]
            [            0,         1,         2,      3,     4,        5,        6]


        Not yet accepted:
        norm_sin2_cos2_hw_5
            [sin2_theta, cos_2theta, height, width, center_y, center_x]
            [        0,           1,      2,     3,        4,        5]

        grasp_success_norm_sin2_cos2_hw_5
            [grasp_success, sin_2theta, cos2_theta, height, width]
            [            0,         1,         2,      3,     4,]


        y_true: a numpy array of features
        y_pred: a numpy array of features
        angle_threshold: The maximum allowed difference in
            angles for a grasp to be considered successful.
            Default of 60 degrees is for 2 * theta, which is 30 degrees for theta.
        theta_multiplier: Either 1.0 or 2.0.
            If it is 1.0 theta angles are compared directly.
            If it is 2.0 (the default), angles that are off by 180 degrees
            are considered equal, which is the case for a gripper with two plates.


    """

    has_grasp_success = prediction_vector_has_grasp_success(y_pred)

    # print('0')
    # round grasp success to 0 or 1
    # note this is not valid and not used if
    # has grasp success is false.
    predicted_success = np.rint(y_pred[0])
    # print('1')
    if has_grasp_success and predicted_success != int(y_true[0]):
        # grasp success prediction doesn't match, return 0 score
        # print('2')
        return 0.0
    elif has_grasp_success and predicted_success == 0:
        # The success prediction correctly matches the ground truth,
        # plus both are False so this is a true negative.
        # Any true negative where failure to grasp is predicted correctly
        # gets credit regardless of box contents
        # print('3')
        return 1.0
    else:
        # We're looking at a successful grasp and we've correctly predicted grasp_success.
        # First check if the angles are close enough to matching the angle_threshold.
        # print('4')

        # denormalize the values from (0, 1) back to (-1, 1 range) and get the array entries
        true_y_sin_theta, true_x_cos_theta, true_rp = decode_prediction_vector(y_true)
        pred_y_sin_theta, pred_x_cos_theta, pred_rp = decode_prediction_vector(y_pred)

        # print('5')
        # if the angle difference isn't close enough to ground truth return 0.0
        if not angle_difference_less_than_threshold(
                true_y_sin_theta, true_x_cos_theta,
                pred_y_sin_theta, pred_x_cos_theta,
                angle_threshold,
                verbose=verbose):
            return 0.0

        # print('6')
        # We passed all the other checks so
        # let's find out if the grasp boxes match
        # via the jaccard distance.
        iou = shapely_intersection_over_union(true_rp, pred_rp)
        if verbose:
            print('iou: ' + str(iou))
        # print('8')
        if iou >= iou_threshold:
            # passed iou threshold
            return 1.0
        else:
            # didn't meet iou threshold
            return 0.0


def grasp_jaccard_batch(y_true, y_pred, verbose=0):
    # print('y_true.shape: ' + str(y_true.shape))
    # print('y_pred.shape: ' + str(y_pred.shape))
    scores = []
    for i in range(y_true.shape[0]):
        # print(' i: ' + str(i))
        # TODO(ahundt) comment the next few lines when not debugging
        verbose = 0
        if np.random.randint(0, 10000) % 10000 == 0:
            verbose = 1
            print('')
            print('')
            print('grasp_metrics.py sample of ground_truth and prediction:')
        this_true = y_true[i, :]
        this_pred = y_pred[i, :]
        score = jaccard_score(this_true, this_pred, verbose=verbose)
        if verbose:
            print('s2t_c2t_hw_cycx_true: ' + str(this_true))
            print('s2t_c2t_hw_cycx_pred: ' + str(this_pred))
            print('score:' + str(score))
        scores += [score]
    scores = np.array(scores, dtype=np.float32)
    # print('scores.shape: ' + str(scores.shape))
    return scores


def grasp_jaccard(y_true, y_pred):
    """ Calculates the jaccard metric score in a manner compatible with tf and keras metrics.

        This is an IOU metric with angle difference and IOU score thresholds.

        Feature formats accepted as a 2d array containing a batch of data ordered as:
        [grasp_success, sin_2theta, cos_2theta, height, width, center_y, center_x]
        [            0,         1,         2,      3,     4,        5,        6]

        [sin_2theta, cos_2theta, height, width, center_y, center_x]
        [        0,         1,      2,     3,        4,        5]

        It is very important to be aware that sin(2*theta) and cos(2*theta) are expected,
        additionally all coordinates and height/width are normalized by the network's input dimensions.
    """
    scores = tf.py_func(func=grasp_jaccard_batch, inp=[y_true, y_pred], Tout=tf.float32, stateful=False)
    return scores


def rotation_to_xyz_theta(rotation, verbose=0):
    """Convert a rotation to an angle theta

    From above, a rotation to the right should be a positive theta,
    and a rotation to the left negative theta. The initial pose is with the
    z axis pointing down, the y axis to the right and the x axis forward.

    This format does not allow for arbitrary rotation commands to be defined,
    and originates from the costar dataset.

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
    # pyquaternion is in xyzw format!
    aa = Quaternion(rotation)
    # angle in radians
    theta = aa.angle
    if aa.axis[2] < 0:
        multiply = 1.0
    else:
        multiply = -1.0
    if verbose > 0:
        print("ANGLE_AXIS_MULTIPLY: ", aa.angle, np.array(aa.axis), multiply)
    theta *= multiply

    return np.concatenate([aa.axis, [theta]], axis=-1)


def normalize_axis(aaxyz, epsilon=1e-5, verbose=0):
    """ Normalize an axis in angle axis format data.

    If axis is all zeros, epsilon is added to the final axis.
    """
    if not np.any(aaxyz):
        # source: https://stackoverflow.com/a/23567941/99379
        # we checked if all values are zero, fix missing axis
        aaxyz[-1] += epsilon
    arr = sklearn.preprocessing.normalize(np.array([aaxyz], dtype=np.float))
    aaxyz = np.squeeze(arr[0, :])
    if verbose:
        print('normalize_axis: ' + str(aaxyz))
    return aaxyz


def encode_xyz_qxyzw_to_xyz_aaxyz_nsc(xyz_qxyzw, rescale_meters=4, rotation_weight=0.001, random_augmentation=None):
    """ Encode a translation + quaternion pose to an encoded xyz, axis, and an angle as sin(theta) cos(theta)

    rescale_meters: Divide the number of meters by this number so
        positions will be encoded between 0 and 1.
        For example if you want to be able to reach forward and back by 2 meters, divide by 4.
    rotation_weight: scale down rotation values by this factor to a smaller range
        so mse gives similar weight to both rotations and translations.
        Use 1.0 for no adjustment. Default of 0.001 makes 1 radian
        about equal weight to 1 millimeter.
    random_augmentation: default None means no data modification,
        otherwise a value between 0.0 and 1.0 for the probability
        of randomly modifying the data with a small translation and rotation.
        Enabling random_augmentation is not recommended.
    """
    xyz = (xyz_qxyzw[:3] / rescale_meters) + 0.5
    length = len(xyz_qxyzw)
    if length == 7:
        # print('xyz: ' + str(xyz))
        rotation = Quaternion(xyz_qxyzw[3:])
        # pose augmentation with no feedback or correspondingly adjusted transform poses
        if random_augmentation is not None and np.random.random() > random_augmentation:
            # random rotation change
            # random = Quaternion.random()
            # # only take rotations less than 5 degrees
            # while random.angle > np.pi / 36.:
            #     # TODO(ahundt) make more efficient and re-enable
            #     random = Quaternion.random()
            # rotation = rotation * random
            # random translation change of up to 0.5 cm
            random = (np.random.random(3) - 0.5) / 10.
            xyz = xyz + random

        aaxyz_theta = rotation_to_xyz_theta(rotation)
        # encode the unit axis vector into the [0,1] range
        # rotation_weight makes it so mse applied to rotation values
        # is on a similar scale to the translation values.
        aaxyz = ((aaxyz_theta[:-1] / 2) * rotation_weight) + 0.5
        nsc = encode_theta(aaxyz_theta[-1])
        # print('nsc: ' + str(nsc))
        xyz_aaxyz_nsc = np.concatenate([xyz, aaxyz, nsc], axis=-1)
        return xyz_aaxyz_nsc
    elif length == 3:
        if random_augmentation is not None and np.random.random() > random_augmentation:
            # random translation change of up to 0.5 cm
            random = (np.random.random(3) - 0.5) / 10.
            xyz = xyz + random

        return xyz
    else:
        raise ValueError('encode_xyz_qxyzw_to_xyz_aaxyz_nsc: unsupported input data length of ' + str(length))


def batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(batch_xyz_qxyzw, rescale_meters=4, rotation_weight=0.001, random_augmentation=None):
    """ Expects n by 7 batch with xyz_qxyzw

    rescale_meters: Divide the number of meters by this number so
        positions will be encoded between 0 and 1.
        For example if you want to be able to reach forward and back by 2 meters, divide by 4.
    rotation_weight: scale down rotation values by this factor to a smaller range
        so mse gives similar weight to both rotations and translations.
        Use 1.0 for no adjustment.
    random_augmentation: default None means no data modification,
        otherwise a value between 0.0 and 1.0 for the probability
        of randomly modifying the data with a small translation and rotation.
        Enabling random_augmentation is not recommended.
    """
    encoded_poses = []
    for xyz_qxyzw in batch_xyz_qxyzw:
        # print('xyz_qxyzw: ' + str(xyz_qxyzw))
        xyz_aaxyz_nsc = encode_xyz_qxyzw_to_xyz_aaxyz_nsc(
            xyz_qxyzw, rescale_meters=rescale_meters, rotation_weight=rotation_weight, random_augmentation=random_augmentation)
        # print('xyz_aaxyz_nsc: ' + str(xyz_aaxyz_nsc))
        encoded_poses.append(xyz_aaxyz_nsc)
    return np.stack(encoded_poses, axis=0)


def decode_xyz_aaxyz_nsc_to_xyz_qxyzw(xyz_aaxyz_nsc, rescale_meters=4, rotation_weight=0.001):
    """ Encode a translation + quaternion pose to an encoded xyz, axis, and an angle as sin(theta) cos(theta)

    rescale_meters: Divide the number of meters by this number so
        positions will be encoded between 0 and 1.
        For example if you want to be able to reach forward and back by 2 meters, divide by 4.
    rotation_weight: scale down rotation values by this factor to a smaller range
        so mse gives similar weight to both rotations and translations.
        Use 1.0 for no adjustment.
    """
    xyz = (xyz_aaxyz_nsc[:3] - 0.5) * rescale_meters
    length = len(xyz_aaxyz_nsc)
    if length == 8:
        theta = decode_sin_cos(xyz_aaxyz_nsc[-2:])
        # decode ([0, 1] * rotation_weight) range to [-1, 1] range
        aaxyz = ((xyz_aaxyz_nsc[3:-2] - 0.5) * 2) / rotation_weight
        # aaxyz is axis component of angle axis format,
        # Note that rotation_weight is automatically removed by normalization step.
        aaxyz = normalize_axis(aaxyz)
        q = Quaternion(axis=aaxyz, angle=theta)
        xyz_qxyzw = np.concatenate([xyz, q.elements], axis=-1)
        return xyz_qxyzw
    elif length != 3:
        raise ValueError('decode_xyz_aaxyz_nsc_to_xyz_qxyzw: unsupported input data length of ' + str(length))
    return xyz


def grasp_acc(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.01, max_rotation=0.261799):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.
    Limits default to 15 degrees and 1cm.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.01 meters, or 1cm,
       translations must be less than this distance away.
    max_rotation: defaults to 15 degrees in radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_5mm_7_5deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.005, max_rotation=0.1308995):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.
    Limits default to 7.5 degrees and 0.5cm.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.005 meters, which is 0.5cm,
       translations must be less than this distance away.
    max_rotation: defaults to 7.5 degrees, which is 0.1308995 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_1cm_15deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.01, max_rotation=0.261799):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.
    Limits default to 15 degrees and 1cm.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.01 meters, which is 1cm,
       translations must be less than this distance away.
    max_rotation: defaults to 15 degrees in radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_2cm_30deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.02, max_rotation=0.523598):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_4cm_60deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.04, max_rotation=1.047196):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_8cm_120deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.08, max_rotation=2.094392):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_16cm_240deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.16, max_rotation=4.188784):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_32cm_360deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.32, max_rotation=6.2832):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_64cm_360deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.64, max_rotation=6.2832):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_128cm_360deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=1.28, max_rotation=6.2832):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_256cm_360deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=2.56, max_rotation=6.2832):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def grasp_acc_512cm_360deg(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=5.12, max_rotation=6.2832):
    """ Calculate 3D grasp accuracy for a single result with grasp_accuracy_xyz_aaxyz_nsc encoding.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.02 meters, which is 2cm,
       translations must be less than this distance away.
    max_rotation: defaults to 30 degrees, which is 0.523598 radians,
       rotations must be less than this angular distance away.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        grasp_accuracy_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation, max_rotation],
        [tf.float32], stateful=False,
        name='py_func/grasp_accuracy_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def cart_error(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
    """ Calculate 3D grasp accuracy for a single result
    grasp_accuracy_xyz_aaxyz_nsc
    max_translation defaults to 0.1 meters, or 1cm.
    max_rotation defaults to 15 degrees in radians.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        absolute_cart_distance_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc],
        [tf.float32], stateful=False,
        name='py_func/absolute_cart_distance_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def angle_error(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
    """ Calculate 3D grasp accuracy for a single result
    Input format is xyz_aaxyz_nsc.
    max_translation defaults to 0.1 meters, or 1cm.
    max_rotation defaults to 15 degrees in radians.
    """
    # TODO(ahundt) make a single, simple call for grasp_accuracy_xyz_aaxyz_nsc, no py_func etc
    [filter_result] = tf.py_func(
        absolute_angle_distance_xyz_aaxyz_nsc_batch,
        [y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc],
        [tf.float32], stateful=False,
        name='py_func/absolute_angle_distance_xyz_aaxyz_nsc_batch')
    filter_result.set_shape(y_true_xyz_aaxyz_nsc.get_shape()[0])
    return filter_result


def absolute_angle_distance_xyz_aaxyz_nsc_single(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
    """ Calculate 3D grasp accuracy for a single result

    max_translation is 0.01 meters, or 1cm.
    max_rotation is 15 degrees in radians.
    Input format is xyz_aaxyz_nsc.

    This version is for a single pair of numpy arrays of length 8.
    """
    length = len(y_true_xyz_aaxyz_nsc)
    if length == 5:
        # workaround rotation distance only,
        # just use [0.5, 0.5, 0.5] for translation component
        # so existing code can be utilized
        fake_translation = np.array([0.5, 0.5, 0.5])
        y_true_xyz_aaxyz_nsc = np.concatenate([fake_translation, y_true_xyz_aaxyz_nsc])
        y_pred_xyz_aaxyz_nsc = np.concatenate([fake_translation, y_pred_xyz_aaxyz_nsc])

    y_true_xyz_qxyzw = decode_xyz_aaxyz_nsc_to_xyz_qxyzw(y_true_xyz_aaxyz_nsc)
    y_pred_xyz_qxyzw = decode_xyz_aaxyz_nsc_to_xyz_qxyzw(y_pred_xyz_aaxyz_nsc)
    y_true_q = Quaternion(y_true_xyz_qxyzw[3:])
    y_pred_q = Quaternion(y_pred_xyz_qxyzw[3:])
    return Quaternion.absolute_distance(y_true_q, y_pred_q)


def absolute_angle_distance_xyz_aaxyz_nsc_batch(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
    """ Calculate 3D grasp accuracy for a single result
    Expects batch of data as an nx8 array. Eager execution / numpy version.

    max_translation defaults to 0.01 meters, or 1cm.
    max_rotation defaults to 15 degrees in radians.
    Input format is xyz_aaxyz_nsc.
    """
    # print('type of y_true_xyz_aaxyz_nsc: ' + str(type(y_true_xyz_aaxyz_nsc)))
    accuracies = []
    for y_true, y_pred in zip(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
        one_accuracy = absolute_angle_distance_xyz_aaxyz_nsc_single(y_true, y_pred)
        # print('one grasp acc: ' + str(one_accuracy))
        accuracies.append(one_accuracy)
    accuracies = np.array(accuracies, np.float32)
    return accuracies


def absolute_cart_distance_xyz_aaxyz_nsc_single(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
    """ Calculate cartesian distance of encoded pose

    This version is for a single pair of numpy arrays of length 8.
    Input format is xyz_aaxyz_nsc.
    """
    y_true_xyz_qxyzw = decode_xyz_aaxyz_nsc_to_xyz_qxyzw(y_true_xyz_aaxyz_nsc)
    y_pred_xyz_qxyzw = decode_xyz_aaxyz_nsc_to_xyz_qxyzw(y_pred_xyz_aaxyz_nsc)
    # translation distance
    return np.linalg.norm(y_true_xyz_qxyzw[:3] - y_pred_xyz_qxyzw[:3])


def absolute_cart_distance_xyz_aaxyz_nsc_batch(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
    """ Calculate 3D grasp accuracy for a single result
    Expects batch of data as an nx8 array. Eager execution / numpy version.

    max_translation defaults to 0.01 meters, or 1cm.
    max_rotation defaults to 15 degrees in radians.
    """
    # print('type of y_true_xyz_aaxyz_nsc: ' + str(type(y_true_xyz_aaxyz_nsc)))
    accuracies = []
    for y_true, y_pred in zip(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
        one_accuracy = absolute_cart_distance_xyz_aaxyz_nsc_single(y_true, y_pred)
        # print('one grasp acc: ' + str(one_accuracy))
        accuracies.append(one_accuracy)
    accuracies = np.array(accuracies, np.float32)
    return accuracies


def grasp_accuracy_xyz_aaxyz_nsc_single(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.01, max_rotation=0.261799):
    """ Calculate 3D grasp accuracy for a single 1D numpy array for the ground truth and predicted value.

    Return 1 if the prediction meets both the translation and rotation accuracy criteria, 0 otherwise.

    Supported formats are translation xyz with length 3,
    aaxyz_nsc which is an axis and normalized sin(theta) cos(theta) with length 5,
    or xyz_aaxyz_nsc which incorporates both of the above with length 8.

    max_translation: defaults to 0.01 meters, or 1cm,
       translations must be less than this distance away.
    max_rotation: defaults to 15 degrees in radians,
       rotations must be less than this angular distance away.
    """
    length = len(y_true_xyz_aaxyz_nsc)
    if length == 3 or length == 8:
        # translation distance
        translation = absolute_cart_distance_xyz_aaxyz_nsc_single(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc)
    if length == 3:
        # translation component only
        if translation < max_translation:
            return 1.
    # translation and rotation
    elif length == 8:
        # rotation distance
        angle_distance = absolute_angle_distance_xyz_aaxyz_nsc_single(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc)
        if angle_distance < max_rotation and translation < max_translation:
            return 1.
    elif length == 5:
        # rotation distance only, just use [0.5, 0.5, 0.5] for translation component so existing code can be utilized
        fake_translation = np.array([0.5, 0.5, 0.5])
        angle_distance = absolute_angle_distance_xyz_aaxyz_nsc_single(
            np.concatenate([fake_translation, y_true_xyz_aaxyz_nsc]),
            np.concatenate([fake_translation, y_pred_xyz_aaxyz_nsc]))
        if angle_distance < max_rotation:
            return 1.
    else:
        raise ValueError('grasp_accuracy_xyz_aaxyz_nsc_single: unsupported label value format of length ' + str(length))
    return 0.


def grasp_accuracy_xyz_aaxyz_nsc_batch(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc, max_translation=0.01, max_rotation=0.261799):
    """ Calculate 3D grasp accuracy for a single result
    Expects batch of data as an nx8 array. Eager execution / numpy version.

    max_translation defaults to 0.01 meters, or 1cm.
    max_rotation defaults to 15 degrees in radians.
    """
    # print('type of y_true_xyz_aaxyz_nsc: ' + str(type(y_true_xyz_aaxyz_nsc)))
    accuracies = []
    for y_true, y_pred in zip(y_true_xyz_aaxyz_nsc, y_pred_xyz_aaxyz_nsc):
        one_accuracy = grasp_accuracy_xyz_aaxyz_nsc_single(
            y_true, y_pred, max_translation=max_translation, max_rotation=max_rotation)
        # print('one grasp acc: ' + str(one_accuracy))
        accuracies.append(one_accuracy)
    accuracies = np.array(accuracies, np.float32)
    return accuracies
