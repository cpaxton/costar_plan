
import os
import sys
import errno
import traceback
import itertools
import six
import glob
import copy
import numpy as np
from random import shuffle

import tensorflow as tf
import re
from scipy.ndimage.filters import median_filter
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras._impl.keras.utils.data_utils import _hash_file
import keras
from keras import backend as K

import grasp_utilities
import grasp_metrics


def draw_grasp(axs, grasp_success, center, theta, x_current=None, y_current=None, z=2, showTextBox=False, show_center=True, title=None):
    """ x_current and y_current are the lists of pairs of line coodinates extracted from the polygon
    """

    widths = [1, 2, 1, 2]
    alphas = [0.25, 0.5, 0.25, 0.5]
    cx, cy = center
    if grasp_success:
        # gray is width, color (purple/green) is plate
        # that's gap, plate, gap plate
        colors = ['gray', 'green', 'gray', 'green']
        success_str = 'pos'
        center_color = 'green'
    else:
        colors = ['gray', 'purple', 'gray', 'purple']
        success_str = 'neg'
        center_color = 'green'

    if show_center:
        axs.scatter(np.array([cx]), np.array([cy]), zorder=z, c=center_color, alpha=0.5, lw=2)
        z += 1

    if showTextBox:
        axs.text(
            cx, cy,
            success_str, size=10, rotation=-np.rad2deg(theta),
            ha="right", va="top",
            bbox=dict(boxstyle="square",
                      ec=(1., 0.5, 0.5),
                      fc=(1., 0.8, 0.8)),
            zorder=z,
            )
        z += 1
    if x_current is not None and y_current is not None:
        for i, (color, width, alpha, x, y) in enumerate(zip(colors, widths, alphas, x_current, y_current)):
            # axs[h, w].text(example['bbox/x'+str(i)], example['bbox/y'+str(i)], "Point:"+str(i))
            axs.add_line(lines.Line2D(x, y, linewidth=width,
                                      color=color, zorder=z, alpha=alpha))

    if title is not None:
        if grasp_success:
            title += ' - positive example - grasp success'
        else:
            title += ' - negative example - grasp failure'
        axs.set_title(title)
    return z


def print_feature(feature_map, feature_name):
    """ Print the contents of a feature map
    """
    print(feature_name + ': ' + str(feature_map[feature_name]))


def get_grasp_polygon_lines_from_example(example):
    """ For use with matplotlib Line2D and draw_grasp()
    """
    num_bboxes = 4
    x_current = [[] for k in range(num_bboxes)]
    y_current = [[] for k in range(num_bboxes)]
    for j in range(num_bboxes):
        x_current[j] += [example['bbox/x'+str(j)], example['bbox/x'+str((j+1) % 4)]]
        y_current[j] += [example['bbox/y'+str(j)], example['bbox/y'+str((j+1) % 4)]]
    return x_current, y_current


def get_grasp_polygon_lines_from_polygon(polygon):
    """ For use with matplotlib Line2D and draw_grasp()

    y, x order!

    really a polygon format rectangle [[yx], [yx], [yx], [yx]]
    """
    num_bboxes = 4
    x_current = [[] for k in range(num_bboxes)]
    y_current = [[] for k in range(num_bboxes)]
    y_idx = 0
    x_idx = 1
    for j in range(num_bboxes):
        y_current[j] += [polygon[j][y_idx], polygon[(j+1) % 4][y_idx]]
        x_current[j] += [polygon[j][x_idx], polygon[(j+1) % 4][x_idx]]
    return y_current, x_current


def unit_coordinates_to_image_coordinates(y_current, x_current, center, height, width):
    """ y, x order
    """
    y_current = [[y0 * height, y1 * height] for y0, y1 in y_current]
    x_current = [[x0 * width, x1 * width] for x0, x1 in x_current]
    center[0] = center[0] * height
    center[1] = center[1] * height
    return y_current, x_current, center


def decode_prediction_for_matplotlib(prediction, img2):
    theta, center, rectangle_polygon = grasp_metrics.decode_prediction_vector_theta_center_polygon(prediction)
    # adjust from unit coordinates to image coordinates
    ih, iw, _ = img2.shape
    y_current, x_current = get_grasp_polygon_lines_from_polygon(rectangle_polygon)
    y_current, x_current, center = unit_coordinates_to_image_coordinates(
        y_current=y_current,
        x_current=x_current,
        center=center,
        height=ih, width=iw)
    return center, theta, y_current, x_current


def draw_grasp_prediction_matplotlib(axs, prediction, image, grasp_success, z, showTextBox, title=None):
    # axs is just a single entry here and not a 2d array
    center, theta, y_current, x_current = decode_prediction_for_matplotlib(prediction, image)
    axs.imshow(image, alpha=1, zorder=z)
    z = draw_grasp(
        axs=axs,
        grasp_success=grasp_success,
        center=center,
        theta=theta,
        z=z,
        y_current=y_current,
        x_current=x_current,
        showTextBox=showTextBox,
        title=title)
    return z


def visualize_redundant_images_example(
        features_dicts, predictions=None, predictions_grasp_success=True, showTextBox=None, figcols=2,
        show=True, blocking=False, save_filename=None, close=True, verbose=1):
    """ Visualize numpy dictionary containing a grasp example.
    """
    # if showTextBox is None:
    #     showTextBox = FLAGS.showTextBox

    if not isinstance(features_dicts, list):
        features_dicts = [features_dicts]

    if predictions is None:
        predictions = [None] * len(features_dicts)
    elif not isinstance(predictions, list):
        predictions = [predictions]

    if isinstance(predictions_grasp_success, bool):
        predictions_grasp_success = [predictions_grasp_success] * len(features_dicts)

    preprocessed_examples = []
    for example in features_dicts:
        if verbose > 1:
            print('original example bbox/theta: ' + str(example['bbox/theta']))
        if ('bbox/preprocessed/cy_cx_normalized_2' in example):
            if verbose > 0:
                print_feature(example, 'bbox/preprocessed/cy_cx_normalized_2')
                print_feature(example, 'bbox/preprocessed/cy')
                print_feature(example, 'bbox/preprocessed/cx')
                print_feature(example, 'bbox/preprocessed/width')
                print_feature(example, 'bbox/preprocessed/height')
                print_feature(example, 'grasp_success_norm_sin2_cos2_hw_yx_7')
            # Reverse the preprocessing so we can visually compare correctness
            decoded_example = copy.deepcopy(example)
            sin_cos_2 = np.squeeze(example['bbox/preprocessed/sin_cos_2'])
            # y, x ordering.
            recovered_theta = np.arctan2(sin_cos_2[0], sin_cos_2[1])
            if 'random_rotation' in example:
                recovered_theta += example['random_rotation']
            cy_cx_normalized_2 = np.squeeze(example['bbox/preprocessed/cy_cx_normalized_2'])
            cy_cx_normalized_2[0] *= np.squeeze(example['image/preprocessed/height'])
            cy_cx_normalized_2[1] *= np.squeeze(example['image/preprocessed/width'])
            if 'random_translation_offset' in example:
                offset = example['random_translation_offset']
                print('offset: ' + str(offset))
            # if np.allclose(np.array(example['bbox/theta']), recovered_theta):
            #     print('WARNING: bbox/theta: ' + str(example['bbox/theta']) +
            #           ' feature does not match bbox/preprocessed/sin_cos_2: '
            #           )
            # # change to preprocessed
            # assert np.allclose(cy_cx_normalized_2[0] + offset[0], example['bbox/cy'])
            # assert np.allclose(cy_cx_normalized_2[1] + offset[1], example['bbox/cx'])
            decoded_example['bbox/theta'] = example['bbox/preprocessed/theta']
            decoded_example['image/decoded'] = example['image/preprocessed']
            decoded_example['bbox/width'] = example['bbox/preprocessed/width']
            decoded_example['bbox/height'] = example['bbox/preprocessed/height']
            decoded_example['bbox/cy'] = example['bbox/preprocessed/cy']
            decoded_example['bbox/cx'] = example['bbox/preprocessed/cx']
            if verbose > 0 and 'random_projection_transform' in example:
                print('random_projection_transform:' + str(example['random_projection_transform']))
                if 'random_rotation' in example:
                    print('random_rotation: ' + str(example['random_rotation']))
                print('bbox/preprocessed/theta: ' + str(example['bbox/preprocessed/theta']))
            preprocessed_examples.append(decoded_example)

    img = np.squeeze(example['image/decoded'])
    center_x_list = [example['bbox/cx'] for example in features_dicts]
    center_y_list = [example['bbox/cy'] for example in features_dicts]
    grasp_success = [example['bbox/grasp_success'] for example in features_dicts]
    gt_plot_height = int(np.ceil(float(len(center_x_list)) / 2))
    fig, axs = plt.subplots(gt_plot_height + 1, figcols, figsize=(15, 15))
    if verbose > 1:
        print('max: ' + str(np.max(img)) + ' min: ' + str(np.min(img)))
    axs[0, 0].imshow(np.squeeze(img), zorder=0)
    axs[0, 0].set_title('Original Image with Grasp')
    # for i in range(4):
    #     feature['bbox/y' + str(i)] = _floats_feature(dict_bbox_lists['bbox/y' + str(i)])
    #     feature['bbox/x' + str(i)] = _floats_feature(dict_bbox_lists['bbox/x' + str(i)])
    # axs[0, 0].arrow(np.array(center_y_list), np.array(center_x_list),
    #                 np.array(coordinates_list[0]) - np.array(coordinates_list[2]),
    #                 np.array(coordinates_list[1]) - np.array(coordinates_list[3]), c=grasp_success)
    axs[0, 0].scatter(np.array(center_x_list), np.array(center_y_list), zorder=2, c=grasp_success, alpha=0.5, lw=2)
    axs[0, 1].imshow(np.squeeze(img), zorder=0)
    axs[0, 1].set_title('Original Image')
    # plt.show()
    # axs[1, 0].scatter(data[0], data[1])
    # axs[2, 0].imshow(gt_image)
    for i, (example, prediction, prediction_grasp_success) in enumerate(zip(preprocessed_examples, predictions, predictions_grasp_success)):
        plot_idx = i * 2
        h, w = plot_coordinate(plot_idx, gt_plot_height)
        h_pred, w_pred = plot_coordinate(plot_idx + 1, gt_plot_height)
        z = 0
        # axs[h, w].imshow(img, zorder=z)
        z += 1

        # this is really the preprocessed image if preprocessed data is present
        img2 = np.squeeze(example['image/decoded'])
        # Assuming 'tf' preprocessing mode! Changing channel range from [-1, 1] to [0, 1]
        img2 /= 2
        img2 += 0.5
        if verbose > 1:
            print('preprocessed max: ' + str(np.max(img2)) + ' min: ' + str(np.min(img2)) + ' shape: ' + str(np.shape(img2)))
        axs[h, w].imshow(img2, alpha=1, zorder=z)
        grasp_success = example['bbox/grasp_success']

        # TODO(ahundt) get xcurrent ycurrent
        z = draw_grasp(
            axs=axs[h, w],
            grasp_success=grasp_success,
            center=(float(example['bbox/cx']), float(example['bbox/cy'])),
            theta=float(example['bbox/theta']),
            z=z,
            showTextBox=showTextBox)

        # Draw the ground truth encoded grasp
        gt_prediction = np.squeeze(example['grasp_success_norm_sin2_cos2_hw_yx_7'])
        if verbose > 0:
            print('gt_encoded grasp_success_norm_sin2_cos2_hw_yx_7: ' + str(gt_prediction))
        z = draw_grasp_prediction_matplotlib(
            axs[h, w],
            prediction=gt_prediction,
            image=img2,
            grasp_success=grasp_success,
            z=z,
            showTextBox=showTextBox,
            title='Ground Truth')
        if prediction is not None:
            if verbose > 0:
                print('encoded nn prediction norm_sin2_cos2_hw_yx_6: ' + str(prediction))
            # draw the actual prediction, note that predictions
            z = draw_grasp_prediction_matplotlib(axs[h_pred, w_pred], prediction, img2, prediction_grasp_success, z, showTextBox,
                                                 title='Prediction')

        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename)
        if show:
            plt.draw()
            plt.pause(0.001)
            if blocking:
                plt.show()
    if close:
        plt.close()
    # axs[1, 1].hist2d(data[0], data[1])


def visualize_example(
        img, bbox_example_features, gt_images=None, showTextBox=False,
        show=True, blocking=True, save_filename=None, close=True,
        figsize=(15, 15)):
    """ Visualize an example from the dataset with multiple grasp labels.

    # Arguments
        image: the rgb image associated with this example.
        bbox_example_features: A list of dictionaries. Each dictionary maps from string
            keys to a value containing example features in basic numerical or text format
            (numpy array, int float, string), with data about the grasp location,
            success status, etc.
        gt_images: optional list one image for each bbox_example_features list item
            containing a pixel-wise ground truth status.
        showTextBox: Should a text box be displayed showing the orientation and pos/neg label.
            This is useful to make sure orientation and rotation math is correct, but sometimes
            gets in the way so it can be enabled and disabled.
    """
    if gt_images is None:
        gt_images = [None] * len(bbox_example_features)

    if isinstance(bbox_example_features, dict):
        example_format = 'dict_of_lists'
        bbox_example_features = [bbox_example_features]
    else:
        example_format = 'list_of_dicts'

    center_x_list = []
    center_y_list = []
    grasp_success = []

    gt_plot_height = None
    bbox_count = None
    for example in bbox_example_features:
        center_x_list += [example['bbox/cx']]
        center_y_list += [example['bbox/cy']]
        grasp_success += [example['bbox/grasp_success']]
        if 'bbox/count' in example:
            bbox_count = example['bbox/count'][0]
            gt_plot_height = bbox_count / 2

    if gt_plot_height is None:
        # The data is stored a list of dictionaries containing single data values
        gt_plot_height = len(center_x_list)/2
    else:
        # The data is stored as lists or arrays in a single dictionary
        # Remove padded values from the lists
        center_x_list = np.squeeze(center_x_list)
        center_y_list = np.squeeze(center_x_list)
        grasp_success = np.squeeze(grasp_success)
        center_x_list = center_x_list[:bbox_count]
        center_y_list = center_x_list[:bbox_count]
        grasp_success = grasp_success[:bbox_count]

    fig, axs = plt.subplots(gt_plot_height + 1, 4, figsize=figsize)
    axs[0, 0].imshow(img, zorder=0)
    # for i in range(4):
    #     feature['bbox/y' + str(i)] = _floats_feature(dict_bbox_lists['bbox/y' + str(i)])
    #     feature['bbox/x' + str(i)] = _floats_feature(dict_bbox_lists['bbox/x' + str(i)])
    # axs[0, 0].arrow(np.array(center_y_list), np.array(center_x_list),
    #                 np.array(coordinates_list[0]) - np.array(coordinates_list[2]),
    #                 np.array(coordinates_list[1]) - np.array(coordinates_list[3]), c=grasp_success)
    axs[0, 0].scatter(np.array(center_x_list), np.array(center_y_list), zorder=2, c=grasp_success, alpha=0.5, lw=2)
    axs[0, 1].imshow(img, zorder=0)
    # axs[1, 0].scatter(data[0], data[1])
    # axs[2, 0].imshow(gt_image)
    if example_format == 'list_of_dicts':
        # The data is stored a list of dictionaries containing single data values
        for i, (gt_image, example) in enumerate(zip(gt_images, bbox_example_features)):
            grasp_success = example['bbox/grasp_success']
            cx = example['bbox/cx']
            cy = example['bbox/cy']
            x_current, y_current = get_grasp_polygon_lines_from_example(example)
            h = i % gt_plot_height + 1
            w = int(i / gt_plot_height)
            z = 0
            axs[h, w].imshow(img, zorder=z)
            z += 1
            if gt_image is not None:
                axs[h, w].imshow(gt_image, alpha=0.25, zorder=z)
            z += 1
            # axs[h, w*2+1].imshow(gt_image, alpha=0.75, zorder=1)
            theta = example['bbox/theta']
            z = draw_grasp(axs[h, w], grasp_success, (cx, cy), theta, x_current, y_current, z=z, showTextBox=showTextBox)
            z = draw_grasp(axs[0, 0], grasp_success, (cx, cy), theta, x_current, y_current, z=z, showTextBox=showTextBox)
    else:
        # The data is stored as lists or arrays in a single dictionary
        for k, (gt_image, example) in enumerate(zip(gt_images, bbox_example_features)):
            print('example[bbox/grasp_success]: ' + str(example['bbox/grasp_success']))
            # assume bbox/count is present
            for i in range(example['bbox/count']):
                grasp_success = np.squeeze(example['bbox/grasp_success'])[i]
                cx = np.squeeze(example['bbox/cx'])[i]
                cy = np.squeeze(example['bbox/cy'])[i]

                num_bboxes = 4
                x_current = [[] for k in range(num_bboxes)]
                y_current = [[] for k in range(num_bboxes)]
                for j in range(num_bboxes):
                    x_current[j] += [np.squeeze(example['bbox/x'+str(j)])[i], np.squeeze(example['bbox/x'+str((j+1) % 4)])[i]]
                    y_current[j] += [np.squeeze(example['bbox/y'+str(j)])[i], np.squeeze(example['bbox/y'+str((j+1) % 4)])[i]]
                # x_current, y_current = get_grasp_polygon_lines_from_example(example)
                h = i % gt_plot_height + 1
                w = int(i / gt_plot_height)
                z = 0
                axs[h, w].imshow(img, zorder=z)
                z += 1
                if gt_image is not None:
                    axs[h, w].imshow(gt_image, alpha=0.25, zorder=z)
                z += 1
                # axs[h, w*2+1].imshow(gt_image, alpha=0.75, zorder=1)
                theta = np.squeeze(example['bbox/theta'])[i]
                z = draw_grasp(axs[h, w], grasp_success, (cx, cy), theta, x_current, y_current, z=z, showTextBox=showTextBox)
                z = draw_grasp(axs[0, 0], grasp_success, (cx, cy), theta, x_current, y_current, z=z, showTextBox=showTextBox)

    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename)
    if show:
        plt.draw()
        plt.pause(0.001)
        if blocking:
            plt.show()
    if close:
        plt.close()


def plot_coordinate(i, gt_plot_height):
    """ Determine the subplot height, width coordinate to use in a figure.
    """
    h = i % gt_plot_height + 1
    w = int(i / gt_plot_height)
    return h, w
