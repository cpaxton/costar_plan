#!/usr/local/bin/python
'''Converts Cornell Grasping Dataset data into TFRecords data format using Example protos.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code based on:
    https://github.com/tnikolla/robot-grasp-detection

The raw data set resides in png and txt files located in the following structure:

    dataset/03/pcd0302r.png
    dataset/03/pcd0302cpos.txt

image count: 885
labeled grasps count: 8019
positive: 5110 (64%)
negative: 2909 (36%)
object count: 244
object category count: 93


From the Cornell dataset Readme
-------------------------------

The dataset file contains 5 types of files:
1. Image files
    Named pcdxxxxr.png
    where xxxx ranges from 0000-1034
    These are the original images of the objects
2. Point cloud files
    Named pcdxxxx.txt
    where xxxx ranges from 0000-1034
3. Handlabeled grasping rectangles
    Named pcdxxxxcpos.txt for positive rectangles
    Named pcdxxxxcneg.txt for negative rectangles
4. Background images
    Named pcdb_xxxx.png
5. A mapping from each image to its background image
    Named backgroundMapping.txt

====================================================
2. Point cloud files
Point cloud files are in .PCD v.7 point cloud data file format
See http://www.pointclouds.org/documentation/tutorials/pcd_file_format.php
for more information. Each uncommented line represents a pixel in the image.
That point in space that intersects that pixel (from pcdxxxxr.png)
has x, y, and z coordinates (relative to the base of the robot that was
taking the images, so for our purposes we call this "global space").

You can tell which pixel each line refers to by the final column in each line
(labelled "index").  That number is an encoding of the row and column number
of the pixel. In all of our images, there are 640 columns and 480 rows.  Use
the following formulas to map an index to a row, col pair.
Note [in matlab] that index = 0 maps to row 1, col 1.
[Note in python that index = 0 maps to row 0, col 0.]

row = floor(index / 640) + 1
col = (index MOD 640) + 1

3. Grasping rectangle files contain 4 lines for each rectangle. Each line
contains the x and y coordinate of a vertex of that rectangle separated by
a space. [warning: we believe this next statement isn't correct]
The first two coordinates of a rectangle define the line
representing the orientation of the gripper plate. Vertices are listed in
counter-clockwise order.

[Begin comments by code authors]
The above description is unclear so we provide more detailed analysis below
from visualizing the actual *cpos.txt and *cneg.txt files,
gripper location and orientation rectangle data:
  - Coordinates are space separated floating point values in (x, y) order
  - Four rows define a single rectangle
  - Some coordinates contain NaN, and those whole rectangles must be skipped
  - The origin (0, 0) is the coordinate at the top left of the image.
  - points p0, p1 aka [(x0, y0), (x1, y1)] defines side 0 of the gap between the gripper plates
  - points p1, p2 aka [(x1, y1), (x2, y2)] defines gripper plate 0
  - points p0, p1 aka [(x2, y2), (x3, y3)] defines side 1 of the gap between the gripper plates
  - points p1, p2 aka [(x3, y3), (x0, y0)] defines gripper plate 1


The "gripper plate" is what we save as "bbox/width", this line actually defines
a range of possible gripper plate positions for successful grasps. For example,
you might imagine that a ruler could be picked up at any point along the length
of the ruler.

The "bbox/height" parameter is how open the gripper is, in other words the distance
between the gripper plates' interior surface.
[End comments by code authors]

5. The backgroundMapping file contains one line for each image in the
dataset, giving the image name and the name of the corresponding
background image separated by a space.

Obviously, in non-research settings, you would not want to force your personal
robot to take a background picture beforehand, so this is not a practical way
to handle identifying objects.  However, for the sake of concentrating only on
grasping, it is a very convenient method to subtract the backgrounds when possible.


From the second Cornell dataset README
--------------------------------------
Each of the files x,y, and z contained here are formatted for the convenience of the user.

Each of the four files contain 7037 lines.  Each line is a data sample using Marcus Lim's feature extraction along with Kerekes and Meusling's data and labels.  The line numbers map to a particular labelled rectangle.  So line 1 in each file refers to the first labelled rectangle, and line 200 in each file refers to the 200th labelled rectangle for example.

features
This file is formatted with rewards and features combined (x and y) and is already in the proper format to plug into SVM-Light.

x.txt
Each line has 1901 space-delimitted floating point values.  Line 1 is the first labelled rectangle's 1901 extracted features.  Line 7037 is the final labelled rectangle's 1901 extracted features.  Each line corresponds to the same line in y.txt and z.txt.


y.txt
Each line has the associated reward for each sample.  Line 1 has the reward for the first labelled rectangle.  Line 7037 has the reward for the final labelled rectangle.  For our purposes, each reward is either +1 or -1, meaning "good grasping rectangle" or "bad grasping rectangle."

z.txt
Each line has four space-delimitted pieces of data.  First is the image id that the rectangle is taken from (0000 through 1034).  Second is the object id (0 through 281), since most objects have multiple images.  Each object id represents a different object.  Three different bowls will have three different object ids.  Third is a short description of what the item is.  Fourth is the identifier for which background image to use if you wish to perform background subtraction.  The background image may or may not be useful for you depending on how you plan to identify the object to grasp in the image.

'''

import os
import sys
import errno
import traceback
import itertools
import six
import glob
import numpy as np

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


flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
flags.DEFINE_string('grasp_dataset', 'all', 'TODO(ahundt): integrate with brainrobotdata or allow subsets to be specified')
flags.DEFINE_boolean('objectwise_split', False,
                     """ If enabled K-Fold split based on object, image of same object will split to same fold. Default to false, doing regular imagewise K-Fold.
                     """)
flags.DEFINE_integer('num_fold', 5, 'number of fold for K-Fold splits, default to 5')
flags.DEFINE_boolean('grasp_download', False,
                     """Download the grasp_dataset to data_dir if it is not already present.""")
flags.DEFINE_boolean('plot', False, 'Plot images and grasp bounding box data in matplotlib as it is traversed')
flags.DEFINE_boolean(
    'showTextBox', False,
    """If plotting is enabled, plot extra text boxes near each grasp box
       so you can check that gripper orientation is correct.
    """)
flags.DEFINE_boolean('verbose', False, 'Print actual features for each image')
flags.DEFINE_boolean('write', False, 'Actually write the tfrecord files if True, simply gather stats if False.')
flags.DEFINE_boolean('shuffle', True, 'shuffle the image order before running')
flags.DEFINE_boolean(
    'redundant', True,
    """Duplicate images for every bounding box so dataset is easier to traverse.
       Please note that this substantially affects the output file size,
       but the dataset parsing code becomes much easier to write.
    """)
flags.DEFINE_float(
    'evaluate_fraction', 0.2,
    """proportion of dataset to be used separately for evaluation,
       use 0 if you want all files to be in one dataset file,
       which makes sense if you're going to do your splits with the tensorflow Dataset API.""")
flags.DEFINE_string('train_filename', 'cornell-grasping-dataset-train.tfrecord', 'filename used for the training dataset')
flags.DEFINE_string('evaluate_filename', 'cornell-grasping-dataset-evaluate.tfrecord', 'filename used for the evaluation dataset')
flags.DEFINE_string('stats_filename', 'cornell-grasping-dataset-stats.md', 'filename used for the dataset statistics file')


FLAGS = flags.FLAGS
FLAGS(sys.argv)


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command

    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def is_sequence(arg):
    """Returns true if arg is a list or another Python Sequence, and false otherwise.

        source: https://stackoverflow.com/a/17148334/99379
    """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


class GraspDataset(object):
    """Cornell Grasping Dataset - about 5GB total size
        http:pr.cs.cornell.edu/grasping/rect_data/data.php

        Downloads to `~/.keras/datasets/cornell_grasping` by default.

        # Arguments

        data_dir: Path to dataset in TFRecord format
            (aka Example protobufs) and feature csv files.
             `~/.keras/datasets/grasping` by default.

        dataset: 'all' to load all the data.

        download: True to actually download the dataset, also see FLAGS.
    """
    def __init__(self, data_dir=None, dataset=None, download=None, verbose=0):
        if data_dir is None:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir
        if dataset is None:
            dataset = FLAGS.grasp_dataset
        self.dataset = dataset
        if download is None:
            download = FLAGS.grasp_download
        if download:
            self.download(data_dir, dataset)
        self.verbose = verbose

    def download(self, data_dir=None, dataset='all'):
        '''Cornell Grasping Dataset - about 5GB total size

        http:pr.cs.cornell.edu/grasping/rect_data/data.php

        Downloads to `~/.keras/datasets/cornell_grasping` by default.
        Includes grasp_listing.txt with all files in all datasets;
        the feature csv files which specify the dataset size,
        the features (data channels), and the number of grasps;
        and the tfrecord files which actually contain all the data.

        If `grasp_listing_hashed.txt` is present, an additional
        hashing step will will be completed to verify dataset integrity.
        `grasp_listing_hashed.txt` will be generated automatically when
        downloading with `dataset='all'`.

        # Arguments

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.

        # Returns

           list of paths to the downloaded files

        '''
        dataset = self._update_dataset_param(dataset)
        if data_dir is None:
            if self.data_dir is None:
                data_dir = FLAGS.data_dir
            else:
                data_dir = self.data_dir
        mkdir_p(data_dir)
        print('Downloading datasets to: ', data_dir)

        url_prefix = ''
        # If a hashed version of the listing is available,
        # download the dataset and verify hashes to prevent data corruption.
        listing_hash = os.path.join(data_dir, 'grasp_listing_hash.txt')
        if os.path.isfile(listing_hash):
            files_and_hashes = np.genfromtxt(listing_hash, dtype='str', delimiter=' ')
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, file_hash=hash_str, extract=True)
                     for fpath, hash_str in tqdm(files_and_hashes)
                     if '_' + str(dataset) in fpath]
        else:
            # If a hashed version of the listing is not available,
            # simply download the dataset normally.
            listing_url = 'https://raw.githubusercontent.com/ahundt/robot-grasp-detection/master/grasp_listing.txt'
            grasp_listing_path = get_file('grasp_listing.txt', listing_url, cache_subdir=data_dir)
            grasp_files = np.genfromtxt(grasp_listing_path, dtype=str)
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, extract=True)
                     for fpath in tqdm(grasp_files)
                     if '_' + dataset in fpath]

            # If all files are downloaded, generate a hashed listing.
            if dataset is 'all' or dataset is '':
                print('Hashing all dataset files to prevent corruption...')
                hashes = []
                for i, f in enumerate(tqdm(files)):
                    hashes.append(_hash_file(f))
                file_hash_np = np.column_stack([grasp_files, hashes])
                with open(listing_hash, 'wb') as hash_file:
                    np.savetxt(hash_file, file_hash_np, fmt='%s', delimiter=' ', header='file_path sha256')
                print('Hashing complete, {} contains each url plus hash, and will be used to verify the '
                      'dataset during future calls to download().'.format(listing_hash))

        return files

    def _update_dataset_param(self, dataset):
        """Internal function to configure which subset of the datasets is being used.
        Helps to choose a reasonable default action based on previous user parameters.
        """
        if dataset is None and self.dataset is None:
            return []
        if dataset is 'all':
            dataset = ''
        if dataset is None and self.dataset is not None:
            dataset = self.dataset
        return dataset


def read_label_file(path):
    """
     based on https://github.com/falcondai/robot-grasp
    """
    with open(path) as f:
        xys = []
        has_nan = False
        for l in f:
            x, y = map(float, l.split())
            # some bounding boxes have invalid NaN coordinates, skip them
            if np.isnan(x) or np.isnan(y):
                has_nan = True
                print('this file has NaN in it')
                print(path)
            xys.append((x, y))
            if len(xys) % 4 == 0 and len(xys) / 4 >= 1:
                if not has_nan:
                    yield xys[-4], xys[-3], xys[-2], xys[-1]
                has_nan = False


def k_fold_split(path=FLAGS.data_dir, is_objectwise=FLAGS.objectwise_split, num_fold=FLAGS.num_fold):
    """ K-Fold on dataset.
        path: path to z.txt, a file match images and objects. And *pos/neg.txt, should
        remain in same folder with z.txt.
        is_objectwise: if True, do splits on different objects. Otherwise do splits on image.
        num_fold: the number of splits.

        Return: List of image_id list for each fold
    """
    if path[-1] != '/':
        path += '/'

    which_splits = [i for i in range(num_fold)]
    num_splits = [num_fold] * num_fold
    unique_image_num_list = [0 for i in range(num_fold)]
    unique_object_num_list = [0 for i in range(num_fold)]
    positive_num_list = [0 for i in range(num_fold)]
    negative_num_list = [0 for i in range(num_fold)]
    total_grasp_list = [0 for i in range(num_fold)]

    fold_last_image = ['' for i in range(num_fold)]
    fold_last_object = ['' for i in range(num_fold)]

    fold_image_id_list = [[] for i in range(num_fold)]

    if not is_objectwise:
        spilt_type_list = ['imagewise'] * num_fold
        result_path = path + 'imagewise_k_fold_stat.csv'
    else:
        spilt_type_list = ['objectwise'] * num_fold
        result_path = path + 'objectwise_k_fold_stat.csv'

    image_counter = 0
    object_counter = 0
    last_image_id = 'first_image'  # anything not '0000'
    last_object_id = 'first_object'  # anything not '0'
    with open(path + 'z.txt') as f:
        for line in f:
            image_id, object_id, _, _ = line.split()
            if image_id == last_image_id:
                continue
            else:
                last_image_id = image_id
                image_counter += 1
            path_pos = path + 'pcd' + image_id + 'cpos.txt'
            path_neg = path + 'pcd' + image_id + 'cneg.txt'
            if os.path.isfile(path_neg) and os.path.isfile(path_pos):
                if last_object_id != object_id:
                    last_object_id = object_id
                    object_counter += 1

                if is_objectwise:
                    dst_fold = (object_counter - 1) % num_fold  # make first idx 0
                else:
                    dst_fold = (image_counter - 1) % num_fold  # make first idx 0

                if fold_last_object[dst_fold] != object_id:
                    fold_last_object[dst_fold] = object_id
                    unique_object_num_list[dst_fold] += 1
                if fold_last_image[dst_fold] != image_id:
                    fold_last_image[dst_fold] = image_id
                    unique_image_num_list[dst_fold] += 1

                _, neg_pos_num = load_bounding_boxes_from_pos_neg_files(path_pos, path_neg)
                negative_num_list[dst_fold] += neg_pos_num[0]
                positive_num_list[dst_fold] += neg_pos_num[1]
                total_grasp_list[dst_fold] += (neg_pos_num[0] + neg_pos_num[1])

                # Store image_ids for each fold
                fold_image_id_list[dst_fold].append(image_id)

        info_lists = [which_splits, num_splits, unique_image_num_list,
                      unique_object_num_list, positive_num_list, negative_num_list,
                      total_grasp_list, spilt_type_list]
        head_line = ('which_splits, num_splits, unique_image, unique_object,'
                     'num_pos, num_neg, num_total_grasp, spilt_type\n')
        # mkdir_p(result_path)
        file_object = open(result_path, 'w+')
        file_object.writelines(head_line)
        for i in range(num_fold):
            cur_line = ''
            for single_list in info_lists:
                cur_line += str(single_list[i]) + ','
            file_object.writelines(cur_line + '\n')
        file_object.close()

    return fold_image_id_list


def k_fold_tfrecord_writer(path=FLAGS.data_dir, kFold_list=None, is_objectwise=FLAGS.objectwise_split):
    """ Write Tfrecord based on image_id stored in kFold_list.

        path: directory of where origin data is stored, not a file path.
        kFold_list: List of image_id list for each fold, returned from kFold_split.
        path_to_store: directory to where tfrecords are stored, not a file path,
        default same as path.
    """
    if path[-1] != '/':
        path += '/'

    if is_objectwise:
        split_type = 'objectwise'
    else:
        split_type = 'imagewise'

    coder = ImageCoder()
    for i, fold in enumerate(kFold_list):
        recordPath = path + 'cornell-grasping-dataset' + split_type + '_fold_' + str(i) + '.tfrecord'
        cur_writer = tf.python_io.TFRecordWriter(recordPath)
        for image_id in fold:
            bbox_pos_path = path + 'pcd' + image_id + 'cpos.txt'
            bbox_neg_path = path + 'pcd' + image_id + 'cneg.txt'
            image_path = path + 'pcd' + image_id + 'r.png'
            image_buffer, height, width = _process_image(image_path, coder)
            examples, _, _ = traverse_examples_in_single_image(
                image_path, bbox_pos_path, bbox_neg_path, image_buffer, height, width)
            for example in examples:
                cur_writer.write(example.SerializeToString())
        cur_writer.close()

    return


def bbox_info(box):
    """ Get the bounding box coordinates, center, tangent, angle, width and height.

        The bounding box consists of a polygon with coordinates
        going around an oriented rectangle. It is important to
        think of how this looks if the rectangle is at a 20 degree
        angle and a 120 degree angle!

        coordinates order: y0, x0, y1, x1, y2, x2, y3, x3
        coordinate index:   0,  1,  2,  3,  4,  5,  6,  7
    """
    box_coordinates = []

    for i in range(4):
        for j in range(2):
            box_coordinates.append(box[i][j])
    box = np.array(box, dtype=np.float32)
    y = np.squeeze(box[:, 1])
    x = np.squeeze(box[:, 0])
    # x0 + x2 / 2
    center_x = (x[0] + x[2])/2

    # y0 + y2 / 2
    center_y = (y[0] + y[2])/2
    center_yx = (center_y, center_x)

    # x1 - x0  (check if the bottom is level?)
    if (x[1] - x[0]) == 0:
        tan = np.pi/2
    else:
        # (y1 - y0)/ (x1 - x0)
        tan = (y[1] - y[0]) / (x[1] - x[0])
    angle = np.arctan2((y[1] - y[0]),
                       (x[1] - x[0]))
    # distance between gripper plates
    width = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
    width2 = np.sqrt((x[2] - x[3]) ** 2 + (y[2] - y[3]) ** 2)
    # gripper plate
    height = np.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2)
    height2 = np.sqrt((x[3] - x[0]) ** 2 + (y[3] - y[0]) ** 2)

    # Ensure the data actually contains rectangles
    assert np.isclose(width, width2, rtol=1e-3, atol=1e-3)
    assert np.isclose(height, height2, rtol=1e-3, atol=1e-3)

    return box_coordinates, center_yx, tan, angle, width, height


def load_bounding_boxes_from_pos_neg_files(path_pos, path_neg):
    # list of list [y0_list, x0_list, y1_list, x1_list, ...]
    coordinates_list = [[], [], [], [], [], [], [], []]
    # list of centers
    center_x_list = []
    center_y_list = []
    # list of angles
    tan_list = []
    angle_list = []
    cos_list = []
    sin_list = []
    # list of width and height
    width_list = []
    height_list = []
    # list of label success/fail, 1/0
    grasp_success = []
    count_fail_success = [0, 0]

    # coordinates_list: a list containing 8 total lists of floats.
    #     Each list contains specific coordinates for the grasping box
    #     at that index.
    #     [x0, y0, x1, y1, x2, y2, x3, y3]
    for path_label, path in enumerate([path_neg, path_pos]):
        for box in read_label_file(path):
            coordinates, center_yx, tan, angle, width, height = bbox_info(box)
            for coordinate, sublist in zip(coordinates, coordinates_list):
                sublist.append(coordinate)
            center_x_list.append(center_yx[1])
            center_y_list.append(center_yx[0])
            tan_list.append(tan)
            angle_list.append(angle)
            cos_list.append(np.cos(angle))
            sin_list.append(np.sin(angle))
            width_list.append(width)
            height_list.append(height)
            grasp_success.append(path_label)
            count_fail_success[path_label] += 1

    bbox_example_features = []
    for i in range(len(center_x_list)):
        # Build an Example proto for an example
        feature = {}
        for j in range(4):
            feature['bbox/y' + str(j)] = coordinates_list[2 * j + 1][i]
            feature['bbox/x' + str(j)] = coordinates_list[2 * j][i]
        feature['bbox/cy'] = center_y_list[i]
        feature['bbox/cx'] = center_x_list[i]
        feature['bbox/tan'] = tan_list[i]
        feature['bbox/theta'] = angle_list[i]
        feature['bbox/sin_theta'] = sin_list[i]
        feature['bbox/cos_theta'] = cos_list[i]
        feature['bbox/width'] = width_list[i]
        feature['bbox/height'] = height_list[i]
        feature['bbox/grasp_success'] = grasp_success[i]
        bbox_example_features += [feature]

    return (bbox_example_features, count_fail_success)


def gaussian_kernel_2D(size=(3, 3), center=None, sigma=1):
    """Create a 2D gaussian kernel with specified size, center, and sigma.

    All coordinates are in (y, x) order, which is (height, width),
    with (0, 0) at the top left corner.

    Output with the default parameters `size=(3, 3) center=None, sigma=1`:

        [[ 0.36787944  0.60653066  0.36787944]
         [ 0.60653066  1.          0.60653066]
         [ 0.36787944  0.60653066  0.36787944]]

    Output with parameters `size=(3, 3) center=(0, 1), sigma=1`:

        [[0.60653067 1.         0.60653067]
        [0.36787945 0.60653067 0.36787945]
        [0.082085   0.13533528 0.082085  ]]

    # Arguments

        size: dimensions of the output gaussian (height_y, width_x)
        center: coordinate of the center (maximum value) of the output gaussian, (height_y, width_x).
            Default of None will automatically be the center coordinate of the output size.
        sigma: standard deviation of the gaussian in pixels

    # References:

            https://stackoverflow.com/a/43346070/99379
            https://stackoverflow.com/a/32279434/99379

    # How to normalize

        g = gaussian_kernel_2d()
        g /= np.sum(g)
    """
    if center is None:
        center = np.array(size) / 2
    yy, xx = np.meshgrid(np.arange(size[0]),
                         np.arange(size[1]),
                         indexing='ij')
    kernel = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2. * sigma ** 2))
    return kernel


class ImageCoder(object):
    # probably based on https://github.com/visipedia/tfrecords
    def __init__(self):
        self._sess = tf.Session()
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def decode_png(self, image_data):
        return self._sess.run(self._decode_png,
                              feed_dict={self._decode_png_data: image_data})


def _process_image(filename, coder):
    # Decode the image
    with open(filename) as f:
        image_data = f.read()
    image = coder.decode_png(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def add_one_gaussian(image, center, grasp_theta, grasp_width, grasp_height, label, sigma_divisor=10):
    """ Compare to ground_truth_image in grasp_img_proc.py
    """
    sigma = max(grasp_width, grasp_height) / sigma_divisor
    # make sure center value for gaussian is 0.5
    gaussian = gaussian_kernel_2D((image.shape[0], image.shape[1]), center=center, sigma=sigma)
    # label 0 is grasp failure, label 1 is grasp success, label 0.5 will have no effect.
    # gaussian center with label 0 should be subtracting 0.5
    # gaussian center with label 1 should be adding 0.5
    gaussian = ((label * 2) - 1.0) * gaussian
    image = image + gaussian
    return image


def ground_truth_images(
        image_shape,
        grasp_cys, grasp_cxs,
        grasp_thetas,
        grasp_heights, grasp_widths,
        labels):
    gt_images = []
    if not isinstance(grasp_cys, list):
        grasp_cys = [grasp_cys]
        grasp_cxs = [grasp_cxs]
        grasp_thetas = [grasp_thetas]
        grasp_heights = [grasp_heights]
        grasp_widths = [grasp_widths]
        labels = [labels]

    for (grasp_cy, grasp_cx, grasp_theta,
         grasp_height, grasp_width, label) in zip(grasp_cys, grasp_cxs,
                                                  grasp_thetas, grasp_heights,
                                                  grasp_widths, labels):
        gt_image = np.zeros(image_shape[:2])
        gt_image = add_one_gaussian(
            gt_image, (grasp_cy, grasp_cx), grasp_theta,
            grasp_height, grasp_width, label)
        max_num = max(np.max(gt_image), 1.0)
        min_num = min(np.min(gt_image), -1.0)
        gt_image = (gt_image - min_num) / (max_num - min_num)
        gt_images += [gt_image]

    return gt_images


def visualize_example(img, bbox_example_features, gt_images, showTextBox=FLAGS.showTextBox):
    width = 3

    center_x_list = [example['bbox/cx'] for example in bbox_example_features]
    center_y_list = [example['bbox/cy'] for example in bbox_example_features]
    grasp_success = [example['bbox/grasp_success'] for example in bbox_example_features]
    gt_plot_height = len(center_x_list)/2
    fig, axs = plt.subplots(gt_plot_height + 1, 4, figsize=(15, 15))
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
    for i, (gt_image, example) in enumerate(zip(gt_images, bbox_example_features)):
        h = i % gt_plot_height + 1
        w = int(i / gt_plot_height)
        z = 0
        axs[h, w].imshow(img, zorder=z)
        z += 1
        axs[h, w].imshow(gt_image, alpha=0.25, zorder=z)
        z += 1
        # axs[h, w*2+1].imshow(gt_image, alpha=0.75, zorder=1)
        widths = [1, 2, 1, 2]
        alphas = [0.25, 0.5, 0.25, 0.5]
        if example['bbox/grasp_success']:
            # that's gap, plate, gap plate
            colors = ['gray', 'green', 'gray', 'green']
            success_str = 'pos'
        else:
            colors = ['gray', 'purple', 'gray', 'purple']
            success_str = 'neg'

        if showTextBox:
            axs[h, w].text(
                    example['bbox/cx'], example['bbox/cy'],
                    success_str, size=10, rotation=-np.rad2deg(example['bbox/theta']),
                    ha="right", va="top",
                    bbox=dict(boxstyle="square",
                              ec=(1., 0.5, 0.5),
                              fc=(1., 0.8, 0.8),
                              ),
                    zorder=z,
                    )
            z += 1
        for i, (color, width, alpha) in enumerate(zip(colors, widths, alphas)):
            x_current = [example['bbox/x'+str(i)], example['bbox/x'+str((i+1)%4)]]
            y_current = [example['bbox/y'+str(i)], example['bbox/y'+str((i+1)%4)]]
            # axs[h, w].text(example['bbox/x'+str(i)], example['bbox/y'+str(i)], "Point:"+str(i))

            axs[h, w].add_line(lines.Line2D(x_current, y_current, linewidth=width,
                               color=color, zorder=z, alpha=alpha))
            axs[0, 0].add_line(lines.Line2D(x_current, y_current, linewidth=width,
                               color=color, zorder=z, alpha=alpha))

    # axs[1, 1].hist2d(data[0], data[1])
    plt.draw()
    plt.pause(0.25)

    plt.show()
    return width


def _process_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.

    https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    """Wrapper for inserting float features into Example proto.

    https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.

    https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""

    if isinstance(text, str):
        return text
    elif isinstance(text, unicode):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)


def _create_examples(
        filename, image_id, image_buffer, height, width, dict_bbox_lists):
    """

    Create a TFRecord example which stores multiple bounding boxe copies with a single image.
    This makes lists of coordinates so that images are never repeated.

    # Arguments

        dict_bbox_lists: A dictionary containing lists of feature values.

    # Returns

      A list of examples
    """

    # Build an Example proto for an example
    feature = {'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(image_buffer),
               'image/height': _int64_feature(height),
               'image/width': _int64_feature(width),
               'image/id': _int64_feature(image_id)}
    for i in range(4):
        feature['bbox/y' + str(i)] = _floats_feature(dict_bbox_lists['bbox/y' + str(i)])
        feature['bbox/x' + str(i)] = _floats_feature(dict_bbox_lists['bbox/x' + str(i)])
    feature['bbox/cy'] = _floats_feature(dict_bbox_lists['bbox/cy'])
    feature['bbox/cx'] = _floats_feature(dict_bbox_lists['bbox/cx'])
    feature['bbox/tan'] = _floats_feature(dict_bbox_lists['bbox/tan'])
    feature['bbox/theta'] = _floats_feature(dict_bbox_lists['bbox/theta'])
    feature['bbox/sin_theta'] = _floats_feature(dict_bbox_lists['bbox/sin_theta'])
    feature['bbox/cos_theta'] = _floats_feature(dict_bbox_lists['bbox/cos_theta'])
    feature['bbox/width'] = _floats_feature(dict_bbox_lists['bbox/width'])
    feature['bbox/height'] = _floats_feature(dict_bbox_lists['bbox/height'])
    feature['bbox/grasp_success'] = _int64_feature(dict_bbox_lists['bbox/grasp_success'])
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return [example]


def _create_examples_redundant(
        filename, image_id, image_buffer, height, width, bbox_example_features):
    """

    Write the same image example repeatedly, once for each single bounding box example

    All lists of coordinates are size 1, makes dataset easier to read because you
    can assume there is exactly one bounding box per image, the dataset
    becomes substantially larger on disk. Note that this larger size is still
    smaller than the original format from the cornell website because it is compressed.

    # Arguments

        bbox_example_features: A list of dictionaries, each containing example features
            in basic numerical format (numpy array, int float).
    """
    examples = []
    for i, bbox_dict in enumerate(bbox_example_features):
        # Build an Example proto for an example
        feature = {'image/filename': _bytes_feature(filename),
                   'image/encoded': _bytes_feature(image_buffer),
                   'image/height': _int64_feature(height),
                   'image/width': _int64_feature(width),
                   'image/id': _int64_feature(image_id)}
        for j in range(4):
            feature['bbox/y' + str(j)] = _floats_feature(bbox_dict['bbox/y' + str(j)])
            feature['bbox/x' + str(j)] = _floats_feature(bbox_dict['bbox/x' + str(j)])
        feature['bbox/cy'] = _floats_feature(bbox_dict['bbox/cy'])
        feature['bbox/cx'] = _floats_feature(bbox_dict['bbox/cx'])
        feature['bbox/tan'] = _floats_feature(bbox_dict['bbox/tan'])
        feature['bbox/theta'] = _floats_feature(bbox_dict['bbox/theta'])
        feature['bbox/sin_theta'] = _floats_feature(bbox_dict['bbox/sin_theta'])
        feature['bbox/cos_theta'] = _floats_feature(bbox_dict['bbox/cos_theta'])
        feature['bbox/width'] = _floats_feature(bbox_dict['bbox/width'])
        feature['bbox/height'] = _floats_feature(bbox_dict['bbox/height'])
        feature['bbox/grasp_success'] = _int64_feature(bbox_dict['bbox/grasp_success'])
        examples += [tf.train.Example(features=tf.train.Features(feature=feature))]

    return examples


def get_image_id_from_filename(filename):
    """ Get the object id from the filename, assumes all digits are part of the object id.
    """
    return int(list(filter(str.isdigit, filename))[0])


def list_of_dicts_to_dict_of_lists(ld):
    """ list of dictionaries to dictionary of lists when all keys are the same.

    source: https://stackoverflow.com/a/23551944/99379
    """
    return {key: [item[key] for item in ld] for key in ld[0].keys()}


def traverse_examples_in_single_image(filename, path_pos, path_neg, image_buffer, height, width, verbose=FLAGS.verbose):
    """
    """
    # get the bounding box information as lists of dictionaries storing feature data as floats and ints
    (bbox_example_features, count_fail_success) = load_bounding_boxes_from_pos_neg_files(path_pos, path_neg)

    attempt_count = len(bbox_example_features)
    dict_bbox_lists = list_of_dicts_to_dict_of_lists(bbox_example_features)
    if verbose:
        print('filename: ' + filename)
        print('image height: ' + str(height))
        print('image width: ' + str(width))
        print(dict_bbox_lists)
        print('-----------------------')

    # visualize this example
    if FLAGS.plot:
        gt_images = ground_truth_images([height, width],
                                        dict_bbox_lists['bbox/cy'],
                                        dict_bbox_lists['bbox/cx'],
                                        dict_bbox_lists['bbox/theta'],
                                        dict_bbox_lists['bbox/height'],
                                        dict_bbox_lists['bbox/width'],
                                        dict_bbox_lists['bbox/grasp_success'])
        # load the image with matplotlib for display
        img = mpimg.imread(filename)
        visualize_example(img, bbox_example_features, gt_images)

    # get the object id from the filename
    image_id = get_image_id_from_filename(filename)

    # create the tfrecord example protobufs
    if FLAGS.redundant:
        examples = _create_examples_redundant(filename, image_id, image_buffer, height, width, bbox_example_features)
    else:
        examples = _create_examples(filename, image_id, image_buffer, height, width, dict_bbox_lists)

    return examples, attempt_count, count_fail_success


def traverse_dataset(filenames, eval_fraction=FLAGS.evaluate_fraction, write=FLAGS.write, train_file=None, validation_file=None):
    coder = ImageCoder()
    image_count = len(filenames)
    train_image_count = 0
    eval_image_count = 0
    train_attempt_count = 0
    eval_attempt_count = 0
    total_attempt_count = 0
    train_fail_success_count = [0, 0]
    eval_fail_success_count = [0, 0]
    steps_per_eval = int(np.ceil(1.0 / eval_fraction))

    if write:
        writer_train = tf.python_io.TFRecordWriter(train_file)
        writer_validation = tf.python_io.TFRecordWriter(validation_file)

    for i, filename in enumerate(tqdm(filenames)):
        bbox_pos_path = filename[:-5]+'cpos.txt'
        bbox_neg_path = filename[:-5]+'cneg.txt'
        image_buffer, height, width = _process_image(filename, coder)
        examples, attempt_count, count_fail_success = traverse_examples_in_single_image(
            filename, bbox_pos_path, bbox_neg_path, image_buffer, height, width)

        # Split the dataset in 80% for training and 20% for validation
        total_attempt_count += attempt_count
        if i % steps_per_eval == 0:
            eval_image_count += 1
            eval_attempt_count += attempt_count
            eval_fail_success_count[0] += count_fail_success[0]
            eval_fail_success_count[1] += count_fail_success[1]
            if write:
                for example in examples:
                    writer_validation.write(example.SerializeToString())
        else:
            train_image_count += 1
            train_attempt_count += attempt_count
            train_fail_success_count[0] += count_fail_success[0]
            train_fail_success_count[1] += count_fail_success[1]
            if write:
                for example in examples:
                    writer_train.write(example.SerializeToString())

    if write:
        writer_train.close()
        writer_validation.close()

    return (image_count, total_attempt_count, train_image_count, eval_image_count,
            train_attempt_count, eval_attempt_count, train_fail_success_count,
            eval_fail_success_count)


def get_cornell_grasping_dataset_filenames(data_dir=FLAGS.data_dir, shuffle=FLAGS.shuffle):
    # Creating a list with all the image paths
    folders = range(1, 11)
    folders = ['0'+str(i) if i < 10 else '10' for i in folders]
    png_filenames = []

    for i in folders:
        for name in glob.glob(os.path.join(data_dir, i, 'pcd' + i + '*r.png')):
            png_filenames.append(name)

    if shuffle:
        # Shuffle the list of image paths
        np.random.shuffle(png_filenames)

    bbox_successful_filenames = []
    bbox_failure_filenames = []
    for filename in png_filenames:
        bbox_pos_path = filename[:-5]+'cpos.txt'
        bbox_neg_path = filename[:-5]+'cneg.txt'
        bbox_successful_filenames += [bbox_pos_path]
        bbox_successful_filenames += [bbox_neg_path]
    return png_filenames, bbox_successful_filenames, bbox_failure_filenames


def get_stat(name, amount, total, percent_description=''):
    return ' - %s %s, ' % (amount, name) + "{0:.2f}".format(100.0 * amount/total) + ' percent' + percent_description


def main():

    # plt.ion()
    gd = GraspDataset()
    if FLAGS.grasp_download:
        gd.download(dataset=FLAGS.grasp_dataset)

    # Creating a list with all the image paths
    png_filenames, _, _ = get_cornell_grasping_dataset_filenames()

    train_file = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
    validation_file = os.path.join(FLAGS.data_dir, FLAGS.evaluate_filename)
    stats_file = os.path.join(FLAGS.data_dir, FLAGS.stats_filename)
    print(train_file)
    print(validation_file)
    if not FLAGS.write:
        print('WARNING: Gathering stats that WILL NOT BE WRITTEN TO A FILE'
              ' training and evaluation stats will not '
              'be valid for any existing tfrecord file.'
              'To write to a file run python build_cgd_dataset.py --write.')

    (image_count, total_attempt_count, train_image_count, eval_image_count,
     train_attempt_count, eval_attempt_count, train_fail_success_count,
     eval_fail_success_count) = traverse_dataset(png_filenames, train_file=train_file, validation_file=validation_file)

    total_success_count = train_fail_success_count[1] + eval_fail_success_count[1]
    total_fail_count = train_fail_success_count[0] + eval_fail_success_count[0]

    stat_string = ''

    stat_string += '\n' + ('Cornell Grasping Dataset')
    stat_string += '\n' + ('------------------------')
    stat_string += '\n' + ('')
    stat_string += '\n' + ('TFRecord generation complete. Saved files:\n\n - %s\n - %s\n - %s' % (train_file, validation_file, stats_file))
    stat_string += '\n' + ('')
    stat_string += '\n' + ('Dataset Statistics')
    stat_string += '\n' + ('---------------')
    stat_string += '\n' + ('')
    stat_string += '\n' + ('### Totals')
    stat_string += '\n' + (' - %s images' % image_count)
    stat_string += '\n' + (' - %s grasp attempts' % total_attempt_count)
    stat_string += '\n' + get_stat('successful grasps', total_success_count, total_attempt_count)
    stat_string += '\n' + get_stat('failed grasps', total_fail_count, total_attempt_count)
    stat_string += '\n' + ('')
    stat_string += '\n' + ('### Training Data')
    stat_string += '\n' + get_stat('images', train_image_count, image_count)
    stat_string += '\n' + get_stat('grasp attempts', train_attempt_count, total_attempt_count, ' of total')
    stat_string += '\n' + get_stat('successful grasps', train_fail_success_count[1], train_attempt_count, ' of training data')
    stat_string += '\n' + get_stat('failed grasps', train_fail_success_count[0], train_attempt_count, ' of training data')
    stat_string += '\n' + ('')
    stat_string += '\n' + ('### Evaluation Data')
    stat_string += '\n' + get_stat('images', eval_image_count, image_count)
    stat_string += '\n' + get_stat('grasp attempts', eval_attempt_count, total_attempt_count, ' of toal')
    stat_string += '\n' + get_stat('successful grasps', eval_fail_success_count[1], eval_attempt_count, ' of eval data')
    stat_string += '\n' + get_stat('failed grasps', eval_fail_success_count[0], eval_attempt_count, ' of eval data')
    stat_string += '\n' + ('')

    if FLAGS.write:
        with open(FLAGS.stats_filename, "w") as text_file:
            text_file.write(stat_string)

    print(stat_string)

if __name__ == '__main__':
    main()
