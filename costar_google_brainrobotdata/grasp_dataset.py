"""Code for loading data from the google brain robotics grasping dataset.

https://sites.google.com/site/brainrobotdata/home/grasping-dataset

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import errno
import traceback
import itertools
import six

import numpy as np
import tensorflow as tf
import re
from scipy.ndimage.filters import median_filter

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

# TODO(ahundt) importing moviepy prevented python from exiting, uncomment lines when fixed.
# try:
#     import moviepy.editor as mpy
# except ImportError:
#     print('moviepy not available, try `pip install moviepy`. '
#           'Skipping dataset gif extraction components.')

import grasp_geometry
import grasp_geometry_tf
import depth_image_encoding
import random_crop as rcp
import inception_preprocessing
from grasp_median_filter import grasp_dataset_median_filter

# DATASET LOADING CONFIGURATION COMMAND LINE PARAMETERS, see GraspDataset()
flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
flags.DEFINE_string('visualization_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'grasping', 'images_extracted_grasp'),
                    """Path to output data visualizations such as image gifs and ply clouds.""")
flags.DEFINE_integer('batch_size', 6, 'batch size per compute device')
flags.DEFINE_integer('sensor_image_width', 640, 'Camera Image Width')
flags.DEFINE_integer('sensor_image_height', 512, 'Camera Image Height')
flags.DEFINE_integer('sensor_color_channels', 3,
                     'Number of color channels (3, RGB)')
flags.DEFINE_boolean('grasp_download', False,
                     """Download the grasp_dataset to data_dir if it is not already present.""")
flags.DEFINE_string('grasp_dataset', '102',
                    """Filter the subset of 1TB Grasp datasets to run.
                    None by default. 'all' will run all datasets in data_dir.
                    '052' and '057' will download the small starter datasets.
                    '102' will download the main dataset with 102 features,
                    around 110 GB and 38k grasp attempts.
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_integer('crop_width', 560,
                     """Width to crop images""")
flags.DEFINE_integer('crop_height', 448,
                     """Height to crop images""")
flags.DEFINE_boolean('random_crop', False,
                     """random_crop will apply the tf random crop function with
                        the parameters specified by crop_width and crop_height.

                        If random crop is disabled a fixed crop will be applied
                        that is centered on the image height and to the far 
                        right on the image width.
                     """)
flags.DEFINE_boolean('median_filter', False,
                     """median filter apply on depth image to remove zero and invalid depth.
                     """)
flags.DEFINE_integer('median_filter_width', 5,
                     """Width of median filter kernel.
                     """)
flags.DEFINE_integer('median_filter_height', 5,
                     """Height of median filter kernel.
                     """)
flags.DEFINE_integer('resize_width', 160,
                     """Width to resize images before prediction, if enabled.""")
flags.DEFINE_integer('resize_height', 128,
                     """Height to resize images before prediction, if enabled.""")
flags.DEFINE_boolean('resize', True,
                     """resize will resize the input images to the desired dimensions specified but the
                        resize_width and resize_height flags. It is suggested that an exact factor of 2 be used
                        relative to the input image directions if random_crop is disabled or the crop dimensions otherwise.
                     """)
flags.DEFINE_float('gripper_z_offset_meters', 0.02,
                   """ Offset of gripper's z axis in meters, so that in the final time
                       step of grasp, the gripper reaches the object surface.
                       default value 0.02.
                   """)
flags.DEFINE_boolean('image_augmentation', True,
                     'image augmentation applies random brightness, saturation, hue, contrast. imagenet_preprocessing must be True.')
flags.DEFINE_boolean('imagenet_preprocessing', True,
                     'subtract the imagenet mean pixel values from the rgb images')
flags.DEFINE_integer('grasp_sequence_max_time_step', None,
                     """The grasp motion time sequence consists of up to 11 time steps.
                        This integer, or None for unlimited specifies the max number of these steps from the last to the first
                        that will be used in training and evaluation. This may be needed to reduce memory utilization.""")
flags.DEFINE_integer('grasp_sequence_min_time_step', None,
                     """The grasp motion time sequence consists of up to 11 time steps.
                        This integer, or None for unlimited specifies the min time step
                        that will be used in training and evaluation. This may be needed
                        to reduce memory utilization or check performance at different
                        stages of a grasping motion.""")
flags.DEFINE_string(
    'grasp_sequence_motion_command_feature',
    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3',
    """Different ways of representing the motion vector parameter.
       'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3'
             [delta_depth, sin(theta), cos(theta)] where delta_depth depth offset for the gripper
                from the measured surface, alongside a single rotation angle theta containing sin(theta), cos(theta).
                This format does not allow for arbitrary commands to be defined, and the rotation component
                is based on the paper and dataset:
                    https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                    https://arxiv.org/abs/1603.02199
       'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_sin_cos_5'
            the real reached gripper pose of the end effector to calculate
            the transform from the current time step to the final time step
            to generate the parameters defined in https://arxiv.org/abs/1603.02199,
            consisting of [x,y,z, sin(theta), cos(theta)].
       'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_quat_7'
            transform from the current endeffector pose to the final endeffector pose.
       'final_pose_orientation_quaternion' directly input the final pose translation and orientation.
       'next_timestep' input the params for the command saved in the dataset with translation,
            sin theta, cos theta from the current time step to the next
       'endeffector_current_T_endeffector_final_vec_quat_7' use
            the real reached gripper pose of the end effector to calculate
            the transform from the current time step to the final time step
            to generate the parameters [x, y, z, qx, qy, qz, qw].
    """)
flags.DEFINE_string(
    'clear_view_image_feature', 'move_to_grasp/time_ordered/clear_view/rgb_image/preprocessed',
    """RGB image input feature for the clear scene view, typically an image from before the robot enters the scene.

        Options include:

        'move_to_grasp/time_ordered/clear_view/rgb_image/preprocessed'
            rgb image after all preprocessing as defined by the other parameters have been applied.

    """)
flags.DEFINE_string(
    'grasp_sequence_image_feature', 'move_to_grasp/time_ordered/rgb_image/preprocessed',
    """RGB image input feature at each time step.

        Options include:

        'move_to_grasp/time_ordered/rgb_image/preprocessed'
            rgb image after all preprocessing as defined by the other parameters have been applied.

    """)
flags.DEFINE_string(
    'grasp_success_label',
    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/grasp_success_yx_3',
    """Algorithm used to generate the grasp_success labels.

    Options include:

        'move_to_grasp/time_ordered/grasp_success':
            binary scalar, 1 for success 0 for failure. Pre-organized by time step to ease training,
            but is identical to 'grasp_success' repeated once for each time step.
        'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/grasp_success_yx_3':
            [grasp_success, y_coordinate, x_coordinate], grasp_success is 1 for success, 0 for failure,
            y_coordinate, x_coordinate is the gripper coordinate in the image at the time step when the gripper closed.
            This is useful for pixel-wise labeling of grasp attempts.
        'grasp_success': binary scalar, 1 for success 0 for failure, this is the single label for each complete grasp attempt.
        grasp_success_binary_2D: Apply a constant label at every input pixel.
            (Not yet implemented.)
        grasp_success_gaussian_2d: Apply a 0 to 1 label at every input pixel
            adjusted by a weight centered on the final gripper position in the image frame.
            (Not yet implemented.)
    """)
flags.DEFINE_string(
    'coordinate_feature',
    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/yx_2',
    """Algorithm used to generate the image coordinate labels.

    Options include:

    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/yx_2':
        the coordinate of the end effector at the final time step, after all preprocessing has been run.
    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/preprocessed/preprocessed/yx_2':
        the coordinate at the current time step, after all preprocessing has been run.
    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/yx_2':
        the coordinate of the end effector at the final time step.
    'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/preprocessed/yx_2':
        the coordinate at the current time step.
    """)

# MULTI DATASET AGGREGATION PARAMETERS, see get_multi_dataset_training_tensors()
flags.DEFINE_string('grasp_datasets_train', '062_b,063,072_a,082_b,102',
                    """Filter multiple subsets of 1TB Grasp datasets to train.

                    For use with get_multi_dataset_training_tensors() when training
                    multiple datasets simultaneously.

                    Comma separated list 062_b,063,072_a,082_b,102 by default,
                    totaling 513,491 grasp attempts.
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_string('grasp_datasets_batch_algorithm', 'proportional',
                    """Use default batch size for each dataset if constant.

                    For use with get_multi_dataset_training_tensors() when training
                    multiple datasets simultaneously.

                    'constant' training on multiple datasets reads `batch_size`
                        elements from each dataset at each training step when this parameter.
                    'proportional' each dataset's batch size will be individually
                        set to int(batch_size*single_batch/max_batch_size)
                        so smaller datasets are run more slowly than larger datasets.""")

FLAGS = flags.FLAGS


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
    """Google Grasping Dataset - about 1TB total size
        https://sites.google.com/site/brainrobotdata/home/grasping-dataset

        Data from https://arxiv.org/abs/1603.02199

        Downloads to `~/.keras/datasets/grasping` by default.

        grasp_listing.txt lists all the files in all grasping datasets;
        *.csv lists the number of features, the number of grasp attempts, and all feature names;
        *.tfrecord and *.tfrecord*-of-* is the actual data stored in the tfrecord.

        If you are using this for the first time simply select the dataset number you would like, such as
        102 in the constructor, and then call `get_training_tensors()`. This will give
        you tensorflow tensors for the original dataset configuration and parameterization. Then initialize
        your model with the tensors and train or predict.

        For analysis and visualization of the dataset use `get_training_dictionaries()`,
        this contains both the raw features stored on disk and preprocessed features needed to visualize
        the dataset in detail. Also see `vrep_grasp.py` to visualize the dataset.

        This interface only supports one dataset at a time (aka 102 feature version or 057).

        # Arguments

        data_dir: Path to dataset in TFRecord format
            (aka Example protobufs) and feature csv files.
             `~/.keras/datasets/grasping` by default.

        dataset: Filter the subset of 1TB Grasp datasets to run.
            None by default, which loads the data specified by the tf flag
            grasp_dataset command line parameter.
            'all' will run all datasets in data_dir.
            '052' and '057' will download the small starter datasets.
            '102' will download the main dataset with 102 features,
            around 110 GB and 38k grasp attempts.
            See https://sites.google.com/site/brainrobotdata/home
            for a full listing.

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
        '''Google Grasping Dataset - about 1TB total size
        https://sites.google.com/site/brainrobotdata/home/grasping-dataset

        Downloads to `~/.keras/datasets/grasping` by default.
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

        url_prefix = 'https://storage.googleapis.com/brain-robotics-data/'
        # If a hashed version of the listing is available,
        # download the dataset and verify hashes to prevent data corruption.
        listing_hash = os.path.join(data_dir, 'grasp_listing_hash.txt')
        if os.path.isfile(listing_hash):
            files_and_hashes = np.genfromtxt(listing_hash, dtype='str', delimiter=' ')
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, file_hash=hash_str)
                     for fpath, hash_str in tqdm(files_and_hashes)
                     if '_' + str(dataset) in fpath]
        else:
            # If a hashed version of the listing is not available,
            # simply download the dataset normally.
            listing_url = 'https://sites.google.com/site/brainrobotdata/home/grasping-dataset/grasp_listing.txt'
            grasp_listing_path = get_file('grasp_listing.txt', listing_url, cache_subdir=data_dir)
            grasp_files = np.genfromtxt(grasp_listing_path, dtype=str)
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir)
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

    def _get_feature_csv_file_paths(self, dataset=None):
        """List feature csv files with full paths in the data_dir.
        Feature csv files identify each dataset, the size, and its data channels.
        One example is: 'features_102.csv'
        """
        dataset = self._update_dataset_param(dataset)
        return gfile.Glob(os.path.join(os.path.expanduser(self.data_dir), '*{}*.csv'.format(dataset)))

    def get_features(self, dataset=None):
        """Get all features associated with the dataset specified when this class
        was constructed. Use when only one dataset will be processed.

        dataset: Filter the subset of 1TB Grasp datasets to run.
            None is the default, which utilizes the currently loaded dataset,
            'all' will run all datasets in data_dir.
            '052' and '057' will download the small starter datasets.
            '102' will download the main dataset with 102 features,
            around 110 GB and 38k grasp attempts.
            See https://sites.google.com/site/brainrobotdata/home
            for a full listing.

        # Returns:

            (features_complete_list, attempt_count)

            features_complete_list: is a list of all feature strings in the
                fixedLengthFeatureDict and sequenceFeatureDict.
            attempt_count: total number of grasp attempts
        """
        dataset = self._update_dataset_param(dataset)
        csv_files = self._get_feature_csv_file_paths(dataset)
        features_complete_list, _, feature_count, attempt_count = self._get_grasp_tfrecord_info(csv_files[-1])

        return features_complete_list, attempt_count

    def _get_tfrecord_path_glob_pattern(self, dataset=None):
        """Get the Glob string pattern for matching the specified dataset tfrecords.

        This will often be used in conjunction with the RecordInput class if you need
        a custom dataset loading function.

        # Arguments
            data_dir: The path to the folder containing the grasp dataset.

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.
        """
        dataset = self._update_dataset_param(dataset)
        return os.path.join(os.path.expanduser(self.data_dir), '*{}.tfrecord*'.format(dataset))

    def _get_grasp_tfrecord_info(self, feature_csv_file):
        """ Get the number of features, examples, and the name of features in a grasp tfrecord dataset.

        # Arguments

            feature_csv_file: path to the feature csv file for this dataset

        # Returns
            features: complete list of all features for this dataset aka tfrecord group
            tfrecord_paths: paths to all tfrecords for this dataset
            feature_count: total number of features
            attempt_count: total number of grasp attempts
        """
        features = np.genfromtxt(os.path.join(os.path.expanduser(self.data_dir), feature_csv_file), dtype=str)
        # need to account for multiple datasets with n features like 062_a and 062_b
        feature_count = int(features[0].split('_')[0])
        attempt_count = int(features[1])
        features = features[2:]
        # Workaround for csv files which may not actually list the key features below,
        # although they have been added to the dataset itself.
        if not any('grasp_success' in s for s in features):
            features = np.append(features, 'grasp_success')
            feature_count += 1
        if not any('gripper/status' in s for s in features):
            features = np.append(features, 'gripper/status')
            feature_count += 1
        # note that the tfrecords are often named '*{}.tfrecord*-of-*'
        tfrecord_paths = gfile.Glob(self._get_tfrecord_path_glob_pattern())
        return features, tfrecord_paths, feature_count, attempt_count

    @staticmethod
    def get_time_ordered_features(features, feature_type='all', step='all', record_type='fixed'):
        """Get list of all image features ordered by time, features are identified by a string path.
            See https://docs.google.com/spreadsheets/d/1GiPt57nCCbA_2EVkFTtf49Q9qeDWgmgEQ6DhxxAIIao/edit#gid=0
            for mostly correct details.
            These slides also help make the order much more clear:
            https://docs.google.com/presentation/d/13RdgkZQ_neqeXwYU3at2fP4RLl4qb1r74CTehq8d_Qc/edit?usp=sharing
            Also see the downloaded grasp_listing.txt and feature_*.csv files that the downloader puts
            into ~/.keras/datasets/grasping by default.

            TODO(ahundt) This function is extremely inefficient, pre-sort by time then extract string matches once.
        # Arguments

            features: list of feature TFRecord strings
            step: string indicating which parts of the grasp sequence should be returned.
                'all' retrieves data from all steps of the grasp sequence,
                'camera' get camera intrinsics matrix and/or camera_T_base transform,
                'robot' get robot name.
                Remaining options in chronological order are:
                'view_clear_scene' shows the clear scene before the arm enters the camera view,
                'move_to_grasp' are the up to 10 steps when the gripper is moving towards an object
                    it will try to grasp, also known as the grasp phase.
                'close_gripper' when the gripper is actually closed.
                 'test_success' with the pre and post drop images,

            feature_type: feature data type, 'all' matches all types, or one of:
                'transforms/base_T_endeffector/vec_quat_7'
                    A pose is a 6 degree of freedom rigid transform represented with 7 values:
                       vector (x, y, z) and quaternion (x, y, z, w).
                       A pose is always annotated with the target and source frames of reference.
                       In this case base_T_endeffector is a transform that takes a point in the endeffector
                       frame of reference and transforms it to the base frame of reference.
                       For example, base_T_camera is a transform that takes a point in the camera frame
                       of reference and transforms it to the base frame of reference.
                'camera/transforms/camera_T_base/matrix44'
                    Same as base_T_endeffector but from the camera center to the robot base,
                    and contains a 4x4 transformation matrix instead of a vector and quaternion.
                    For example, camera_T_base is a transform that takes a point in the base frame
                    of reference and transforms it to the camera frame of reference.
                'camera/intrinsics/matrix33'
                    The 3x3 camera intrinsics matrix.
                'camera/cropped/intrinsics/matrix33'
                    The 3x3 camera intrinsics matrix modified to matched the `/cropped`
                    features that have undergone random cropping with a consistent offset
                    and size across a whole grasp attempt.
                'commanded_pose'
                    Commanded pose is the input to the inverse kinematics computation,
                    which is used to determine desired joint positions.
                'reached_pose'
                    Pose calculated by forward kinematics from current robot state in status updates.
                    Please note that this is the pose reached at the end of the time step indicated by
                    the rest of the feature path, which makes it nearest to the image at the start of
                    the next time step. In other words, if you'd like to use this data as an input,
                    you must use the result from the previous time step for an accurate option.
                'joint/commanded_torques'
                    Commanded torques are the torque values calculated from the inverse kinematics
                    of the commanded pose and the robot driver.
                'joint/external_torques'
                    External torques are torques at each joint that are a result of gravity and
                    other external forces. These values are reported by the robot status query.
                'joint/positions'
                    Robot joint positions as reported by the robot status query.
                'joint/velocities'
                    Robot joint velocities as reported by the robot status query.
                'depth_image/encoded'
                    Depth image is encoded as an RGB PNG image where the RGB 24-bit value is an
                    integer depth with scale 1/256 of a millimeter. See `depth_image_encoding.py`.
                'depth_image/decoded'
                    This feature is calculated at runtime if `_image_decode()` is called.
                    Uncompressed 'depth_image/encoded'.
                    There may also be uncompressed ops 'depth_image/decoded'
                'depth_image/cropped'
                    This feature is calculated at runtime if `_image_decode()` then `image_crop()` is called.
                    Cropped 'depth_image/decoded', with dimensions and corner offset equal to those of all other
                    cropped image in this grasp attempt.
                'xyz_image/decoded'
                    This feature is calculated at runtime if `_image_decode()` is called.
                    Projected xyz point cloud from 'depth_image/encoded' and 'camera/intrinsics/matrix33'
                    with the same width and height as the depth image, plus 3 channels for float XYZ values.
                'xyz_image/cropped'
                    This feature is calculated at runtime if `_image_decode()` then `image_crop()` is called.
                    Cropped 'xyz_image/decoded', with dimensions and corner offset equal to those of all other
                    cropped image in this grasp attempt.
                    Projected xyz point cloud from 'depth_image/encoded' and 'camera/cropped/intrinsics/matrix33'
                    with the same width and height as the depth image, plus 3 channels for float XYZ values.
                '/image/encoded'
                    Camera RGB images are stored in JPEG format, be careful not to mismatch on
                    depth_image/encoded, so include the leading slash.
                    There may also be uncompressed ops '/image/decoded' and
                    if `_image_decode()` is called.
                '/image/decoded'
                    This feature is calculated at runtime if `_image_decode()` is called.
                    Uncompressed '/image/encoded'.
                    There may also be uncompressed ops '/image/decoded'
                '/image/cropped'
                    This feature is calculated at runtime if `_image_decode()` then `image_crop()` is called.
                    Cropped '/image/decoded', with dimensions and corner offset equal to those of all other
                    cropped image in this grasp attempt.
                'params'
                    This is a simplified representation of a commanded robot pose and gripper status.
                    These are the values that were solved for in the network, the output of
                    the Cross Entropy Method. This contains the command from one timestep to the next,
                    not the data that was trained on in the original paper.
                'name'
                    A string indicating the name of the robot
                'status'
                    'gripper/status' is a 0 to 1 value which indicates how open or closed the
                    gripper was at the end of a grasp attempt.
                'grasp_success'
                    1 if the gripper is determined to have successfully grasped an object on this attempt,
                    0 otherwise. This incorporates a combination of 'gripper/status', the open closed angle
                    of the gripper as the dominant indicator, with image differencing between the final
                    two object drop image features as a secondary indicator of grasp success.
                'image_coordinate'
                    An integer tuple (x, y) indicating an image coordinate.

            record_type:
                'all' will match any feature type.
                'fixed' matches fixed length record types (most records, use this if unsure)
                'sequence' matches sequence records (ongoing variable length updates of robot pose)
                Used to separate sequence data types from non sequence data types, since they can be
                variable length and need to be handled separately. In particular this separates out
                variable length vectors of poses.

                This is because data sequences must be read separately from fixed length features
                see `tf.parse_single_sequence_example()` which can do both but they must be defined separately and
                `tf.parse_single_example()` which can only do fixed length features

        # Returns
            list of features organized by time step in a single grasp
        """
        matching_features = []

        def _match_feature(features, feature_name_regex, feature_type='', exclude_substring=None, exclude_regex=None):
            """Get sll features from the list that meet requirements.

            Used to ensure correct ordering and easy selection of features.
            some silly tricks to get the feature names in the right order while
            allowing variation between the various datasets.
            For example, we need to make sure 'grasp/image/encoded comes'
            before grasp/0/* and post_grasp/*, but with a simple substring
            match 'grasp/', all of the above will be matched in the wrong order.
            `r'/\\d+/'` (single backslash) regex will exclude /0/, /1/, and /10/.
            """
            if ('image/encoded' in feature_type) and ('depth' not in feature_type):
                feature_type = '/image/encoded'
            ifnames = []
            for ifname in features:
                if (bool(re.search(feature_name_regex, ifname)) and  # feature name matches
                        ((exclude_substring is None) or (exclude_substring not in ifname)) and
                        ((exclude_regex is None) or not bool(re.search(exclude_regex, ifname))) and
                        (feature_type in ifname)):
                    ifnames.append(str(ifname))
            return ifnames

        # see feature type docstring for details
        if record_type is 'fixed':
            exclude_substring = 'sequence'
        else:
            exclude_substring = None

        # 'all' really just matches anything, so set to empty string
        if feature_type is 'all':
            feature_type = ''

        if step in ['camera', 'all', '']:
            # the r'^ in r'^camera/' makes sure the start of the string matches
            matching_features.extend(_match_feature(features, r'^camera/', feature_type))
        if step in ['robot', 'all', '']:
            matching_features.extend(_match_feature(features, r'^robot/', feature_type))
        # r'/\d/' is a regex that will exclude things like /0/, /1/, through /10/
        # this is the first pre grasp image, even though it is called grasp
        if step in ['view_clear_scene', 'all', '']:
            matching_features.extend(_match_feature(features, r'^initial/', feature_type))
            matching_features.extend(_match_feature(features, r'^approach/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^approach_sequence/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^pregrasp/', feature_type, 'post', r'/\d+/'))
            matching_features.extend(_match_feature(features, r'^grasp/', feature_type, 'post', r'/\d+/'))

        # up to 11 grasp steps in the datasets
        if step in ['move_to_grasp', 'all', '']:
            max_grasp_steps = 11  # 0 through 10
            for i in range(max_grasp_steps):
                matching_features.extend(_match_feature(features, r'^grasp/{}/'.format(i), feature_type, 'post'))
                # note that this feature is user created and not directly present in the stored dataset files
                # see _endeffector_current_T_endeffector_final for details
                # This is where most surface relative transform features will match, typically including the word 'depth'
                matching_features.extend(_match_feature(features, r'^move_to_grasp/{:03}/'.format(i), feature_type, 'post'))

        # closing the gripper
        if step in ['close_gripper', 'all', '']:
            matching_features.extend(_match_feature(features, r'^gripper/', feature_type, exclude_substring))
            # Not totally sure, but probably the transforms and angles from as the gripper is closing (might shift a bit)
            matching_features.extend(_match_feature(features, r'^gripper_sequence/', feature_type, exclude_substring))
            # Withdraw the closed gripper from the bin vertically upward, roughly 10 cm above the bin surface.
            matching_features.extend(_match_feature(features, r'^withdraw_sequence/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^post_grasp/', feature_type))

        if step in ['test_success', 'all', '']:
            matching_features.extend(_match_feature(features, r'^present/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^present_sequence/', feature_type, exclude_substring))
            # After presentation (or after withdraw, if present is skipped),
            # the object is moved to a random position about 10 cm above the bin to drop the object.
            matching_features.extend(_match_feature(features, r'^raise/', feature_type))
            # Open the gripper and drop the object (if any object is held) into the bin.
            matching_features.extend(_match_feature(features, r'^drop/', feature_type))
            # Images recorded after withdraw, raise, and the drop.
            matching_features.extend(_match_feature(features, r'^post_drop/', feature_type))
            matching_features.extend(_match_feature(features, r'^grasp_success', feature_type))
        return matching_features

    @staticmethod
    def _parse_grasp_attempt_protobuf(serialized_grasp_attempt_proto, features_complete_list):
        """ Parses an Example protobuf containing a training example of an image.

        The output of the build_image_data.py image preprocessing script is a dataset
        containing serialized Example protocol buffers. Each Example proto contains
        the fields described in get_time_ordered_features.

        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.

            list of tensors corresponding to images, actions, and states. Each images
            tensor is 4D, batch x height x width x channels. The state and
            action tensors are 3D, batch x time x dimension.

        Returns:
            feature_op_dict: dictionary of Tensors for every feature,
                use `get_time_ordered_features` to select the subsets you need.
            sequence_feature_op_dict: dictionary of sequence tensors for every features,
                contains base to end effector transforms.
        """
        with tf.name_scope('parse_grasp_attempt_protobuf') as scope:
            # Dense features in Example proto.
            num_grasp_steps_name = 'num_grasp_steps'
            camera_to_base_name = 'camera/transforms/camera_T_base/matrix44'
            camera_intrinsics_name = 'camera/intrinsics/matrix33'
            grasp_success_name = 'grasp_success'
            gripper_status = 'gripper/status'

            # setup one time features like the camera and number of grasp steps
            features_dict = {
                num_grasp_steps_name: tf.FixedLenFeature([1], tf.string),
                camera_to_base_name: tf.FixedLenFeature([4, 4], tf.float32),
                camera_intrinsics_name: tf.FixedLenFeature([3, 3], tf.float32),
                grasp_success_name: tf.FixedLenFeature([1], tf.float32),
                gripper_status: tf.FixedLenFeature([1], tf.float32),
            }

            # load all the images
            ordered_rgb_image_feature_names = GraspDataset.get_time_ordered_features(
                features_complete_list,
                feature_type='/image/encoded')
            features_dict.update({image_name: tf.FixedLenFeature([1], tf.string)
                                  for image_name in ordered_rgb_image_feature_names})

            # load all the depth images
            ordered_depth_feature_names = GraspDataset.get_time_ordered_features(
                features_complete_list,
                feature_type='depth_image')
            features_dict.update({image_name: tf.FixedLenFeature([1], tf.string)
                                  for image_name in ordered_depth_feature_names})

            # load all vec/quat base to end effector transforms that aren't sequences
            vec_quat_transform_names = GraspDataset.get_time_ordered_features(
                features_complete_list,
                feature_type='vec_quat_7'
                )
            features_dict.update({x_form: tf.FixedLenFeature([7], tf.float32)
                                  for x_form in vec_quat_transform_names})

            # load all vec/quat base to end effector transforms that are sequences
            base_to_endeffector_sequence_names = GraspDataset.get_time_ordered_features(
                features_complete_list,
                feature_type='sequence/transforms/base_T_endeffector/vec_quat_7',
                record_type='sequence'  # Don't exclude sequences
                )
            sequence_features_dict = {x_form: tf.FixedLenSequenceFeature([7], tf.float32, allow_missing=True)
                                      for x_form in base_to_endeffector_sequence_names}

            # load transforms that were modified for their network from the original paper
            transform_adapted_for_network = GraspDataset.get_time_ordered_features(
                features_complete_list,
                feature_type='params'
                )
            features_dict.update({x_form: tf.FixedLenFeature([5], tf.float32)
                                  for x_form in transform_adapted_for_network})

            # extract all the features from the file, return two dicts
            return tf.parse_single_sequence_example(serialized_grasp_attempt_proto,
                                                    context_features=features_dict,
                                                    sequence_features=sequence_features_dict)

    def _get_transform_tensors(
            self,
            feature_op_dicts=None,
            features_complete_list=None,
            time_ordered_feature_name_dict=None,
            num_samples=None,
            batch_size=FLAGS.batch_size,
            gripper_z_offset=FLAGS.gripper_z_offset_meters,
            median_filter=FLAGS.median_filter,
            verbose=0):
        """Get runtime generated 3D transform feature tensors as a dictionary, including depth surface relative transforms.

        @TODO(ahundt) update this documentation.

        Includes surface relative transform features,
        see `grasp_geometry.grasp_dataset_to_transforms_and_features()`.

        The surface relative transform is from the clear view depth image
        pixel world coordinate to the final gripper coordinate.
        This applies the surface relative transforms.
        The point cloud point is selected by using the (x, y) pixel
        coordinate of the final gripper pose in the camera frame.
        The depth value is then taken at this coordinate from the
        clear view frame's depth image, and this depth pixel is projected
        into its 3D space point cloud point. A transform is calculated
        from this point to the final gripper pose, which is the input
        to the training algorithm.

        Note: below ### will substituted with an actual time step number in the real data, such as 000, 001.

        Generate feature ops which define a transform from the current time step's
        reached endeffector pose to the final time step's reached endeffector pose.
        'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/vec_quat_7'

        This function defines new features:

        Generate feature ops which define a transform from the current time
        step's reached endeffector pose to the final time step's reached endeffector pose.
        'move_to_grasp/###/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_quat_7'
        'move_to_grasp/###/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_sin_cos_5'

        'move_to_grasp/###/reached_pose/transforms/depth_pixel_T_endeffector_final/vec_quat_7'
            This is the transform from the depth pixel point to the end effector point at the final time step.
            The depth value is taken from the clear view frame's depth image at the yx_2 coordinate described below.

        'move_to_grasp/###/reached_pose/image_coordinates/depth_pixel_T_endeffector_final/yx_2'
            This is the x,y coordinate of the relevant depth pixel in the depth image frame from the feature above.
            This is expressed in a tensor as [x, y].

        'move_to_grasp/###/depth_pixel_T_endeffector_final/delta_depth_sin_cos_3'
            This is a surface relative transform based of the distance between
            the clear view depth pixel and the gripper wrist coordinate,
            as well as the change in angle theta from the current time step to the end.
            This is expressed in a tensor as [delta_depth, sin(theta), cos(theta)].

        'move_to_grasp/###/depth_pixel_T_endeffector_final/delta_depth_quat_5'
            This is a surface relative transform based of the distance between
            the clear view depth pixel and the gripper wrist coordinate,
            as well as the change in orientation from the current time step to
            the final move_to_grasp time step when the gripper was closed.
            This is expressed in a tensor as [delta_depth, qx, qy, qz, qw].

        'move_to_grasp/###/reached_pose/depth_pixel_T_endeffector_final/image_coordinate/yx_2'
            The camera image coordinate of the final pose of the gripper.

        'camera/transforms/base_T_camera/vec_quat_7'
            A more convenient transform from the robot base to the camera frame.

        TODO(ahundt) finish description of all features listed below
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

        pixel_coordinate_param_names is ['surface_relative_grasp/reached_pose/image_coordinates/depth_pixel_T_endeffector_final/yx_2'].

        See also: grasp_dataset_to_transforms_and_features() and surface_relative_transform() in grasp_geometry.py.


        # Parameters

        feature_op_dicts: A list containing a tuple of (fixed_feature_dict, sequence_feature_dict) from string to ops.
            See _get_simple_parallel_dataset_ops for details.
        features_complete_list: A list of all feature strings.
            See _get_simple_parallel_dataset_ops for details.
        feature_type: The feature type to be used for this calculation. Options include:
            'delta_depth_sin_cos_3' [delta_depth, sin(theta), cos(theta)] where delta_depth depth offset for the gripper
                from the measured surface, alongside a single rotation angle theta containing sin(theta), cos(theta).
                This format does not allow for arbitrary commands to be defined, and the rotation component
                is based on the paper and dataset:
                    https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                    https://arxiv.org/abs/1603.02199
            'delta_depth_quat_5' A numpy array with 5 total entries including depth offset and 4 entry quaternion.
            'vec_quat_7' Create the 7 entry vector quaternion feature [dx, dy, dz, qx, qy, qz, qw]
            'vec_sin_cos_5' Create the 5 entry vector quaternion feature [dx, dy, dz, sin(theta), cos(theta)]
            '' don't get any specific param names
        offset: random crop offset for depth preprocessing.

        # Returns

            [new_feature_op_dicts, features_complete_list, time_ordered_feature_name_dict, num_samples]

            new_feature_op_dicts: updated list of feature op dictionaries, fixed and sequence to replace the originally passed version.

            features_complete_list: updated list of all features.

            time_ordered_feature_name_dict:
            dictionary to access time ordered lists of the specified feature type.
            These types will be identical to 'move_to_grasp/###/', but instead bundle
            the names together in a list prefixed with 'move_to_grasp/time_ordered/'.
            {
                'move_to_grasp/time_ordered/reached_pose/transforms/camera_T_endeffector/vec_quat_7':
                    ['move_to_grasp/000/...', 'move_to_grasp/001/...', ...],
                'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/vec_quat_7':
                    ['move_to_grasp/000/...', 'move_to_grasp/001/...', ...],

            }

            num_samples: the number of grasp attempts in the dataset.
        """
        if feature_op_dicts is None:
            feature_op_dicts, features_complete_list, num_samples = self._get_simple_parallel_dataset_ops(batch_size=batch_size)
        elif features_complete_list is None or num_samples is None:
            raise ValueError('GraspDataset._get_transform_tensors(): feature_op_dicts was specified so '
                             'you must also set features_complete_list and num_samples.')
        if time_ordered_feature_name_dict is None:
            time_ordered_feature_name_dict = {}

        new_feature_op_dicts = []

        # The reached poses are the end of each time step and the start of the next,
        # we need to shift everything over by one!
        all_base_to_endeffector_transforms = self.get_time_ordered_features(
            features_complete_list,
            feature_type='reached_pose/transforms/base_T_endeffector/vec_quat_7')

        timed_base_to_endeffector_transforms = self.get_time_ordered_features(
            all_base_to_endeffector_transforms,
            feature_type='reached_pose/transforms/base_T_endeffector/vec_quat_7',
            step='move_to_grasp')

        if verbose:
            print('all_base_to_endeffector_transforms: ', all_base_to_endeffector_transforms)
        first_timed_index = all_base_to_endeffector_transforms.index(timed_base_to_endeffector_transforms[0])
        base_to_endeffector_transforms = ['approach/transforms/base_T_endeffector/vec_quat_7'] + timed_base_to_endeffector_transforms[:-1]

        final_base_to_endeffector_transform_name = all_base_to_endeffector_transforms[-1]

        # Get different image features depending on if
        # cropping is enabled or not.
        # Note that depth images can be converted to xyz images
        # when the camera/intrinsics/matrix33 is available.
        xyz_image_feature_type = 'xyz_image/decoded'
        depth_image_feature_type = 'depth_image/decoded'
        camera_intrinsics_name = 'camera/intrinsics/matrix33'

        if median_filter:
            xyz_image_feature_type = 'xyz_image/median_filtered'
            depth_image_feature_type = 'depth_image/median_filtered'

        xyz_image_clear_view_name = 'pregrasp/' + xyz_image_feature_type
        depth_image_clear_view_name = 'pregrasp/' + depth_image_feature_type

        def add_feature_op(fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict, new_op, shape, name, batch_i, time_step_j):
            """Helper function to extend the dict containing feature ops

               TODO(ahundt) expand docs of this function

                unique: special handling for features that are constant across all time steps or are needed for the final time step only
            """
            if new_op is None or shape is None or name is None:
                raise ValueError('GraspDataset._get_transform_features.add_feature_op() new_op, shape, and name parameters '
                                 'cannot be None must each have a valid value')
            # prefix for every transform/coordinate feature
            move_to_grasp_prefix = 'move_to_grasp/{:03}/reached_pose/transforms/'.format(time_step_j)
            feature_name = move_to_grasp_prefix + name
            time_ordered_name = 'move_to_grasp/time_ordered/reached_pose/transforms/' + name
            # TODO(ahundt) are assigned numbers off by 1 skipping /grasp/ without a number?
            new_op.set_shape(shape)
            # Ops that can be any time step use none, unique per attempt time steps use 0
            fixed_feature_op_dict[feature_name] = new_op
            if batch_i == 0:
                # assume all batches have the same features
                features_complete_list = np.append(features_complete_list, feature_name)
                if time_ordered_name in time_ordered_feature_name_dict:
                    time_ordered_feature_name_dict[time_ordered_name] = np.append(time_ordered_feature_name_dict[time_ordered_name], feature_name)
                else:
                    time_ordered_feature_name_dict[time_ordered_name] = np.array([feature_name])
            return fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict

        # loop through all grasp attempts in this batch
        for batch_i, (fixed_feature_op_dict, sequence_feature_op_dict) in enumerate(tqdm(feature_op_dicts, desc='get_transform_tensors')):

            # Note that the XYZ image computation can be enabled/disabled,
            # so if no XYZ image is available supply the depth image.
            # Performance may actually be better with the xyz image
            # computed on the fly in
            # grasp_geometry.grasp_dataset_to_transforms_and_features()
            # due to the python global interpreter lock.
            # see image_decode() for more details.
            if xyz_image_clear_view_name in fixed_feature_op_dict:
                cartesian_clear_view_op = fixed_feature_op_dict[xyz_image_clear_view_name]
            else:
                cartesian_clear_view_op = fixed_feature_op_dict[depth_image_clear_view_name]
            final_base_to_endeffector_transform_op = fixed_feature_op_dict[final_base_to_endeffector_transform_name]
            camera_intrinsics_matrix = fixed_feature_op_dict[camera_intrinsics_name]
            camera_T_base = fixed_feature_op_dict['camera/transforms/camera_T_base/matrix44']

            # loop through all time steps in this grasp attempt
            for time_step_j, base_to_endeffector_transform_name in enumerate(base_to_endeffector_transforms):
                base_to_endeffector_op = fixed_feature_op_dict[base_to_endeffector_transform_name]
                # call the python function that extracts all features for the surface relative transforms
                [current_base_T_camera_vec_quat_7_array,
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
                 delta_depth_quat_5] = tf.py_func(
                     grasp_geometry.grasp_dataset_to_transforms_and_features,
                     # parameters for grasp_dataset_to_transforms_and_features() function call
                     [cartesian_clear_view_op, camera_intrinsics_matrix, camera_T_base,
                      base_to_endeffector_op, final_base_to_endeffector_transform_op,
                      gripper_z_offset],
                     # return type data formats to expect
                     [tf.float32] * 14,
                     stateful=False, name='py_func/grasp_dataset_to_transforms_and_features')

                # define pixel image coordinate as an integer type
                image_coordinate_current = tf.cast(image_coordinate_current, tf.int32)
                image_coordinate_final = tf.cast(image_coordinate_final, tf.int32)

                # camera_T_endeffector_current_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    camera_T_endeffector_current_vec_quat_7_array, [7],
                    'camera_T_endeffector/vec_quat_7',
                    batch_i, time_step_j)

                # camera_T_depth_pixel_current_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    camera_T_depth_pixel_current_vec_quat_7_array, [7],
                    'camera_T_depth_pixel/vec_quat_7',
                    batch_i, time_step_j)

                # depth_pixel_T_endeffector_current_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict, depth_pixel_T_endeffector_current_vec_quat_7_array, [7],
                    'endeffector_clear_view_depth_pixel_T_endeffector/vec_quat_7',
                    batch_i, time_step_j)

                # image_coordinate_current,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    image_coordinate_current, [2],
                    'endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/yx_2',
                    batch_i, time_step_j)

                # depth_pixel_T_endeffector_final_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    depth_pixel_T_endeffector_final_vec_quat_7_array, [7],
                    'endeffector_final_clear_view_depth_pixel_T_endeffector_final/vec_quat_7',
                    batch_i, time_step_j)

                # camera_T_endeffector_final_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    camera_T_endeffector_final_vec_quat_7_array, [7],
                    'camera_T_endeffector_final/vec_quat_7',
                    batch_i, time_step_j)

                # camera_T_depth_pixel_final_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    camera_T_depth_pixel_final_vec_quat_7_array, [7],
                    'camera_T_depth_pixel_final/vec_quat_7',
                    batch_i, time_step_j)

                # image_coordinate_final,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    image_coordinate_final, [2],
                    'endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/yx_2',
                    batch_i, time_step_j)

                # current_base_T_camera_vec_quat_7_array,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    current_base_T_camera_vec_quat_7_array, [7],
                    'camera/transforms/base_T_camera/vec_quat_7',
                    batch_i, time_step_j)

                # eectf_vec_quat_7_array, aka endeffector_current_T_endeffector_final
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    eectf_vec_quat_7_array, [7],
                    'endeffector_current_T_endeffector_final/vec_quat_7',
                    batch_i, time_step_j)

                # sin_cos_2,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    sin_cos_2, [2],
                    'endeffector_final_clear_view_depth_pixel_T_endeffector_final/sin_cos_2',
                    batch_i, time_step_j)

                #  eectf_vec_quat_7_array, aka endeffector_current_T_endeffector_final
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    vec_sin_cos_5, [5],
                    'endeffector_current_T_endeffector_final/vec_sin_cos_5',
                    batch_i, time_step_j)

                # delta_depth_sin_cos_3,
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    delta_depth_sin_cos_3, [3],
                    'endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3',
                    batch_i, time_step_j)

                # delta_depth_quat_5
                (fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict) = add_feature_op(
                    fixed_feature_op_dict, features_complete_list, time_ordered_feature_name_dict,
                    delta_depth_quat_5, [5],
                    'endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_quat_5',
                    batch_i, time_step_j)

            # assemble the updated feature op dicts
            new_feature_op_dicts.append((fixed_feature_op_dict, sequence_feature_op_dict))

        return new_feature_op_dicts, features_complete_list, time_ordered_feature_name_dict, num_samples

    def _get_simple_parallel_dataset_ops(self, dataset=None, batch_size=1, buffer_size=300, parallelism=20, shift_ratio=0.01):
        """ Simple unordered & parallel TensorFlow ops that go through the whole dataset.

        # Returns

            A list of tuples ([(fixedLengthFeatureDict, sequenceFeatureDict)], features_complete_list, num_samples).
            fixedLengthFeatureDict maps from the feature strings of most features to their TF ops.
            sequenceFeatureDict maps from feature strings to time ordered sequences of poses transforming
            from the robot base to end effector.
            features_complete_list: a list of all feature strings in the fixedLengthFeatureDict and sequenceFeatureDict,
                and a parameter for get_time_ordered_features().
            num_samples: the number of samples in the dataset, used for configuring the size of one training epoch
            shift_ratio: The order the files are read will be shifted each epoch by shift_amount so that the data
                is presented in a different order every epoch, 0 means the order always stays the same.


        """
        tf_glob = self._get_tfrecord_path_glob_pattern(dataset=dataset)
        record_input = data_flow_ops.RecordInput(tf_glob, batch_size,
                                                 buffer_size, parallelism, shift_ratio=shift_ratio)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_size, 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        features_complete_list, num_samples = self.get_features()
        feature_op_dicts = [self._parse_grasp_attempt_protobuf(serialized_protobuf, features_complete_list)
                            for serialized_protobuf in tqdm(records_op, desc='get_simple_parallel_dataset_ops.parse_protobuf')]
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go on cpu and gpu via prefetching in a staging area
        # staging_area = tf.contrib.staging.StagingArea()
        dict_and_feature_tuple_list = []
        # Get all image features to finish extracting image data '/image/encoded' 'depth_image/decoded' 'xyz_image/decoded'))
        image_features = GraspDataset.get_time_ordered_features(features_complete_list, '/image/encoded')
        image_features = np.append(image_features, GraspDataset.get_time_ordered_features(features_complete_list, 'depth_image/encoded'))
        for feature_op_dict, sequence_op_dict in tqdm(feature_op_dicts, desc='get_simple_parallel_dataset_ops.image_decode_batches'):
            new_feature_op_dict, new_feature_list = GraspDataset._image_decode(feature_op_dict, image_features=image_features)
            dict_and_feature_tuple_list.append((new_feature_op_dict, sequence_op_dict))
        # the new_feature_list should be the same for all the ops
        features_complete_list = np.append(features_complete_list, new_feature_list)

        return dict_and_feature_tuple_list, features_complete_list, num_samples

    def get_simple_tfrecordreader_dataset_ops(self, batch_size=1, decode_depth_as='depth'):
        """ Get a dataset reading op from tfrecordreader.

        You will have to call tf.train.batch and tf.train.start_queue_runners(sess), see create_gif.

            decode_depth_as:
               The default 'depth' turns png encoded depth images into 1 channel depth images, the format for training.
               'rgb' keeps the png encoded format so it can be visualized, particularly in create_gif().
                     If you are unsure, keep the default 'depth'.
        # Returns

            A list of tuples [(fixedLengthFeatureDict, sequenceFeatureDict, features_complete_list)].
            fixedLengthFeatureDict maps from the feature strings of most features to their TF ops.
            sequenceFeatureDict maps from feature strings to time ordered sequences of poses transforming
            from the robot base to end effector.
            features_complete_list: is a list of all feature strings in the fixedLengthFeatureDict and sequenceFeatureDict.
            feature_count: total number of features
            attempt_count: total number of grasp attempts

        """
        [feature_csv_file] = self._get_feature_csv_file_paths()
        features_complete_list, tfrecord_paths, feature_count, attempt_count = self._get_grasp_tfrecord_info(feature_csv_file)
        if not tfrecord_paths:
            raise RuntimeError('No tfrecords found for {}.'.format(feature_csv_file))
        filename_queue = tf.train.string_input_producer(tfrecord_paths, shuffle=False)
        # note that this API is deprecated
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        feature_op_dicts = [self._parse_grasp_attempt_protobuf(serialized_example, features_complete_list)]
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()
        dict_and_feature_tuple_list = []
        for feature_op_dict, sequence_op_dict in feature_op_dicts:
            features_op_dict, new_feature_list = GraspDataset._image_decode(feature_op_dict, decode_depth_as=decode_depth_as)
            dict_and_feature_tuple_list.append((features_op_dict, sequence_op_dict))
        # the new_feature_list should be the same for all the ops
        features_complete_list = np.append(features_complete_list, new_feature_list)

        return dict_and_feature_tuple_list, features_complete_list, feature_count, attempt_count

    @staticmethod
    def _image_decode(feature_op_dict, sensor_image_dimensions=None, image_features=None, decode_depth_as='depth',
                      point_cloud_fn='tensorflow', median_filter=FLAGS.median_filter, 
                      median_filter_height=FLAGS.median_filter_height, median_filter_width=FLAGS.median_filter_width):
        """ Add features to dict that supply decoded png and jpeg images for any encoded images present.

        Any feature path that is 'image/encoded' will also now have 'image/decoded', and 'image/xyz' when
        both 'depthimage/endoced' and 'camera/intrinsics/matrix33' are in the dictionary.

        # Arguments

            feature_op_dict: dictionary of strings to fixed feature tensors.
            sensor_image_dimensions: [batch, height, width, channels], defaults to
                [1, FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
            image_features: list of image feature strings to modify, generated automatically if not supplied,
                improves performance.
            point_cloud_fn: Choose the function to convert depth images to point clouds.

                    Performance may actually be better with the xyz image computed on the fly in
                    grasp_geometry.grasp_dataset_to_transforms_and_features()
                    due massive slowdowns encountered due to the python global interpreter lock.
                    Note that the XYZ image computation can be enabled/disabled,
                    so if no XYZ image is available supply the depth image.

                None: Do not generate XYZ point clouds from the depth image during a call to image_decode,
                    and the feature is not created. (the default)
                'numpy': calls grasp_geometry.depth_image_to_point_cloud().
                'tensorflow' calls grasp_geometry_tf.depth_image_to_point_cloud().
                    The tensorflow option has bugs at the time of writing, but may be the ideal choice
                    once the bugs are resolved.
            decode_depth_as:
               The default 'depth' turns png encoded depth images into 1 channel depth images, the format for training.
               'rgb' keeps the png encoded format so it can be visualized, particularly in create_gif().
                     If you are unsure, keep the default 'depth'.

        # Returns

            updated feature_op_dict, new_feature_list
        """
        with tf.name_scope('image_decode'):
            new_feature_list = []
            if sensor_image_dimensions is None:
                    sensor_image_dimensions = [FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
            height, width, rgb_channels = sensor_image_dimensions
            # create a feature tensor for the dimensions
            rgb_sensor_image_dimensions = tf.constant(sensor_image_dimensions[1:], name='rgb_sensor_image_dimensions')
            depth_sensor_image_dimensions = tf.constant([height, width, 1], name='depth_sensor_image_dimensions')

            feature_op_dict['rgb_sensor_image_dimensions'] = rgb_sensor_image_dimensions
            new_feature_list = np.append(new_feature_list, 'rgb_sensor_image_dimensions')
            feature_op_dict['depth_sensor_image_dimensions'] = depth_sensor_image_dimensions
            new_feature_list = np.append(new_feature_list, 'depth_sensor_image_dimensions')

            if image_features is None:
                # the list of feature strings isn't supplied by the user so generate them here
                features = [feature for (feature, tf_op) in six.iteritems(feature_op_dict)]
                image_features = GraspDataset.get_time_ordered_features(features, '/image/encoded')
                image_features.extend(GraspDataset.get_time_ordered_features(features, 'depth_image/encoded'))

            for image_index, image_feature in enumerate(tqdm(image_features, desc='image_decode')):
                image_buffer = tf.reshape(feature_op_dict[image_feature], shape=[])
                if 'depth_image' in image_feature:
                    with tf.name_scope('depth'):
                        image = tf.image.decode_png(image_buffer, channels=sensor_image_dimensions[-1])
                        # extract as rgb without any additional processing for creating gifs
                        image.set_shape(sensor_image_dimensions)
                        image = tf.reshape(image, sensor_image_dimensions)
                        decoded_image_feature = image_feature.replace('encoded', 'rgb_encoded')
                        feature_op_dict[decoded_image_feature] = image
                        new_feature_list = np.append(new_feature_list, decoded_image_feature)
                        # convert depth from the rgb depth image encoding to float32 depths
                        # https://sites.google.com/site/brainrobotdata/home/depth-image-encoding
                        # equivalent to depth_image_encoding.ImageToFloatArray
                        image = tf.cast(image, tf.float32)
                        image = tf.reduce_sum(image * [65536, 256, 1], axis=2)
                        RGB_SCALE_FACTOR = 256000.0
                        image = image / RGB_SCALE_FACTOR
                        image.set_shape([height, width])
                        # apply median filter to depth image
                        median_filtered_image = grasp_dataset_median_filter(image, median_filter_height, median_filter_width)
                        median_filtered_image_feature = image_feature.replace('encoded', 'median_filtered')
                        # depth images have one channel
                        if 'camera/intrinsics/matrix33' in feature_op_dict and point_cloud_fn is not None:
                            with tf.name_scope('xyz'):
                                # generate xyz point cloud image feature
                                if point_cloud_fn == 'tensorflow':
                                    # should be more efficient than the numpy version
                                    xyz_image = grasp_geometry_tf.depth_image_to_point_cloud(
                                        image, feature_op_dict['camera/intrinsics/matrix33'])
                                    median_filtered_xyz_image = grasp_geometry_tf.depth_image_to_point_cloud(
                                        median_filtered_image, feature_op_dict['camera/intrinsics/matrix33'])
                                elif point_cloud_fn == 'numpy':
                                    [xyz_image] = tf.py_func(
                                        grasp_geometry.depth_image_to_point_cloud,
                                        # parameters for function call
                                        [image, feature_op_dict['camera/intrinsics/matrix33']],
                                        [tf.float32],
                                        stateful=False, name='py_func/depth_image_to_point_cloud'
                                    )
                                    [median_filtered_xyz_image] = tf.py_func(
                                        grasp_geometry.depth_image_to_point_cloud,
                                        # parameters for function call
                                        [median_filtered_image, feature_op_dict['camera/intrinsics/matrix33']],
                                        [tf.float32],
                                        stateful=False, name='py_func/depth_image_to_point_cloud'
                                    )
                                else:
                                    raise ValueError('point_cloud_fn must be one of tensorflow, numpy, or None')
                                # add xyz image to the dictionary and feature list
                                xyz_image.set_shape([height, width, 3])
                                xyz_image = tf.reshape(xyz_image, [height, width, 3])
                                xyz_feature = image_feature.replace('depth_image/encoded', 'xyz_image/decoded')
                                feature_op_dict[xyz_feature] = xyz_image
                                new_feature_list = np.append(new_feature_list, xyz_feature)
                                # add median filtered xyz image to the dictionary and feature list
                                median_filtered_xyz_image.set_shape([height, width, 3])
                                median_filtered_xyz_image = tf.reshape(median_filtered_xyz_image, [height, width, 3])
                                median_filtered_xyz_feature = image_feature.replace('depth_image/encoded', 'xyz_image/median_filtered')
                                feature_op_dict[median_filtered_xyz_feature] = median_filtered_xyz_image
                                new_feature_list = np.append(new_feature_list, median_filtered_xyz_feature)
                        image = tf.reshape(image, [height, width, 1])
                        median_filtered_image = tf.reshape(median_filtered_image, [height, width, 1])
                        feature_op_dict[median_filtered_image_feature] = median_filtered_image
                        new_feature_list = np.append(new_feature_list, median_filtered_image_feature)
                else:
                    with tf.name_scope('rgb'):
                        image = tf.image.decode_jpeg(image_buffer, channels=sensor_image_dimensions[-1])
                        image.set_shape(sensor_image_dimensions)
                        image = tf.reshape(image, sensor_image_dimensions)
                decoded_image_feature = image_feature.replace('encoded', 'decoded')
                feature_op_dict[decoded_image_feature] = image
                new_feature_list = np.append(new_feature_list, decoded_image_feature)

            return feature_op_dict, new_feature_list

    @staticmethod
    def _image_random_crop(
            feature_op_dict, sensor_image_dimensions=None,
            random_crop_dimensions=None,
            random_crop_offset=None,
            image_features=None,
            coordinate_features=None,
            seed=None,
            verbose=0):
        """ Crop all images and update parameters in accordance with a single random_crop.

        All images will be cropped in an identical fashion for the entire feature_op_dict,
        thus defining one crop size and offset for a single example because data is used
        across time and must thus be generated consistently. Across separate grasp attempts examples,
        crop dimensions will vary including across multiple iterations of a single grasp attempt.

        Adds 'image/cropped', 'depth_image/cropped', 'xyz_image/cropped'.

        Intrinsics matrices and image coordinates will also be updated based
        on the new coordinate system of the cropped images.

        # Prerequisites

        _image_decode() must have been called, or image features must contain the string 'image/decoded'

        # Params

            feature_op_dict: dictionary of strings to fixed feature tensors.
            sensor_image_dimensions: [height, width, channels], defaults to
                [FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
            random_crop_dimensions: [height, width, sensor_color_channels], defaults to
                [FLAGS.crop_height, FLAGS.crop_width, rgb_channels], and for depth
                images the number of channels is automatically set to 1.

        # Returns

            updated feature_op_dict, new_feature_list
        """
        with tf.name_scope('image_random_crop'):
            new_feature_list = []
            if sensor_image_dimensions is None:
                if 'rgb_sensor_image_dimensions' in feature_op_dict:
                    rgb_sensor_image_dimensions = feature_op_dict['rgb_sensor_image_dimensions']
                if 'depth_sensor_image_dimensions' in feature_op_dict:
                    depth_sensor_image_dimensions = feature_op_dict['depth_sensor_image_dimensions']
            if sensor_image_dimensions is None:
                    sensor_image_dimensions = [FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
            height, width, rgb_channels = sensor_image_dimensions

            # get dimensions of random crop if enabled
            if random_crop_dimensions is None:
                random_crop_dimensions = tf.constant([FLAGS.crop_height, FLAGS.crop_width, rgb_channels], name='rgb_random_crop_dimensions')
                depth_crop_dim_tensor = tf.constant([FLAGS.crop_height, FLAGS.crop_width, 1], name='depth_random_crop_dimensions')
            else:
                random_crop_dimensions = tf.constant(random_crop_dimensions, dtype=tf.int32)
                depth_crop_dim_tensor = tf.concat([random_crop_dimensions[:2], tf.constant([1], tf.int32)], 0)
            feature_op_dict['rgb_random_crop_dimensions'] = random_crop_dimensions
            feature_op_dict['depth_random_crop_dimensions'] = depth_crop_dim_tensor
            if random_crop_offset is None:
                # get random crop offset parameters so that cropping will be done consistently.
                random_crop_offset = rcp.random_crop_offset(sensor_image_dimensions, random_crop_dimensions, seed=seed)
            else:
                random_crop_offset = tf.convert_to_tensor(random_crop_offset)
            feature_op_dict['random_crop_offset'] = random_crop_offset
            # add the modified image intrinsics, applying the changes that occur when a crop is performed
            if 'camera/intrinsics/matrix33' in feature_op_dict:
                cropped_camera_intrinsics_matrix = rcp.crop_image_intrinsics(feature_op_dict['camera/intrinsics/matrix33'], random_crop_offset)
                feature_op_dict['camera/cropped/intrinsics/matrix33'] = cropped_camera_intrinsics_matrix
                new_feature_list = np.append(new_feature_list, 'camera/cropped/intrinsics/matrix33')

            # if features aren't supplied by the user (for performance reasons) generate them here
            if image_features is None:
                features = [feature for (feature, tf_op) in six.iteritems(feature_op_dict)]
                image_features = GraspDataset.get_time_ordered_features(features, '/image/decoded')
                image_features.extend(GraspDataset.get_time_ordered_features(features, 'depth_image/decoded'))
                image_features.extend(GraspDataset.get_time_ordered_features(features, 'xyz_image/decoded'))
                image_features.extend(GraspDataset.get_time_ordered_features(features, 'depth_image/median_filtered'))
                image_features.extend(GraspDataset.get_time_ordered_features(features, 'xyz_image/median_filtered'))

            for image_feature in image_features:
                image = feature_op_dict[image_feature]
                if verbose:
                    print('_image_random_crop image:', image, 'random_crop_offset:', random_crop_offset)
                if 'depth_image' in image_feature and 'xyz' not in image_feature:
                    image = rcp.crop_images(image_list=image, offset=random_crop_offset, size=depth_crop_dim_tensor)
                else:
                    # crop rgb and xyz tensor, which each have 3 channels
                    image = rcp.crop_images(image_list=image, offset=random_crop_offset, size=random_crop_dimensions)
                # two image feature types, replace the name in both cases, as appropriate
                cropped_image_feature = image_feature.replace('decoded', 'cropped')
                cropped_image_feature = cropped_image_feature.replace('median_filtered', 'median_filter_cropped')
                feature_op_dict[cropped_image_feature] = image
                new_feature_list = np.append(new_feature_list, cropped_image_feature)

            coordinate_feature_names = []
            # get all image coordinate features so they can be adjusted by the crop offset
            if coordinate_features is None:
                coordinate_feature_names = [feature for (feature, tf_op) in six.iteritems(feature_op_dict)
                                            if 'image_coordinate/yx_2' in feature]

            # apply the cropping offsets to any image coordinates stored in the feature maps
            for coordinate_feature in coordinate_feature_names:
                offset_coordinate_op = feature_op_dict[coordinate_feature] - random_crop_offset[:2]
                # add the cropped feature
                offset_coordinate_cropped_name = coordinate_feature.replace('image_coordinate/yx_2', 'image_coordinate/cropped/yx_2')
                feature_op_dict[offset_coordinate_cropped_name] = offset_coordinate_op
                new_feature_list = np.append(new_feature_list, offset_coordinate_cropped_name)

            return feature_op_dict, new_feature_list

    def _image_random_crop_grasp_attempt(
            self, features_complete_list, feature_op_dicts, sensor_image_dimensions,
            random_crop_offset, seed, time_ordered_feature_name_dict):
        """ Performs the crops for a given grasp attempt.

            Helper for get_training_dictionaries() which extracts image cropping features.
        """
        # Do the random crop preprocessing
        dict_and_feature_tuple_list = []
        # list of all *decoded* image features available
        # note that new features become available later in this function
        image_features = GraspDataset.get_time_ordered_features(features_complete_list, '/image/decoded')
        image_features.extend(GraspDataset.get_time_ordered_features(features_complete_list, 'depth_image/decoded'))
        image_features.extend(GraspDataset.get_time_ordered_features(features_complete_list, 'xyz_image/decoded'))
        image_features.extend(GraspDataset.get_time_ordered_features(features_complete_list, 'depth_image/median_filtered'))
        image_features.extend(GraspDataset.get_time_ordered_features(features_complete_list, 'xyz_image/median_filtered'))
        # loop through each grasp attempt in a batch and crop the images
        for feature_op_dict, sequence_op_dict in tqdm(feature_op_dicts, desc='get_training_dictionaries._image_random_crop()'):
            feature_op_dict, new_feature_list = GraspDataset._image_random_crop(
                feature_op_dict, sensor_image_dimensions, random_crop_offset, seed, image_features=image_features)
            dict_and_feature_tuple_list.append((feature_op_dict, sequence_op_dict))
        # the new_feature_list should be the same for all the ops
        features_complete_list = np.append(features_complete_list, new_feature_list)
        feature_op_dicts = dict_and_feature_tuple_list

        # update the time ordered features with the cropped versions,
        # This inserts new time ordered image coordinate features with 'cropped' as an identifier
        self._remap_time_ordered_feature_names(time_ordered_feature_name_dict, features_complete_list)
        return feature_op_dicts, features_complete_list, time_ordered_feature_name_dict

    @staticmethod
    def _remap_time_ordered_feature_names(
            time_ordered_feature_name_dict, features_complete_list,
            input_key_substring='image_coordinate/yx_2', output_key_substring='image_coordinate/cropped/yx_2',
            input_feature_substring='image_coordinate/yx_2', output_feature_substring='image_coordinate/cropped/yx_2'):
        """ Take a dictionary from a string key to a list of string features, and create a new list with the specified replacements.
        """
        # update the time ordered features with the cropped versions
        time_ordered_cropped_feature_names = {}
        for feature_key, features in six.iteritems(time_ordered_feature_name_dict):
            if input_key_substring in feature_key:
                cropped_features = []
                cropped_feature_key = feature_key.replace(input_key_substring, output_key_substring)
                for feature in features:
                    feature = feature.replace(input_feature_substring, output_feature_substring)
                    if feature in features_complete_list:
                        cropped_features.append(feature)
                    else:
                        print('warning, expected feature missing:' + str(feature))
                time_ordered_cropped_feature_names[cropped_feature_key] = cropped_features

        # combine the new dictionary with the provided dictionary
        time_ordered_feature_name_dict.update(time_ordered_cropped_feature_names)

    def _rgb_preprocessing(
            self, rgb_image_op,
            image_augmentation=FLAGS.image_augmentation,
            imagenet_preprocessing=FLAGS.imagenet_preprocessing,
            resize=FLAGS.resize,
            resize_height=FLAGS.resize_height,
            resize_width=FLAGS.resize_width):
        """Preprocess an rgb image into a float image, applying image augmentation and imagenet mean subtraction if desired.

        Please note that cropped images are generated in `_image_decode()` and given separate feature names.
        Also please be very careful about resizing the rgb image
        """
        with tf.name_scope('rgb_preprocessing'):
            # make sure the shape is correct
            rgb_image_op = tf.squeeze(rgb_image_op)
            # apply image augmentation and imagenet preprocessing steps adapted from keras
            if resize:
                rgb_image_op = tf.image.resize_images(rgb_image_op,
                                                      tf.constant([resize_height, resize_width],
                                                                  name='resize_height_width'))
            if imagenet_preprocessing:
                rgb_image_op = tf.image.per_image_standardization(rgb_image_op)
                rgb_image_op = inception_preprocessing.preprocess_image(
                    rgb_image_op,
                    is_training=image_augmentation,
                    fast_mode=False)
            else:
                rgb_image_op = tf.cast(rgb_image_op, tf.float32)

            return rgb_image_op

    @staticmethod
    def _resize_coordinate(coordinate, input_shape, output_shape):
        """ Update a coordinate based on the current input shape and a new updated output shape.
        """
        coordinate = coordinate * output_shape / input_shape[:2]
        return coordinate

    @staticmethod
    def to_tensors(feature_op_dicts, features):
        """Convert a list or dict of feature strings to tensors.

        # Arguments

        feature_op_dicts: list of (fixed_op_dict, sequence_op_dict) pairs, where each dict is from strings to tensors.
            feature_op_dicts expects data in the following format [({\'feature_name\': tensor},{\'feature_name\': sequence_tensor}).
        features: list of strings, or dict where keys are strings and values are lists of strings

        # Returns
        If 'features' is a str:
            list of list of tensors, one for each dictionary in 'feature_op_dicts'.
        If 'features' is a list:
            list of list of tensors, one for each dictionary in 'feature_op_dicts'.
        If 'features' is a dict:
            list of dicts from strings to tensors, one for each dictionary in 'feature_op_dicts'.
        """
        if isinstance(features, str):
            # motion commands, such as pose or transform features
            if features not in feature_op_dicts[0]:
                available_features = [k for k, v in six.iteritems(feature_op_dicts[0])]
                raise ValueError(
                    'Unknown feature selected: {}'.format(features) +
                    ' Available features include: ' + str(available_features))
            # TODO(ahundt): follow the time step range limits again [grasp_sequence_min_time_step:grasp_sequence_max_time_step]
            simplified_op_list = [t_o_dict[features] for t_o_dict in feature_op_dicts]
            return simplified_op_list
        elif isinstance(features, dict):
            list_of_tensor_dicts = []

            if isinstance(feature_op_dicts, list):
                for fixed_op_dict, seq_op_dict in feature_op_dicts:
                    tensor_dict = {}
                    for time_ordered_key, feature_value_list in features.items():
                        tensor_list = [fixed_op_dict[feature_name] for feature_name in feature_value_list]
                        tensor_dict[time_ordered_key] = tensor_list
                    list_of_tensor_dicts.append(tensor_dict)
                return list_of_tensor_dicts
            else:
                raise ValueError(
                    'feature_op_dicts expects data in the following format: '
                    '[({\'feature_name\': tensor},{\'feature_name\': sequence_tensor})')
        else:
            # assume features is a list, go through and get the list of lists that contain tensors
            return [[fixed_dict[feature] for feature in features] for (fixed_dict, seq_dict) in feature_op_dicts]

    @staticmethod
    def _confirm_expected_feature(features, feature_name, select=0, expected=1):
        """ Returns a single feature out of a list for when one is supposed to be selected out of many.

            The purpose of this function is to help detect and handle inconsistencies in the dataset.
            If the expected number of features isn't present, a warning and stack trace is printed.
        """
        if len(features) == 0:
            raise ValueError('There should be at least one xyz view_clear_scene image feature such as grasp/xyz_image/decoded, but it is missing!')
        selected = features[select]
        if len(features) != expected:
            print('Warning: unexpected number of data sources for feature type:' + feature_name + ' found: ' + str(features) +
                  'confirm you have the correct one. Selected default: ' + selected)
            print(''.join(traceback.format_stack()))
        return selected

    def get_training_dictionaries(
            self,
            feature_op_dicts=None,
            features_complete_list=None,
            time_ordered_feature_name_dict=None,
            num_samples=None,
            batch_size=FLAGS.batch_size,
            image_augmentation=FLAGS.image_augmentation,
            imagenet_preprocessing=FLAGS.imagenet_preprocessing,
            median_filter=FLAGS.median_filter,
            random_crop=FLAGS.random_crop,
            sensor_image_dimensions=None,
            random_crop_dimensions=None,
            random_crop_offset=None,
            resize=FLAGS.resize,
            resize_height=FLAGS.resize_height,
            resize_width=FLAGS.resize_width,
            seed=None,
            verbose=0):
        """Get feature dictionaries containing ops and time ordered feature lists.

        This function is for advanced use cases and aims to make it easy to perform custom training,
        evaluation, or visualization with the loaded dataset.

           # Returns

               [new_feature_op_dicts, features_complete_list, time_ordered_feature_name_dict, num_samples]
        """
        if sensor_image_dimensions is None:
            sensor_image_dimensions = [FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
        if feature_op_dicts is None:
            feature_op_dicts, features_complete_list, num_samples = self._get_simple_parallel_dataset_ops(batch_size=batch_size)
        if time_ordered_feature_name_dict is None:
            time_ordered_feature_name_dict = {}

        # image type to load
        preprocessed_suffix = 'decoded'
        preprocessed_depth_suffix = 'decoded'

        # Ordered sequence of preprocessing operations performed in this function
        preprocessing_sequence = [preprocessed_suffix]
        if verbose:
            print('feature_complete_list before crop len:', len(features_complete_list), 'list:', features_complete_list)

        # Get the surface relative transform tensors
        #
        # Get tensors that load the dataset from disk plus features
        # calculated from the raw data, including transforms and point clouds
        (feature_op_dicts, features_complete_list,
         time_ordered_feature_name_dict, num_samples) = self._get_transform_tensors(
            feature_op_dicts=feature_op_dicts, features_complete_list=features_complete_list,
            time_ordered_feature_name_dict=time_ordered_feature_name_dict,
            num_samples=num_samples, batch_size=batch_size)

        if verbose:
            # print('feature_op_dicts_after_transform_tensors, len', len(feature_op_dicts), 'dicts:', feature_op_dicts)
            print('feature_complete_list after transforms len:', len(features_complete_list), 'list:', features_complete_list)
            print('END DICTS AFTER TRANSFORMS')

        # get the clear view rgb, depth, and xyz image names
        rgb_clear_view_name = 'grasp/image/decoded'
        depth_clear_view_name = 'pregrasp/depth_image/decoded'
        median_filtered_depth_clear_view_name = 'pregrasp/depth_image/median_filtered'
        xyz_clear_view_name = 'pregrasp/xyz_image/decoded'

        # get median filter of depth image if enabled
        if median_filter:
            preprocessed_depth_suffix = 'median_filtered'
            preprocessing_sequence = preprocessing_sequence + [preprocessed_suffix]

        # get dimensions of random crop if enabled
        if random_crop_dimensions is None:
            random_crop_dimensions = [FLAGS.crop_height, FLAGS.crop_width, 3]

        if random_crop_offset is None:
             # do random cropping of depth and rgb images if enabled
            if random_crop:
                # The preprocessed suffix is 'cropped' if cropping is enabled.
                random_crop_offset = rcp.random_crop_offset(sensor_image_dimensions, random_crop_dimensions, seed=seed)
            else:
                random_crop_offset = np.array([0, 0, 3])
                random_crop_offset[:2] = np.array(sensor_image_dimensions[:2]) - np.array(random_crop_dimensions[:2])
                random_crop_offset[0] = random_crop_offset[0] / 2

        preprocessed_suffix = 'cropped'
        preprocessed_depth_suffix = 'cropped'
        preprocessing_sequence = preprocessing_sequence + [preprocessed_suffix]
        # Do the random crop preprocessing
        # This updates rgb images, depth, images, and coordinates of the gripper
        feature_op_dicts, features_complete_list, time_ordered_feature_name_dict = self._image_random_crop_grasp_attempt(
            features_complete_list, feature_op_dicts, sensor_image_dimensions,
            random_crop_offset, seed, time_ordered_feature_name_dict)

        if verbose:
            # print('feature_op_dicts_after_crop len:', len(feature_op_dicts), 'dicts:', feature_op_dicts)
            print('feature_complete_list after crop len:', len(features_complete_list), 'list:', features_complete_list)
            print('END DICTS AFTER CROP')

        preprocessed_depth_suffix = 'median_filter_cropped'
        # get the feature names for the sequence of xyz images
        # in which movement towards the close gripper step is made
        xyz_move_to_grasp_steps_preprocessed = self.get_time_ordered_features(
            features_complete_list,
            feature_type='xyz_image/' + preprocessed_depth_suffix,
            step='move_to_grasp'
        )

        # the feature names vary depending on the user configuration,
        # the random_crop boolean flag in particular
        preprocessed_image_feature_type = '/image/' + preprocessed_suffix

        preprocessed_rgb_clear_view_name = rgb_clear_view_name.replace(
            '/image/decoded', preprocessed_image_feature_type)

        preprocessed_depth_clear_view_name = depth_clear_view_name.replace(
            'depth_image/decoded', 'depth_image/' + preprocessed_depth_suffix)

        preprocessed_xyz_clear_view_name = xyz_clear_view_name.replace(
            'xyz_image/decoded', 'xyz_image/' + preprocessed_depth_suffix)

        # get the feature names for the sequence of rgb images
        # in which movement towards the close gripper step is made
        preprocessed_rgb_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type=preprocessed_image_feature_type,
            step='move_to_grasp'
        )

        # get the feature names for the sequence of rgb images
        # in which movement towards the close gripper step is made
        rgb_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/decoded',
            step='move_to_grasp'
        )

        # get the feature names for the sequence of xyz images
        # in which movement towards the close gripper step is made
        xyz_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type='xyz_image/decoded',
            step='move_to_grasp'
        )

        # get the feature names for the sequence of xyz images
        # in which movement towards the close gripper step is made
        xyz_move_to_grasp_steps_median_filtered = self.get_time_ordered_features(
            features_complete_list,
            feature_type='xyz_image/median_filtered',
            step='move_to_grasp'
        )

        # get the feature names for the sequence of xyz images
        # in which movement towards the close gripper step is made
        xyz_preprocessed_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type='xyz_image/' + preprocessed_depth_suffix,
            step='move_to_grasp'
        )

        # get the feature names for the sequence of depth images
        # in which movement towards the close gripper step is made
        depth_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/decoded',
            step='move_to_grasp'
        )

        # get the feature names for the sequence of filtered depth images
        # in which movement towards the close gripper step is made
        preprocessed_depth_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/' + preprocessed_depth_suffix,
            step='move_to_grasp'
        )

        # get the feature names for the sequence of filtered depth images
        # in which movement towards the close gripper step is made
        median_filtered_depth_move_to_grasp_steps = self.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/median_filtered',
            step='move_to_grasp'
        )

        # These are the command params when the dataset was originally gathered
        # as part of the levine 2016 paper.
        params_names = self.get_time_ordered_features(
            features_complete_list,
            feature_type='params',
            step='move_to_grasp'
        )

        preprocessed_rgb_move_to_grasp_steps_names = []
        new_feature_op_dicts = []

        # go through every element in the batch
        for batch_i, (fixed_feature_op_dict, sequence_feature_op_dict) in enumerate(tqdm(feature_op_dicts,
                                                                                         desc='get_training_dictionaries.preprocess_images')):
            # print('fixed_feature_op_dict: ', fixed_feature_op_dict)
            # get the pregrasp image, and squeeze out the extra batch dimension from the tfrecord
            pregrasp_image_rgb_op = fixed_feature_op_dict[preprocessed_rgb_clear_view_name]
            raw_image_rgb_op = fixed_feature_op_dict[rgb_move_to_grasp_steps[0]]

            if resize:
                preprocessing_sequence = preprocessing_sequence + ['resize']

            # Coordinate updates that correspond to image resizing
            img_shape = K.int_shape(pregrasp_image_rgb_op)
            raw_img_shape = K.int_shape(raw_image_rgb_op)
            output_shape = np.array([resize_height, resize_width])

            resized_coordinate_dict = {}
            for feature_key, feature in six.iteritems(fixed_feature_op_dict):
                if 'yx_2' in feature_key and 'image_coordinate' in feature_key:
                    # Keep in mind multiple sizes of image features may be present,
                    # resize needs to be done based on the correct input and output image size,
                    # not just the pregrasp_image_rgb_op's size.
                    if 'cropped' in feature_key:
                        coordinate = self._resize_coordinate(feature, img_shape, output_shape)
                        coordinate_name = feature_key.replace(preprocessed_suffix, 'cropped_resized')
                        resized_coordinate_dict[coordinate_name] = coordinate
                        if batch_i == 0:
                            features_complete_list = np.append(features_complete_list, coordinate_name)
                    elif 'image_coordinate/yx_2' in feature_key:
                        coordinate = self._resize_coordinate(feature, raw_img_shape, output_shape)
                        coordinate_name = feature_key.replace('image_coordinate/yx_2', 'image_coordinate/original_resized/yx_2')
                        resized_coordinate_dict[coordinate_name] = coordinate
                        if batch_i == 0:
                            features_complete_list = np.append(features_complete_list, coordinate_name)
                    else:
                        print('Warning: unsupported feature_key detected in resize: ' + str(feature_key))
            fixed_feature_op_dict.update(resized_coordinate_dict)

            # Image resizing and other rgb preprocessing,
            # such as changes in brightness and saturation
            pregrasp_image_rgb_op = self._rgb_preprocessing(
                pregrasp_image_rgb_op,
                image_augmentation=image_augmentation,
                imagenet_preprocessing=imagenet_preprocessing,
                resize=resize,
                resize_height=resize_height,
                resize_width=resize_width)
            fully_preprocessed_rgb_clear_view_name = preprocessed_rgb_clear_view_name.replace(
                preprocessed_image_feature_type, '/image/preprocessed')
            fixed_feature_op_dict[fully_preprocessed_rgb_clear_view_name] = pregrasp_image_rgb_op

            grasp_success_op = tf.squeeze(fixed_feature_op_dict['grasp_success'])
            if self.verbose > 2:
                print('\nrgb_move_to_grasp_steps: ', rgb_move_to_grasp_steps)

            for time_step_j, (grasp_step_rgb_feature_name) in enumerate(preprocessed_rgb_move_to_grasp_steps):
                # do preprocessing and add new preprocessed version of image to fixed_feature_op_dict
                grasp_step_rgb_feature_op = self._rgb_preprocessing(
                    fixed_feature_op_dict[grasp_step_rgb_feature_name],
                    image_augmentation=image_augmentation,
                    imagenet_preprocessing=imagenet_preprocessing,
                    resize=resize,
                    resize_height=resize_height,
                    resize_width=resize_width)
                grasp_step_rgb_feature_name = grasp_step_rgb_feature_name.replace(
                    preprocessed_image_feature_type, '/image/preprocessed')
                fixed_feature_op_dict[grasp_step_rgb_feature_name] = grasp_step_rgb_feature_op
                if batch_i == 0:
                    # add to list of names, assume all batches have the same steps so only need to append names on batch 0
                    preprocessed_rgb_move_to_grasp_steps_names.append(grasp_step_rgb_feature_name)

            # assemble the updated feature op dicts
            new_feature_op_dicts.append((fixed_feature_op_dict, sequence_feature_op_dict))

        # Determine the image coordinate feature name based on user selected preprocessing options
        preprocessed_final_coordinate_suffix = 'yx_2'
        if resize:
            preprocessed_final_coordinate_suffix = 'cropped_resized/yx_2'
        else:
            preprocessed_final_coordinate_suffix = 'cropped/yx_2'

        # get the sequence of preprocessed coordinates
        preprocessed_final_coordinate_names = self.get_time_ordered_features(
            features_complete_list,
            feature_type='endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/' + preprocessed_final_coordinate_suffix,
            step='move_to_grasp'
        )

        # get the sequence of preprocessed coordinates
        preprocessed_current_coordinate_names = self.get_time_ordered_features(
            features_complete_list,
            feature_type='endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/' + preprocessed_final_coordinate_suffix,
            step='move_to_grasp'
        )

        num_time_steps = len(rgb_move_to_grasp_steps)
        grasp_success_names = ['grasp_success'] * num_time_steps

        grasp_success_yx_3_names = []
        # create grasp_success_yx_3 tensors for pixel-wise training loss and accuracy labels
        for batch_i, (fixed_feature_op_dict, sequence_feature_op_dict) in enumerate(tqdm(new_feature_op_dicts,
                                                                                         desc='get_training_dictionaries.grasp_success_yx_3')):
            for time_step_j, (grasp_success_name, coordinate_name) in enumerate(zip(grasp_success_names, preprocessed_final_coordinate_names)):
                grasp_success_op = tf.cast(fixed_feature_op_dict[grasp_success_name], tf.int32)
                grasp_success_final_coordinate_op = tf.cast(fixed_feature_op_dict[coordinate_name], tf.int32)
                grasp_success_yx_3_op = K.concatenate([grasp_success_op, grasp_success_final_coordinate_op])
                grasp_success_yx_3_name = coordinate_name.replace('yx_2', 'grasp_success_yx_3')
                fixed_feature_op_dict[grasp_success_yx_3_name] = grasp_success_yx_3_op
                if batch_i == 0:
                    grasp_success_yx_3_names.append(grasp_success_yx_3_name)

        new_time_ordered_feature_name_dict = {
            'move_to_grasp/time_ordered/clear_view/rgb_image/decoded': [rgb_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/clear_view/rgb_image/preprocessed': [fully_preprocessed_rgb_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/clear_view/depth_image/decoded': [depth_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/clear_view/depth_image/median_filtered': [median_filtered_depth_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/clear_view/depth_image/preprocessed': [preprocessed_depth_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/clear_view/xyz_image/decoded': [xyz_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/clear_view/xyz_image/preprocessed': [preprocessed_xyz_clear_view_name] * num_time_steps,
            'move_to_grasp/time_ordered/rgb_image/decoded': rgb_move_to_grasp_steps,
            'move_to_grasp/time_ordered/rgb_image/preprocessed': preprocessed_rgb_move_to_grasp_steps_names,
            'move_to_grasp/time_ordered/xyz_image/decoded': xyz_move_to_grasp_steps,
            # note: at the time of writing preprocessing of xyz images only includes cropping
            'move_to_grasp/time_ordered/xyz_image/preprocessed': xyz_move_to_grasp_steps_preprocessed,
            'move_to_grasp/time_ordered/xyz_image/median_filtered': xyz_move_to_grasp_steps_median_filtered,
            'move_to_grasp/time_ordered/depth_image/preprocessed': preprocessed_depth_move_to_grasp_steps,
            'move_to_grasp/time_ordered/depth_image/decoded': depth_move_to_grasp_steps,
            'move_to_grasp/time_ordered/depth_image/median_filtered': median_filtered_depth_move_to_grasp_steps,
            'move_to_grasp/time_ordered/grasp_success': grasp_success_names,
            'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/preprocessed/yx_2':
                preprocessed_current_coordinate_names,
            'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/yx_2':
                preprocessed_final_coordinate_names,
            'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/grasp_success_yx_3':
                grasp_success_yx_3_names
        }

        # combine the new dictionary with the provided dictionary
        time_ordered_feature_name_dict.update(new_time_ordered_feature_name_dict)

        if verbose:
            print('END OF GET TRAINING_DICTIONARIES')
            print('features_complete_list', features_complete_list)
            features = [feature for (feature, time_ordered_features) in six.iteritems(time_ordered_feature_name_dict)]
            print('time_ordered_feature_name_dict keys: ', features)

        # All time ordered features must have the same length for ease of use
        for (feature, time_ordered_features) in six.iteritems(time_ordered_feature_name_dict):
            if not len(time_ordered_features) == num_time_steps:
                print('WARNING: expected ' + str(num_time_steps) + ' time steps but found ' + str(len(time_ordered_features)) +
                      ' in feature: ' + feature + ' in dataset ' + self.dataset +
                      ' check get_training_dictionaries().'.join(traceback.format_stack()))

        return new_feature_op_dicts, features_complete_list, time_ordered_feature_name_dict, num_samples

    def get_training_tensors(
            self, batch_size=FLAGS.batch_size,
            imagenet_preprocessing=FLAGS.imagenet_preprocessing,
            image_augmentation=FLAGS.image_augmentation,
            random_crop=FLAGS.random_crop,
            sensor_image_dimensions=None,
            random_crop_dimensions=None,
            random_crop_offset=None,
            resize=FLAGS.resize,
            resize_height=FLAGS.resize_height,
            resize_width=FLAGS.resize_width,
            motion_command_feature=FLAGS.grasp_sequence_motion_command_feature,
            grasp_sequence_image_feature=FLAGS.grasp_sequence_image_feature,
            clear_view_image_feature=FLAGS.grasp_sequence_image_feature,
            grasp_success_label=FLAGS.grasp_success_label,
            grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
            grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
            seed=None):
        """Get tensors configured for training on grasps from a single dataset.

         TODO(ahundt) 2017-12-05 update get_training_tensors docstring, now expects 'move_to_grasp/time_ordered/' feature strings.

         # Features

         The arguments motion_command_feature, grasp_sequence_image_feature, clear_view_image_feature, and grasp_success_label
         each accept a feature name string as their parameter. The available feature types are outlined below:
                 'delta_depth_sin_cos_3' [delta_depth, sin(theta), cos(theta)] where delta_depth depth offset for the gripper
                     from the measured surface, alongside a single rotation angle theta containing sin(theta), cos(theta).
                     This format does not allow for arbitrary commands to be defined, and the rotation component
                     is based on the paper and dataset:
                                 https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                                 https://arxiv.org/abs/1603.02199
                     This is a surface relative transform parameterization.
                 'delta_depth_quat_5' A numpy array with 5 total entries including depth offset and 4 entry quaternion.
                     This is a surface relative transform parameterization.
                 'final_pose_orientation_quaternion' directly input the final pose translation and orientation.
                     Vector and quaternion representing the absolute end effector position at the end of the grasp attempt,
                     the actual final value will vary based on the dataset being used. This is the same as the
                     'grasp/final/reached_pose/transforms/base_T_endeffector/vec_quat_7' feature.
                 'next_timestep' input the params for the command saved in the dataset with translation,
                     sin theta, cos theta from the current time step to the next. This is the same as the 'params' feature.
                 'endeffector_final_clear_view_depth_pixel_T_endeffector_final'
                     Surface relative transform from the final gripper pixel depth position
                     This is fairly complex, so please inquire if additional details are needed.
                     Determine the 'final_pose_orientation_quaternion', the specific feature will vary based on the dataset,
                     for example in dataset 102 it is `grasp/10/reached_pose/transforms/base_T_endeffector/vec_quat_7`.
                     Using this pose, we determine the x,y pixel coordinate of the gripper's reached pose at the final time step
                     in the camera frame, and use this to look up the depth value in the initial clear view image.
                     'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/vec_quat_7'.
                 'endeffector_current_T_endeffector_final_vec_sin_cos_5' use
                     the real reached gripper pose of the end effector to calculate
                     the transform from the current time step to the final time step
                     to generate the parameters defined in https://arxiv.org/abs/1603.02199,
                     consisting of [x,y,z, sin(theta), cos(theta)].
                 'endeffector_current_T_endeffector_final_vec_quat_7'
                     vector and quaternion representing the transform from the current time step's pose
                     to the pose at the final time step in the current time step's end effector frame.
                     This also generates the new feature
                     'move_to_grasp/###/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_quat_7',
                     where ### is a number for each step from 000 to the number of time steps in the example.
                     use the real reached gripper pose of the end effector to calculate
                     the transform from the current time step to the final time step
                     to generate the parameters [x, y, z, qx, qy, qz, qw].

         ## Feature Names

             feature: move_to_grasp/time_ordered/clear_view/depth_image/decoded
             feature: move_to_grasp/time_ordered/clear_view/xyz_image/preprocessed
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_sin_cos_3
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/vec_quat_7
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/cropped/yx_2
             feature: move_to_grasp/time_ordered/reached_pose/transforms/camera_T_endeffector/vec_quat_7
             feature: move_to_grasp/time_ordered/reached_pose/transforms/camera/transforms/base_T_camera/vec_quat_7
             feature: move_to_grasp/time_ordered/xyz_image/decoded
             feature: move_to_grasp/time_ordered/reached_pose/transforms/camera_T_depth_pixel/vec_quat_7
             feature: move_to_grasp/time_ordered/clear_view/xyz_image/decoded
             feature: move_to_grasp/time_ordered/grasp_success
             feature: move_to_grasp/time_ordered/clear_view/depth_image/preprocessed
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/cropped/yx_2
             feature: move_to_grasp/time_ordered/clear_view/rgb_image/decoded
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/sin_cos_2
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/delta_depth_quat_5
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_quat_7
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/vec_quat_7
             feature: move_to_grasp/time_ordered/xyz_image/preprocessed
             feature: move_to_grasp/time_ordered/clear_view/rgb_image/preprocessed
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/grasp_success_yx_3
             feature: move_to_grasp/time_ordered/depth_image/decoded
             feature: move_to_grasp/time_ordered/rgb_image/decoded
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/yx_2
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/preprocessed/yx_2
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_sin_cos_5
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_final_clear_view_depth_pixel_T_endeffector_final/image_coordinate/preprocessed/yx_2
             feature: move_to_grasp/time_ordered/rgb_image/preprocessed
             feature: move_to_grasp/time_ordered/reached_pose/transforms/camera_T_endeffector_final/vec_quat_7
             feature: move_to_grasp/time_ordered/reached_pose/transforms/camera_T_depth_pixel_final/vec_quat_7
             feature: move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/yx_2

         # Arguments

         grasp_sequence_max_time_step: Grasp examples consist of time steps from 0 to up to a max of 11.
             To train on specific range of data set this value to the maximum desired time step.
             A value of None gives unlimited range.
         grasp_sequence_min_time_step: Grasp examples consist of time steps from 0 to up to a max of 11.
             To train on specific range of data set this value to the minimum desired time step.
             A value of None gives unlimited range.
         motion_command_feature: The grasp command parameter to be supplied during training.

        # Returns

            (pregrasp_op_batch, grasp_step_op_batch, simplified_grasp_command_op_batch, grasp_success_op_batch, num_samples)
        """
        # Get tensors that load the dataset from disk plus features calculated from the raw data, including transforms and point clouds
        feature_op_dicts, features_complete_list, time_ordered_feature_name_dict, num_samples = self.get_training_dictionaries(
            batch_size=batch_size, random_crop=random_crop, sensor_image_dimensions=sensor_image_dimensions,
            imagenet_preprocessing=imagenet_preprocessing, image_augmentation=image_augmentation,
            random_crop_dimensions=random_crop_dimensions, random_crop_offset=random_crop_offset)

        time_ordered_feature_tensor_dicts = GraspDataset.to_tensors(feature_op_dicts, time_ordered_feature_name_dict)

        # motion commands, such as pose or transform features
        simplified_grasp_command_op_batch = self.to_tensors(time_ordered_feature_tensor_dicts, motion_command_feature)

        # image of a clear scene view, originally from 'view_clear_scene' step,
        # There is also a move_to_grasp versions copied from view_clear_scene then repeated once for each time step.
        pregrasp_op_batch = self.to_tensors(time_ordered_feature_tensor_dicts, clear_view_image_feature)

        # image from the current time step
        grasp_step_op_batch = self.to_tensors(time_ordered_feature_tensor_dicts, grasp_sequence_image_feature)

        # grasp success labels from the motion
        grasp_success_op_batch = self.to_tensors(time_ordered_feature_tensor_dicts, grasp_success_label)

        # make one long list from each list of lists
        pregrasp_op_batch = list(itertools.chain.from_iterable(pregrasp_op_batch))
        grasp_step_op_batch = list(itertools.chain.from_iterable(grasp_step_op_batch))
        simplified_grasp_command_op_batch = list(itertools.chain.from_iterable(simplified_grasp_command_op_batch))
        grasp_success_op_batch = list(itertools.chain.from_iterable(grasp_success_op_batch))

        # stack all the data in a way that will let it run in parallel
        pregrasp_op_batch = tf.parallel_stack(pregrasp_op_batch)
        grasp_step_op_batch = tf.parallel_stack(grasp_step_op_batch)
        simplified_grasp_command_op_batch = tf.parallel_stack(simplified_grasp_command_op_batch)
        grasp_success_op_batch = tf.parallel_stack(grasp_success_op_batch)

        return pregrasp_op_batch, grasp_step_op_batch, simplified_grasp_command_op_batch, grasp_success_op_batch, num_samples

    def npy_to_gif(self, npy, filename, fps=2):
        """Convert a numpy array into a gif file at the location specified by filename.
        """
        # TODO(ahundt) currently importing moviepy prevents python from exiting. Once this is resolved remove the import below.
        import moviepy.editor as mpy
        clip = mpy.ImageSequenceClip(list(npy), fps)
        clip.write_gif(filename)

    def create_gif(self, tf_session=None,
                   visualization_dir=FLAGS.visualization_dir,
                   rgb_feature_type='move_to_grasp/time_ordered/rgb_image/preprocessed',
                   depth_feature_type='depth_image/rgb_encoded',
                   draw='circle_on_gripper',
                   coordinate_feature='move_to_grasp/time_ordered/reached_pose/transforms/endeffector_clear_view_depth_pixel_T_endeffector/image_coordinate/preprocessed/yx_2',
                   verbose=1):
        """Create gif images of each grasp attempt and write them to visualization_dir

        # Arguments

            tf_session: the TensorFlow Session to use
            visualization_dir: where to save the gif files
            rgb_feature_type: save rgb gif files with the feature type provided, see get_time_ordered_features().
            depth_feature type: save depth gif files with the feature type provided, see get_time_ordered_features().
            pipeline: Determines what preprocessing you will see when the images are saved.
                None saves gifs containing the raw dataset data,
                'training' will save the images after preprocessing with the training pipeline.
            draw: visualize data and feature calculations. Options include:
                None to disable this option and simply output the image sequences as a gif.
                'circle_on_gripper' to draw a circle at the gripper position.
        """
        """Create input tfrecord tensors.

        Args:
        novel: whether or not to grab novel or seen images.
        Returns:
        list of tensors corresponding to images. The images
        tensor is 5D, batch x time x height x width x channels.
        Raises:
        RuntimeError: if no files found.
        """
        if tf_session is None:
            tf_session = tf.Session()
        mkdir_p(FLAGS.visualization_dir)

        batch_size = 1
        (feature_op_dicts, features_complete_list,
         time_ordered_feature_name_dict, num_samples) = self.get_training_dictionaries(batch_size=batch_size, verbose=verbose)

        tf_session.run(tf.global_variables_initializer())

        for attempt_num in tqdm(range(num_samples / batch_size), desc='dataset'):
            attempt_num_string = 'attempt_' + str(attempt_num).zfill(4) + '_'
            print('dataset_' + self.dataset + '_' + attempt_num_string + 'starting')
            # batch shize should actually always be 1 for this visualization
            output_features_dicts = tf_session.run(feature_op_dicts)
            # reorganize is grasp attempt so it is easy to walk through
            [time_ordered_feature_data_dict] = self.to_tensors(output_features_dicts, time_ordered_feature_name_dict)
            # features_dict_np contains fixed dimension features, sequence_dict_np contains variable length sequences of data
            # We're assuming the batch size is 1, which is why there are only two elements in the list.
            [(features_dict_np, sequence_dict_np)] = output_features_dicts

            if rgb_feature_type:
                if rgb_feature_type in time_ordered_feature_name_dict:
                    video = time_ordered_feature_data_dict[rgb_feature_type]
                    if attempt_num == 0:
                        print("Saving rgb features to a gif in the following order: " +
                              str(time_ordered_feature_name_dict[rgb_feature_type]))
                else:
                    ordered_rgb_image_features = GraspDataset.get_time_ordered_features(
                        features_complete_list,
                        feature_type=rgb_feature_type)
                    if attempt_num == 0:
                        print("Saving rgb features to a gif in the following order: " +
                              str(ordered_rgb_image_features))
                    video = np.concatenate(self.to_tensors(output_features_dicts, ordered_rgb_image_features), axis=0)

                video_min = np.min(video)
                video_max = np.max(video)
                if video_min < 0 or video_max < 10:
                    # video_max = 1.0
                    # normalize the values for display if they aren't in 0 to 255 range
                    video = (video - video_min) / (video_max - video_min) * 254
                    print('Warning: normalizing video image channels for gif with min: ' +
                          str(video_min) + ' and max: ' + str(video_max))

                if draw == 'circle_on_gripper':
                    coordinates = time_ordered_feature_data_dict[coordinate_feature]
                    if 'random_crop_offset' in features_dict_np:
                        offset = features_dict_np['random_crop_offset']
                    else:
                        offset = np.array([0, 0])

                    num_frames = len(video)
                    num_coordinates = len(coordinates)
                    # fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
                    circle_vid = []
                    # the coordinates aren't available for every time step
                    # so this offset ensures they are drawn on the correct time step
                    if num_coordinates == num_frames:
                        coordinate_time_offset = 0
                    else:
                        # when showing all rgb images there isn't
                        # a coordinate for the clear view image
                        coordinate_time_offset = 1
                    for i, frame in enumerate(zip(video)):
                        frame = np.array(frame, dtype=np.uint8)
                        # TODO(ahundt) fix hard coded range
                        if((coordinate_time_offset == 0) or
                           (i > 1 and i < len(coordinates) + coordinate_time_offset)):

                            coordinate = coordinates[i - coordinate_time_offset]
                            # offset_coordinate = coordinate - offset[:2]
                            frame = grasp_geometry.draw_circle(
                                frame,
                                np.array(coordinate, dtype=np.int32),
                                color=[0, 255, 255])
                        circle_vid.append(frame)
                        # plt.show()
                    video = np.concatenate(circle_vid)
                gif_filename = (os.path.basename(str(self.dataset) + '_grasp_' + str(int(attempt_num)) +
                                '_rgb_success_' + str(int(features_dict_np['grasp_success'])) + '.gif'))
                gif_path = os.path.join(visualization_dir, gif_filename)
                self.npy_to_gif(video, gif_path)

            if depth_feature_type:
                ordered_depth_image_features = GraspDataset.get_time_ordered_features(
                    features_complete_list,
                    feature_type=depth_feature_type)
                if attempt_num == 0:
                    print("Saving depth features to a gif in the following order: " + str(ordered_depth_image_features))
                frame_list = self.to_tensors(output_features_dicts, ordered_depth_image_features)
                if FLAGS.median_filter:
                    for single_frame in frame_list[0]:
                            median_filter(single_frame, size=(FLAGS.median_filter_height,FLAGS.median_filter_width, 1), output=single_frame)
                video = np.concatenate(frame_list, axis=0)
                gif_filename = (os.path.basename(str(self.dataset) + '_grasp_' + str(int(attempt_num)) +
                                '_depth_success_' + str(int(features_dict_np['grasp_success'])) + '.gif'))
                gif_path = os.path.join(visualization_dir, gif_filename)
                self.npy_to_gif(video, gif_path)

    def count_success_failure_number(self, tf_session=tf.Session(), save_file=True, save_dir=FLAGS.data_dir, filename=None, verbose=1):
        """ Count the total number of grasp attempts, grasp successes, and grasp failures in the loaded dataset.

        Counts, prints, and saves a file containing:

         1. the total number of grasp attempts in the dataset
         2. the number of times an object was successfully grasped
         3. the number of times an attempt failed and no object was grasped

        The printout and saved file also writes:
            grasp successes / total attempts
            grasp failures / total attempts

        # Returns

            [attempt_num, success_num, failure_num]

            attempt_num: total number of grasp attempts
            success_num: number of success attempts
            failure_num: number of failed attempts
        """
        batch_size = 1
        (feature_op_dicts, _, _, num_samples) = self.get_training_dictionaries(batch_size=batch_size)

        text_lines = ('Statistics for grasp_dataset_' + self.dataset + ':\n' +
                      'Total grasp attempts: ' + str(num_samples) + '\n')
        if verbose:
            print(str(text_lines))

        [(fixed_feature_op_dict, _)] = feature_op_dicts
        grasp_success_op = fixed_feature_op_dict['grasp_success']
        tf_session.run(tf.global_variables_initializer())

        success_num = 0
        failure_num = 0
        for attempt_num in tqdm(range(num_samples), desc='loading grasp success frequency statictics'):
            grasp_success = tf_session.run(grasp_success_op)
            # grasp_success is 1 if successful, 0 otherwise
            success_num += int(grasp_success)

        failure_num = num_samples - success_num
        success_ratio = success_num * 1.0 / num_samples

        text_lines = text_lines + (
                      'grasp successes: ' + str(success_num) + '\n' +
                      'grasp failures: ' + str(failure_num) + '\n' +
                      'grasp successes / total attempts: ' + str(success_ratio) + '\n' +
                      'grasp failures / total attempts: ' + str(1 - success_ratio) + '\n')

        if verbose:
            print(str(text_lines))

        text_lines = text_lines + calculated_text_lines

        if save_file is True:
            save_dir = FLAGS.data_dir
            mkdir_p(save_dir)
            if filename is None:
                filename = 'grasp_dataset_' + self.dataset + '_statistics.txt'
            complete_path = os.path.join(save_dir, filename)
            file_object = open(complete_path, 'w')
            file_object.writelines(text_lines)
            file_object.close()
            if verbose:
                print('Results saved to: ' + complete_path)

        return num_samples, success_num, failure_num


def get_multi_dataset_training_tensors(
        datasets=FLAGS.grasp_datasets_train,
        batch_size=FLAGS.batch_size,
        imagenet_preprocessing=FLAGS.imagenet_preprocessing,
        grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
        grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
        random_crop=FLAGS.random_crop,
        resize=FLAGS.resize,
        resize_height=FLAGS.resize_height,
        resize_width=FLAGS.resize_width,
        grasp_datasets_batch_algorithm=FLAGS.grasp_datasets_batch_algorithm):
    """Aggregate multiple datasets into combined training tensors.

    # TODO(ahundt) parameterize this function properly, don't just use FLAGS defaults in get_training_tensors

    The Grasping data is divided into multiple datasets with varying numbers
    of features. In particular, individually each dataset has a fixed number
    of time steps, but the number of time steps varies between datasets.
    Each dataset is named based on the number of features it contains.

    This function aggregates these datasets such that a 'batch' is number
    of grasp examples, and a 'minibatch' is the number of time steps
    in one grasp example.

    The number of training samples per step for a single dataset will be:

    samples_per_step_for_one_dataset = (minibatch size * batch size)

    The total number of training samples per step for multiple datsets is
    the sum of the samples per step in each dataset:

    samples_per_step_for_all_datasets = sum(list_of_samples_per_step_for_one_dataset)

    Dataset 102 has 10 time steps, for example,
    and dataset 097 (used for eval) has 9 time steps. This means
    a batch size of 5 with the 'constant' training algorithm
    will mean dataset 102 has  10 * batch size
    5 = 50 individual training time step samples per step,
    and 097 will have 9  * 5 = 45

    # Arguments

    datasets:
        datasets parameter must be a list of strings or a
        comma separated string identifying the datasets to load
        such as 062_b,063,072_a,082_b,102.
        see FLAGS.grasp_datasets_train
    grasp_datasets_batch_algorithm: Use default batch if constant.
        'constant' training on multiple datasets reads `batch_size`
            elements from each dataset at each training step when this parameter.
        'proportional' each dataset's batch size will be individually
            set to int(batch_size*single_batch/max_batch_size)
            so smaller datasets are run more slowly than larger datasets.

    Note that one limitation of this setup is that we will
    iterate over samples according to the largest dataset,
    which means we see smaller datasets more than once in
    a single epoch. Try not to aggregate a very small dataset
    with a very large one with 'constant' batch algorithm or a small
    batch size and the 'proportional' algorithm!

    # Returns

    Time-aligned list of data tensors of equal batch size for training on multiple datasets simultaneously.

    [pregrasp_op_batch,
     grasp_step_op_batch,
     simplified_grasp_command_op_batch,
     grasp_success_op_batch,
     steps_per_epoch]

    pregrasp_op_batch:
        4d [batch, height, width, channels]
        rgb image feature tensor containing the clear view scene.
    grasp_step_op_batch:
        4d [batch, height, width, channels] rgb image feature tensor
        containing the current time step's scene (gripper visible).
    simplified_grasp_command_op_batch:
        2d [batch, channels] feature tensor containing the
        "proposed" command which the network should evaluate.
        this parameter is configurable based on the
        FLAGS.grasp_sequence_motion_command_feature command line parameter
    grasp_success_op_batch:
        2d [batch, 1] tensor containing labels,
        including a 1 if the grasp attempt succeeded
        and a 0 if the grasp attempt failed.
    steps_per_epoch:
        A python integer with the number of steps to traverse
        the largest of the aggregated datasets once.
        This means that not all datasets will be covered at a perfectly equal pace.

    """
    if grasp_datasets_batch_algorithm != 'constant' and grasp_datasets_batch_algorithm != 'proportional':
        raise ValueError('grasp_datasets_batch_algorithm string value must be either constant or proportional.')

    if isinstance(datasets, str):
        datasets = dataset.split(',')
    elif not isinstance(datasets, list):
        raise ValueError('datasets parameter must be a list of strings or '
                         'a comma separated string such as 062_b,063,072_a,082_b,102')
    max_num_samples = 0
    grasp_datasets = []
    pregrasp_op_batch = []
    grasp_step_op_batch = []
    # simplified_network_grasp_command_op
    simplified_grasp_command_op_batch = []
    grasp_success_op_batch = []

    dataset_batch_sizes = []
    grasp_dataset_list = []

    # initialize all datasets
    for single_dataset in datasets:
        data = GraspDataset(dataset=single_dataset)
        grasp_dataset_list.append(data)
        dataset_batch_sizes.append(data.get_features()[1])

    max_batch_size = max(dataset_batch_sizes)
    # Not sure why any thing assigned to max_batch_size, it can pass
    for single_dataset, single_batch in zip(grasp_dataset_list, tqdm(dataset_batch_sizes, desc='load_selected_datasets')):
        proportional_batch_size = batch_size
        if(grasp_datasets_batch_algorithm == 'proportional'):
            # scale batch size of each dataset according to the number of samples,
            # while ensuring there is a minimum batch size of 1.
            proportional_batch_size = max(int(batch_size * single_batch / max_batch_size), 1)
        data = single_dataset
        # list of dictionaries the length of batch_size
        (pregrasp_op,
         grasp_step_op,
         simplified_grasp_command_op,
         grasp_success_op,
         num_samples) = data.get_training_tensors(
             batch_size=proportional_batch_size,
             imagenet_preprocessing=imagenet_preprocessing,
             random_crop=random_crop,
             resize=resize,
             grasp_sequence_min_time_step=grasp_sequence_min_time_step,
             grasp_sequence_max_time_step=grasp_sequence_max_time_step)

        max_num_samples = max(num_samples, max_num_samples)
        pregrasp_op_batch.append(pregrasp_op)
        grasp_step_op_batch.append(grasp_step_op)
        simplified_grasp_command_op_batch.append(simplified_grasp_command_op)
        grasp_success_op_batch.append(grasp_success_op)
        # make sure we visit every image once

    steps_per_epoch = int(np.ceil(float(max_num_samples)/float(batch_size)))

    pregrasp_op_batch = tf.concat(pregrasp_op_batch, 0)
    grasp_step_op_batch = tf.concat(grasp_step_op_batch, 0)
    simplified_grasp_command_op_batch = tf.concat(simplified_grasp_command_op_batch, 0)
    print('grasp_success_op_batch before concat: ', grasp_success_op_batch)
    grasp_success_op_batch = tf.concat(grasp_success_op_batch, 0)
    print('pregrasp_op_batch:', pregrasp_op_batch,
          'grasp_step_op_batch:', grasp_step_op_batch,
          'simplified_grasp_command_op_batch:', simplified_grasp_command_op_batch,
          'grasp_success_op_batch:', grasp_success_op_batch)

    return pregrasp_op_batch, grasp_step_op_batch, simplified_grasp_command_op_batch, grasp_success_op_batch, steps_per_epoch

if __name__ == '__main__':
    with tf.Session() as sess:
        gd = GraspDataset()
        if FLAGS.grasp_download:
            gd.download(dataset=FLAGS.grasp_dataset)
        gd.create_gif(sess)
