# -*- coding: utf-8 -*-
"""Code for visualizing grasp attempt examples from the google brain robotics grasping dataset.

https://sites.google.com/site/brainrobotdata/home/grasping-dataset

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0
"""
import os
import errno
import traceback

import numpy as np
import six  # compatibility between python 2 + 3 = six
import matplotlib.pyplot as plt

try:
    import vrep
except Exception as e:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in PYTHONPATH folder relative to this file,')
    print ('or appropriately adjust the file "vrep.py. Also follow the"')
    print ('ReadMe.txt in the vrep remote API folder')
    print ('--------------------------------------------------------------')
    print ('')
    raise
    # raise e

import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from keras.utils import get_file
from ply import write_xyz_rgb_as_ply
from PIL import Image

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

import grasp_geometry
import grasp_geometry_tf
from grasp_train import GraspTrain
from grasp_train import choose_make_model_fn
from depth_image_encoding import ClipFloatValues
from depth_image_encoding import FloatArrayToRgbImage
from depth_image_encoding import FloatArrayToRawRGB
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import img_as_uint
from skimage.color import grey2rgb

try:
    import eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
    import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
except ImportError:
    print('eigen and sva python modules are not available. To install run the script at:'
          'https://github.com/ahundt/robotics_setup/blob/master/robotics_tasks.sh'
          'or follow the instructions at https://github.com/jrl-umi3218/Eigen3ToPython'
          'and https://github.com/jrl-umi3218/SpaceVecAlg. '
          'When you build the modules make sure python bindings are enabled.')

tf.flags.DEFINE_string('vrepConnectionAddress', '127.0.0.1', 'The IP address of the running V-REP simulation.')
tf.flags.DEFINE_integer('vrepConnectionPort', 19999, 'ip port for connecting to V-REP')
tf.flags.DEFINE_boolean('vrepWaitUntilConnected', True, 'block startup call until vrep is connected')
tf.flags.DEFINE_boolean('vrepDoNotReconnectOnceDisconnected', True, '')
tf.flags.DEFINE_integer('vrepTimeOutInMs', 5000, 'Timeout in milliseconds upon which connection fails')
tf.flags.DEFINE_integer('vrepCommThreadCycleInMs', 5, 'time between communication cycles')
tf.flags.DEFINE_integer('vrepVisualizeGraspAttempt_min', 0, 'min grasp attempt to display from dataset, or -1 for no limit')
tf.flags.DEFINE_integer('vrepVisualizeGraspAttempt_max', 100, 'max grasp attempt to display from dataset, exclusive, or -1 for no limit')
tf.flags.DEFINE_boolean('vrepVisualizeRGBD', True, 'display the rgbd images and point cloud')
tf.flags.DEFINE_integer('vrepVisualizeRGBD_min', 0, 'min time step on each grasp attempt to display, or -1 for no limit')
tf.flags.DEFINE_integer('vrepVisualizeRGBD_max', -1, 'max time step on each grasp attempt to display, exclusive, or -1 for no limit')
tf.flags.DEFINE_boolean('vrepVisualizeSurfaceRelativeTransform', True, 'display the surface relative transform frames')
tf.flags.DEFINE_boolean('vrepVisualizeSurfaceRelativeTransformLines', True, 'display lines from the camera to surface depth points')
tf.flags.DEFINE_string('vrepParentName', 'LBR_iiwa_14_R820', 'The default parent frame name from which to base all visualized transforms.')
tf.flags.DEFINE_boolean('vrepVisualizeDilation', False, 'Visualize result of dilation performed on depth image used for point cloud.')
# The commented flags below can now be found in vrep/visualization.py
#
# TODO(ahundt) clean the moved code up
#
# tf.flags.DEFINE_string('vrepVisualizeDepthFormat', 'vrep_depth_encoded_rgb',
#                        """ Controls how Depth images are displayed. Options are:
#                            None: Do not modify the data and display it as-is for rgb input data (not working properly for float depth).
#                            'depth_rgb': convert a floating point depth image to a straight 0-255 encoding of depths less than 3m
#                            'depth_encoded_rgb': convert a floating point depth image to the rgb encoding used by
#                                the google brain robot data grasp dataset's raw png depth image encoding,
#                                see https://sites.google.com/site/brainrobotdata/home/depth-image-encoding.
#                            'vrep': add a vrep prefix to any of the above commands to
#                                rotate image by 180 degrees, flip left over right, then invert the color channels
#                                after the initial conversion step.
#                                This is due to a problem where V-REP seems to display images differently.
#                                Examples include 'vrep_depth_rgb' and 'vrep_depth_encoded_rgb',
#                                see http://www.forum.coppeliarobotics.com/viewtopic.php?f=9&t=737&p=27805#p27805.
#                        """)
# tf.flags.DEFINE_string('vrepVisualizeRGBFormat', 'vrep_rgb',
#                        """  Controls how images are displayed. Options are:
#                         None: Do not modify the data and display it as-is for rgb input data (not working properly for float depth).
#                         'depth_rgb': convert a floating point depth image to a straight 0-255 encoding of depths less than 3m
#                         'depth_encoded_rgb': convert a floating point depth image to the rgb encoding used by
#                             the google brain robot data grasp dataset's raw png depth image encoding,
#                             see https://sites.google.com/site/brainrobotdata/home/depth-image-encoding.
#                         'vrep': add a vrep prefix to any of the above commands to
#                             rotate image by 180 degrees, flip left over right, then invert the color channels
#                             after the initial conversion step.
#                             This is due to a problem where V-REP seems to display images differently.
#                             Examples include 'vrep_depth_rgb' and 'vrep_depth_encoded_rgb',
#                             see http://www.forum.coppeliarobotics.com/viewtopic.php?f=9&t=737&p=27805#p27805.
#                        """)
tf.flags.DEFINE_string('vrepVisualizationPipeline', 'tensorflow',
                       """Options are: python, tensorflow.
                           'tensorflow' tensorflow loads the raw data from the dataset and
                               calculates all features before they are rendered with vrep via python,
                           'python' loads the raw data from tensorflow,
                               then the visualize_python function calculates the features
                               before they are rendered with vrep.
                       """)
tf.flags.DEFINE_boolean('vrepVisualizePredictions', False,
                        """Visualize the predictions of weights defined in grasp_train.py,
                           If loss is pixel-wise, prediction will be 2d image of probabilities.
                           Otherwise it's boolean indicate success or failure.

                           Model weights must be available to load, and the weights must match
                           the model being created for this to work correctly.
                           If you only wish to visualize the data, set this to False.
                        """
                        )
tf.flags.DEFINE_boolean('vrepVisualizeMatPlotLib', True,
                        """Visualize the predictions with a matplotlib heat map.
                        """
                        )

# the following line is needed for tf versions before 1.5
# flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS


class VREPGraspVisualization(object):
    """ Visualize the google brain robot data grasp dataset in the V-REP robot simulator.
    """

    def __init__(self):
        """Start the connection to the remote V-REP simulation

           Once initialized, call visualize().
        """
        print('VREPGraspVisualization: Object started, attempting to connect to V-REP')
        # just in case, close all opened connections
        vrep.vrep.simxFinish(-1)
        # Connect to V-REP
        self.client_id = vrep.vrep.simxStart(
            FLAGS.vrepConnectionAddress,
            FLAGS.vrepConnectionPort,
            FLAGS.vrepWaitUntilConnected,
            FLAGS.vrepDoNotReconnectOnceDisconnected,
            FLAGS.vrepTimeOutInMs,
            FLAGS.vrepCommThreadCycleInMs)

        self.dataset = FLAGS.grasp_dataset
        self.parent_name = FLAGS.vrepParentName
        self.visualization_pipeline = FLAGS.vrepVisualizationPipeline
        self.visualization_dir = FLAGS.visualization_dir

        if self.client_id != -1:
            print('VREPGraspVisualization: Connected to remote API server')
        else:
            print('VREPGraspVisualization: Error connecting to remote API server')
        return

    def visualize(self, tf_session=None, dataset=None, batch_size=1, parent_name=None,
                  visualization_pipeline=None,
                  visualization_dir=None):
        """ Perform dataset visualization on the selected data pipeline
        """

        if dataset is not None:
            self.dataset = dataset
        if parent_name is not None:
            self.parent_name = parent_name
        if visualization_pipeline is not None:
            self.visualization_pipeline = visualization_pipeline
        if visualization_dir is not None:
            self.visualization_dir = visualization_dir

        if self.visualization_pipeline == 'python':
            self.visualize_python(tf_session, self.dataset, batch_size, self.parent_name)
        elif self.visualization_pipeline == 'tensorflow':
            self.visualize_tensorflow(tf_session, self.dataset, batch_size, self.parent_name)
        else:
            raise ValueError('VREPGraspVisualization.visualize(): unsupported vrepVisualizationPipeline: ' + str(self.visualization_pipeline))

    def visualize_tensorflow(self, tf_session=None, dataset=None, batch_size=1, parent_name=None,
                             visualization_pipeline=None,
                             visualization_dir=None, verbose=0):
        """Visualize one dataset in V-REP from performing all preprocessing in tensorflow.

            tensorflow loads the raw data from the dataset and also calculates all
            features before they are rendered with vrep via python,
        """

        if dataset is not None:
            self.dataset = dataset
        if parent_name is not None:
            self.parent_name = parent_name
        if visualization_pipeline is not None:
            self.visualization_pipeline = visualization_pipeline
        if visualization_dir is not None:
            self.visualization_dir = visualization_dir

        raise NotImplementedError

    def visualize_python(self, tf_session=None, dataset=None, batch_size=1, parent_name=None,
                         visualization_pipeline=None,
                         visualization_dir=None):
        """Visualize one dataset in V-REP from raw dataset features, performing all preprocessing manually in this function.
        """

        if dataset is not None:
            self.dataset = dataset
        if parent_name is not None:
            self.parent_name = parent_name
        if visualization_pipeline is not None:
            self.visualization_pipeline = visualization_pipeline
        if visualization_dir is not None:
            self.visualization_dir = visualization_dir

        raise NotImplementedError

    def __del__(self):
        vrep.vrep.simxFinish(-1)


def vrep_grasp_main(_):
    with tf.Session() as sess:
        viz = VREPGraspVisualization()
        viz.visualize(sess)

if __name__ == '__main__':
    tf.app.run(main=vrep_grasp_main)

