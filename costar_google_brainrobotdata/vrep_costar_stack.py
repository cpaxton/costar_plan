# -*- coding: utf-8 -*-
"""Code for visualizing data from the costar stacking dataset in a web browser and V-REP.

https://sites.google.com/site/brainrobotdata/home/grasping-dataset

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0

The bokeh and HoloViews libraries are utilized to display the data
in the browser and show the controls, while V-REP is used for 3-D
display.

The app can be served using:

    bokeh serve --show vrep_costar_stack.py --args --data_dir /Users/athundt/.keras/datasets/costar_block_stacking_dataset_v0.4/2018-05-23-20-46-09_example000001.success.h5f
    bokeh serve --show vrep_costar_stack.py --args  --data-dir "~/.keras/datasets/costar_block_stacking_dataset_v0.4/*success.h5f"

Note that the data dir with * uses glob syntax, so this example will only load the data which has been labeled as successful grasps.
"""

try:
    # don't require tensorflow for reading, it only speeds things up
    import tensorflow as tf
    tf.enable_eager_execution()
except ImportError:
    tf = None

import vrep_grasp
import os
import errno
import traceback
import sys

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
    raise e

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
from depth_image_encoding import ClipFloatValues
from depth_image_encoding import FloatArrayToRgbImage
from depth_image_encoding import FloatArrayToRawRGB
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import img_as_uint
from skimage.color import grey2rgb
import json

try:
    import eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
    import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
except ImportError:
    print('eigen and sva python modules are not available. To install run the script at:'
          'https://github.com/ahundt/robotics_setup/blob/master/robotics_tasks.sh'
          'or follow the instructions at https://github.com/jrl-umi3218/Eigen3ToPython'
          'and https://github.com/jrl-umi3218/SpaceVecAlg. '
          'When you build the modules make sure python bindings are enabled.')

import numpy as np
import holoviews as hv
import os
import glob

import h5py
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider
from bokeh.models import Button
from bokeh.models.widgets import TextInput

import numpy as np
import io
from PIL import Image
import argparse
from functools import partial

try:
    # don't require vrep, only use it if it is available
    import vrep
except ImportError:
    vrep = None

from vrep_grasp import VREPGraspVisualization


# flags.DEFINE_string('data_dir',
#                     os.path.join(os.path.expanduser("~"),
#                                  '.keras', 'datasets', 'costar_block_stacking_dataset_v0.4'),
#                     """Path to directory containing the dataset.""")

# the following line is needed for tf versions before 1.5
# flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS
FLAGS(sys.argv)


class VREPCostarStackingVisualization(VREPGraspVisualization):
    """ Visualize the google brain robot data grasp dataset in the V-REP robot simulator.
    """

    def __init__(self):
        """Start the connection to the remote V-REP simulation

           Once initialized, call visualize().
        """
        super(VREPCostarStackingVisualization, self).__init__()

    def visualize_tensorflow(self, tf_session, dataset=FLAGS.grasp_dataset, batch_size=1, parent_name=FLAGS.vrepParentName,
                             visualization_dir=FLAGS.visualization_dir, verbose=0):
        """Visualize one dataset in V-REP from performing all preprocessing in tensorflow.

            tensorflow loads the raw data from the dataset and also calculates all
            features before they are rendered with vrep via python,
        """

    def visualize_python(self, tf_session=None, dataset=FLAGS.grasp_dataset, batch_size=1, parent_name=FLAGS.vrepParentName,
                         visualization_dir=FLAGS.visualization_dir):
        """Visualize one dataset in V-REP from raw dataset features, performing all preprocessing manually in this function.
        """
        # Visualize clear view point cloud
        # if FLAGS.vrepVisualizeRGBD:

        #     vrep.visualization.create_point_cloud(
        #         self.client_id, 'clear_view_cloud',
        #         depth_image=np.copy(clear_frame_depth_image),
        #         camera_intrinsics_matrix=camera_intrinsics_matrix,
        #         transform=base_to_camera_vec_quat_7,
        #         color_image=clear_frame_rgb_image, parent_handle=parent_handle,
        #         rgb_sensor_display_name='kcam_rgb_clear_view',
        #         depth_sensor_display_name='kcam_depth_clear_view')

    def show_step(self, data, numpy_data, step, prefix='', parent_name=None):
        if parent_name is None:
            parent_name = FLAGS.vrepParentName

        error_code, parent_handle = vrep.vrep.simxGetObjectHandle(self.client_id, parent_name, vrep.vrep.simx_opmode_blocking)
        if error_code is -1:
            parent_handle = -1
            print('could not find object with the specified name, so putting objects in world frame:', parent_name)

        single_frame_json_str = str(data["all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json"][step])
        frames = json.loads(single_frame_json_str)
        for name, frame in six.iteritems(frames):
            display_name = prefix + name.replace('/', '_')
            print('creating dummmy: ' + str(display_name) + ' for frame: ' + name)
            frame = np.array(frame)
            vrep.visualization.create_dummy(
                self.client_id,
                display_name=display_name,
                transform=frame,
                parent_handle=parent_handle
            )
            print('past create_dummy')



def vrep_grasp_main(_):
    with tf.Session() as sess:
        viz = VREPCostarStackingVisualization()
        viz.visualize(sess)


# parser = argparse.ArgumentParser(description='Process additional parameters for stack player')

# parser.add_argument('--data-dir', type=str, action='store', default='~/.keras/datasets/costar_block_stacking_dataset_v0.4',
#                     help='directory path containing the data')

# args = parser.parse_args()


def GetJpeg(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(np.uint8(img))
    output = io.BytesIO()
    im.save(output, format="JPEG", quality=80)
    return output.getvalue()


def GetPng(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(img)
    output = io.BytesIO()
    # enabling optimized file size
    # increases saving time to ~0.4 seconds per image.
    #im.save(output, format="PNG", optimize=True)
    im.save(output, format="PNG")
    return output.getvalue()

def JpegToNumpy(jpeg):
    if tf is not None:
        image = tf.image.decode_jpeg(jpeg)
    else:
        stream = io.BytesIO(jpeg)
        image = Image.open(stream)
    return np.asarray(image, dtype=np.uint8)

def ConvertImageListToNumpy(data, format='numpy', data_format='NHWC', dtype=np.uint8):
    """ Convert a list of binary jpeg or png files to numpy format.

    # Arguments

    data: a list of binary jpeg images to convert
    format: default 'numpy' returns a 4d numpy array,
        'list' returns a list of 3d numpy arrays
    """
    length = len(data)
    images = []
    for raw in data:
        img = JpegToNumpy(raw)
        if data_format == 'NCHW':
            img = np.transpose(img, [2, 0, 1])
        images.append(img)
    if format == 'numpy':
        images = np.array(images, dtype=dtype)
    return images

def generate_holo_map(rgb_images, height, width):
    frame_map = {}
    for i, image in enumerate(rgb_images):

        # print('image type: ' + str(type(image)))
        hv_rgb = hv.RGB(np.array(image))
        shape = image.shape
        frame_map[i] = hv_rgb
    holomap = hv.HoloMap(frame_map)
    holomap = holomap.options(width=int(width), height=int(height))
    return holomap

def load_example(file_path):
    """ Update the example, loading images and other data
    """
    data = h5py.File(file_path, 'r')
    rgb_images = list(data['image'])
    depth_images = list(data['depth_image'])
    frame_indices = np.arange(len(rgb_images))
    gripper_status = list(data['gripper'])
    action_status = list(data['label'])
    gripper_action_goal_idx = []

    # print("gripper ",gripper_status)
    # print("frames ", frame_indices)

    # generate new action labels and goal action indices
    gripper_action_label, gripper_action_goal_idx = generate_gripper_action_label(data, action_status, gripper_status, gripper_action_goal_idx)

    rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='list')
    depth_images = ConvertImageListToNumpy(np.squeeze(depth_images), format='list')
    numpy_data = {
        'rgb_images': rgb_images,
        'depth_images': depth_images,
        'frame_indices': frame_indices,
        'gripper_status': gripper_status,
        'action_status': action_status,
        'gripper_action_label': gripper_action_label,
        'gripper_action_goal_idx': gripper_action_goal_idx
    }

    return data, numpy_data


def generate_gripper_action_label(data, action_status, gripper_status, gripper_action_goal_idx):
    """ generate new action labels and goal action indices based on the gripper open/closed state
    """
    if "gripper_action_label" in list(data.keys()):
        # load the gripper action label from the hdf5 file
        gripper_action_label = list(data['gripper_action_label'])
        gripper_action_goal_idx = list(data['gripper_action_goal_idx'])
        print(gripper_action_goal_idx)
        print("gripper_action_labels already exist..")
    else:
        # compute the gripper action label on the fly
        unique_actions, indices = np.unique(action_status, return_index=True)
        unique_actions = [action_status[index] for index in sorted(indices)]
        action_ind = 0
        gripper_action_label = action_status[:]
        for i in range(len(gripper_status)):
            if (gripper_status[i] > 0.1 and gripper_status[i-1] < 0.1) or (gripper_status[i] < 0.5 and gripper_status[i-1] > 0.5):
                action_ind += 1
                print(i)
                gripper_action_goal_idx.append(i)
            if len(unique_actions) <= action_ind or len(gripper_action_label) <= i:
                break
            else:
                gripper_action_label[i] = unique_actions[action_ind]
    return gripper_action_label, gripper_action_goal_idx

def load_data_plot(renderer, frame_indices, gripper_status, action_status, gripper_action_label, height, width):
    # load the gripper data
    gripper_data = hv.Table({'Gripper': gripper_status, 'Frame': frame_indices},
                            ['Gripper', 'Frame'])
    gripper_curves = gripper_data.to.curve('Frame', 'Gripper')
    gripper_curves = gripper_curves.options(width=width, height=height//4)
    gripper_plot = renderer.get_plot(gripper_curves)

    # load the action data
    action_data = hv.Table({'Action': action_status, 'Frame': frame_indices},
                           ['Action', 'Frame'])
    action_curves = action_data.to.curve('Frame', 'Action')
    action_curves = action_curves.options(width=width, height=height//4)
    action_plot = renderer.get_plot(action_curves)

    # load the gripper action label

    gripper_action_data = hv.Table({'Gripper Action': gripper_action_label, 'Frame': frame_indices},
                           ['Gripper Action', 'Frame'])
    gripper_action_curves = gripper_action_data.to.curve('Frame', 'Gripper Action')
    gripper_action_curves = gripper_action_curves.options(width=width, height=height//4)
    gripper_action_plot = renderer.get_plot(gripper_action_curves)

    return gripper_plot, action_plot, gripper_action_plot


def check_errors(file_list, index, action='next'):
    """ Checks the file for valid data and returns the index of the file to be read.

    # Arguments

    file_list: a list of file names in the directory
    index: index of the file to check
    action: action to identify the button task
    """
    file_list_copy = file_list[:]
    index_copy = index
    print(file_list[index])
    flag = 0
    while flag == 0:
        with h5py.File(file_list[index], 'r') as data:
            if len(list(data['q'])) == 0:
                print("-------File Empty------")
                if len(file_name_list) > 1:
                    if action == 'next':
                        index = (index + 1) % len(file_list_copy)
                    else:
                        index = (index - 1) % len(file_list_copy)
                else:
                    print("Closing...")
                    exit()
            else:
                flag = 1
    return index

########################################################################################################################################################
## Start the bokeh rendering script portion
renderer = hv.renderer('bokeh')

#example_filename = "C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.4/2018-05-23-20-18-25_example000002.success.h5f"
#file_name_list = glob.glob("C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.4/*success.h5f")

FLAGS(sys.argv)
if vrep is not None:
    vrep_viz = VREPCostarStackingVisualization()
else:
    vrep_viz = None

data_dir = os.path.expanduser(FLAGS.data_dir)
if os.path.isdir(data_dir):
    data_dir = os.path.join(data_dir, '*.h5f')
print('Loading data_dir: ' + str(data_dir))
file_name_list = glob.glob(data_dir)
index = 0

index = check_errors(file_name_list, index)

data, numpy_data = load_example(file_name_list[index])
rgb_images = numpy_data['rgb_images']
frame_indices = numpy_data['frame_indices']
gripper_status = numpy_data['gripper_status']
action_status = numpy_data['action_status']
gripper_action_label = numpy_data['gripper_action_label']
gripper_action_goal_idx = numpy_data['gripper_action_goal_idx']
print('images loaded')
# Declare the HoloViews object
start = 0
end = len(rgb_images)
# TODO(ahundt) resize image, all of this size code had no effect
width = int(640*1.5)
height = int(480*1.5)
if end > 0:
    height = int(rgb_images[0].shape[0])
    width = int(rgb_images[0].shape[1])
print('width: ' + str(width) + ' height: ' + str(height))
# hv.opts(plot_width=width, plot_height=height)

holomap = generate_holo_map(rgb_images, height, width)

# Convert the HoloViews object into a plot
plot = renderer.get_plot(holomap)
# bokeh.plotting.curplot().plot_width=800
# plot.plot_width=width

print('holomap loaded')

gripper_plot, action_plot, gripper_action_plot = load_data_plot(renderer, frame_indices, gripper_status, action_status, gripper_action_label, height, width)

def animate_update():
    year = slider.value + 1
    if year > end:
        year = start
    slider.value = year

def slider_update(attrname, old, new):
    global data, numpy_data, vrep_viz
    plot.update(slider.value)
    if vrep_viz is not None:
        print('vrep_viz update slider value: ' + str(slider.value))
        vrep_viz.show_step(data, numpy_data, slider.value)


slider = Slider(start=start, end=end, value=0, step=1, title="Frame", width=width)
slider.on_change('value', slider_update)

def animate():
    if button.label == ' Play':
        button.label = ' Pause'
        curdoc().add_periodic_callback(animate_update, 10)
    else:
        button.label = ' Play'
        curdoc().remove_periodic_callback(animate_update)

def next_example(files, action):
    """ load the next example in the dataset
    """
    global file_textbox, button, button_next, button_prev, index, vrep_viz, data, numpy_data
    print("next clicked")
    file_textbox.value = "Processing..."
    renderer = hv.renderer('bokeh')
    if action == 'next':
        index = (index + 1) % len(files)
    else:
        index = (index - 1) % len(files)
    #print("it ", iterator)
    print("index before check", index)
    index = check_errors(files, index, action)
    print("index after check", index)
    print("len", len(files))

    file_name = files[index]
    data, numpy_data = load_example(file_name_list[index])
    rgb_images = numpy_data['rgb_images']
    frame_indices = numpy_data['frame_indices']
    gripper_status = numpy_data['gripper_status']
    action_status = numpy_data['action_status']
    gripper_action_label = numpy_data['gripper_action_label']
    gripper_action_goal_idx = numpy_data['gripper_action_goal_idx']
    print("image loaded")
    print("action goal idx", gripper_action_goal_idx)
    height = int(rgb_images[0].shape[0])
    width = int(rgb_images[0].shape[1])
    start = 0
    end = len(rgb_images)
    print(end)

    def slider_update(attrname, old, new):
        plot.update(slider.value)

    slider = Slider(start=start, end=end, value=0, step=1, title="Frame", width=width)
    slider.on_change('value', slider_update)

    holomap = generate_holo_map(rgb_images, height, width)
    print("generated holomap")
    plot = renderer.get_plot(holomap)
    print("plot rendered")
    gripper_plot, action_plot, gripper_action_plot = load_data_plot(renderer, frame_indices, gripper_status, action_status, gripper_action_label, height, width)
    print("plot loaded..")
    plot_list = [[plot.state], [gripper_plot.state], [action_plot.state]]

    widget_list = [[slider, button, button_prev, button_next], [file_textbox]]

    # "gripper_action" plot, labels based on the gripper opening and closing
    plot_list.append([gripper_action_plot.state])
    layout_child = layout(plot_list + widget_list, sizing_mode='fixed')
    curdoc().clear()
    file_textbox.value = file_name.split("\\")[-1]
    #curdoc().remove_root(layout_child)
    #layout_root.children[0] = layout_child
    curdoc().add_root(layout_child)

#iterator = iter(file_name_list)

button = Button(label=' Play', width=60)
button.on_click(animate)

button_next = Button(label='Next', width=60)
button_next.on_click(partial(next_example, files=file_name_list, action='next'))
button_prev = Button(label='Prev', width=60)
button_prev.on_click(partial(next_example, files=file_name_list, action='prev'))

# https://bokeh.pydata.org/en/latest/docs/reference/models/widgets.inputs.html
# TODO(ahundt) switch to AutocompleteInput with list of files
file_textbox = TextInput(value=file_name_list[index].split('\\')[-1], width=width)


# TODO(ahundt) load another file when it changes
# def textbox_update(attrname, old, new):
#     plot.update(slider.value)

# file_textbox.on_change(textbox_update)

# Combine the bokeh plot on plot.state with the widgets
plot_list = [
    [plot.state],
    [gripper_plot.state],
    [action_plot.state]]

widget_list = [[slider, button, button_prev, button_next], [file_textbox]]

# "gripper_action" plot, labels based on the gripper opening and closing
plot_list.append([gripper_action_plot.state])

layout_root = layout(plot_list+widget_list, sizing_mode='fixed')

#print(layout)



curdoc().add_root(layout_root)
if __name__ == '__main__':
    tf.app.run(main=vrep_grasp_main)

