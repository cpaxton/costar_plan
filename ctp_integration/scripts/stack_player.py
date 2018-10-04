# -*- coding: utf-8 -*-
"""
A simple player widget for visualizing an example in the dataset.

Start in the costar_plan directory:

    ~/src/costar_plan/ctp_integration

The app can be started (served) using the following line, it will use glob to find matching files:

    bokeh serve --show scripts/stack_player.py --args  --data-dir "~/.keras/datasets/costar_block_stacking_dataset_v0.4/*success.h5f"

Alternate command:

    bokeh serve --show scripts/stack_player.py --args --data-dir ~/.keras/datasets/costar_block_stacking_dataset_v0.4/

Note that the data dir with * uses glob syntax, so this example will only load the data which has been labeled as successful grasps.
"""
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
    # don't require tensorflow for reading, it only speeds things up
    import tensorflow as tf
except ImportError:
    tf = None

try:
    # don't require vrep, only use it if it is available
    import vrep
except ImportError:
    vrep = None


parser = argparse.ArgumentParser(description='Process additional parameters for stack player')

parser.add_argument('--data-dir', type=str, action='store', default='~/.keras/datasets/costar_block_stacking_dataset_v0.4',
                    help='directory path containing the data')

args = parser.parse_args()


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

def process_image(file_path):
    """ Update the example, loading images and other data
    """
    data = h5py.File(file_path, 'r')
    rgb_images = list(data['image'])
    frame_indices = np.arange(len(rgb_images))
    gripper_status = list(data['gripper'])
    action_status = list(data['label'])
    gripper_action_goal_idx = []

    # print("gripper ",gripper_status)
    # print("frames ", frame_indices)

    # generate new action labels and goal action indices
    gripper_action_label, gripper_action_goal_idx = generate_gripper_action_label(data, action_status, gripper_status, gripper_action_goal_idx)

    rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='list')
    return rgb_images, frame_indices, gripper_status, action_status, gripper_action_label, gripper_action_goal_idx


def generate_gripper_action_label(data, action_status, gripper_status, gripper_action_goal_idx):
    """ generate new action labels and goal action indices based on the gripper open/closed state
    """
    if "gripper_action_label" in list(data.keys()):
        # load the gripper action label from the hdf5 file
        gripper_action_label = list(data['gripper_action_label'])
        gripper_action_goal_idx = list(data['gripper_action_goal_idx'])
        print(gripper_action_goal_idx)
        print("gripper_action_labels already exist..")
        print("gripper_action_label length: " + str(len(gripper_action_label)))
        print("gripper_action_goal_idx length: " + str(len(gripper_action_goal_idx)))
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
    if not file_list:
        raise ValueError('List of files to load is empty! Quitting')
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

if tf is not None:
    tf.enable_eager_execution()
renderer = hv.renderer('bokeh')

#example_filename = "C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.4/2018-05-23-20-18-25_example000002.success.h5f"
#file_name_list = glob.glob("C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.4/*success.h5f")

path = os.path.expanduser(args.data_dir)

if '.h5f' in path:
    filenames = glob.glob(args.data_dir)
else:
    filenames = os.listdir(path)
    # use the full path name
    filenames = [os.path.join(path, filename) for filename in filenames]

# filter out files that aren't .h5f files
ignored_files = [filename for filename in filenames if '.h5f' not in filename]
filenames = [filename for filename in filenames if '.h5f' in filename]

# Report ignored files to the user
if ignored_files:
    print('Ignoring the following files which do not contain ".h5f": \n\n' + str(ignored_files) + '\n\n')

file_name_list = filenames

index = 0

index = check_errors(file_name_list, index)

rgb_images, frame_indices, gripper_status, action_status, gripper_action_label, gripper_action_goal_idx = process_image(file_name_list[index])
print('images loaded')
# Declare the HoloViews object
start = 0
end = len(rgb_images) - 1
print(' End Index of RGB images: ' + str(end))
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
    plot.update(slider.value)

slider = Slider(start=start, end=end, value=0, step=1, title="Frame", width=width)
slider.on_change('value', slider_update)

def animate():
    if button.label == ' Play':
        button.label = ' Pause'
        curdoc().add_periodic_callback(animate_update, 10)
    else:
        button.label = ' Play'
        curdoc().remove_periodic_callback(animate_update)

def next_image(files, action):
    global file_textbox, button, button_next, button_prev, index
    print("next clicked")
    file_textbox.value = "Processing..."
    renderer = hv.renderer('bokeh')
    if action == 'next':
        index=(index + 1) % len(files)
    else:
        index=(index - 1) % len(files)
    #print("it ", iterator)
    print("index before check",index)
    index = check_errors(files, index, action)
    print("index after check", index)
    print("len", len(files))

    file_name = files[index]
    rgb_images, frame_indices, gripper_status, action_status, gripper_action_label, gripper_action_goal_idx = process_image(file_name)
    print("image loaded")
    print("action goal idx", gripper_action_goal_idx)
    height = int(rgb_images[0].shape[0])
    width = int(rgb_images[0].shape[1])
    start = 0
    end = len(rgb_images) - 1
    print(' End Index of RGB images: ' + str(end))

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
button_next.on_click(partial(next_image, files=file_name_list, action='next'))
button_prev = Button(label='Prev', width=60)
button_prev.on_click(partial(next_image, files=file_name_list, action='prev'))

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
