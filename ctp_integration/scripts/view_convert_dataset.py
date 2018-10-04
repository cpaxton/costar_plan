#!/usr/bin/env python

'''
View and Convert dataset lets you look at the video data in a costar stacking dataset.

show a video from an h5f file:

    python view_convert_dataset.py --path <path/to/data/folder/or/file> --preview True

Convert video from an h5f file into a gif:

    python view_convert_dataset --path <path/to/data/folder/or/file> --preview --convert gif

Preprocess data that was just collected to include action labels
'gripper_action_goal_idx' and 'gripper_action' based on when the gripper moves:

    export CUDA_VISIBLE_DEVICES="" && python2 ctp_integration/scripts/view_convert_dataset.py --path "~/.keras/datasets/costar_plush_block_stacking_dataset_v0.1/" --preprocess_inplace gripper_action --write

Relabel "success" data in a dataset:

    python2 ctp_integration/scripts/view_convert_dataset.py --path ~/.keras/datasets/costar_block_stacking_dataset_v0.4 --label_correction --fps 60 --ignore_failure True --ignore_error True

'''
import argparse
import os
import sys
import traceback
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
# https://github.com/tanyaschlusser/array2gif
# from array2gif import write_gif

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

# from costar_models.datasets.image import JpegToNumpy
# from costar_models.datasets.image import ConvertImageListToNumpy
import pygame
import io
from PIL import Image
import moviepy
import moviepy.editor as mpye
# import skimage
try:
    # don't require tensorflow for viewing
    import tensorflow as tf
except ImportError:
    tf = None


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
    images = []
    for raw in data:
        img = JpegToNumpy(raw)
        if data_format == 'NCHW':
            img = np.transpose(img, [2, 0, 1])
        images.append(img)
    if format == 'numpy':
        images = np.array(images, dtype=dtype)
    return images


def npy_to_video(npy, filename, fps=10, preview=True, convert='gif'):
    """Convert a numpy array into a gif file at the location specified by filename.

        # Arguments

        convert: Default empty string is no conversion, options are gif and mp4.
        preview: pop open a preview window to view the video data.
    """
    # Useful moviepy instructions https://github.com/Zulko/moviepy/issues/159
    # TODO(ahundt) currently importing moviepy prevents python from exiting. Once this is resolved remove the import below.
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(npy), fps)
    if preview:
        # https://stackoverflow.com/a/41771413
        clip.preview()
    if convert == 'gif':
        clip.write_gif(filename)
    elif convert:
        clip.write_videofile(filename)


def _parse_args():
    """Read the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=os.path.join(os.path.expanduser("~"), '.costar', 'data'),
                        help='path to dataset h5f file or folder containing many files')
    parser.add_argument("--convert", type=str, default='',
                        help='format to convert images to. Default empty string is no conversion, options are gif and mp4.')
    parser.add_argument("--success_only", action='store_true', default=False, help='only visit stacking data labeled as successful')
    parser.add_argument("--ignore_failure", action='store_true', default=False, help='skip grasp failure cases')
    parser.add_argument("--ignore_success", action='store_true', default=False, help='skip grasp success cases')
    parser.add_argument("--extra_cool_example", action='store_true', default=False,
                        help='The human labeled label_correction_csv can store notes identifying particularly '
                             'interesting examples with the string super_cool_example. With this flag we will '
                             ' skip all except the examples which are particularly interesting')
    parser.add_argument("--ignore_error", action='store_true', default=False, help='skip grasp attempts that are both failures and contain errors')
    parser.add_argument("--preview", action='store_true', help='pop open a preview window to view the video data')
    parser.add_argument("--preview_initial_frame", type=int, default=0,
                        help='initial frame to view in video, negative numbers are the distance in frames from the final frame.')
    parser.add_argument("--preview_final_frame", type=int, default=-1,
                        help='final frame to view in video, negative numbers are the distance in frames from the final frame.')
    parser.add_argument("--label_correction", action='store_true',
                        help="""Change dataset filenames which have incorrect success or failure labels.

                                (1) You will be shown the last few frames for each file, just
                                    press 1 for a success label, 2 for a failure label.
                                (2) A csv defined by --label_correction_csv will be created
                                    and re-saved with each new human label update.
                                    If the csv already exists, that file will be loaded and edited.

                                The csv will be formatted with the following columns:

                                    original_filename, corrected_filename, human_labeling_status

                                original_filename: Filename of the original example data file.
                                corrected_filename: Either the same as the original filename, or defines the new corrected filename.
                                human_labeling_status: One of 'unconfirmed', 'confirmed', or an underscore separated string
                                   containing 'error' plus the type of error message. Example status values include
                                   'error_file_ioerror_encountered' and 'error_exception_encountered'. Have a look at the code
                                   for more details.

                                Once you have a complete label_correction_csv ready, running this program
                                with --write will load the csv file and immediately do an automatic
                                rename of all the files containing a 'confirmed' human_labeling_status
                                to the new corrected filenames.

                                Warning: Make sure you have a backup for any existing label correction CSV file,
                                because the csv may be modified even if you don't pass --write!
                             """)
    parser.add_argument("--label_correction_reconfirm", action='store_true',
                        help='Same as --label_correction, but we will reconfirm every single row with you.')
    parser.add_argument("--label_correction_csv", type=str, default='rename_dataset_labels.csv',
                        help='File from which to load & update the label correction csv file, expected to be in --path folder.')
    parser.add_argument("--label_correction_initial_frame", type=int, default=-3,
                        help='labinitial frame to view in video, negative numbers are the distance in frames from the final frame.')
    parser.add_argument("--label_correction_final_frame", type=int, default=-1,
                        help='final frame to view in video, negative numbers are the distance in frames from the final frame.')
    parser.add_argument("--goal_to_jpeg", action='store_true', default=False,
                        help='Convert the rgb images from each goal time step to a jpeg saved in a folder called '
                             'goal_images which will be created right next to the dataset files.')
    parser.add_argument("--gripper", action='store_true', default=False, help='print gripper data channel')
    parser.add_argument("--depth", action='store_true', default=True, help='process depth data', dest='depth')
    parser.add_argument("--no-depth", action='store_false', default=True, help='do not process depth data', dest='depth')
    parser.add_argument("--rgb", action='store_true', default=True, help='process rgb data')
    parser.add_argument("--no-rgb", action='store_false', default=True, help='do not process rgb data', dest='rgb')
    parser.add_argument("--fps", type=int, default=10, help='framerate to process images in frames per second')
    parser.add_argument("--matplotlib", type=bool, default=False,
                        help='preview data with matplotlib, slower but you can do pixel lookups')
    parser.add_argument("--print", type=str, default='',
                        help=('Comma separated list of data channels to convert to a list and print as a string.'
                              'Options include: label, gripper, pose, nsecs, secs, q, dq, labels_to_name, all_tf2_frames_as_yaml, '
                              'all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json, and more. See collector.py for more key strings.'))
    parser.add_argument('--preprocess_inplace', type=str, action='store', default='',
                        help="""Currently the only option is gripper_action, which generates new labels
                                gripper_action_label and gripper_action_goal_idx based on the timestep at which the gripper opened and closed,
                                and inserts them directly into the hdf5 file.
                             """)
    parser.add_argument("--write", action='store_true', help='Actually write out the changes specified in preprocess_inplace, or label_correction.')

    return vars(parser.parse_args())

def draw_matplotlib(depth_images, fps):
    for image in tqdm(depth_images):
        plt.imshow(image)
        plt.pause(1.0 / fps)
        plt.draw()


def wait_for_keypress_to_select_label(progress_bar):
    """
    # Returns

      description, comment.
    """
    progress_bar.write(
        "\nPress a key to label the file: 1. success, 2. failure, 4. skip, 5. Extra Cool Example, 6. Problem with this Example 0. whoops! make previous file unconfirmed \n"
        "What to look for:\n"
        " - A successful stack is 3 blocks tall or 4 blocks tall with the gripper completely removed from the field of view.\n"
        " - If the tower is 3 blocks tall and blocks will clearly slide off if not for the wall press 2 for 'failure',\n"
        "   if it is merely in contact with a wall, press 1 for 'success'."
        " - When the robot doesn't move but there is already a visible successful stack, that's an error.failure.falsely_appears_correct, so press 1 for 'success'!\n"
        " - If you can see the gripper, the example is a failure even if the stack is tall enough!\n")
    # , 3: error.failure
    flag = 0
    comment = 'none'
    mark_previous_unconfirmed = None
    while flag == 0:
        # pygame.event.pump()
        events = pygame.event.get()
        # if len(events) > 0:
        #     print(events)
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    progress_bar.write("label set to success")
                    flag = 1
                    return "success", comment, mark_previous_unconfirmed
                elif event.key == pygame.K_2:
                    progress_bar.write("label set to failure")
                    flag = 1
                    return "failure", comment, mark_previous_unconfirmed
                # elif event.key == pygame.K_3:
                #     progress_bar.write("label set to error.failure")
                #     flag = 1
                #     return "error.failure"
                elif event.key == pygame.K_4:
                    flag = 1
                    return 'skip', comment, mark_previous_unconfirmed
                elif event.key == pygame.K_5:
                    comment = 'extra_cool_example'
                    progress_bar.write('comment added: extra_cool_example (this note will remove past notes)')
                elif event.key == pygame.K_6:
                    comment = 'problem_with_example'
                    progress_bar.write('comment added: problem_with_example (this note will remove past notes)')
                elif event.key == pygame.K_0:
                    mark_previous_unconfirmed = True
                    progress_bar.write(
                        'Thanks for mentioning there is a problem with the selection for the previous example.'
                        'We will clear that data so the example will appear again when you re-run the label correction.'
                        'Jut to be extra safe, we also suggest you write down the exact filename '
                        'of the previous example so you can check it manually, ')


def save_label_correction_csv_file(label_correction_csv_path, label_correction_table):
    """ Save the file with all corrections to filenames.
    """
    with open(label_correction_csv_path, 'wb') as label_correction_csv_file:
        np.savetxt(label_correction_csv_file, label_correction_table, fmt='%s', delimiter=', ',
                   header='original_filename, corrected_filename, human_labeling_status, comment')


def main(args, root="root"):

    clip = None
    label_correction_table = None
    path = os.path.expanduser(args['path'])
    if '.h5f' in path:
        filenames = [args['path']]
    else:
        if os.path.isdir(path):
            filenames = os.listdir(path)
        else:
            filenames = glob.glob(path)

    # filter out files that aren't .h5f files
    ignored_files = [filename for filename in filenames if '.h5f' not in filename]
    filenames = [filename for filename in filenames if '.h5f' in filename]

    if args['label_correction'] or args['goal_to_jpeg']:
        # make sure they're sorted in sorted order
        # this is done in a weird way to ensure it matches
        # the label correction csv file
        filenames = np.expand_dims(np.array(filenames), -1)
        filenames = filenames[filenames[:, 0].argsort(kind='mergesort')]
        filenames = np.squeeze(filenames)

    # Read data
    progress_bar = tqdm(filenames)

    # Report ignored files to the user
    if ignored_files:
        progress_bar.write('Ignoring the following files which do not contain ".h5f": \n\n' + str(ignored_files) + '\n\n')

    # label_correction_reconfirm forces all label_correction steps
    if args['label_correction_reconfirm']:
        args['label_correction'] = True

    # if args['goal_to_jpeg']:
    #     label_correction_csv_path = os.path.join(path, args['label_correction_csv'])
    #     print(label_correction_csv_path)
    #     renamed_file_data = np.genfromtxt(label_correction_csv_path, dtype=str, delimiter=',')
    #     path_to_goal = os.path.join(path, 'goal_images')
    #     # print(renamed_file_data)
    #     file_list = []
    #     for i, file_attr_list in enumerate(renamed_file_data[1:]):
    #         if 'extra_cool_example' in file_attr_list[-1]:
    #             file_list.append(os.path.join(path, file_attr_list[0]))
    #     print(len(file_list))
    #     progress_bar = tqdm(file_list)

    if args['label_correction'] or args['goal_to_jpeg'] or args['extra_cool_example']:
        label_correction_csv_path = os.path.join(path, args['label_correction_csv'])
        if os.path.isfile(label_correction_csv_path):
            progress_bar.write('Loading existing label correction csv file:\n    ' + str(label_correction_csv_path))
            label_correction_table = np.genfromtxt(label_correction_csv_path, dtype='str', delimiter=', ')
        else:
            progress_bar.write('Creating new label correction csv file:\n    ' + str(label_correction_csv_path))
            label_correction_table = np.column_stack([filenames, filenames, ['unconfirmed'] * len(filenames), ['none'] * len(filenames)])
            save_label_correction_csv_file(label_correction_csv_path, label_correction_table)

        if label_correction_table.shape[0] != len(filenames):
            raise RuntimeError(
                'The code cannot currently handle mismatched lists of files, so we will exit the program.\n'
                'The the label correction table csv file:\n'
                '    ' + str(label_correction_csv_path) +
                '\n\nhas ' + str(label_correction_table.shape[0]) + ' rows, but there are ' + str(len(filenames)) +
                ' filenames containing .h5f in the list of filenames to be processed. '
                'To solve this problem you might want to try one of the following options:\n'
                '    (1) Start fresh with no CSV file and all unconfirmed values.\n'
                '    (2) Manually edit the csv or file directories so the number of rows matches the number of .h5f files.\n'
                '    (3) Edit the code to handle this situation by perhaps adding any missing files to the list and re-sorting.')
        # make sure they're sorted in the same order as filenames
        label_correction_table = label_correction_table[label_correction_table[:, 0].argsort(kind='mergesort')]
        save_label_correction_csv_file('sorted_label_correction_filenames.csv', filenames)
        save_label_correction_csv_file('sorted_label_correction.csv', label_correction_table)

    # keep track of the previous index that was not skipped when looping through the data
    previous_i_not_skipped = None
    current_i_not_skipping = None

    for i, filename in enumerate(progress_bar):
        # skip certain files based on command line parameters
        if filename.startswith('.') or '.h5' not in filename:
            continue
        if args['success_only'] and 'success' not in filename:
            progress_bar.write('Skipping example not labeled success: ' + filename)
            continue
        if args['ignore_error'] and 'error' in filename:
            progress_bar.write('Skipping example containing errors: ' + filename)
            continue
        if args['ignore_failure'] and 'failure' in filename:
            progress_bar.write('Skipping example containing failure: ' + filename)
            continue
        if args['ignore_success'] and 'success' in filename:
            progress_bar.write('Skipping example containing success: ' + filename)
            continue
        if args['extra_cool_example']:
            comment_idx = 3
            if 'extra_cool_example' not in label_correction_table[i, comment_idx]:
                progress_bar.write('Skipping since it is not an extra_cool_example: ' + filename)
                continue

        progress_bar.write('Current File: ' + filename)
        # keep track of which indices we actually visit and which we skip
        previous_i_not_skipped = current_i_not_skipping
        current_i_not_skipping = i

        if args['path'] not in filename:
            # prepend the path if it isn't already present
            example_filename = os.path.join(args['path'], filename)
        else:
            example_filename = filename

        if args['label_correction']:
            # Label correction needs some special data loading logic
            # so we can skip data that already has human confirmation
            original_idx = 0
            corrected_idx = 1
            status_idx = 2
            comment_idx = 3
            status_string = label_correction_table[i, status_idx]

            example_filename_folder, example_filename_base = os.path.split(example_filename)
            original = label_correction_table[i, original_idx]
            corrected = label_correction_table[i, corrected_idx]
            # check that we are working with the right file
            if example_filename_base not in label_correction_table[i, :status_idx]:
                raise ValueError(
                    '\n' + ('-' * 80) + '\n\n'
                    'Files may have been added and/or removed from the dataset folder '
                    'but not updated in the label correction csv file:\n'
                    '    ' + str(label_correction_csv_path) + '\n\n'
                    'The code cannot currently handle mismatched lists of files, so we will exit the program.\n\n'
                    'The current example filename:\n'
                    '    ' + str(example_filename) +
                    '\ndoes not match the corresponding original entry at row ' + str(i) +
                    ' in the label correction csv:\n'
                    '    ' + str(label_correction_table[i, :]) + '\n\n'
                    'To solve this problem you might want to try one of the following options:\n'
                    '    (1) Start fresh with no CSV file and all unconfirmed values.\n'
                    '    (2) Manually edit the csv or file directories so the number of rows matches the number of .h5f files.\n'
                    '    (3) Edit the code to handle this situation by perhaps adding any missing files to the list and re-sorting.')
            # next commented line is for debug
            # progress_bar.write('i: ' + str(i) + ' status: ' + status_string)
            if args['write']:
                if(example_filename_base == original and
                        'confirmed_rename' in status_string and
                        'unconfirmed' not in status_string):
                    progress_bar.write(
                        'Performing rename at row i: ' + str(i) + ' filename: ' + str(filename) +
                        ' table: ' + str(str(label_correction_table[i, :])))

                    if original == corrected:
                        raise ValueError(
                            'Rename confirmed but filenames are identical '
                            'please correct this entry. '
                            '[source, destination, status] entry at row ' + str(i) +
                            ' in the label correction csv:\n'
                            '    ' + str(label_correction_table[i, :]))

                    corrected = os.path.join(example_filename_folder, corrected)
                    original = os.path.join(example_filename_folder, original)

                    if os.path.isfile(corrected):
                        raise ValueError(
                            'Trying to rename a file, but the destination filename already exists!'
                            '[source, destination, status] entry at row ' + str(i) +
                            ' in the label correction csv:\n'
                            '    ' + str(label_correction_table[i, :]))

                    if not os.path.isfile(original):
                        raise ValueError(
                            'Trying to rename a file, but the original filename either '
                            'does not exist or is not a file!'
                            '[source, destination, status] entry at row ' + str(i) +
                            ' in the label correction csv:\n'
                            '    ' + str(label_correction_table[i, :]))

                    progress_bar.write(original + ' -> ' + corrected)
                    # we've ensured the user wants to write the new filenames,
                    # there was no error in the human labeling stage,
                    # this rename has been confirmed by a human,
                    # the source and destination filenames aren't equal,
                    # and the destination filename doesn't already exist.
                    # All looks good so let's finally rename it!
                    os.rename(original, corrected)
                    progress_bar.write('One rename completed!')
                # loading the data would take a long time,
                # plus the filename just changed so skip
                continue
            if status_string != 'unconfirmed' and not args['label_correction_reconfirm']:
                # loading the data would take a long time, so skip
                continue

        # We haven't run into any errors yet for this file,
        # if we do, this will identify files with errors
        # in logs and the label correction csv
        error_encountered = None

        description = "Processing %s" % example_filename
        progress_bar.set_description(description)
        pygame.display.set_caption(description)

        mode = 'r'
        # open the files in a writing mode
        if args['write']:
            mode = 'r+'

        # open the file
        try:
            with h5py.File(example_filename, mode) as data:
                fps = args['fps']
                # check if the data is there to load
                load_depth = args['depth'] and 'depth_image' in data and len(data['depth_image']) > 0
                load_rgb = args['rgb'] and 'image' in data and len(data['image']) > 0
                # print('load_depth: ' + str(load_depth) + ' load_rgb: ' + str(load_rgb))

                if args['gripper']:
                    # print the gripper data channel
                    progress_bar.write(filename + ': ' + str(list(data['gripper'])))

                if args['print']:
                    # comma separated keys from which to create a list and
                    # print string values from the h5f file
                    data_to_print = args['print'].split(',')
                    for data_str in data_to_print:
                        progress_bar.write(filename + ' ' + data_str + ': ' + str(list(data[data_str])))

                if args['preprocess_inplace'] == 'gripper_action':
                    if 'gripper' not in data or 'label' not in data:
                        progress_bar.write('Skipping file because the feature string '
                                           'gripper  and/or label is not present: ' +
                                           str(filename))
                        continue
                    # if 'label' in data:
                    #     progress_bar.write("frames ", len(list(data['label'])))
                    # generate new action labels based on when the gripper opens and closes
                    gripper_action_label, gripper_action_goal_idx = generate_gripper_action_label(data)
                    # add the new action label and goal indices based on when the gripper opens/closes
                    if args['write']:
                        # cannot write without deleting existing data
                        if "gripper_action_label" in list(data.keys()):
                                progress_bar.write('Deleting existing gripper action labels for file: ' + str(filename))
                                del data['gripper_action_label']
                                del data['gripper_action_goal_idx']

                        data['gripper_action_label'], data['gripper_action_goal_idx'] = np.array(gripper_action_label), np.array(gripper_action_goal_idx)
                        # progress_bar.write("data on file",list(data['gripper_action_goal_idx']))
                    else:
                        progress_bar.write(
                            'gripper_action_label test run, use --write to change the files in place. gripper_action_label: ' +
                            str(gripper_action_label) + ' gripper_action_goal_idx: ' + str(gripper_action_goal_idx))

                    # skip other steps like video viewing,
                    # so this conversion runs 1000x faster
                    continue

                if args['goal_to_jpeg']:
                    # Visit all the goal timesteps and write out a jpeg file in the 'goal_images' folder
                    progress_bar.write('-' * 80)
                    image_list = []
                    tiling_list = []
                    total_frames = len(data['image'])
                    if total_frames == 0:
                        progress_bar.write('Skipping file without image frames: ' + filename)
                        continue
                    goal_frames = np.unique(data['gripper_action_goal_idx'])
                    data_gripper_action_label = list(data['gripper_action_label'])
                    goal_label_idx = np.unique(data_gripper_action_label, return_index=True)[1]
                    goal_label_idx = [data_gripper_action_label[index] for index in sorted(goal_label_idx)]
                    progress_bar.write('goal_label_idx: ' + str(goal_label_idx))
                    data_labels_to_name = list(data['labels_to_name'])
                    goal_labels_name = np.array(data_labels_to_name)[goal_label_idx]
                    progress_bar.write('goal_labels_name: ' + str(goal_labels_name))
                    progress_bar.write("writing frames:" + str(goal_frames) + ' total frames: ' + str(total_frames))
                    if len(goal_frames) > 1:
                        image_list = np.array(data['image'])[goal_frames]
                    else:
                        progress_bar.write("WARNING: printing first and final frame, but 0 or 1 actual goal frames for: " + str(filename))
                        image_list = []
                    images = ConvertImageListToNumpy(image_list, format='list')
                    example_folder_path, name = os.path.split(example_filename)
                    progress_bar.write(example_folder_path)
                    name = name.replace('.h5f', '')
                    example_folder_path = os.path.join(example_folder_path, 'goal_images')
                    if not os.path.exists(example_folder_path):
                        os.makedirs(example_folder_path)
                    # extract the clear view image
                    image = ConvertImageListToNumpy(np.array(data['image'][0:1]), format='list')
                    im = Image.fromarray(image[0])
                    goal_image_path = os.path.join(example_folder_path, name + '_clear_view_' + "0" + '.jpg')
                    progress_bar.write('Saving jpeg: ' + str(goal_image_path))
                    im.save(goal_image_path)
                    tiling_list = image + images
                    # extract the final image
                    final_frame = total_frames - 1
                    image = ConvertImageListToNumpy(np.array(data['image'][-2:]), format='list')
                    im = Image.fromarray(image[0])
                    goal_image_path = os.path.join(example_folder_path, name + '_z_final_frame_' + str(final_frame) + '.jpg')
                    progress_bar.write('Saving jpeg: ' + str(goal_image_path))
                    im.save(goal_image_path)
                    if len(goal_frames) == 0 or (len(goal_frames) > 0 and final_frame != goal_frames[-1]):
                        # only append the final frame if it isn't already the last goal frame
                        tiling_list = tiling_list + image
                    # build up a tiled version of all the images and save that first
                    tiled_image = np.squeeze(np.hstack(tiling_list))
                    progress_bar.write('tiled_image shape 1: ' + str(tiled_image.shape))
                    im = Image.fromarray(tiled_image)
                    goal_image_path = os.path.join(example_folder_path, name + '_tiled.jpg')
                    progress_bar.write('Saving jpeg: ' + str(goal_image_path))
                    im.save(goal_image_path)
                    # save out each individual image
                    for i, image in enumerate(images):
                        # write out this specific image
                        im = Image.fromarray(image)
                        goal_label_name = 'unknown_label'
                        if i < len(goal_labels_name):
                            goal_label_name = str(goal_labels_name[i])

                        goal_image_path = os.path.join(example_folder_path, name + '_goal_frame_' + str(goal_frames[i]) + '_' + goal_label_name + '.jpg')
                        progress_bar.write('Saving jpeg: ' + str(goal_image_path))
                        im.save(goal_image_path)

                    # skip other steps like video viewing,
                    # so this conversion runs 1000x faster
                    continue

                # Video display and conversion
                try:
                    # define where to start and end reading images
                    start_frame = args['preview_initial_frame']
                    end_frame = args['preview_final_frame']
                    if args['label_correction']:
                        # only show the last few frames when correcting labels
                        start_frame = args['label_correction_initial_frame']
                        end_frame = args['label_correction_final_frame']
                        if args['write']:
                            # Don't show video to the user when writing.
                            load_depth = False
                            load_rgb = False
                        if len(data['image']) == 0:
                            clip = None
                            error_encountered = 'no_images'

                    if load_depth:
                        depth_images = list(data['depth_image'][start_frame:end_frame])
                        depth_images = ConvertImageListToNumpy(np.squeeze(depth_images), format='list')
                        if args['matplotlib']:
                            draw_matplotlib(depth_images, fps)
                        depth_clip = mpye.ImageSequenceClip(depth_images, fps=fps)
                        clip = depth_clip
                    if load_rgb:
                        rgb_images = list(data['image'][start_frame:end_frame])
                        rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='list')
                        if args['matplotlib']:
                            draw_matplotlib(rgb_images, fps)
                        rgb_clip = mpye.ImageSequenceClip(rgb_images, fps=fps)
                        clip = rgb_clip

                    if load_depth and load_rgb:
                        clip = mpye.clips_array([[rgb_clip, depth_clip]])

                    if (load_depth or load_rgb) and args['preview'] and not args['label_correction']:
                        clip.preview()

                    save_filename = example_filename.replace('.h5f', '.' + args['convert'])
                    if 'gif' in args['convert']:
                        clip.write_gif(save_filename, fps=fps)
                    elif args['convert']:
                        clip.write_videofile(save_filename, fps=fps)
                except KeyboardInterrupt as ex:
                    progress_bar.write('Keyboard interrupt detected. Exiting')
                    break
                except Exception as ex:
                    error_encountered = 'error_exception_encountered'
                    ex_type, ex2, tb = sys.exc_info()
                    traceback.print_tb(tb)
                    # deletion must be explicit to prevent leaks
                    # https://stackoverflow.com/a/16946886/99379
                    del tb
                    progress_bar.write(
                        'Warning: Skipping File. Exception encountered while processing ' + example_filename +
                        ' please edit the code of view_convert_dataset.py to debug the specifics: ' + str(ex))

        except IOError as ex:
            error_encountered = 'error_file_ioerror_encountered'
            ex_type, ex2, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
            progress_bar.write(
                'Error: Skipping file due to IO error when opening ' +
                example_filename + ': ' + str(ex))

        if args['label_correction']:
            label_correction_table = label_correction(
                label_correction_table, i, example_filename, args,
                progress_bar, label_correction_csv_path, error_encountered,
                clip, previous_i_not_skipped)

    if args['label_correction']:
        progress_bar.write('Run complete! Label correction csv:\n' + str(label_correction_csv_path))


def label_correction(
        label_correction_table, i, example_filename, args, progress_bar,
        label_correction_csv_path, error_encountered, clip, previous_i_not_skipped):
    original_idx = 0
    corrected_idx = 1
    status_idx = 2
    comment_idx = 3
    status_string = label_correction_table[i, status_idx]
    example_filename_base = os.path.basename(example_filename)
    # check that we are working with the right file
    if example_filename_base not in label_correction_table[i, :status_idx]:
        raise ValueError(
            '\n' + ('-' * 80) + '\n\n'
            'Files may have been added and/or removed from the dataset folder '
            'but not updated in the label correction csv file:\n'
            '    ' + str(label_correction_csv_path) + '\n\n'
            'The code cannot currently handle mismatched lists of files, so we will exit the program.\n\n'
            'The current example filename:\n'
            '    ' + str(example_filename) +
            '\ndoes not match the corresponding original entry at row ' + str(i) +
            ' in the label correction csv:\n'
            '    ' + str(label_correction_table[i, :]) + '\n\n'
            'To solve this problem you might want to try one of the following options:\n'
            '    (1) Start fresh with no CSV file and all unconfirmed values.\n'
            '    (2) Manually edit the csv or file directories so the number of rows matches the number of .h5f files.\n'
            '    (3) Edit the code to handle this situation by perhaps adding any missing files to the list and re-sorting.')

    if not args['write']:
        if error_encountered is None and (status_string == 'unconfirmed' or args['label_correction_reconfirm']):
            progress_bar.write('-' * 80)
            progress_bar.write('Current row ' + str(i) + ' [original, corrected, status, comment]:\n    ' + str(label_correction_table[i, :]) + '\n')
            # show the clip
            clip.preview()
            # Get the human corrected label
            label, comment, mark_previous_unconfirmed = wait_for_keypress_to_select_label(progress_bar)
            if mark_previous_unconfirmed and i > 0:
                if previous_i_not_skipped is not None:
                    # If some labels are skipped such as in success only,
                    # the previous label viewed by the user can be several indices back!
                    label_correction_table[previous_i_not_skipped, status_idx] = 'unconfirmed'
                else:
                    progress_bar.write(
                        'Sorry, we could not mark the previous value because none existed, '
                        'you should manually fix whatever problem you encountered.')
            original_filename = label_correction_table[i, original_idx]
            if label not in original_filename:
                if label == 'success':
                    if 'error' in original_filename:
                        # there was some sort of program/system error so the actions were not completed.
                        # However for the purpose of detecting if there is a stack 3 tall present, this would be marked correct.
                        # Therefore we have the special label error.failure.falsely_appears_correct!
                        # Sorry it is complicated but I want to be specific so I don't have to go through all the examples again...
                        # TODO(ahundt) BUG!!! filenames are becoming something like: 2018-05-10-14-33-26_example000019.error.failure.fal instead of 2018-05-10-14-33-26_example000019.error.failure.falsely_appears_correct.h5f
                        label_correction_table[i, corrected_idx] = original_filename.replace('failure', 'failure.falsely_appears_correct')
                    else:
                        # replace failure with success
                        label_correction_table[i, corrected_idx] = original_filename.replace('failure', label)
                    label_correction_table[i, status_idx] = 'confirmed_rename'
                elif label == 'failure':
                    label_correction_table[i, corrected_idx] = original_filename.replace('success', label)
                    label_correction_table[i, status_idx] = 'confirmed_rename'
                elif label == 'skip':
                    pass
                else:
                    raise ValueError(
                        'Unsupported label: ' + str(label) +
                        ' this should not happen so please consider having a look at the code.')
            else:
                label_correction_table[i, status_idx] = 'confirmed_no_change'

            label_correction_table[i, comment_idx] = comment
            # we've now confirmed this label with a human
            # save the updated csv file
            save_label_correction_csv_file(label_correction_csv_path, label_correction_table)
            progress_bar.write('Updated row ' + str(i) + ' [original, corrected, status, comment]:\n    ' + str(label_correction_table[i, :]) + '\n')
            progress_bar.write('-' * 80)
        elif error_encountered is not None and error_encountered != label_correction_table[i, status_idx]:
            label_correction_table[i, status_idx] = error_encountered
            # save the updated csv file
            save_label_correction_csv_file(label_correction_csv_path, label_correction_table)

    return label_correction_table


def generate_gripper_action_label(data):
    """ generate new action labels and goal action indices based on the gripper open/closed state

    This performs an in place modification of the data argument sets 'gripper_action_label' and 'gripper_action_goal_idx' in the data argument.

    Use stack_player.py to visualize the dataset, that makes it easier to understand what this data is.

    # Returns

        gripper_action_label, gripper_action_goal_idx

        numpy array containing integer label values,
        and numpy array containing integer goal timestep indices.
    """
    gripper_status = list(data['gripper'])
    action_status = list(data['label'])
    # print("goal",list(data["goal_idx"]))
    gripper_action_goal_idx = []
    unique_actions, indices = np.unique(action_status, return_index=True)
    unique_actions = [action_status[index] for index in sorted(indices)]
    action_ind = 0
    gripper_action_label = action_status[:]
    for i in range(len(gripper_status)):

        if (gripper_status[i] > 0.1 and gripper_status[i-1] < 0.1) or (gripper_status[i] < 0.5 and gripper_status[i-1] > 0.5):
            action_ind += 1
            # print(i)
            gripper_action_goal_idx.append(i)

        # For handling error files having improper data
        if len(unique_actions) <= action_ind or len(gripper_action_label) <= i:
            break
        else:
            gripper_action_label[i] = unique_actions[action_ind]

    gripper_ind = 0
    goal_list = []
    goal_state = False
    if len(gripper_action_goal_idx) == 0:
        goal_to_add = 0
    else:
        goal_to_add = gripper_action_goal_idx[0] - 1
    if goal_to_add != 0:
        for i in range(len(gripper_action_label)):
            # print(i)
            if(i < goal_to_add):
                # print("goal_len", len(goal_list))
                goal_state = False
                goal_list.append(goal_to_add)

            else:
                gripper_ind += 1
                goal_state = True

            if gripper_ind < len(gripper_action_goal_idx):
                goal_to_add = gripper_action_goal_idx[gripper_ind] - 1
            else:
                goal_to_add = len(gripper_status)-1
            if goal_state is True:
                goal_list.append(goal_to_add)
    else:
        goal_list = [goal_state]

    gripper_action_goal_idx = goal_list

    return gripper_action_label, gripper_action_goal_idx


if __name__ == "__main__":
    if tf is not None:
        tf.enable_eager_execution()
    args = _parse_args()
    main(args)
