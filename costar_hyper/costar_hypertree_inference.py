'''
Run model inference for HyperTree Model and generate plots for distance vs error and attempts vs error
See main for object initialization to evaluate model from hyperparams and weights, generate plots from the evaluated data.
'''
from block_stacking_reader import CostarBlockStackingSequence
from costar_inference_plot_generator import CostarInferencePlotGenerator
from costar_inference_plot_generator import inference_mode_gen
import h5py
import os
import numpy as np
import glob
import keras
from grasp_utilities import load_hyperparams_json
from cornell_grasp_train import get_compiled_model
from cornell_grasp_train import choose_features_and_metrics
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class CostarHyperTreeInference():

    def __init__(self, filenames_text, hyperparams_json, load_weights, problem_name, feature_combo_name, image_shape, pose_name):
        '''
        Initialization.
        Setting up the CostarBlockStackingSequence, loading weights and hyperparameters from the url/path specified.

        #Arguments
        filenames: List of file paths to be read
        hyperparams_json: Path/url of the model json to be read
        load_weights: Path/url of the weights h5 file to be used
        problem_name: As used in cornell_grasp_train.py
        feature_combo_name: feature_combo_name as used in cornell_grasp_train.py
        pose_name: Which pose to use as the robot 3D position in space. Options include:
            'pose' is the end effector ee_link pose at the tip of the connector
                of the robot, which is the base of the gripper wrist.
            'pose_gripper_center' is a point in between the robotiq C type gripping plates when the gripper is open
                with the same orientation as pose.

        '''
        print('loading data from: ' + str(filenames_text))
        self.filenames = np.genfromtxt(filenames_text, dtype='str', delimiter=', ')
        self.hyperparams_json = hyperparams_json
        if not os.path.isfile(hyperparams_json):
            self.hyperparams_json = self.get_file_from_url(hyperparams_json)
        self.problem_name = problem_name
        self.load_weights = load_weights
        if load_weights is None:
            print('No weights passed')
        elif not os.path.isfile(load_weights):
            self.load_weights = self.get_file_from_url(load_weights)
        self.gripper_action_goal_idx = []
        self.image_shape = image_shape
        self.file_list_updated, self.file_len_list, self.gripper_action_goal_idx = inference_mode_gen(self.filenames)
        self.file_counter=0
        self.generator = self.initialize_generator(pose_name)

    def initialize_generator(self, pose_name):
        '''
        Initializes and returns CostarBlockStacking generator.
  
        #Arguments:
        pose_name: See init for description
        '''

        filenames = self.filenames
        output_shape = self.image_shape
        # image_shapes, vector_shapes, data_features, model_name, monitor_loss_name, label_features, _
        features_and_metrics = choose_features_and_metrics(feature_combo_name, problem_name, image_shapes=output_shape)
        label_features = features_and_metrics[5]
        data_features = features_and_metrics[2]
        generator = CostarBlockStackingSequence(
            filenames, batch_size=1, verbose=1,
            output_shape=output_shape,
            label_features_to_extract=label_features,
            data_features_to_extract=data_features,
            blend_previous_goal_images=False, inference_mode=True, pose_name=pose_name, is_training=False)
        return generator


    def block_stacking_generator(self, sequence):
        epoch_size = len(self.file_list_updated)
        sequence.on_epoch_end()
        step = 0
        while True:
            # step = self.file_counter
            # if step > epoch_size:
            #     step = 0
            #     sequence.on_epoch_end()
            batch = sequence.__getitem__(step)
            step += 1
            yield batch
    def evaluate_model(self, result_filename):
        '''
        Evaluates the initialized model and stores all metrics as per cases in cornell_grasp_train.py. 
        See choose_features_and_metrics in cornell_grasp_train.py for more details on metrics

        #Arguments
        generator: Generator object for feeding data for evaluations
        result_filename: The filename for evaluated metrics to be stored.
        '''
        generator = self.generator
        filenames_updated, file_len_list = self.file_list_updated, self.file_len_list
        hyperparams = load_hyperparams_json(self.hyperparams_json)
        hyperparams.pop('checkpoint', None)
        model = get_compiled_model(**hyperparams, problem_name=self.problem_name, load_weights=self.load_weights)
        bsg = self.block_stacking_generator(generator)
        with open(result_filename, 'w') as fp:
            cw = csv.writer(fp, delimiter=',', lineterminator='\n')
            cw.writerow(['example', 'frame_no'] + model.metrics_names)
            # fp.write("\n")
        frame_counter = 0
        self.file_counter = 0
        frame_len = file_len_list[self.file_counter]
        for i in range(len(generator)):
            data = next(bsg)
            if filenames_updated[i] != self.filenames[self.file_counter]:
                self.file_counter += 1
                frame_len = file_len_list[self.file_counter]
                frame_counter = 0

            # print("len of X---", len(data[0]))
            frame_counter += 1 % frame_len
            score = model.evaluate(data[0], data[1])
            with open("inference_results_per_frame.csv", 'a') as fp:
                cw = csv.writer(fp, delimiter=',', lineterminator='\n')
                score = [self.file_counter] + [frame_counter] + score
                cw.writerow(score)

    def extract_filename_from_url(self, url):
        # note this is almost certainly insecure,
        # and the url has to exactly match a filename,
        # no extra string contents at the end
        filename = url[url.rfind("/")+1:]
        return filename

    def get_file_from_url(self, url, extract=True, file_hash=None, cache_subdir='models'):
        filename = self.extract_filename_from_url(url)

        found_extension = None
        if extract:
            for extension in ['.tar', '.tar.gz', '.tar.bz', '.zip']:
                if extension in filename:
                    found_extension = extension

        path = keras.utils.get_file(filename, url, extract=extract, file_hash=file_hash, cache_subdir=cache_subdir)
        if found_extension is not None:
            # strip the file extension
            path = path.replace(found_extension, '')

        if not os.path.isfile(path):
            raise ValueError(
                'get_file_from_url() tried extracting the url: ' + str(url) +
                ' and we expected this compression option: ' + str(found_extension) +
                ' and the file directly at the url to match this hash option: ' + str(file_hash) +
                ' . However, the final file is not at the expected location: ' + str(path) +
                ' One possible problem is with compression, it is optional'
                ' but when there is compression we expect'
                ' a filename in the archive that matches the filename in the url.'
                ' You may need to debug the code, or if your use case is different'
                ' try get_file() in Keras.')
        return path



if __name__ == "__main__":

    # path to txt file containing validation/test filenames
    filenames = r"C:\Users\Varun\JHU\LAB\Projects\costar_task_planning_stacking_dataset_v0.1\train.txt"
    output_shape = (224, 224, 3)
    # url or path to weights and hyperparams
    weights_url = "https://github.com/ahundt/costar_dataset/releases/download/v0.2/2018-09-04-20-17-25_train_v0.3_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-412-val_loss-0.002-val_angle_error-0.279.h5.zip"
    hyperparams = "https://github.com/ahundt/costar_dataset/releases/download/v0.2/2018-09-04-20-17-25_train_v0.3_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5_hyperparams.json"

    problem_name = 'semantic_rotation_regression'
    feature_combo_name = 'image/preprocessed'

    # object initalization
    hypertree_inference = CostarHyperTreeInference(filenames_text=filenames, hyperparams_json=hyperparams, load_weights=weights_url, problem_name=problem_name, feature_combo_name=feature_combo_name, image_shape=output_shape, pose_name='pose')

    # pass filename where you want inference data to be stored
    hypertree_inference.evaluate_model('inference_results_per_frame.csv')

    # pass filename and the metric_name to be used to generate the plots
    # inference_generator = CostarInferencePlotGenerator(filenames)
    # inference_generator.generate_plots('inference_results_per_frame.csv', 'angle_error')
    # hypertree_inference.generate_plots('inference_results_per_frame.csv', 'angle_error')

    # exit()
    # filenames_updated, file_len_list = inference_mode_gen(filenames[:2])
    # print(len(filenames))
    # training_generator = CostarBlockStackingSequence(
    #     filenames[:4], batch_size=1, verbose=1,
    #     output_shape=output_shape,
    #     label_features_to_extract='grasp_goal_aaxyz_nsc_5',
    #     data_features_to_extract=['current_xyz_aaxyz_nsc_8'],
    #     blend_previous_goal_images=False, inference_mode=True, pose_name="pose")
    # # bsg = block_stacking_generator(training_generator)
    # # print('bsg len', len(training_generator))
    # file_mode = 'w'
    # hyperparams = "2018-09-04-20-17-25_train_v0.3_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5_hyperparams.json"
    # problem_name = 'semantic_rotation_regression'
    # hypertree_inference = CostarHyperTreeInference(filenames=filenames, hyperparams_json=hyperparams, load_weights=load_weights, problem_name=problem_name)
    # hypertree_inference.evaluate_model(training_generator, 'inference_results_per_frame.csv')
    # hypertree_inference.generate_plots('inference_results_per_frame.csv', 'angle_error')
