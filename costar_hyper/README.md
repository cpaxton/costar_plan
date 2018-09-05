
# CoSTAR Hyper


Author and maintainer: `Andrew Hundt <ATHundt@gmail.com>`

## Costar Block Stacking Dataset

The Block stacking dataset is currently hosted on google drive. It can be downloaded with [rclone](https://github.com/ncw/rclone).

Follow the [google drive costar block stacking dataset v0.2 download link](https://drive.google.com/drive/folders/1k7N1EvxRh7GwS5tbax8RAeUU_dcPGWOw?usp=sharing), and add the folder to your drive. This should not use up any of your storage limit.

```
rclone copy drive:costar_block_stacking_dataset_v0.3 ~/.keras/dataset/costar_block_stacking_dataset_v0.3 --exclude="*failure*"
```

Two lines to read one file from the dataset:
```
h5 = h5py.File(filename, 'r')
x = np.array(h5[key])
```

Detailed documentation is in the dataset README.md.

More precisely:

```#!/usr/bin/env python

import sys

filename = sys.argv[1]

import h5py
import matplotlib.pyplot as plt
import numpy as np
from image import *

horizon = 10
i0 = 0

h5 = h5py.File(filename, 'r')
string = list(h5["label_to_string"])
label = np.array(h5["label"])
imgs = ConvertPNGListToNumpy(np.array(h5["images_rgb"]))

changed = np.abs(np.diff(label, axis=-1)) > 0
selected = np.zeros_like(label)
selected[1:] = changed
selected[-1] = 1
selected[0] = 1
selected = selected > 0

selected_action = np.zeros_like(label)
selected_action[:-1] = changed
selected_action[-1] = 1
selected_action[0] = 1
selected_action = selected_action > 0

print(selected)
selected_imgs = imgs[selected]

plt.figure()
for i, img in enumerate(selected_imgs):
    plt.subplot(1,len(selected_imgs),i+1)
    plt.axis("off")
    plt.imshow(img)

plt.figure()
plt.axis("off")
plt.imshow(selected_imgs[4])

plt.show()```

some of those fields will vary for different use cases.

## Google Brain Grasp Dataset APIs

<img width="1511" alt="2017-12-16 surface relative transforms correct" src="https://user-images.githubusercontent.com/55744/34134058-5846b59e-e426-11e7-92d6-699883199255.png">
This version should be ready to use when generating data real training.

Plus now there is a flag to draw a circle at the location of the gripper as stored in the dataset:
![102_grasp_0_rgb_success_1](https://user-images.githubusercontent.com/55744/34133964-ccf57caa-e425-11e7-8ab1-6bba459a5408.gif)

A new feature is writing out depth image gifs:
![102_grasp_0_depth_success_1](https://user-images.githubusercontent.com/55744/34133966-d0951f28-e425-11e7-85d1-aa2706a4ba05.gif)

Image data can be resized:

![102_grasp_1_rgb_success_1](https://user-images.githubusercontent.com/55744/34430739-3adbd65c-ec36-11e7-84b5-3c3712949914.gif)

The blue circle is a visualization, not actual input, which marks the gripper stored in the dataset pose information.

Color augmentation is also available:

![102_grasp_2_rgb_success_1](https://user-images.githubusercontent.com/55744/34698561-ba2bd61e-f4a6-11e7-88d9-5091aed500fe.gif)
![102_grasp_3_rgb_success_1](https://user-images.githubusercontent.com/55744/34698564-bef2fba0-f4a6-11e7-9547-06b4410d86aa.gif)

### How to view the vrep dataset visualization

1. copy the .ttt file and the .so file (.dylib on mac) into the `costar_google_brainrobotdata/vrep` folder.
2. Run vrep with -s file pointing to the example:
```
./vrep.sh -s ~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/vrep/kukaRemoteApiCommandServerExample.ttt
```
4. vrep should load and start the simulation
5. make sure the folder holding `vrep_grasp.py` is on your PYTHONPATH
6. cd to `~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/`, or wherever you put the repository
7. run `export CUDA_VISIBLE_DEVICES="" && python2 vrep_grasp.py`

## Hyperparameter search


### Costar Block Stacking Dataset

The Block stacking dataset is currently hosted on google drive. It can be downloaded with [rclone](https://github.com/ncw/rclone).

Follow the [google drive costar block stacking dataset v0.2 download link](https://drive.google.com/drive/folders/1k7N1EvxRh7GwS5tbax8RAeUU_dcPGWOw?usp=sharing), and add the folder to your drive. This should not use up any of your storage limit.

```
rclone copy drive:costar_block_stacking_dataset_v0.3 ~/.keras/dataset/costar_block_stacking_dataset_v0.3 --exclude="*failure*"
```

Configure `cornell_hyperopt.py` to run on the problem you'd like, the script supports both the cornell and the costar block stacking ataset. Sorry for the bad naming, this a historical artifact that should be resolved at some point in the future.

Options include `semantic_grasp_regression`, `semantic_translation_regression`, `semantic_rotation_regression`. Once you have edited the file for your use case run the following command:

```
while true; do export CUDA_VISIBLE_DEVICES="0" && python2 cornell_hyperopt.py; done
```

This will run the hyperopt search on a loop with 100 random models and 10 Bayesian models each run until you stop it with `ctrl + c`. These limits are set due to an unresolved memory leak in the block stacking hdf5 reading code which has not yet been resolved. If you run out of memory lower this number. Your GPU should have at least 7GB memory, but preferably 10-12GB of memory.

After running hyperopt, you will have collected a dataset of folders with results for each model, collate the results as follows:

```
± python hyperopt_rank.py  --log_dir hyperopt_logs_costar_block_stacking_train_ranked_regression --sort_by val_grasp_acc --ascending=False --nofilter_epoch --filter_unique  && csvtotable hyperopt_logs_costar_block_stacking_train_ranked_regression/hyperopt_rank.csv hyperopt_logs_costar_block_stacking_train_ranked_regression/hyperopt_rank.html -o
```

`hyperopt_rank.py` produces a file hyperopt_rank.csv with all of the best models sorted by the chosen metric, in the above case `val_grasp_acc`.

[csvtotable](https://github.com/vividvilla/csvtotable) is used to convert the csv files to html for easy viewing on a remote machine. You can also simply open the csv file in openoffice, google sheets, excel, etc, or the html file in a web browser.

Next, directly edit `costar_block_stacking_train_ranked_regression.py` to configure a run through the top ranked models for perhaps 30 epochs, to see which ones perform best with more training. `costar_block_stacking_train_ranked_regression.py` will load the `hyperopt_rank.csv` file, sort it again based on the chosen configuration, then train each model for the specified number of epochs:

```
export CUDA_VISIBLE_DEVICES="0" && python2 costar_block_stacking_train_ranked_regression.py --log_dir hyperopt_logs_costar_block_stacking_train_ranked_regression --run_name 30_epoch
```

Re-run the `hyperopt_rank.py` command above to update the model rankings with the new data. Do a final run on the absolute best models for 300 epochs, again editing the file for the new number of epochs:

```
export CUDA_VISIBLE_DEVICES="0" && python2 costar_block_stacking_train_ranked_regression.py --log_dir hyperopt_logs_costar_block_stacking_train_ranked_regression --run_name 300_epoch
```

You may wish to use the ` --learning_rate_schedule triangular` flag for one run and then the  `--learning_rate_schedule triangular2 --load_weights path/to/previous_best_weights.h5` for a second run. These learning rate schedules use the [keras_contrib](github.com/keras-team/keras-contrib) cyclical learning rate callback, see [Cyclical learning rate repo](https://github.com/bckenstler/CLR) for a detailed description and paper links.


### Google Brain Grasping Dataset

To run the search execute the following command

```
export CUDA_VISIBLE_DEVICES="0" && python2 google_grasp_hyperopt.py --run_name single_prediction_all_transforms
```

Generating a hyperparameter search results summary for google brain grasping dataset classification:

```
python hyperopt_rank.py --log_dir hyperopt_logs_google_brain_classification --sort_by val_acc
```

### Cornell Dataset

These are instructions for training on the [cornell grasping dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php).


#### Downloading the dataset

To download the dataset and generate the tensorflow tfrecord dataset files simply run the following command:

```
python cornell_grasp_dataset_writer.py
```

This should take about 5-6 GB of space, and by default the dataset files will go in:

```

flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
```

#### Where results can be found

Check the value of the following parameter with each command:

```--log_dir```

Files will be created containing training results in that file.

By default it is `./logs_cornell/`:

#### Regression

Regression tries to predict a grasp position and orientation based on an image alone.

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_grasp_train_regression.py --pipeline_stage k_fold --run_name 60_epoch_real_run
```

The following commands will generate a hyperopt summary.

Generating a hyperparameter search results summary for cornell regression

```
python hyperopt_rank.py --log_dir hyperopt_logs_cornell_regression --sort_by val_grasp_jaccard
```

Please note that there are currently bugs in the loss with the regression problem. Additonally,
a custom command must be run to find the correct intersection over union score.

#### Classification

Takes a grasp position, orientation, gripper width, etc as input, then returns a score indicating if it will be a successful grasp of an object or a failed grasp.
Given the grasp command input, the proposed grasp image
is rotated and centered on the grasp, so that all grasps have the gripper
in a left-right orientation to the image and the gripper is at the center.

 - `cornell_hyperopt.py` is basically a configuration file used to configure hyperparameter optimization for the cornell grasping dataset.
 - `hyperopt.py` is another configuration file that sets up the range of models you should search during hyperopt.
     - shared by google and cornell for both classification and regression training
     - you may also want to modify this to change what kind of models are being searched.

In `cornell_hyperopt.py` make sure the following variable is set correctly:

```
    FLAGS.problem_type = 'classification'
```

This will run hyperopt search for a couple of days on a GTX 1080 or Titan X to find the best model.

Hyperparameter search to find the best model:

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_hyperopt.py --log_dir hyperopt_logs_cornell_classification
```

Result will be in files with names like these:

```

2018-05-08-17-55-00__bayesian_optimization_convergence_plot.png
2018-05-08-17-55-00__hyperoptions.json
2018-05-08-17-55-00__optimized_hyperparams.json
2018-05-23-19-08-31__hyperoptions.json
```


Generating a hyperparameter search results summary for cornell classification:

```
python hyperopt_rank.py --log_dir hyperopt_logs_cornell_classification --sort_by val_binary_accuracy
```

This will generate a hyperopt ranking file:

```
hyperopt_rank.csv
```

Look at the CSV file above, and it will have different model configurations listed
as well as how well each performed. Select the best performing model, typically the
top entry of the list, and look for the json filename in that row which stores the best configuration.
With this file, we need to run k-fold cross validation to determine how well the model
actually performs.

Perform K-fold cross validation on the model for a more robust comparison of your model against other papers and models:

 - `cornell_grasp_train_classification.py` configuration file to train a single classification model completely.
     - Can do train/test/val splits
     - Can do k-fold cross validation
     - Be sure to manually edit the file with the json file specifying the hyperparameters for the model you wish to load, number of epochs, etc.

Here is one example configuration:

```
        FLAGS.load_hyperparams = ('hyperparams/classification/2018-05-03-17-02-01_inception_resnet_v2_classifier_model-'
                                  '_img_inception_resnet_v2_vec_dense_trunk_vgg_conv_block-dataset_cornell_grasping-grasp_success_hyperparams.json')
```

Here is the command to actually run k-fold training:

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_grasp_train_classification.py  --run_name 2018-04-08-21-04-19_s2c2hw4 --pipeline_stage k_fold
```

After it finishes running there should be a file created named `*summary.json` with your final results.