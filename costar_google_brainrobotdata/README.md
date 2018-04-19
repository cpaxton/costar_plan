
# Google Brain Grasp Dataset APIs

Author and maintainer: `Andrew Hundt <ATHundt@gmail.com>`

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

#### Regression
```

```

The following commands will generate a hyperopt summary.

Generating a hyperparameter search results summary for cornell regression

```
python hyperopt_rank.py --log_dir hyperopt_logs_cornell_regression --sort_by val_grasp_jaccard
```

#### Classification

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_hyperopt.py --log_dir hyperopt_logs_cornell_classification
```


Generating a hyperparameter search results summary for cornell classification

```
python hyperopt_rank.py --log_dir hyperopt_logs_cornell_classification --sort_by val_binary_accuracy
```


## TF Object Detection API


1. follow the installation instructions for tensorflow_models.sh
in github.com/ahundt/robotics_setup.
2. That will install the [tf object detection
API](https://github.com/tensorflow/models/blob/master/research/object_detection)
3. Perform cornell dataset tfrecord generation for use in object detection

```
python2 cornell_grasp_dataset_writer --write True --padding False --angle_classes True --include_negative_examples False --angle_classes_zero_is_background_class True
```

4. Run `python tf_object_detection.py`, this will download models from the [tf object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Model files will be downloaded to

- `~/.keras/datasets/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/`
- `~/.keras/datasets/ssd_mobilenet_v2_coco_2018_03_29/`
- `~/.keras/datasets/faster_rcnn_nas_coco_2018_01_28/`

Next step is to set up configuration files for a training run.

Some reference commands might be at https://github.com/tensorflow/models/issues/3909

TODO(ahundt) how to specify multiple tfrecord files?
(the next batch of text is from [train.py](https://github.com/tensorflow/models/blob/master/research/object_detection/train.py))

The executable `research/object_detection/train.py` is used to train DetectionModels. There are two ways of
configuring the training job:
1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.
Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt
2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.
Example usage:
```
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
```

Here are two example commands to run one after the other from [github issue #1854](https://github.com/tensorflow/models/issues/1854) train with:

```
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=~/.virtualenvs/Project/lib/python2.7/site-packages/tensorflow/models/object_detection/samples/configs/faster_rcnn_resnet101_pets_learn.config \
    --train_dir=~/.virtualenvs/Project/models/model/train
```

Then run the following to evaluate:

```
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=~/.virtualenvs/Project/lib/python2.7/site-packages/tensorflow/models/object_detection/samples/configs/faster_rcnn_resnet101_pets_learn.config \
    --checkpoint_dir=~/.virtualenvs/Project/models/model/train \
    --eval_dir=~/.virtualenvs/Project/models/model/eval

```