
Costar stacking dataset v0.4
----------------------------

Authors:

- Andrew Hundt <ATHundt@gmail.com> or <ahundt@jhu.edu>
- Chris Paxton <cpaxton@jhu.edu>

Special Thanks To:

- Chunting Jiao for starting & resetting the robot when needed.


v0.4 is a **beta** version of the dataset.
We are using an open source license which requires
attribution, so we require that users do not publically
release information based on this dataset until the paper
announcing it is ready, expected Sep 2018. Contact Andrew Hundt
at the email address above for details, if you would like
to release something sooner, or if you would like to run code
on the robot.

Please report any inconsistencies in the documentation below,
and pull requests with preprocessing scripts or corrections
will be very much appreciated!


About this Dataset
------------------

The robot attempts to stack 3 of 4 colored blocks (red, green, yellow, blue) in a pre-specified order.
The blocks are cubes that are 51mm on each side.

- 2293 successful stacking examples `*.success.h5f`
- 6253 failure stacking examples `*.failure.h5f` consisting of:
        - 1396 failures where the run completed but the stack was not at least 3 blocks tall
	- 4857 failures (`*.error.failure.h5f`) where an error occurred and the example terminated early
            - Causes include arm hardware errors such as safety stops and software crashes.

TODO(ahundt) Some failure examples are successes that hae been mislabled, because the z height of the block was used to determine success, and the robot pounded the floor surface down enough to turn success labels into failures.

How to count the data:

```
ls -1 ~/.costar/data/ | grep success | wc -l
ls -1 ~/.costar/data/ | grep failure | wc -l
ls -1 ~/.costar/data/ | grep error | wc -l
```

### Filenames

Filenames are in the hdf5 format and take the following form:

```
YYYY-MM-DD-HH-MM-SS_example######.[success,failure,error.failure].h5f
```

Here are several examples:

```
2018-05-29-15-00-28_example000001.success.h5f
2018-05-30-15-33-16_example000031.success.h5f
2018-06-04-10-01-33_example000001.error.failure.h5f
2018-06-04-10-01-33_example000001.error.failure.h5f
```

The filenames are defined based on the `save()` function in `collector.py`:

```python
    def save(self, seed, result, log=None):
        '''
        Save function that wraps dataset access, dumping the data
        which is currently buffered in memory to disk.

        seed: An integer identifer that should go in the filename.
        result: Options are 'success' 'failure' or 'error.failure'
            error.failure should indicate a failure due to factors
            outside the purposes for which you are collecting data.
            For example, if there is a network communication error
            and you are collecting data for a robot motion control
            problem, that can be considered an error based failure,
            since the goal task may essentially already be complete.
            You may also pass a numerical reward value, for numerical
            reward values, 0 will name the output file 'failure', and
            anything > 0 will name the output file 'success'.
        log: A string containing any log data you want to save,
            For example, if you want to store information about errors
            that were encounterd to help figure out if the problem
            is one that is worth including in the dataset even if it
            is an error.failure case. For example, if the robot
            was given a motion command that caused it to crash into the
            floor and hit a safety stop, this could be valuable training
            data and the error string will make it easier to determine
            what happened after the fact.
'''


Loading Data
------------

More information and code can be found in [costar_plan/ctp_integration/README.md](https://github.com/cpaxton/costar_plan/tree/master/ctp_integration).

### [view_convert_dataset.py](https://github.com/cpaxton/costar_plan/blob/master/ctp_integration/scripts/view_convert_dataset.py)
    - example of how to walk through the dataset
    - TODO(ahundt) modify the dataset to add 'gripper_action' labels based on when the gripper opens and closes

### [stack_player.py](https://github.com/cpaxton/costar_plan/blob/2ef9b194a507b24e479c037c8e7121209c5013fe/ctp_integration/scripts/stack_player.py)
    - view the dataset
    - scroll through individual timesteps and image frames
    - plot state such as action labels over time
    - TODO(ahundt) manually label data

### [collector.py](https://github.com/cpaxton/costar_plan/blob/master/ctp_integration/python/ctp_integration/collector.py)
    - Saves the data from the robot to disk



### Dataset Features and Time Steps

Every example can be loaded easily with [h5py](https://www.h5py.org/) and all features has an equal number of frames collected at 10 Hz, or 0.1 second per frame. The number of frames will vary with each example, including examples with zero frames which are typically in the `*.error.failure.h5f` examples.


Here is a complete list of features:

 - "nsecs" - nanosecond component of the timestamp for this entry.
 - "secs" - second component of the timestamp for this entry.
 - "q" - The 6 joint angles of the UR5 arm from base to tip in radians.
 - "dq" - Change in joint angle `q` from the previous times step
 - "pose" - Pose of the gripper end effector.
 - "camera" - Pose of the camera.
 - "image" - rgb image encoded as binary data in the jpeg format, it has already been rectified and calibrated.
 - "depth_image" - [encoded depth images](https://sites.google.com/site/brainrobotdata/home/depth-image-encoding) in png format with measurements in milimeters, it has already been rectified and calibrated
 - "goal_idx" - The timestep index in the data list at which the goal is reached. For grasps and placements this changes after the gripper respectively opens or closes and then backs off the target.
 - "gripper" - open/closed griper float where 0 (~0.055 in practice) is completely open and 1 (in practice TBD) is completely closed
 - "label" - current integer label as defined by labels_to_name
 - "info" - string description of current step
 - "depth_info" - currently empty.
 - "rgb_info" - currently empty.
 - "object" - Identifier of the object the robot will be interacting with.
 - "object_pose" - Pose of entry in "object" feature detected via objrecransac.
 - "labels_to_name" - list of action description strings. The string index corresponds to the integer label for that action, i.e. if `data["labels_to_name"][0] is "grab_blue", then its corresponding integer index is 0.
 - "rgb_info_D" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "rgb_info_K" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "rgb_info_R" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "rgb_info_P" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "rgb_info_distortion_model" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "depth_info_D" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "depth_info_K" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "depth_info_R" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "depth_info_P" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "depth_distortion_model" - camera calibration param, may be empty. see the yaml file in the dataset for the values.
 - "all_tf2_frames_as_yaml"
 - "all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json"] - list of json strings that when loaded define a dictionary mapping from coordinate frame names to a list of doubles in [x, y, z, qx, qw, qy, qz] translation + quaternion order. All transforms are specified relative to the robot base.
 - "visualization_marker" - transform from the robot base to the AR tag marker. TODO(ahundt )Needs to be validated, and if it is the one on the robot it will vary over time because the mount broke.
 - "camera_rgb_optical_frame_pose" - the pose of the camera rgb image optical frame relative to the robot base
 - "camera_depth_optical_frame_pose" - the pose of the camera depth image optical frame relative to the robot base

 Additional features may be added with preprocessing.

 TODO(ahundt) preprocessing to create new goal labels which change at the moment gripper open/close actions are triggered is currently in progress.

Camera Calibration
------------------

A yml file is included in the dataset with camera calibration parameters. See the ROS documentation for details.


Collecting Data
---------------

See [costar_plan/ctp_integration/README.md](https://github.com/cpaxton/costar_plan/tree/master/ctp_integration) for details on how to run data collection.


License
-------

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)



Notes and Limitations
---------------------

TODO(ahundt) Successes may have been marked as failures in the dataset, they will need to be double checked.


### Error logs

error.failure indicates a problem we cannot currently recover from, such as a security stop or a ROS planning error.
About halfway through the dataset we saved the final error string on error.failure cases, hopefully that
will assist in diagnosing/classifiying mroe detailed reasons for why failures occur.

### Broken AR Tag mount

The AR tag mount broke around 2018-05-15.
Some of the dataset was collected with the tag shifting around.

Starting with:

```
2018-05-17-16-39-30_example000001.success.h5f
```
- re-glued the ar tag so the hand-eye calibration is different
    - did not re-generate hand eye calibration since objects are still being grasped ok at this point
- At least some successes were being reported as failures at this point.


```
from

2018-05-29-15-00-28_example000001.success.h5f

to

2018-05-29-19-55-55_example000001.success.h5f
```

- Added wood_cube.stl to planning scene in `~/data/mesh/collision_mesh/`, which was previously disabled.
This will mean motions include collision planning so behavior of the robot will differ from that
which was collected previously.
- Start is Approximate filename, the change may have been a few minutes before/after, end is exact.
- run with planning includes about 162 successes.

### RGB/Depth time synchronization

The RGB and depth data (in fact all data) is not always perfectly time synchronized,
and in some cases many depth frames are missing.
This is much more common in examples containing failures and errors `*.failure.h5f` and `*.error.failure.h5f`
examples than it is in successes `*.success.h5f`.

Some bugs were fixed during the collection process which improved the synchronization
dramatically, so use the filenames to choose more recently collected data, try after 2018-05-17-16-39, if you need to minimize the synchronization errors.

### AR Tag mount broken

TODO(ahundt) get date of this break

The AR Tag mount broke a second time, and we simply left it detached for collection until the first data in september. In general, we advise not making any assumptions about the precision of the AR tag position relative to the robot. It can vary by a few centimeters and vary noticeably in angle. It should only be used if very rough values are needed.
The hand eye calibration itself seems to remain OK.

### 2018-08-31 Gripper Failure


The gripper failed on 2018-08-31 at the time 22:33:34, the example:

    ~/.keras/datasets/costar_plush_block_stacking_dataset_v0.1/2018-08-31-22-33-34_example000002.error.failure.h5f

and all following it on 2018-08-31 have the gripper locked in the closed position, and data may or may not be recorded for the gripper state.

### 2018-09-07 Gripper Repaired, appearance change

We repaired the connector and added a piece to minimize flexing of the wire, and changed where the gripper wire is attached. This means there will be an appearance change which may affect predictions.



