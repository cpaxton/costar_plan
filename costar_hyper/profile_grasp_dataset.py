
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.contrib.hooks import ProfilerHook
# from tensorflow import SingularMonitoredSession
from grasp_dataset import GraspDataset
from grasp_dataset import get_multi_dataset_training_tensors
from tqdm import tqdm  # progress bars https://github.com/tqdm/tqdm


def profile_grasp_dataset():
    # To profile, run this program then open
    # the address chrome://tracing/ in google chrome,
    # other browsers will not work.
    # Use the load button on that website to open the file
    #
    #     /tmp/profiling/timeline-0.json
    #
    # The tracing results should load immediately!
    dataset = '102'
    batch_size = 10
    num_batches_to_traverse = 1000

    grasp_dataset_object = GraspDataset(dataset=dataset)

    (feature_op_dicts,
     features_complete_list,
     time_ordered_feature_name_dict,
     num_samples_in_dataset) = grasp_dataset_object.get_training_dictionaries(batch_size=batch_size)

    config = tf.ConfigProto()
    config.inter_op_parallelism_threads = 40
    config.intra_op_parallelism_threads = 40
    config.gpu_options.allow_growth = True
    global_step = tf.train.get_or_create_global_step()
    hooks = [ProfilerHook(save_secs=30, output_dir="/tmp/profiling")]
    with tf.train.SingularMonitoredSession(hooks=hooks, config=config) as tf_session:
        for _ in tqdm(range(num_batches_to_traverse)):
            tf_session.run(feature_op_dicts)
            if tf_session.should_stop():
                break


if __name__ == '__main__':
    profile_grasp_dataset()
