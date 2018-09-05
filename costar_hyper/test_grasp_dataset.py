
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.contrib.hooks import ProfilerHook
# from tensorflow import SingularMonitoredSession
from grasp_dataset import GraspDataset
from grasp_dataset import get_multi_dataset_training_tensors
from tqdm import tqdm  # progress bars https://github.com/tqdm/tqdm


def test_grasp_dataset():
    dataset = '102'
    batch_size = 10
    num_batches_to_traverse = 10

    grasp_dataset_object = GraspDataset(dataset=dataset)
    (feature_op_dicts,
     features_complete_list,
     time_ordered_feature_name_dict,
     num_samples_in_dataset) = grasp_dataset_object.get_training_dictionaries(batch_size=batch_size)

    config = tf.ConfigProto()
    config.inter_op_parallelism_threads = 40
    config.intra_op_parallelism_threads = 40
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as tf_session:
        tf_session.run(tf.global_variables_initializer())
        for batch_num in tqdm(range(num_batches_to_traverse), desc='dataset'):
            attempt_num_string = 'batch_' + str(batch_num).zfill(4) + '_'
            print('dataset_' + dataset + '_' + attempt_num_string + 'starting')
            output_features_dicts = tf_session.run(feature_op_dicts)
            assert(len(output_features_dicts) > 0)


if __name__ == '__main__':
    test_grasp_dataset()
    pytest.main([__file__])