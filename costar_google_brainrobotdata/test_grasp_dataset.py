
import numpy as np
import pytest
import tensorflow as tf
from grasp_dataset import GraspDataset
from grasp_dataset import get_multi_dataset_training_tensors
from tqdm import tqdm  # progress bars https://github.com/tqdm/tqdm


def test_grasp_dataset():
    dataset = '102'
    grasp_dataset_object = GraspDataset(dataset=dataset)
    batch_size = 10
    num_batches_to_traverse = 1000

    with tf.Session() as tf_session:
        (feature_op_dicts,
         features_complete_list,
         time_ordered_feature_name_dict,
         num_samples_in_dataset) = grasp_dataset_object.get_training_dictionaries(batch_size=batch_size)

        tf_session.run(tf.global_variables_initializer())

        for batch_num in tqdm(range(num_batches_to_traverse), desc='dataset'):
            attempt_num_string = 'batch_' + str(batch_num).zfill(4) + '_'
            print('dataset_' + dataset + '_' + attempt_num_string + 'starting')
            output_features_dicts = tf_session.run(feature_op_dicts)
            # reorganize is grasp attempt so it is easy to walk through
            time_ordered_feature_data_dicts = grasp_dataset_object.to_tensors(
                output_features_dicts, time_ordered_feature_name_dict)

            assert(len(output_features_dicts) == len(time_ordered_feature_data_dicts))

            # for features_dict_np, sequence_dict_np in output_features_dicts
            # time_ordered_feature_data_dicts
            # [()] = output_features_dicts


if __name__ == '__main__':
    test_grasp_dataset()
    pytest.main([__file__])