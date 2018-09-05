import tensorflow as tf
import os.path
from grasp_dataset import GraspDataset
from grasp_dataset import FLAGS
from grasp_dataset import mkdir_p

""" Counts the grasp successes, failures, and ratio of succsesses vs the total number of attempts.

    Writes to the following location by default:

    ~/.keras/datasets/grasping/grasp_dataset_<dataset number>_statistics.txt'

    To run:

    python2 success_fail_statistic.py --grasp_dataset 102

    Output:

    grasp_dataset_<dataset number>_statistics.txt

"""
if __name__ == '__main__':
    with tf.Session() as sess:
        gd = GraspDataset()
        gd.count_success_failure_number()
