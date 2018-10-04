#!/usr/local/bin/python
"""
Generate a training, validaton, and test list of files from a directory.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

"""

import os
import six
import glob
import numpy as np
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


flags.DEFINE_string(
    'data_dir',
    '~/.keras/datasets/costar_plush_block_stacking_dataset_v0.4/',
    'Directory for collecting the dataset files'
)

flags.DEFINE_string(
    'glob_filename',
    '*success*.h5f',
    'File path to glob for dataset files.'
)

flags.DEFINE_boolean(
    'ascending',
    False,
    'Sort in ascending (1 to 100) or descending (100 to 1) order.'
)

flags.DEFINE_string(
    'save_txt_prefix',
    'costar_block_stacking_v0.4_success_only_',
    'Prefix with which to to save the sorted output txt file with the train test and validation sets'
)

flags.DEFINE_string(
    'save_dir',
    None,
    'Where to save the txt files, defaults to data_dir'
)

flags.DEFINE_integer(
    'seed',
    0,
    'numpy seed for shuffling the data, so you can generate this list in a repeatable way'
)

flags.DEFINE_integer(
    'val_test_size',
    64,
    'the size of the validation and test subsets'
)

FLAGS = flags.FLAGS


def main(_):
    file_names = gfile.Glob(os.path.join(os.path.expanduser(FLAGS.data_dir), FLAGS.glob_filename))

    # Shuffle the list of files
    np.random.seed(FLAGS.seed)
    np.random.shuffle(file_names)
    val_test_size = FLAGS.val_test_size

    # Separate the new train/val/test split
    train_data = file_names[val_test_size*2:]
    validation_data = file_names[val_test_size:val_test_size*2]
    test_data = file_names[:val_test_size]

    # save the lists out to disk
    if FLAGS.save_dir is None:
        FLAGS.save_dir = FLAGS.data_dir

    save_dir = os.path.expanduser(FLAGS.save_dir)

    train_filename = os.path.join(save_dir, FLAGS.save_txt_prefix + 'train_files.txt')
    val_filename = os.path.join(save_dir, FLAGS.save_txt_prefix + 'val_files.txt')
    test_filename = os.path.join(save_dir, FLAGS.save_txt_prefix + 'test_files.txt')
    with open(train_filename, mode='w') as set_file:
        set_file.write('\n'.join(train_data))
    with open(val_filename, mode='w') as set_file:
        set_file.write('\n'.join(validation_data))
    with open(test_filename, mode='w') as set_file:
        set_file.write('\n'.join(test_data))

    print('Processing complete. Results saved to files:\n' + train_filename + '\n' + val_filename + '\n' + test_filename)

if __name__ == '__main__':
    app.run(main=main)
