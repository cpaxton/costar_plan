import sys
import re
import numpy as np
import os
import json
import datetime
import errno
import json
import six


class NumpyEncoder(json.JSONEncoder):
    """ json encoder for numpy types

    source: https://stackoverflow.com/a/49677241/99379
    """
    def default(self, obj):
        if isinstance(obj,
            (np.int_, np.intc, np.intp, np.int8,
             np.int16, np.int32, np.int64, np.uint8,
             np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj,
           (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rotate(data, shift=1):
    """ Rotates indices up 1 for a list or numpy array.

    For example, [0, 1, 2] will become [1, 2, 0] and
    [4, 3, 1, 0] will become [3, 1, 0, 4].
    The contents of index 0 becomes the contents of index 1,
    and the final entry will contain the original contents of index 0.
    Always operates on axis 0.
    """
    if isinstance(data, list):
        return data[shift:] + data[:shift]
    else:
        return np.roll(data, shift, axis=0)


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command

    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def load_hyperparams_json(hyperparams_file, fine_tuning=False, learning_rate=None, feature_combo_name=None):
    """ Load hyperparameters from a json file

    # Returns

    Hyperparams
    """
    kwargs = {}
    hyperparams = None
    if hyperparams_file is not None and hyperparams_file:
        with open(hyperparams_file, mode='r') as hyperparams:
            kwargs = json.load(hyperparams)
            hyperparams = kwargs
    if fine_tuning:
        kwargs['trainable'] = True
        kwargs['learning_rate'] = learning_rate
        # TODO(ahundt) should we actually write the fine tuning settings out to the hyperparams log?
        # hyperparams = kwargs

    if (kwargs is not None and feature_combo_name is not None and
            'feature_combo_name' in kwargs and
            kwargs['feature_combo_name'] != feature_combo_name):
        print('Warning: overriding old hyperparam feature_combo_name: %s'
              ' with new feature_combo_name: %s. This means the network '
              'structure and inputs will be different from what is defined '
              'in the hyperparams file: %s' %
              (kwargs['feature_combo_name'], feature_combo_name, hyperparams_file))
        kwargs.pop('feature_combo_name')
        if 'feature_combo_name' in hyperparams:
            hyperparams.pop('feature_combo_name')
    return kwargs


def is_sequence(arg):
    """Returns true if arg is a list or another Python Sequence, and false otherwise.

        source: https://stackoverflow.com/a/17148334/99379
    """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


def find_best_weights(fold_log_dir, match_string='', verbose=0, out_file=sys.stdout):
    """ Find the best weights file with val_*0.xxx out in a directory
    """
    # Now we have to load the best model
    # '200_epoch_real_run' is for backwards compatibility before
    # the fold nums were put into each fold's log_dir and run_name.
    directory_listing = os.listdir(fold_log_dir)
    fold_checkpoint_files = []
    for name in directory_listing:
        name = os.path.join(fold_log_dir, name)
        if not os.path.isdir(name) and '.h5' in name:
            if '200_epoch_real_run' in name or match_string in name:
                fold_checkpoint_files += [name]

    # check the filenames for the highest val score
    fold_checkpoint_file = None
    best_val = 0.0
    for filename in fold_checkpoint_files:
        if 'val_' in filename:
            # pull out all the floating point numbers
            # source: https://stackoverflow.com/a/4703409/99379
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", filename)
            if len(nums) > 0:
                # don't forget about the .h5 at the end...
                cur_num = np.abs(float(nums[-2]))
                if verbose > 0:
                    out_file.write('old best ' + str(best_val) + ' current ' + str(cur_num))
                if cur_num > best_val:
                    if verbose > 0:
                        out_file.write('new best: ' + str(cur_num) + ' file: ' + filename)
                    best_val = cur_num
                    fold_checkpoint_file = filename

    if fold_checkpoint_file is None:
        raise ValueError('\n\nSomething went wrong when looking for model checkpoints, '
                         'you need to take a look at model_predict_k_fold() '
                         'in cornell_grasp_train.py. Here are the '
                         'model checkpoint files we were looking at: \n\n' +
                         str(fold_checkpoint_files))
    return fold_checkpoint_file


def make_model_description(run_name, model_name, hyperparams, dataset_names_str, label_features=None):
    """ Put several strings together for a model description used in file and folder names
    """
    model_description = ''
    if run_name:
        model_description += run_name + '-'
    if model_name:
        model_description += model_name + '-'

    # if hyperparams is not None:
    #     if 'image_model_name' in hyperparams:
    #         model_description += '_img_' + hyperparams['image_model_name']
    #     if 'vector_model_name' in hyperparams:
    #         model_description += '_vec_' + hyperparams['vector_model_name']
    #     if 'trunk_model_name' in hyperparams:
    #         model_description += '_trunk_' + hyperparams['trunk_model_name']
    ########################################################
    # End tensor configuration, begin model configuration and training
    model_description += '-dataset_' + dataset_names_str

    if label_features is not None:
        model_description += '-' + label_features

    run_name = timeStamped(model_description)
    return run_name


def multi_run_histories_summary(
        run_histories,
        save_filename=None,
        metrics='val_binary_accuracy',
        description_prefix='k_fold_average_',
        results_prefix='k_fold_results',
        multi_history_metrics='mean',
        verbose=1):
    """ Find the k_fold average of the best model weights on each fold, and save the results.

    This can be used to summarize multiple runs, be they on different models or the same model.

    Please note that currently this should only be utilized with classification models,
    or regression models with absolute thresholds.
    it will not calculated grasp_jaccard regression models' scores correctly.

    # Arguments

    run_histories: A dictionary from training run description strings to keras history objects.
    multi_history_metric: 'mean', 'min', 'max',
        used to summarize the data from multiple training runs.

    # Returns

    results disctionary including the max value of metric for each fold,
    plus the average of all folds in a dictionary.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(multi_history_metrics, str):
        multi_history_metrics = [multi_history_metrics]
    results = {}
    for metric, multi_history_metric in zip(metrics, multi_history_metrics):
        best_metric_scores = []
        for history_description, history_object in six.iteritems(run_histories):
            if 'loss' in metric or 'error' in metric:
                best_score = np.min(history_object.history[metric])
                results[history_description + '_min_' + metric] = best_score
            else:
                best_score = np.max(history_object.history[metric])
                results[history_description + '_max_' + metric] = best_score
            best_metric_scores += [best_score]
        if multi_history_metric == 'mean' or multi_history_metric == 'average':
            k_fold_average = np.mean(best_metric_scores)
        elif multi_history_metric == 'min':
            k_fold_average = np.min(best_metric_scores)
        elif multi_history_metric == 'max':
            k_fold_average = np.max(best_metric_scores)
        else:
            raise ValueError(
                'multi_run_histories_summary(): Unsupported multi_history_metric: ' +
                str(multi_history_metric))
        result_key = description_prefix + '_' + multi_history_metric + '_' + metric
        results[result_key] = k_fold_average

    if verbose:
        print(str(results_prefix) + ':\n ' + str(results))

    if save_filename is not None:
        with open(save_filename, 'w') as fp:
            # save out all kfold params so they can be reloaded in the future
            json.dump(results, fp)
    return results
