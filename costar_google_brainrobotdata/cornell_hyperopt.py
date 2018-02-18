import os
import sys
import copy
import six
import GPy
import GPyOpt
import numpy as np
import cornell_grasp_train
import cornell_grasp_dataset_reader
import tensorflow as tf
import traceback
import keras
from tensorflow.python.platform import flags

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

FLAGS = flags.FLAGS


def add_param(name, domain, domain_type='discrete', search_space=None, index_dict=None, enable=True, required=True, default=None, verbose=1):
    """

    # Arguments

        search_space: list of hyperparameter configurations required by BayseanOptimizer
        index_dict: dictionary that will be used to lookup real values
            and types when we get the hyperopt callback with ints or floats
        enable: this parameter will be part of hyperparameter search
        required: this parameter must be passed to the model
        default: default value if required

    """
    if search_space is None:
        search_space = []
    if index_dict is None:
        index_dict = {'current_index': 0}
    if 'current_index' not in index_dict:
        index_dict['current_index'] = 0

    if enable:
        param_index = index_dict['current_index']
        numerical_domain = domain
        needs_reverse_lookup = False
        lookup_as = float
        # convert string domains to a domain of integer indexes
        if domain_type == 'discrete':
            if isinstance(domain, list) and isinstance(domain[0], str):
                numerical_domain = [i for i in range(len(domain))]
                lookup_as = str
                needs_reverse_lookup = True
            elif isinstance(domain, list) and isinstance(domain[0], bool):
                numerical_domain = [i for i in range(len(domain))]
                lookup_as = bool
                needs_reverse_lookup = True
            elif isinstance(domain, list) and isinstance(domain[0], float):
                lookup_as = float
            else:
                lookup_as = int

        opt_dict = {
            'name': name,
            'type': domain_type,
            'domain': numerical_domain}

        if enable:
            search_space += [opt_dict]
            # create a second version for us to construct the real function call
            opt_dict = copy.deepcopy(opt_dict)
            opt_dict['lookup_as'] = lookup_as
        else:
            opt_dict['lookup_as'] = None

        opt_dict['enable'] = enable
        opt_dict['required'] = required
        opt_dict['default'] = default
        opt_dict['index'] = param_index
        opt_dict['domain'] = domain
        opt_dict['needs_reverse_lookup'] = needs_reverse_lookup
        index_dict[name] = opt_dict
        index_dict['current_index'] += 1
    return search_space, index_dict


def params_to_args(x, index_dict):
    """ Convert GPyOpt Bayesian Optimizer params back into function call arguments

    Arguments:

        x: the callback parameter of the GPyOpt Bayesian Optimizer
        index_dict: a dictionary with all the information necessary to convert back to function call arguments
    """
    # x is a funky 2d numpy array, so we convert it back to normal parameters
    kwargs = {}
    for key, opt_dict in six.iteritems(index_dict):
        if key == 'current_index':
            continue

        if opt_dict['enable']:
            arg_name = opt_dict['name']
            optimizer_param_column = opt_dict['index']
            if optimizer_param_column > x.shape[-1]:
                raise ValueError('Attempting to access optimizer_param_column' + str(optimizer_param_column) +
                                 ' outside parameter bounds' + str(x.shape) +
                                 ' of optimizer array with index dict: ' + str(index_dict) +
                                 'and array x: ' + str(x))
            param_value = x[:, optimizer_param_column]
            if opt_dict['type'] == 'discrete':
                # the value is an integer indexing into the lookup dict
                if opt_dict['needs_reverse_lookup']:
                    domain_index = int(param_value)
                    domain_value = opt_dict['domain'][domain_index]
                    value = opt_dict['lookup_as'](domain_value)
                else:
                    value = opt_dict['lookup_as'](param_value)

            else:
                # the value is a param to use directly
                value = opt_dict['lookup_as'](param_value)

            kwargs[arg_name] = value
        elif opt_dict['required']:
            kwargs[opt_dict['name']] = opt_dict['default']
    return kwargs


def optimize(train_file=None, validation_file=None, seed=1, verbose=1):
    np.random.seed(seed)
    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works

    # it seems loading the dataset in advance leads
    # to longer and longer loading times as previous
    # models are not cleared. Extra time is required to
    # load the dataset each time around, however, so
    # there is a tradeoff and we need to figure out
    # what works best
    #
    # Note: feature_combo_name only works
    # when load_dataset_in_advance is False
    load_dataset_in_advance = False
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    batch_size = FLAGS.batch_size
    feature_combo_name = 'image_preprocessed_height_1'
    top = 'classification'
    train_data = None
    validation_data = None

    # Configuring hyperparameters
    search_space, index_dict = add_param('learning_rate', (0.0001, 0.1), 'continuous')
    # disabled dropout rate because in one epoch tests a dropout rate of 0 allows exceptionally fast learning.
    # TODO(ahundt) run a separate search for the best dropout rate after finding a good model
    search_space, index_dict = add_param('dropout_rate', [0.0, 0.125, 0.2, 0.25, 0.5, 0.75], search_space=search_space, index_dict=index_dict,
                                         enable=False, required=True, default=0.25)
    search_space, index_dict = add_param('vector_dense_filters', [2**x for x in range(6, 13)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('vector_branch_num_layers', [x for x in range(0, 5)], search_space=search_space, index_dict=index_dict)
    # leaving out 'resnet' for now, it is causing too many crashes, and nasnet_large because it needs different input dimensions.
    search_space, index_dict = add_param('image_model_name', ['vgg', 'densenet', 'nasnet_mobile'], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('vector_model_name', ['dense', 'dense_block'], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('trainable', [True, False], search_space=search_space, index_dict=index_dict,
                                         enable=True)
    # TODO(ahundt) add a None option for trunk_filters, [None] + [2**x for x in range(5, 12)], because it will automatically match input data's filter count
    search_space, index_dict = add_param('trunk_filters', [2**x for x in range(5, 12)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('trunk_layers', [x for x in range(0, 8)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('top_block_filters', [2**x for x in range(5, 12)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('batch_size', [2**x for x in range(2, 4)], search_space=search_space, index_dict=index_dict,
                                         enable=False, required=True, default=batch_size)
    search_space, index_dict = add_param('feature_combo_name', ['image_preprocessed_height_1', 'image_preprocessed_sin_cos_height_3'],
                                         search_space=search_space, index_dict=index_dict,
                                         enable=True, required=True, default=feature_combo_name)

    # Load dataset if that's done only once in advance
    if load_dataset_in_advance:
        train_file = None
        validation_file = None
        # we load the dataset in advance here because it takes a long time!
        if train_file is None:
            train_file = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
        if validation_file is None:
            validation_file = os.path.join(FLAGS.data_dir, FLAGS.evaluate_filename)

        [samples_in_val_dataset, steps_per_epoch_train,
         steps_in_val_dataset, val_batch_size] = cornell_grasp_train.epoch_params(batch_size)

        [image_shapes, vector_shapes, data_features, model_name,
         monitor_loss_name, label_features, monitor_metric_name,
         loss, metrics] = cornell_grasp_train.choose_features_and_metrics(feature_combo_name, top)

        train_data, validation_data = cornell_grasp_train.load_dataset(
            validation_file, label_features, data_features,
            samples_in_val_dataset, train_file, batch_size,
            val_batch_size)

    # number of samples to take before trying hyperopt
    initial_num_samples = 100
    num_cores = 15
    baysean_batch_size = 1
    # deep learning algorithms don't give exact results
    algorithm_gives_exact_results = False
    # how many optimization steps to take after the initial sampling
    maximum_hyperopt_steps = 200
    total_max_steps = initial_num_samples + maximum_hyperopt_steps

    # defining a temporary variable scope for the callbacks
    class ProgUpdate():
        hyperopt_current_update = 0
        progbar = tqdm(desc='hyperopt', total=total_max_steps)

    def train_callback(x):
        # x is a funky 2d numpy array, so we convert it back to normal parameters
        kwargs = params_to_args(x, index_dict)

        if verbose:
            # update counts by 1 each step
            ProgUpdate.progbar.update()
            ProgUpdate.progbar.write('Training with hyperparams: \n' + str(kwargs))
        ProgUpdate.hyperopt_current_update += 1

        history = None

        try:
            history = cornell_grasp_train.run_training(
                train_data=train_data,
                validation_data=validation_data,
                hyperparams=kwargs,
                **kwargs)
        except tf.errors.ResourceExhaustedError as exception:
            print('Hyperparams caused algorithm to run out of resources, '
                  'will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you might set more memory sensitive hyperparam ranges,'
                  'or add constraints to your hyperparam search so it does not choose'
                  'huge values for all the parameters at once'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
        except ValueError as exception:
            print('Hyperparams encountered a model that failed with an invalid combination of values, '
                  'we will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you will need to debug your model w.r.t. '
                  'the current hyperparam choice.'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb

        # TODO(ahundt) consider shutting down dataset generators and clearing the session when there is an exception
        # https://github.com/tensorflow/tensorflow/issues/4735#issuecomment-363748412
        if not load_dataset_in_advance:
            keras.backend.clear_session()

        if history is not None:
            # hyperopt seems to be done on val_loss
            # may try 1-val_acc sometime (since the hyperopt minimizes)
            loss = history.history['val_loss'][-1]
            if verbose > 0:
                if 'val_binary_accuracy' in history.history:
                    acc = history.history['val_binary_accuracy'][-1]
                    ProgUpdate.progbar.write('val_binary_accuracy: ' + str(acc))
        else:
            # we probably hit an exception so consider this infinite loss
            loss = float('inf')

        return loss

    hyperopt = GPyOpt.methods.BayesianOptimization(
        f=train_callback,  # function to optimize
        domain=search_space,  # where are we going to search
        initial_design_numdata=initial_num_samples,
        model_type='GP_MCMC',
        acquisition_type='EI_MCMC',  # EI
        evaluator_type="predictive",  # Expected Improvement
        batch_size=baysean_batch_size,
        num_cores=num_cores,
        exact_feval=algorithm_gives_exact_results)

    # run the optimization, this will take a long time!
    hyperopt.run_optimization(max_iter=maximum_hyperopt_steps)
    x_best = hyperopt.x_opt
    # myBopt.X[np.argmin(myBopt.Y)]
    print('Hyperparameter Optimization final best result:\n' + str(params_to_args(x_best, index_dict)))
    print("Optimized loss: {0}".format(hyperopt.fx_opt))

    hyperopt.plot_convergence()
    hyperopt.plot_acquisition()
    ProgUpdate.progbar.close()


def main(_):
    optimize()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
