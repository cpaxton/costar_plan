#!/usr/local/bin/python
"""
Manage hyperparameter optimization.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

"""

import os
import sys
import copy
import six
import json
import GPy
import GPyOpt
import numpy as np
import tensorflow as tf
import traceback
import keras
import grasp_utilities

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


class HyperparameterOptions(object):

    def __init__(self, verbose=0):
        self.index_dict = {}
        self.search_space = []
        self.verbose = verbose

    def add_param(self, name, domain, domain_type='discrete', enable=True, required=True, default=None):
        """

        # Arguments

            search_space: list of hyperparameter configurations required by BayseanOptimizer
            index_dict: dictionary that will be used to lookup real values
                and types when we get the hyperopt callback with ints or floats
            enable: this parameter will be part of hyperparameter search
            required: this parameter must be passed to the model
            default: default value if required

        """
        if self.search_space is None:
            self.search_space = []
        if self.index_dict is None:
            self.index_dict = {'current_index': 0}
        if 'current_index' not in self.index_dict:
            self.index_dict['current_index'] = 0

        if enable or required:
            param_index = self.index_dict['current_index']
            numerical_domain = domain
            needs_reverse_lookup = False
            lookup_as = 'float'
            # convert string domains to a domain of integer indexes
            if domain_type == 'discrete':
                if isinstance(domain, list) and isinstance(domain[0], str):
                    numerical_domain = [i for i in range(len(domain))]
                    lookup_as = 'str'
                    needs_reverse_lookup = True
                elif isinstance(domain, list) and isinstance(domain[0], bool):
                    numerical_domain = [i for i in range(len(domain))]
                    lookup_as = 'bool'
                    needs_reverse_lookup = True
                elif isinstance(domain, list) and isinstance(domain[0], float):
                    lookup_as = 'float'
                else:
                    lookup_as = 'int'

            opt_dict = {
                'name': name,
                'type': domain_type,
                'domain': numerical_domain}

            if enable:
                self.search_space += [opt_dict]
                # create a second version for us to construct the real function call
                opt_dict = copy.deepcopy(opt_dict)
                opt_dict['lookup_as'] = lookup_as
            else:
                opt_dict['lookup_as'] = None

            opt_dict['enable'] = enable
            opt_dict['required'] = required
            opt_dict['default'] = default
            opt_dict['domain'] = domain
            opt_dict['needs_reverse_lookup'] = needs_reverse_lookup
            self.index_dict[name] = opt_dict
            if enable:
                opt_dict['index'] = param_index
                self.index_dict['current_index'] += 1

    def params_to_args(self, x):
        """ Convert GPyOpt Bayesian Optimizer params back into function call arguments

        Arguments:

            x: the callback parameter of the GPyOpt Bayesian Optimizer
            index_dict: a dictionary with all the information necessary to convert back to function call arguments
        """
        if len(x.shape) == 1:
            # if we get a 1d array convert it to 2d so we are consistent
            x = np.expand_dims(x, axis=0)

        def lookup_as(name, value):
            """ How to lookup internally stored values.
            """
            if name == 'float':
                return float(value)
            elif name == 'int':
                return int(value)
            elif name == 'str':
                return str(value)
            elif name == 'bool':
                return bool(value)
            else:
                raise ValueError('Trying to lookup unsupported type: ' + str(name))

        # x is a funky 2d numpy array, so we convert it back to normal parameters
        kwargs = {}
        if self.verbose > 0:
            print('INDEX DICT: ' + str(self.index_dict))
        for key, opt_dict in six.iteritems(self.index_dict):
            if key == 'current_index':
                continue

            if opt_dict['enable']:
                arg_name = opt_dict['name']
                optimizer_param_column = opt_dict['index']
                if optimizer_param_column > x.shape[-1]:
                    raise ValueError('Attempting to access optimizer_param_column' + str(optimizer_param_column) +
                                     ' outside parameter bounds' + str(x.shape) +
                                     ' of optimizer array with index dict: ' + str(self.index_dict) +
                                     'and array x: ' + str(x))
                param_value = x[:, optimizer_param_column]
                if opt_dict['type'] == 'discrete':
                    # the value is an integer indexing into the lookup dict
                    if opt_dict['needs_reverse_lookup']:
                        domain_index = int(param_value)
                        domain_value = opt_dict['domain'][domain_index]
                        value = lookup_as(opt_dict['lookup_as'], domain_value)
                    else:
                        value = lookup_as(opt_dict['lookup_as'], param_value)

                else:
                    # the value is a param to use directly
                    value = lookup_as(opt_dict['lookup_as'], param_value)

                kwargs[arg_name] = value
            elif opt_dict['required']:
                if self.verbose > 0:
                    print('REQUIRED NAME: ' + str(opt_dict['name']) + ' DEFAULT: ' + str(opt_dict['default']))
                kwargs[opt_dict['name']] = opt_dict['default']
        return kwargs

    def get_domain(self):
        """ Get the hyperparameter search space in the gpyopt domain format.
        """
        return self.search_space

    def save(self, filename):
        """ Save the HyperParameterOptions search space and argument index dictionary to a json file.
        """
        data = {}
        data['search_space'] = self.search_space
        data['index_dict'] = self.index_dict
        with open(filename, 'w') as fp:
            json.dump(data, fp)


def optimize(
        run_training_fn,
        feature_combo_name,
        seed=1,
        verbose=1,
        initial_num_samples=300,
        maximum_hyperopt_steps=100,
        num_cores=15,
        baysean_batch_size=1,
        problem_type=None,
        log_dir='./',
        run_name='',
        param_to_optimize='val_acc',
        maximize=None,
        variable_trainability=False,
        learning_rate_enabled=False,
        min_top_block_filter_multiplier=6,
        batch_size=2,
        hyperoptions=None,
        **kwargs):
    """ Run hyperparameter optimization

    hyperoptions: an instance of thee hyperopt.HyperparameterOptions class,
        default of None will create one automatically

    kwargs: these are passed to the run_training_fn but *not* saved as hyperparameters.
        The current example use case is to disable model checkpointing if it takes too much space
        with the parameter checkpoint=False (assuming the run_training_fn accepts that parameter).
    """
    np.random.seed(seed)

    optimize_loss = False
    if maximize is None:
        if 'loss' in param_to_optimize:
            maximize = False
        elif 'acc' in param_to_optimize:
            maximize = True
            optimize_loss = True
        else:
            raise ValueError(
                'The parameter "maximize" operates with respect to param_to_optimize, but it '
                'is not one of the valid options. Here is what you can do: '
                'If the string "loss" is in param_to_optimize we will minimize by default, '
                'if the string "acc" is in param_to_optimize we will maximize by default, '
                'alternately you can set the maximize flag to True or False and we '
                'will go with your choice.')

    if hyperoptions is None:
        hyperoptions = HyperparameterOptions()
    # Configuring hyperparameters

    if variable_trainability:
        enable_trainability = True
        # Proportion of layer depths that are trainable.
        # set variable_trainability to True to utilizing the proportional version
        # enables training for a number of layers proportional to the total,
        # starting at the top, which is the final layer before output
        hyperoptions.add_param('trainable', (0.0, 1.0), 'continuous',
                               enable=True, required=True, default=0.0)
    else:
        enable_trainability = False
        # The trainable flag refers to the imagenet pretrained network being trainable or not trainable.
        # We are defaulting to a reasonable learning rate found by prior searches and disabling trainability so that
        # we can restrict the search to more model improvements due to the outsized effects of these changes on performance
        # during the random search phase. We plan to run a separate search on enabling trainable models on one of the best models
        # found when trainable is false. This is due to limitations in processing time availability at the time of writing and
        # multi stage training not yet being configurable during hyperopt.
        #
        # 2018-06-10: except for cornell classification, the best models seem to have trainable=True, so we are locking that in.
        hyperoptions.add_param('trainable', [True, False], enable=enable_trainability, required=True, default=True)

    # Learning rates are exponential so we take a uniform random
    # input and map it from 1 to 3e-5 on an exponential scale.
    # with a base of 0.9.
    # Therefore the value 50 is 0.9^50 == 0.005 (approx).
    #
    # 2018-06-10: We vary the learning rate search space depending on if trainability is variable or not.
    if enable_trainability is False:
        # learning rates from 1 to 0.01
        lr_range = (0.0, 40.0)
    else:
        # learning rates from 1 to about 1e-5
        lr_range = (0.0, 100)
    hyperoptions.add_param('learning_rate', lr_range, 'continuous',
                           enable=learning_rate_enabled, required=True, default=1.0)
    # disabled dropout rate because in one epoch tests a dropout rate of 0 allows exceptionally fast learning.
    #
    # 2018-06-10: Ran a separate search for the best dropout rate after finding a good model, and 0.2 seems to generally be a decent option.
    hyperoptions.add_param('dropout_rate', [0.0, 0.125, 0.2, 0.25, 0.5, 0.75],
                           enable=False, required=True, default=0.2)

    if problem_type == 'grasp_regression':
        # Right now only grasp_regression can configure the loss function
        # Make sure you don't optimize loss when it is being used for your objective function!
        if optimize_loss:
            hyperoptions.add_param('loss', ['mse', 'mae', 'logcosh'], default='mae', enable=False)
    else:
        # There is no vector branch for grasp regression so we only add it in the other cases
        # Handle motion command inputs and search for the best configuration
        hyperoptions.add_param('vector_dense_filters', [2**x for x in range(min_top_block_filter_multiplier - 1, 14)])
        hyperoptions.add_param('vector_branch_num_layers', [x for x in range(0, 5)])
        hyperoptions.add_param('vector_model_name', ['dense', 'dense_block'])
        # Disabled, see hidden_activation instead of vector_hidden_activation.
        # hyperoptions.add_param('vector_hidden_activation', ['linear', 'relu', 'elu'],
        #                        default='linear', enable=True, required=True)
        hyperoptions.add_param('vector_normalization', ['none', 'batch_norm', 'group_norm'],
                               default='batch_norm', enable=True, required=True)

    # enable this if you're search for grasp classifications
    hyperoptions.add_param('feature_combo_name', ['image_preprocessed_width_1', 'image_preprocessed_sin_cos_width_3'],
                           enable=False, required=True, default=feature_combo_name)
    # other supported options: 'densenet', 'nasnet_mobile', 'resnet', 'inception_resnet_v2', 'mobilenet_v2', 'nasnet_large', 'vgg19',
    # TODO(ahundt) Debug 'mobilenet_v2' which might not be working correctly, it has been removed for now.
    # leaving out 'nasnet_large' and 'inception_resnet_v2' for now because they're
    # very large networks that don't train quickly, and may not currently be fast enough for a robot.
    # Please note that inception_resnet_v2 did quite well in both the cornell dataset and the costar dataset!
    hyperoptions.add_param('image_model_name', ['vgg', 'nasnet_mobile'],
                           enable=False, required=True, default='nasnet_mobile')
    # TODO(ahundt) map [0] to the None option for trunk_filters we need an option to automatically match the input data's filter count
    hyperoptions.add_param('trunk_filters', [2**x for x in range(min_top_block_filter_multiplier - 1, 12)])
    hyperoptions.add_param('trunk_layers', [x for x in range(0, 11)])
    # Choose the model for the trunk, options are 'vgg_conv_block', 'dense_block', 'nasnet_normal_a_cell', 'resnet_conv_identity_block'
    # an additional option is 'resnet_conv_identity_block', but it never really did well.
    # dense_block does really well for the cornell dataset, vgg_conv_block for costar stacking dataset
    hyperoptions.add_param('trunk_model_name', ['vgg_conv_block', 'dense_block', 'nasnet_normal_a_cell'],
                           default='nasnet_normal_a_cell', enable=True, required=True)
    # Enable or disable coordconv
    # https://eng.uber.com/coordconv/
    # https://github.com/titu1994/keras-coordconv
    # TODO(ahundt) fix bugs in , 'coord_conv_img'
    # valid options include ['none', 'coord_conv_trunk', 'coord_conv_img']
    # hyperoptions.add_param('coordinate_data', ['none', 'coord_conv_trunk'],
    hyperoptions.add_param('coordinate_data', ['none', 'coord_conv_trunk', 'coord_conv_img'],
                           default='none', enable=True, required=True)
    # Disabled, see hidden_activation instead of trunk_hidden_activation.
    # trunk hidden activation currently only applies to the vgg case
    # hyperoptions.add_param('trunk_hidden_activation', ['relu', 'elu', 'linear'],
    #                        default='relu', enable=True, required=True)
    # 'trunk_normalization' currently only applies in vgg case and after the first conv in the nasnet case
    hyperoptions.add_param('trunk_normalization', ['none', 'batch_norm', 'group_norm'],
                           default='batch_norm', enable=True, required=True)
    hyperoptions.add_param('top_block_filters', [2**x for x in range(min_top_block_filter_multiplier, 14)])
    # number of dense layers before the final dense layer that performs classification in the top block
    hyperoptions.add_param('top_block_dense_layers', [0, 1, 2, 3, 4])
    # Disabled, see hidden_activation instead of top_block_hidden_activation.
    # hyperoptions.add_param('top_block_hidden_activation', ['relu', 'elu', 'linear'],
    #                        default='relu', enable=True, required=True)
    hyperoptions.add_param('batch_size', [2**x for x in range(2, 4)],
                           enable=False, required=True, default=batch_size)
    # The appropriate preprocessing mode must be chosen for each model.
    # This should now be done correctly in cornell_grasp_train.py.
    hyperoptions.add_param('preprocessing_mode', ['tf', 'caffe', 'torch'],
                           enable=False, required=False, default='tf')

    # Hidden activation to use for all hidden layers,
    # note that for the trunk and image portion of models
    # this will only apply under limited circumstances
    # due to the current choose_hypertree_model implementation.
    # ['linear', 'relu', 'elu']
    hyperoptions.add_param('hidden_activation', ['relu', 'elu'],
                           default='relu', enable=True, required=True)
    # deep learning algorithms don't give exact results
    algorithm_gives_exact_results = False
    # how many optimization steps to take after the initial sampling

    total_max_steps = initial_num_samples + maximum_hyperopt_steps
    # defining a temporary variable scope for the callbacks
    class ProgUpdate():
        hyperopt_current_update = 0
        progbar = tqdm(desc='hyperopt', total=total_max_steps)

    def train_callback(x):
        # x is a funky 2d numpy array, so we convert it back to normal parameters
        training_arguments = hyperoptions.params_to_args(x)

        if learning_rate_enabled:
            # Learning rates are exponential so we take a uniform random
            # input and map it from 1 to 3e-5 on an exponential scale.
            training_arguments['learning_rate'] = 0.9 ** training_arguments['learning_rate']

        if verbose:
            # update counts by 1 each step
            ProgUpdate.progbar.update()
            ProgUpdate.progbar.write('Training with hyperparams: \n' + str(training_arguments))
        ProgUpdate.hyperopt_current_update += 1

        history = None

        # hyperparams need to be kept separate from some extra training arguments
        # not relevant to hyperparameter optimization
        hyper_params = training_arguments
        training_arguments.update(kwargs)

        try:
            # call the function that performs actual training and returns a history object
            history = run_training_fn(
                hyperparams=hyper_params,
                **training_arguments)
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
        except (ValueError, tf.errors.FailedPreconditionError, tf.errors.OpError) as exception:
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
        except KeyboardInterrupt as e:
            print('Evaluation of this model canceled based on a user request. '
                  'We will continue to next stage and return infinity loss for the canceled model.')
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb

        # TODO(ahundt) consider shutting down dataset generators and clearing the session when there is an exception
        # https://github.com/tensorflow/tensorflow/issues/4735#issuecomment-363748412
        keras.backend.clear_session()

        if history is not None:
            # hyperopt seems to be done on val_loss
            # may try 1-val_acc sometime (since the hyperopt minimizes)
            if param_to_optimize in history.history:
                # Take the best performance regardless of the epoch
                if maximize:
                    loss = np.max(history.history[param_to_optimize])
                else:
                    loss = np.min(history.history[param_to_optimize])
            else:
                raise ValueError('A hyperopt step completed, but the parameter '
                                 'being optimized over is %s and it '
                                 'was missing from the history'
                                 'so hyperopt must exit. Here are the contents '
                                 'of the history.history dictionary:\n\n %s' %
                                 (param_to_optimize, str(history.history)))
            if verbose > 0:
                if 'val_binary_accuracy' in history.history:
                    acc = np.max(history.history['val_binary_accuracy'])
                    ProgUpdate.progbar.write('val_binary_accuracy: ' + str(acc))
        else:
            # we probably hit an exception so consider this infinite loss
            loss = float('inf')
            if maximize:
                # use negative infinity if we are maximizing!
                loss = -loss

        return loss

    log_run_prefix = os.path.join(log_dir, run_name)
    grasp_utilities.mkdir_p(log_run_prefix)
    print('Hyperopt log run results prefix directory: ' + str(log_run_prefix))
    hyperoptions.save(log_run_prefix + '_hyperoptions.json')

    # model_type chosen based on https://github.com/SheffieldML/GPyOpt/issues/152
    # also see https://github.com/SheffieldML/GPyOpt/issues/107
    # Previous choice before 2018-04-06 was as follows, but became too slow past 300 samples:
    # model_type='GP_MCMC',
    # acquisition_type='EI_MCMC',  # EI

    bayesian_optimization = GPyOpt.methods.BayesianOptimization(
        f=train_callback,  # function to optimize
        domain=hyperoptions.get_domain(),  # where are we going to search
        initial_design_numdata=initial_num_samples,
        model_type='sparseGP',
        acquisition_type='EI',  # Expected Improvement
        evaluator_type="predictive",
        batch_size=baysean_batch_size,
        num_cores=num_cores,
        exact_feval=algorithm_gives_exact_results,
        report_file=log_run_prefix + '_bayesian_optimization_report.txt',
        models_file=log_run_prefix + '_bayesian_optimization_models.txt',
        evaluations_file=log_run_prefix + '_bayesian_optimization_evaluations.txt',
        maximize=maximize)

    # run the optimization, this will take a long time!
    bayesian_optimization.run_optimization(max_iter=maximum_hyperopt_steps)
    x_best = bayesian_optimization.x_opt
    # myBopt.X[np.argmin(myBopt.Y)]
    best_hyperparams = hyperoptions.params_to_args(x_best)
    result_file = os.path.join(log_dir, run_name + '_optimized_hyperparams.json')
    with open(result_file, 'w') as fp:
        json.dump(best_hyperparams, fp)
    print('Hyperparameter Optimization final best result:\n' + str(best_hyperparams))
    print('Optimized ' + param_to_optimize + ': {0}'.format(bayesian_optimization.fx_opt))
    print('Hyperopt log run results prefix directory: ' + str(log_run_prefix))

    bayesian_optimization.plot_convergence(log_run_prefix + '_bayesian_optimization_convergence_plot.png')
    bayesian_optimization.plot_acquisition(log_run_prefix + '_bayesian_optimization_acquisition_plot.png')
    return best_hyperparams
