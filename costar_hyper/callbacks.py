
import sys
from timeit import default_timer
import numpy as np
import keras


class EvaluateInputTensor(keras.callbacks.Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`
    Instead, create a second model configured with input tensors for validation
    and add it to the `EvaluateInputTensor` callback to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    #TODO(ahundt) replace when https://github.com/keras-team/keras/pull/9105 is resolved

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


class EvaluateInputGenerator(keras.callbacks.Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`
    Instead, create a second model configured with input tensors for validation
    and add it to the `EvaluateInputTensor` callback to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    #TODO(ahundt) replace when https://github.com/keras-team/keras/pull/9105 is available

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, generator, steps, metrics_prefix='test', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputGenerator, self).__init__()
        self.generator = generator
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        results = self.model.evaluate_generator(self.generator, steps=int(self.num_steps))
        metrics_str = '\n'
        for result, name in zip(results, self.model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


class PrintLogsCallback(keras.callbacks.Callback):
    """ Prints the log data at the end of each epoch.
    """
    def on_epoch_end(self, epoch, logs={}):
        print('')
        print('logs: ' + str(logs))


class FineTuningCallback(keras.callbacks.Callback):
    """ Switch to fine tuning mode at the specified epoch

    Unlocks layers to make them trainable and resets the learning rate
    to a new initial value.

    # TODO(ahundt) update when https://github.com/keras-team/keras/issues/9477 is resolved.

    # Arguments

        epoch: The epoch at which to enable fine tuning
        layers: Integer for the min index in model.layers which will
            have trainable set to True, along with all layers after it.
        learning_rate: The new fine tuning learning rate to reset to.
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, epoch=100, layers=0, learning_rate=0.0001, verbose=1, output_file=sys.stderr):
        super(FineTuningCallback, self).__init__()
        self.epoch = epoch
        self.layers = layers
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.output_file = output_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)
        # fine_tuning = epoch >= self.epoch
        # logs['fine_tuning'] = fine_tuning
        if epoch == self.epoch:
            if self.verbose > 0:
                print('\n\n--------------------------------------------------------\n'
                      'Epoch %05d Fine tuning initialized with a new '
                      'learning rate of %s.' % (epoch + 1, self.learning_rate))
            for layer in self.model.layers[self.layers:]:
                layer.trainable = True
            self.model.compile(self.model.optimizer, self.model.loss, self.model.metrics)
            if self.verbose > 1:
                print('\n\nepoch:' + str(epoch) + ' self.epoch: ' + str(self.epoch) + ' lr: ' + str(logs['lr']) +
                      ' self.learning_rate: ' + str(self.learning_rate) + ' float(K.get_value(self.model.optimizer.lr)): ' +
                      str(float(keras.backend.get_value(self.model.optimizer.lr))) + ' what is going on?0\n\n')
            keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)

        if self.verbose > 1:
            print('\n\nepoch:' + str(epoch) + ' self.epoch: ' + str(self.epoch) + ' lr: ' + str(logs['lr']) +
                  ' self.learning_rate: ' + str(self.learning_rate) + ' float(K.get_value(self.model.optimizer.lr)): ' +
                  str(float(keras.backend.get_value(self.model.optimizer.lr))) + ' what is going on?1\n\n')


class SlowModelStopping(keras.callbacks.Callback):
    """ Stop a model from training if it runs too slowly.
    """

    def __init__(
            self,
            min_batch_patience=30,
            max_batch_time_seconds=1,
            max_epoch_time_seconds=7000,
            max_epoch_to_check_batch_time=2,
            verbose=0):
        self._min_batch_patience = min_batch_patience
        self._max_batch_time_seconds = max_batch_time_seconds
        self._max_epoch_time_seconds = max_epoch_time_seconds
        self._epoch_elapsed = 0
        self.verbose = verbose
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.max_epoch_to_check_batch_time = max_epoch_to_check_batch_time
        # start timers on init just in case of
        # atypical situations like 0 training steps
        self._epoch_start = default_timer()
        self._batch_start = default_timer()
        self._batch_elapsed = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = default_timer()
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self._batch_start = default_timer()
        self._batch_elapsed = []

    def on_batch_end(self, batch, logs=None):
        batch_elapsed = default_timer() - self._batch_start
        self._batch_elapsed.append(batch_elapsed)
        if(self._min_batch_patience is not None and
           self._max_batch_time_seconds is not None and
           self.current_epoch < self.max_epoch_to_check_batch_time and
           batch > self._min_batch_patience and
           batch < self._min_batch_patience + 10):

            mean_batch = np.mean(self._batch_elapsed)
            if mean_batch > self._max_batch_time_seconds:
                raise ValueError('Batches took too long: ' + str(mean_batch) +
                                 ' vs max allowed: ' + str(self._max_batch_time_seconds))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        # Only log results if there are more than 0 train steps
        if self._batch_elapsed:
            self._epoch_elapsed = default_timer() - self._epoch_start
            logs['epoch_elapsed'] = np.array(self._epoch_elapsed)
            mean_batch = np.mean(self._batch_elapsed)
            # keras will check for value.item(), so don't cast this to a float, leave it as a numpy array
            logs['mean_batch_elapsed'] = mean_batch
            if mean_batch > self._max_batch_time_seconds:
                self.model.stop_training = True
                self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: slow model, stopping early' % (self.stopped_epoch + 1))


class InaccurateModelStopping(keras.callbacks.Callback):
    """ Stop a model from training if it ends up on perverse solutions like always 0 or 1.
    """

    def __init__(self, min_batch_patience=300, min_pred=0.05, max_pred=0.95, metric='mean_pred', verbose=0):
        self._min_batch_patience = min_batch_patience
        self._max_pred = max_pred
        self._min_pred = min_pred
        self._epoch_elapsed = 0
        self.verbose = verbose
        self.stopped_epoch = 0
        self.metric = metric
        self.metric_values = []

    def on_batch_end(self, batch, logs=None):
        if self.metric in logs:
            value = logs[self.metric]
            self.metric_values += [value]
            if(self._min_batch_patience is not None and
               self._max_pred is not None and
               batch > self._min_batch_patience):
                value = np.mean(self.metric_values)

                if value > self._max_pred or value < self._min_pred:
                    raise ValueError(str(self.metric) + ' was inaccurate: ' + str(value) +
                                     ' vs allowed range [ ' + str(self._min_pred) +
                                     ', ' + str(self._max_pred) + ']')

    def on_epoch_end(self, epoch, logs=None):
        self.metric_values = []
