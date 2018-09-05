"""Override some keras functionality so you can get fetches back from keras.predict.

Warning! This file is a super hacky workaround to get fetches back when predict()
gets called see https://github.com/keras-team/keras/pull/9121 for details.
This file is adapted from https://github.com/keras-team/keras. Usage:

    model._make_predict_function = keras_workaround._make_predict_function_get_fetches

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0

"""
import tensorflow as tf
import keras
import keras.backend as K
from keras.backend.tensorflow_backend import *

# Function adapted from K.backend.tensorflow_backend
# periodically check keras itself for changes to these classes and functions


class FunctionGetFetches(object):
    """Runs a computation graph.

    It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
    In particular additional operations via `fetches` argument and additional
    tensor substitutions via `feed_dict` arguments. Note that given
    substitutions are merged with substitutions from `inputs`. Even though
    `feed_dict` is passed once in the constructor (called in `model.compile()`)
    we can modify the values in the dictionary. Through this feed_dict we can
    provide additional substitutions besides Keras inputs.

    # Arguments
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
        session_kwargs: arguments to `tf.Session.run()`: `fetches`, `feed_dict`,
        `options`, `run_metadata`
    """

    def __init__(self, inputs, outputs, updates=None, name=None, **session_kwargs):
        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)
        self.name = name
        # additional tensor substitutions
        self.feed_dict = session_kwargs.pop('feed_dict', {})
        # additional operations
        self.fetches = session_kwargs.pop('fetches', [])
        if not isinstance(self.fetches, list):
            self.fetches = [self.fetches]
        self.session_kwargs = session_kwargs

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')
        feed_dict = self.feed_dict.copy()
        for tensor, value in zip(self.inputs, inputs):
            if is_sparse(tensor):
                sparse_coo = value.tocoo()
                indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                          np.expand_dims(sparse_coo.col, 1)), 1)
                value = (indices, sparse_coo.data, sparse_coo.shape)
            feed_dict[tensor] = value
        fetches = self.outputs + [self.updates_op] + self.fetches
        session = get_session()
        updated = session.run(fetches=fetches, feed_dict=feed_dict,
                              **self.session_kwargs)
        return updated


# function_get_fetches adapted from function() in K.backend.tensorflow_backend


def function_get_fetches(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Keras function.

    # Arguments
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: Passed to `tf.Session.run`.

    # Returns
        Output values as Numpy arrays.

    # Raises
        ValueError: if invalid kwargs are passed in.
    """
    if kwargs:
        for key in kwargs:
            if not (has_arg(tf.Session.run, key, True) or has_arg(Function.__init__, key, True)):
                msg = 'Invalid argument "%s" passed to K.function with TensorFlow backend' % key
                raise ValueError(msg)
    return FunctionGetFetches(inputs, outputs, updates=updates, **kwargs)


# _make_predict_function_get_fetches adapted from _make_predict_function() in K.backend.tensorflow_backend


def _make_predict_function_get_fetches(self):
    if not hasattr(self, 'predict_function'):
        self.predict_function = None
    if self.predict_function is None:
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs = self._feed_inputs + [K.learning_phase()]
        else:
            inputs = self._feed_inputs
        # Gets network outputs. Does not update weights.
        # Does update the network states.
        kwargs = getattr(self, '_function_kwargs', {})
        self.predict_function = function_get_fetches(inputs,
                                                     self.outputs,
                                                     updates=self.state_updates,
                                                     name='predict_function',
                                                     **kwargs)
