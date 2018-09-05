import os
import multiprocessing

import numpy as np
import pytest
from csv import reader
from csv import Sniffer
import shutil
from keras import optimizers
from keras import initializers
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, add
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.utils.test_utils import get_test_data
from keras.utils.test_utils import keras_test
from keras import backend as K
from keras.utils import np_utils
try:
    from unittest.mock import patch
except:
    from mock import patch


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow backend')
@keras_test
def test_EvaluateInputTensor():
    """We test building a model with a TF variable as input.
    We should be able to call fit with the EvaluateInputTensor callback.
    """
    import tensorflow as tf

    input_a = K.variable(np.random.random((10, 3)))
    input_b = K.variable(np.random.random((10, 3)))

    output_a = K.variable(np.random.random((10, 4)))
    output_b = K.variable(np.random.random((10, 3)))

    a = Input(tensor=input_a)
    b = Input(tensor=input_b, name='input_b')

    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    model = Model([a, b], [a_2, b_2])
    model.summary()

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer, loss, metrics=['mean_squared_error'],
                  loss_weights=loss_weights,
                  sample_weight_mode=None,
                  target_tensors=[output_a, output_b])

    eval_model = Model([a, b], [a_2, b_2])
    eval_model.compile(optimizer, loss, metrics=['mean_squared_error'],
                       loss_weights=loss_weights,
                       sample_weight_mode=None,
                       target_tensors=[output_a, output_b])

    eval_callback = callbacks.EvaluateInputTensor(model=eval_model, steps=1)

    # test fit
    out = model.fit(None, None, epochs=1,
                    callbacks=[eval_callback],
                    steps_per_epoch=1)