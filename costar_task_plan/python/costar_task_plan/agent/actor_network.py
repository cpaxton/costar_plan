#adapted from:
# https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ActorNetwork.py

import numpy as np
import math
import keras
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, convolutional=False, output_activation='sigmoid'):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.convolutional = convolutional
        self.output_activation = output_activation

        #K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):

        if self.convolutional:
            print "<<<<<<<<<<<<<<insdide actor", state_size[0], state_size[1]
            #S = Flatten(input_shape=(1,) + (state_size)

            S = Input(shape=(state_size[0], state_size[1], 1))
            #x = BatchNormalization(axis=1)(S)   
            #100x100 -> 31x31
            x = Conv2D(32, (7,7), strides=(3,3), activation="relu")(S)
            #x = BatchNormalization(axis=1)(x)
            #31x31 -> 10x10
            x = Conv2D(32, (5,5), strides=(3,3), activation="relu")(x)
            #x = BatchNormalization(axis=1)(x)
            #10x10 -> 4x4
            x = Conv2D(32, (4,4), strides=(2,2), activation="relu")(x)
            #x = BatchNormalization(axis=1)(x)
            x = Flatten()(x)
            x = Dense(128, activation="relu")(x)
            #x = BatchNormalization()(x)

        else:
            S = Input(shape=(state_size[0],))
            #x = BatchNormalization()(S)
            x = Dense(128, activation="relu")(S)
            x = Dense(128, activation="relu")(x)

        x = Dense(128, activation="relu")(x)
        #x = BatchNormalization()(x)
        init = keras.initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=None)
        A = Dense(action_dim, activation=self.output_activation, kernel_initializer=init, bias_initializer='zeros')(x)          
        model = Model(inputs=S,outputs=A)
        return model, model.trainable_weights, S
