from abstract import AbstractAgent

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, Lambda
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


class KerasDDPGAgent(AbstractAgent):

    name = "keras_ddpg"

    def __init__(self, env, *args, **kwargs):
        super(KerasDDPGAgent, self).__init__(*args, **kwargs)
        self.env = env
        
        #assert len(env.action_space.shape) == 1
        #TODO: is there a way to output a tuple (6,1)
        nb_actions = sum(sum(1 for i in row if i) for row in self.env.action_space.sample())
        
        
        #TODO: terminology? feature or observation?
        observation = env.reset()
        
        print ">>>>>>>>>>>>>>>>>>>", observation.shape

        # TODO: find a way to customize network
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + observation.shape))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        actor.add(Lambda(lambda x: x * 3.14159))

        print(actor.summary())
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        
        observation_input = Input(shape=(1,) + observation.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = merge([action_input, flattened_observation], mode='concat')
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(input=[action_input, observation_input], output=x)
        print(critic.summary())
        
       
        memory = SequentialMemory(limit=500000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        


    def fit(self, num_iter=50000):
        self.agent.fit(self.env, nb_steps=num_iter, visualize=False, verbose=1, nb_max_episode_steps=200)
        
        # After training is done, we save the final weights.
        #self.agent.save_weights('ddpg_{}_weights.h5f'.format(self.name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        self.agent.test(self.env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)
        
