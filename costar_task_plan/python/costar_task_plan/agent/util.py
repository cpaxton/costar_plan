
from .null import *
from .random_agent import *
from .task import *
from .ff import *

from .keras_ddpg import *
from .keras_naf import *
from .apl_ddpg import *
from .nn_planner import *

def GetAgents():
    return ["none", "null",
            "random", # random actions
            "task", # supervised task model
            "keras_ddpg", # keras DDPG
            "keras_naf", # not implemented
            "ff", # FF regression agent
            "apl_ddpg", # custom DDPG implementation
            "nn_planner",
            ]

def MakeAgent(env, name, *args, **kwargs):
    try:
        return {
                'no': lambda: NullAgent(env, *args, **kwargs),
                'none': lambda: NullAgent(env, *args, **kwargs),
                'null': lambda: NullAgent(env, *args, **kwargs),
        		'albert': lambda: AlbertAgent(env, *args, **kwargs),
                'albert2': lambda: Albert2Agent(env, *args, **kwargs),
                'albert3': lambda: Albert3Agent(env, *args, **kwargs),
                'random': lambda: RandomAgent(env, *args, **kwargs),
                'task': lambda: TaskAgent(env, *args, **kwargs),
                'apl_ddpg': lambda: APLDDPGAgent(env, *args, **kwargs),
            		'keras_ddpg': lambda: KerasDDPGAgent(env, *args, **kwargs),
                'ff': lambda: FeedForwardAgent(env, *args, **kwargs),
                'nn_planner': lambda: NeuralNetworkPlannerAgent(
                    env,
                    *args,
                    **kwargs),
                }[name.lower()]()
    except KeyError, e:
        raise NotImplementedError('Agent "%s" not implemented'%name)

