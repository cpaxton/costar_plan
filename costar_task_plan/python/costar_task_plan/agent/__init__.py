
from goal import *
from keras_ddpg import *
from keras_naf import *
from null import *
from random import *
from task import *
from util import *

__all__ = [
        # ============================
        # TODO: re-enable albert code
        #"Albert2Agent",
        #"Albert3Agent",
        #"AlbertAgent",
        "RandomAgent",
        "NullAgent",
        "TaskAgent",
        "RandomGoalAgent",
        "KerasDDPGAgent",
        "KerasNAFAgent",
        # ============================
        "MakeAgent", "GetAgents",
        ]

