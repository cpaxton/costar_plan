
import argparse
import importlib
from plot_utils import graphStatesAndActions
import matplotlib.pyplot as plt

def _getParser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu",
            action="store_true",
            help="Force use of cpu0")
    parser.add_argument("--device",
            default=None,
            help="specify device for tensorflow to use")
    parser.add_argument("--verbose",
            action="store_true",
            help="Print out causes of termination conditions and other information.")
    parser.add_argument("--seed",
            type=int,
            default=None,
            help="Random seed to use.")
    return parser

def _eval_policy(world, policy):

    states = []
    actions = []
    # loop world until terminal, storing action and state at each point
    world.updateFeatures()
    while not world.done:
        actor = world.actors[0]
        state = actor.state

        action = policy(world, state, actor)
        states.append(state)
        actions.append(action)
        world.tick(action)

    return states, actions

def plotOption(world, policy, plots, rename, rows, cols, fs=[]):
    states, actions = _eval_policy(world, policy)
    graphStatesAndActions(states, actions, plots, rename, rows, cols, fs)
    
    plt.show()
