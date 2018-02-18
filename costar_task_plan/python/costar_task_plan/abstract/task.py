# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from __future__ import print_function

from option import AbstractOption, NullOption
from world import AbstractWorld

import copy
import numpy as np

ROOT_NAME = "ROOT"
ROOT_TAG = "ROOT()"


class TaskTemplate(object):
    '''
    Template for a particular task. Stores small amount of information about 
    parents and availavle options.
    '''
    def __init__(self, name, parents):
        self.name = name
        if not isinstance(parents, list):
            parents = [parents]
        self.parents = parents
        self.options = []

    def add(self, *args):
        self.options.append(args)


class Task(object):
    '''
    Model of a task as a state machine. We can specify this in any number of
    different ways: in CoSTAR, we use a Behavior Tree; PDDL or LTL can give us a
    slightly different task plan.
    '''

    def __init__(self, subtask_name=None, subtask_parents=None):
        '''
        Create the task.
        '''
        self.option_templates = {None: NullOptionTemplate()}
        self.template_connections = []

        # for subtasks
        self.subtask_name = subtask_name
        self.subtask_parents = subtask_parents

        self.compiled = False
        self.nodes = {}
        self.children = {}
        self.weights = {}

        # Dictionary from the action string names to the
        # unique integer id (label) of each action
        # node (like grasp_block)
        self.indices = {}

        # Store name by integer index in case we ever want to recover that

        # Dictionary from the unique integer id (label) of each action
        # to the associated string names
        self.names = {}
        self.generic_names = {}

        self.conditions = []

        # List of checks to make -- call these functions on possible arg dicts,
        # they should return true/false.
        self.option_checks = []

    def addCheck(self, check):
        '''
        Do certain combinations of arguments just not make sense? Then add a
        check here. Checks are assumed to be functors, called with:

          check(*arg_set)

        Where arg_set is a list of pairs (arg_type, arg_value). So, if you wanted
        to check if block1 and block2 were the same, you would write a function
        with the form:

          def check_blocks(block1, block2, **kwargs):
            ...

        Parameters:
        -----------
        check: functor to add before adding any particular branch/node to the
        task graph.
        '''
        self.option_checks.append(check)

    def mergeTask(self, task, name, inodes):
        '''
        Add subtasks and options to the current compiled task graph.

        Parameters:
        ----------
        task: the subtask to merge
        inodes: dictionary of nodes to combine

        Returns:
        --------
        inodes: updated node dictionary
        '''

        # add templates in
        for name, template in task.option_templates.items():
            self.option_templates[name] = template

        # add connections in
        for parent, child, frequency in task.template_connections:
            if parent in self.option_templates:
                self.option_templates[parent].connect(child, frequency)

        for iname, option in task.nodes.items():
            name = task.generic_names[iname]
            if iname in self.nodes:
                if name not in inodes:
                    inodes[name] = []
                if iname not in inodes[name]:
                    inodes[name].append(iname)
                for child in task.children[iname]:
                    self.children[iname].add(child)
            else:
                inodes[name] = [iname]
                self.nodes[iname] = option
                self.children[iname] = task.children[iname]

        return inodes

    def add(self, name, parents, option_args, frequencies=[]):
        '''
        Note: we assume that each name uniquely identifies an action, and each
        action has a unique name.
        '''

        ignore_args = False
        if name in self.option_templates and option_args is not None:
            raise RuntimeError('Arguments for option "%s" already set!' % name)
        elif option_args is None:
            ignore_args = True

        if not ignore_args:
            self.option_templates[name] = OptionTemplate(
                subtask_name=self.subtask_name,
                **option_args)

        if parents is None or len(parents) == 0:
            parents = [None]
        elif not isinstance(parents, list):
            parents = [parents]

        # By default, weight everything equally
        if len(frequencies) == 0:
            frequencies = [1] * len(parents)

        for parent, frequency in zip(parents, frequencies):
            self.template_connections.append((parent, name, frequency))

    def getChildren(self, node):
        '''
        Return the children associated with this particular node.
        '''
        if node in self.children:
            return self.children[node], self.weights[node]
        else:
            return [], []

    def getOption(self, node):
        if node in self.nodes:
            return self.nodes[node]
        else:
            raise RuntimeError('node %s does not exist' % node)
    def compile(self, arg_dict, unroll_depth=None):
        '''
        Instantiate this task for a particular world. This takes the task model and
        turns it into a "real" task, with appropriate options being created.

        Procedure:
         - loop over all options
         - for each option: loop over all args
         - create option with those args
        '''

        assert not self.compiled

        if isinstance(arg_dict, AbstractWorld):
            arg_dict = arg_dict.getObjects()

        # Make list of args
        arg_sets = get_arg_sets(arg_dict)

        # First: connect templates (parent -> child)
        for parent, child, frequency in self.template_connections:
            if parent in self.option_templates:
                self.option_templates[parent].connect(child, frequency)

        # Possible assignments to arguments. Add children here, not outside.
        for arg_set in arg_sets:
            # List of instantiated options and subtasks, used for connecting children
            # to parents.
            # "inodes" is the list of instantiated nodes to connect. It is scoped, so
            # that for any particular set of options/substasks created, they will all
            # share the same values. This lets us create interesting hierarchical
            # structures.
            inodes = {}

            ok = True
            for check in self.option_checks:
                if not check(**arg_set):
                    ok = False
                    break
            if not ok:
                print("Skipping invalid args:", arg_set)
                continue

            # create the nodes
            for name, template in self.option_templates.items():
                iname, option = template.instantiate(name, arg_set)
                if option is None:
                    continue
                elif isinstance(option, Task):
                    # this was a subtask, and must be merged into the full version of
                    # the task model.
                    inodes = self.mergeTask(option, name, inodes)
                else:
                    self._addInstantiatedNode(name, iname, option, inodes)

            # Connect nodes and their children, plus update the list of weights
            # associated with each parent-child pair.
            for name, template in self.option_templates.items():
                if template.task is not None:
                    # This subtask has been removed -- we no longer need it
                    # Its associated options have been merged into the graph as a whole
                    continue
                elif name not in inodes:
                    # This activity was never created -- we have no examples of
                    # this action or of the necessary objects to create this
                    # action in the real world.
                    print ("Skipping missing option:", name)
                    continue
                for iname in inodes[name]:
                    self.generic_names[iname] = name
                    self.weights[iname] = []
                    # loop over all templated (abstract) actions
                    for child, frequency in zip(template.children, template.frequencies):
                        # If this child is in the set of instantiated nodes...
                        if child in inodes:
                            # Count the instantiated children and scale down weight
                            # accordingly
                            num_ichildren = len(inodes[child])
                            for ichild in inodes[child]:
                                # For every instantiated child in this high-level option set,
                                # add it to the children of this node. Then update weights.
                                self.children[iname].add(ichild)
                                self.weights[iname].append(
                                    float(frequency) / num_ichildren)
                    self.weights[iname] = (np.array(self.weights[iname]) / 
                                           np.sum(self.weights[iname]))

        if self.subtask_name == None:
                # WARNING: this is kind of terrible and might be sort of inefficient.
                # Convert to a list
            for i, (node, children) in enumerate(self.children.items()):
                children = [child for child in children]
                self.children[node] = children
                self.indices[node] = i
                self.names[i] = node

        self.compiled = True
        return arg_sets

    def _addInstantiatedNode(self, name, iname, option, inodes):
        if iname in self.nodes:  # and iname is ROOT_TAG:
            if name not in inodes:
                inodes[name] = []
            if iname not in inodes[name]:
                inodes[name].append(iname)
        else:
            if name in inodes:
                inodes[name].append(iname)
            else:
                inodes[name] = [iname]
            self.nodes[iname] = option
            self.children[iname] = set()

    def clear(self):
        self.nodes = {}
        self.children = {}
        self.weights = {}
        self.indices = {}
        self.names = {}
        self.compiled = False
        self.generic_names = {}

    def makeTree(self, world, max_depth=10):
        '''
        Make the root of a tree search. This creates the whole tree structure that
        we are going to explore, including all connections.
        '''
        depth = 0
        pass

    def index(self, name):
        '''
        Look up the index associated with a particular node ("action") so we can
        easily map to a discrete action space.
        '''
        return self.indices[name]

    def numIndices(self):
        '''
        Get length.
        '''
        return len(self.indices.keys())

    def name(self, index):
        '''
        Get name from embedding
        '''
        return self.names[index]

    def numActions(self):
        '''
        Indices directly indexes the array of actions so this shouldn't really
        change much.
        '''
        return len(self.names.keys())

    def nodeSummary(self):
        if not self.compiled:
            raise RuntimeError(
                'Cannot print nodes from Task before compile() has been called!')
        summary = ''
        for name, node in self.nodes.items():
            summary += "%s --> %s\n" % (name, str(self.children[name]))
        return summary

    def sampleSequence(self):
        '''
        Sample a random sequence of options starting from the root of the task
        tree. This can be used in conjunction with many different things to get
        different executions.
        '''
        names = []
        sequence = []
        tag = ROOT_TAG
        while True:
            children, weights = self.getChildren(tag)
            if children is None or len(children) == 0:
                break
            else:
                idx = np.random.randint(len(children))
                tag = children[idx]
                names.append(tag)
                sequence.append(self.getOption(tag))
        return names, sequence


''' =======================================================================
                        HELPERS AND INTERNAL CLASSES
    ======================================================================= '''


class OptionTemplate(object):
    '''
    Internal class that represents a single templated, non-instantiated Option.
    '''

    def __init__(self, args=[], constructor=None, remap=None, task=None,
                 subtask_name=None, semantic_remap=None, postconditions=[], semantic_args=[],
                 name_template="%s(%s)"):
        self.constructor = constructor
        self.subtask_name = subtask_name
        self.task = task
        self.args = args
        self.semantic_args = semantic_args
        self.remap = remap
        self.semantic_remap = semantic_remap
        self.name_template = name_template
        self.children = []
        self.postconditions = postconditions
        self.frequencies = []

    def instantiate(self, name, arg_dict):
        '''
        Compute the actual lists of things that get turned into a task graph.

        Parameters:
        -----------
        name: the basic name of the action being performed
        arg_dict: set of options that can fill out the parameters of the action
        '''

        filled_args = {}
        name_args = {}

        # ==================================================================
        for arg in self.args:
            if self.remap is not None and arg in self.remap:
                filled_arg_name = self.remap[arg]
            else:
                filled_arg_name = arg
            if self.semantic_remap is not None and arg in self.semantic_remap:
                semantic_arg_name = self.semantic_remap[arg]
            else:
                semantic_arg_name = arg
            if not arg in arg_dict:
                continue               
            filled_args[filled_arg_name] = arg_dict[arg]
            name_args[semantic_arg_name] = arg_dict[arg]

        # ==================================================================
        for arg in self.semantic_args:
            if self.semantic_remap is not None and arg in self.semantic_remap:
                semantic_arg_name = self.semantic_remap[arg]
            else:
                semantic_arg_name = arg
            name_args[arg] = arg_dict[arg]

        if name is None:
            name = ROOT_NAME

        if self.task is None:
            iname = self.name_template % (name, make_str(name_args))
            #try:
            option = self.constructor(**filled_args)
            for pc in self.postconditions:
                option.addPostCondition(pc)
            #except Exception as e:
            #    print("Warning: failed to create option %s (instance of %s)" % (iname, name))
            #    print("args =", arg_dict)
            #    print("name_args =", name_args)
            #    print(e)
            #    option = None
        else:
            option = Task(subtask_name=self.task.name)
            for args in self.task.options:
                option.add(*args)
            option.compile(arg_dict)
            iname = self.name_template % (name,
                                          make_str(filled_args))

        return iname, option

    def connect(self, child, count=0):
        '''
        Add a connection from this node to the child, but only if we have not
        already done so (limit 1 connection per parent-child pair).
        '''
        if child not in self.children:
            self.children.append(child)
            self.frequencies.append(float(count))


class NullOptionTemplate(OptionTemplate):
    '''
    For the root node
    '''

    def __init__(self):
        super(NullOptionTemplate, self).__init__(
            constructor=NullOption,
            args={})


class TaskNode(object):
    '''
    Internal class that represents branches in our task search.
    '''

    def __init__(self, option, children=[]):
        self.option = option
        self.children = children


def get_arg_sets(arg_dict):

    # empty set of arguments
    arg_sets = [{}]

    # loop over all arguments
    for arg, vals in arg_dict.items():
        prev_arg_sets = arg_sets
        arg_sets = []

        if isinstance(vals, list):
            # loop over possible argument assignments
            for val in vals:

                # add a version where arg=val
                for prev_arg_set in prev_arg_sets:
                    arg_set = copy.copy(prev_arg_set)
                    arg_set[arg] = val
                    arg_sets.append(arg_set)
        else:
            val = vals
            for prev_arg_set in prev_arg_sets:
                arg_set = copy.copy(prev_arg_set)
                arg_set[arg] = val
                arg_sets.append(arg_set)

    # return the set of populated assignments
    return arg_sets


def make_str(filled_args, subtask=None):
    assignment_list = []
    for arg, val in filled_args.items():
        if subtask is not None:
            assignment_list.append("%s_%s=%s") % (subtask, arg, val)
        else:
            assignment_list.append("%s=%s" % (arg, val))
    return str(assignment_list)[1:-1]
