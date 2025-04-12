# inference process for irreducible processes
import random
import numpy as np
from statistics import variance
from math import sqrt
from scipy.stats import multivariate_normal
import os
import time

# Import file from the parent folder of the current script
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulator.mbt_bisse_simulator as mbt_bisse_simulator
import simulator.comp_sumsta as comp_sumsta
from .io import save

# write a function for the inference algorithm
def infer(obs_trees, experiment, trial_path):
    # assign values to all arguments in the inference process
    if experiment.n_trees == 1:
        _reducible_infer_onetree(obs_trees, experiment, trial_path)
    else:
        _reducible_infer_multitree(obs_trees, experiment, trial_path)


def _reducible_infer_onetree(obs_tree, experiment, trial_path):
    """
    obs_tree: a single observed tree
    experiment: an ExpDescriptor object
    """
    pass



def _reducible_infer_multitree(obs_trees, experiment, trial_path):
    """
    obs_tree: a list of observed trees
    experiment: an ExpDescriptor object
    """
    pass