import os
import numpy as np

import exp_descriptor
import inference.irreducible_infer as irreducible_infer
import inference.reducible_infer as reducible_infer
import simulator.mbt_bisse_simulator as mbt_bisse_simulator
from inference.io import save

from ete3 import Tree
import csv

# write a function for parameter inference for irreducible processes
class ExpRunner:
    """
    Run experiments using ABC
    """
    def __init__(self, experiment):
        """
        experiment: an ExpDescriptor object
        """
        assert isinstance(experiment, exp_descriptor.ExpDescriptor)
        self.exp = experiment
        self.file_path = os.path.join('experiments', self.exp.get_dir())

    def run(self, trial=0, sim_obs=True, tree_file=None, phase_file=None):
        """
        Runs the experiment.
        """

        print('\n' + '-' * 80)
        print(f'RUNNING EXPERIMENT, TRIAL {trial}:\n')
        print(self.exp.info())

        trial_path = os.path.join(self.file_path, str(trial))

        # Check if the folder exists
        if os.path.exists(trial_path):
            raise FileExistsError(f"Error: The folder '{trial_path}' already exists. Experiments already exist.")
        os.makedirs(trial_path)

        
        # check model to see which algorithm we will use
        # input for the algorithm: trial_path, sample_gt (add for future)
        if self.exp.model == 'irreducible':
            self._run_irreducible(trial_path, sim_obs=sim_obs, tree_file=tree_file, phase_file=phase_file)
        elif self.exp.model == 'reducible':
            self._run_reducible(trial_path, sim_obs=sim_obs, tree_file=tree_file, phase_file=phase_file)
        else:
            raise TypeError('Unknown MBT process')



    def _run_irreducible(self, trial_path, sim_obs, tree_file=None, phase_file=None):
        # simulation process (save them in obs)
        if sim_obs:
            # check if the process is supercritical
            birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = self.exp.true_ps
            save(os.path.join(trial_path, 'obs', "true_ps.csv"), self.exp.true_ps)
            w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])
            if w<=0:
                raise ValueError("The process needs to be supercritical.")
            tree_obs, flag_obs = mbt_bisse_simulator.birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim, q21sim], nsize=self.exp.n_leaves, start=self.exp.start_phase, r=20)
            if flag_obs==0:
                raise ValueError("The selected parameters have high chance of being extinct before reaching the desired number of (survived) lineages. The pseudo-observed process has gone extinct after 20 attempts.")
            else:
                if self.exp.n_trees == 1:
                    tree_obs_all = tree_obs
                    # save the observed samples (true_ps, trees, tip_phase), tip_phase in a csv file (n_trees, n_leaves), trees
                    leaf_state = np.zeros(self.exp.n_leaves)
                    for index, leaf in enumerate(tree_obs.get_leaves()):
                        leaf.name = index+1 # leaf name starts from 1
                        leaf_state[index] = leaf.phase
                    save(os.path.join(trial_path, 'obs', "obs_trees_phases.csv"), leaf_state)
                    tree_obs.write(format=5, outfile=os.path.join(trial_path, 'obs', "obs_trees.nwk"))
                elif self.exp.n_trees > 1:
                    tree_obs_all = [tree_obs]
                    for _ in range(1, self.exp.n_trees):
                        flag_sim = 0
                        while flag_sim==0:
                            tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim, q21sim], nsize=self.exp.n_leaves, start=self.exp.start_phase, r=20)
                        tree_obs_all.append(tree_sim)
                    # save the observed samples (true_ps, trees, tip_phase), tip_phase in a csv file (n_trees, n_leaves), trees
                    leaf_state = np.zeros((self.exp.n_trees,self.exp.n_leaves))
                    for tree_i in range(self.exp.n_trees):
                        for index, leaf in enumerate(tree_obs_all[tree_i].get_leaves()):
                            leaf.name = index+1 # leaf name starts from 1
                            leaf_state[tree_i,index] = leaf.phase
                    save(os.path.join(trial_path, 'obs', "obs_trees_phases.csv"), leaf_state)
                    with open(os.path.join(trial_path, 'obs', "obs_trees.nwk"), "w") as f:
                        for tree in tree_obs_all:
                            f.write(tree.write(format=5) + "\n")
                else:
                    raise ValueError("Invalid n_trees, the number of trees has to be a positive integer.")

                # inference process
                irreducible_infer.infer(tree_obs_all, self.exp, trial_path)
        else:
            # if we infer from the given tree(s)
            assert tree_file is not None, f"Files about trees for inference do not exist"
            assert phase_file is not None, f"Files about the tips' phases do not exist"

            # read trees
            obs_trees = []
            with open(tree_file) as f:
                for line in f:
                    if line: # skip empty line
                        obs_trees.append(Tree(line, format=0))
            # if the observation is simulated
            if self.exp.obs_simulated is True: # if the tip phases are recorded in the matrix form, like in our simulations
                # Load the tip phases
                obs_trees_phases = np.loadtxt(phase_file, delimiter=",") 

                if len(obs_trees_phases.shape)==1:
                    obs_trees_phases = obs_trees_phases[np.newaxis,:]
                # like in the simulated results, each row corresponds a tree, each column refers to the leaf according to its leaf name
                assert len(obs_trees)==obs_trees_phases.shape[0]
                assert len(obs_trees[0].get_leaves()) == obs_trees_phases.shape[1]
                # assign phases to trees
                for ind_tree, tree in enumerate(obs_trees):
                    for ind_leaf, leaf in enumerate(tree.get_leaves()):
                        # if I don't have a name for leave, by default, leaves have a number as indices
                        assert ind_leaf == int(leaf.name)-1
                        leaf.phase = obs_trees_phases[ind_tree, int(leaf.name)-1]
            else: # if the tip phases are recorded in a data frame, where we can search the phase by the leaf's name
                with open(phase_file, 'r', encoding='utf-8') as f: 
                    name_dict = {line[1]: line[0] for line in csv.reader(f)} # suppose the first column gives the leaf name, the second column gives the phase (1/2)
                    flag_missing = 0 # check if we have leaf with unknown phase
                    for ind_tree, tree in enumerate(obs_trees):
                        for leaf in tree.get_leaves():
                            phase = name_dict.get(leaf.name)
                            if phase == '1':
                                leaf.phase = 1
                            elif phase == '2':
                                leaf.phase = 2
                            else:
                                leaf.phase = 'missing'
                                flag_missing = 1

                    if flag_missing == 1: 
                        if self.exp.n_trees>1:
                            # we have multiple trees, we don't allow missing phases in one of the trees (prune all tree accordingly to make sure the size is the same)
                            raise ValueError(f"The phase of leaf {leaf.name} is missing.")
                        self.prune_missing(obs_trees[0]) # remove the leaves with missing phases
                        self.delete_single_child_internal_dist(obs_trees[0]) # remove unary nodes
                        # update the treesize based on the missing values
                        print(f"Due to missing phases for some leaves, the tree has been pruned and the tree size changed from {self.exp.n_leaves} to {len(obs_trees[0].get_leaves())}.")
                        self.exp.n_leaves = len(obs_trees[0].get_leaves())
                    
            # inference process
            irreducible_infer.infer(obs_trees, self.exp, trial_path)


    def _run_reducible(self, trial_path, sim_obs, tree_file, phase_file):
        # simulation process (save them in obs)
        if sim_obs:
            # check if the process is supercritical
            birth1sim,birth2sim,death1sim,death2sim,q12sim = self.exp.true_ps
            save(os.path.join(trial_path, 'obs', "true_ps.csv"), self.exp.true_ps)
            w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])
            if w<=0:
                raise ValueError("The process needs to be supercritical.")
            tree_obs, flag_obs = mbt_bisse_simulator.birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim, 0], nsize=self.exp.n_leaves, start=self.exp.start_phase, r=20)
            if flag_obs==0:
                raise ValueError("The pseudo-observed process becomes extinct for all 20 attempts.")
            else:
                if self.exp.n_trees < 3:
                    raise ValueError("{n_trees} observed trees are insufficient for inference under reducible processes.")
                else:
                    tree_obs_all = [tree_obs]
                    for _ in range(1, self.exp.n_trees):
                        flag_sim = 0
                        while flag_sim==0:
                            tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim, 0], nsize=self.exp.n_leaves, start=self.exp.start_phase, r=20)
                        tree_obs_all.append(tree_sim)
                    # save the observed samples (true_ps, trees, tip_phase), tip_phase in a csv file (n_trees, n_leaves), trees
                    leaf_state = np.zeros((self.exp.n_trees,self.exp.n_leaves))
                    for tree_i in range(self.exp.n_trees):
                        for index, leaf in enumerate(tree_obs_all[tree_i].get_leaves()):
                            leaf.name = index+1 # leaf name starts from 1
                            leaf_state[tree_i,index] = leaf.phase
                    save(os.path.join(trial_path, 'obs', "obs_trees_phases.csv"), leaf_state)
                    with open(os.path.join(trial_path, 'obs', "obs_trees.nwk"), "w") as f:
                        for tree in tree_obs_all:
                            f.write(tree.write(format=5) + "\n")

                # inference process
                reducible_infer.infer(tree_obs_all, self.exp, trial_path)
        else:
            # if we infer from the given tree(s)
            assert tree_file is not None, f"Files about trees for inference do not exist"
            assert phase_file is not None, f"Files about the tips' phases do not exist"

            # read trees
            obs_trees = []
            with open(tree_file) as f:
                for line in f:
                    if line: # skip empty line
                        obs_trees.append(Tree(line, format=0))
            if self.exp.n_trees < 3:
                    raise ValueError("{n_trees} observed trees are insufficient for inference under reducible processes.")
            # if the observation is simulated
            if self.exp.obs_simulated is True: # if the tip phases are recorded in the matrix form, like in our simulations
                # Load the tip phases
                obs_trees_phases = np.loadtxt(phase_file, delimiter=",") 

                if len(obs_trees_phases.shape)==1:
                    obs_trees_phases = obs_trees_phases[np.newaxis,:]
                # like in the simulated results, each row corresponds a tree, each column refers to the leaf according to its leaf name
                assert len(obs_trees)==obs_trees_phases.shape[0]
                assert len(obs_trees[0].get_leaves()) == obs_trees_phases.shape[1]
                # assign phases to trees
                for ind_tree, tree in enumerate(obs_trees):
                    for ind_leaf, leaf in enumerate(tree.get_leaves()):
                        # if I don't have a name for leave, by default, leaves have a number as indices
                        assert ind_leaf == int(leaf.name)-1
                        leaf.phase = obs_trees_phases[ind_tree, int(leaf.name)-1]
            else: # if the tip phases are recorded in a data frame, where we can search the phase by the leaf's name
                with open(phase_file, 'r', encoding='utf-8') as f: 
                    name_dict = {line[0]: line[1] for line in csv.reader(f)} # suppose the first column gives the leaf name, the second column gives the phase (1/2)
                    for ind_tree, tree in enumerate(obs_trees):
                        for leaf in tree.get_leaves():
                            phase = name_dict.get(leaf.name)
                            if phase == '1':
                                leaf.phase = 1
                            elif phase == '2':
                                leaf.phase = 2
                            else:
                                raise ValueError(f"The phase of leaf {leaf.name} is missing.")
                    
            # inference process
            reducible_infer.infer(obs_trees, self.exp, trial_path)


    @staticmethod
    def prune_missing(tree):
        """
        Utility function that keeps all nodes that have at least one descendant with known phase while removing the other nodes
        """
        for node in tree.traverse():
            flag = 0
            for leaf in node.get_leaves():
                if leaf.phase!='missing':
                    flag = 1 
            if flag == 0:
                node.detach()

    @staticmethod
    def delete_single_child_internal_dist(tree):
        """Utility function that removes internal nodes
        with a single child from tree"""

        for node in tree.traverse("postorder"):
            
            if (not node.is_leaf() and (not node.is_root()) and len(node.get_children())<2):
                child = node.get_children()[0]
                child.dist = child.dist + node.dist
                node.delete()

        if len(tree.get_children()) == 1:
            tree.dist = tree.children[0].dist
            tree.children[0].delete() 


        