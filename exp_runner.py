import os
import numpy as np

import exp_descriptor
import inference.irreducible_infer as irreducible_infer
import inference.reducible_infer as reducible_infer
import simulator.mbt_bisse_simulator as mbt_bisse_simulator
from inference.io import save

# run experiments with an experiment descriptor object

# create a folder (path) to store the outputs

# create a class for each model (reducible, irreducible, reducible_equaldeath)


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

    def run(self, trial=0, sim_obs=True):
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
            self._run_irreducible(trial_path, sim_obs=sim_obs)
        elif self.exp.model == 'reducible':
            self._run_reducible(trial_path, sim_obs=sim_obs)
        else:
            raise TypeError('Unknown MBT process')



    def _run_irreducible(self, trial_path, sim_obs):
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
                raise ValueError("The pseudo-observed process becomes extinct for all 20 attempts.")
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


    def _run_reducible(self, trial_path, sim_obs):
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
                else:
                    raise ValueError("Invalid n_trees, the number of trees has to be a positive integer.")

                # inference process
                reducible_infer.infer(tree_obs_all, self.exp, trial_path)
        