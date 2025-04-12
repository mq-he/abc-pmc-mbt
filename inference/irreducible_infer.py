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
        _irreducible_infer_onetree(obs_trees, experiment, trial_path)
    else:
        _irreducible_infer_multitree(obs_trees, experiment, trial_path)


def _irreducible_infer_onetree(obs_tree, experiment, trial_path):
    """
    obs_tree: a single observed tree
    experiment: an ExpDescriptor object
    """
    trial_path = os.path.join(trial_path, 'results') # the path to the results

    # inference parameters
    T_iter = experiment.n_iter # the number of iterations
    T_nltt = experiment.T_nltt # the number of iteration where we start to consider nLTT statistic
    M_array = experiment.num_accept # a list for the number of accepted samples at each iteration
    startphase = experiment.start_phase # the starting phase
    treesize = experiment.n_leaves # the number of leaves for each tree
    #n_trees_iter = experiment.n_trees_iter # the number of trees at each iteration

    prior_bound_l = experiment.prior_bound_l
    prior_bound_u = experiment.prior_bound_u

    threshold_rate = experiment.threshold_rate # threshold acceptance rate
    decrease_factor = experiment.decrease_factor
    tol_nltt = experiment.tol_nltt # the tolerance value for nLTT statistic

    # break the large tree into smaller pieces
    size1, size2 = 20, 50
    subtrees = comp_sumsta.break_tree(obs_tree,size1,size2)
    n_subtrees = len(subtrees)
    assert n_subtrees >= 3, f"Insufficient amount of subtrees have been observed. We only have {n_subtrees} subtrees with leaves ranging between {size1} and {size2} in the observed tree.\nConsider adjust the accepted size for subtrees."
    subtrees_size = np.zeros(n_subtrees)
    sumsta_subtrees = np.zeros((n_subtrees,8))
    for ind, tree in enumerate(subtrees):
        subtrees_size[ind] = len(tree.get_leaves())
        sumsta_subtrees[ind] = comp_sumsta.generate_mbt_ind_update(tree)

    ex_stats_sub = np.mean(sumsta_subtrees, axis=0)
    max_stats_sub = np.max(sumsta_subtrees, axis=0)
    min_stats_sub = np.min(sumsta_subtrees, axis=0)
    tol_stats_sub = [2*sqrt(variance(sumsta_subtrees[:,i])) for i in range(8)]
    tol_stats_sub = np.array(tol_stats_sub)

    accepted = np.zeros((M_array[0], 6+1+1+8)) # accepted parameter values, growth rate, weights, and associated statistics
    count_accepted = 0 # count the number of accepted samples
    N = 0 # count the number of simulations
    k = 0 # the iteration index
    start = time.time()
    while count_accepted < M_array[0]:
        do = True
        while do:
            do = False
            birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim = [random.uniform(prior_bound_l[i],prior_bound_u[i]) for i in range(6)]
            w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            # make sure it is a supercritical process (w>0)
            while w<=0:
                birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim = [random.uniform(prior_bound_l[i],prior_bound_u[i]) for i in range(6)]
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            sumsta_vals = comp_sumsta.generate_mbt_data_update([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim], num_tree=1, treesize = subtrees_size[0], startphase = startphase, r=20)
            if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                do = True
        N += 1 # only count the survived processes

        if (min(max_stats_sub-sumsta_vals)>=0 and min(sumsta_vals-min_stats_sub)>=0) or (abs(sumsta_vals-ex_stats_sub)<=tol_stats_sub).all():
            accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim])
            accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]), 1/M_array[0]]))
            accepted_ind = np.concatenate((accepted_ind, sumsta_vals))
            accepted[count_accepted] = accepted_ind
            count_accepted += 1

    end = time.time()
    accept_rate = count_accepted/N
    rate = [accept_rate]
    array = accepted

    # print rate and save results on the path, including the printed results (acceptance rate)
    save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
    save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
    print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s.', flush=True)

    z = 0 # record the tolerance level (count the number of times we decrease the tolerance values)
    accept_rate = 0 # initialise rate

    for k in range(1,3):
        index = np.arange(M_array[k-1]) # label the accepted samples in the previous iteration for proposal distribution in PMC

        array_par = array[:,:6]
        weight = array[:,7]
        covmat = 2*np.cov(array_par.T, aweights=weight)
        if accept_rate >= threshold_rate:
            # tolerance value progression
            tol_stats_sub = np.exp(-decrease_factor)*tol_stats_sub
            accept_rate = 0
            z += 1
        
        accepted = np.zeros((M_array[k], 6+1+1+8)) # accepted parameter values, growth rate, weights, and associated statistics
        count_accepted = 0 # count the number of accepted samples
        N = 0 # count the number of simulations
        start = time.time()
        while count_accepted < M_array[k]:
            do = True
            while do:
                do = False
                ind = random.choices(index, weights=weight, k=1)[0]
                sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                # make sure it is a supercritical process (w>0)
                while w<=0 or (sim_par-prior_bound_l<=0).any() or (prior_bound_u-sim_par<=0).any():
                    ind = random.choices(index, weights=weight, k=1)[0]
                    sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                    w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                sumsta_vals = comp_sumsta.generate_mbt_subtrees_update([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim], subtrees_size=subtrees_size, startphase = startphase, r=20)
                if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                    do = True
            N += 1 # only count the survived processes

            if (abs(sumsta_vals-ex_stats_sub)<=tol_stats_sub).all():
                accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim])
                accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim])]))

                invw=0
                parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
                for item in range(M_array[k-1]):
                    par = array_par[item]
                    invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                weightnew_sim = 1/invw

                accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                accepted[count_accepted] = accepted_ind
                count_accepted += 1

        end = time.time()
        accept_rate = count_accepted/N
        accepted[:,7] = accepted[:,7]/sum(accepted[:,7]) # normalised the weight
        array = accepted

        rate.append(accept_rate)
        # print rate and save results on the path, including the printed results (acceptance rate)
        save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
        save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
        print(f'At iteration {k}, where we use subtrees with sizes ranging from {size1} to {size2}: rate={accept_rate}, duration is {end-start}s, tolerance has decreased {z} times.', flush=True)

    z = 0 # record the tolerance level (count the number of times we decrease the tolerance values)
    accept_rate = 0 # initialise rate

    # From the third iteration, we use the full tree instead of subtrees
    # compute the observed summary statistics
    ex_stats = comp_sumsta.generate_mbt_ind_update(obs_tree)
    max_tspan_bound = 5*ex_stats[3] # set an upper bound for the tree height to avoid the process runs too long
    start_nLTT = 20
    obs_array, obs_resp = comp_sumsta.nabsdiff(obs_tree, start_nLTT)  # nLTT for the first tree, if we have a single observation, the nLTT statistic is computed starting with 20 lineages (averaging over these lineages)
    # the tolerance values are computed from simulations
    # use the parameter estimates from the previous posterior
    nsim4tol = 20
    birth1, birth2, death1, death2, q12, q21 = [comp_sumsta.weighted_mean(array[:,i], weights=array[:,7]) for i in range(6)]
    sumsta_sim4tol = np.zeros((nsim4tol,8))
    for i in range(nsim4tol):
        flag = 0
        while flag == 0:
            t_temp,flag = mbt_bisse_simulator.birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=treesize, start=startphase)
        sumsta_sim4tol[i] = comp_sumsta.generate_mbt_ind_update(t_temp)
    tol_stats = [40*sqrt(variance(sumsta_sim4tol[:,i])) for i in range(8)] # we use large inital tolerance at the start, then shrink them together gradually
    tol_stats = np.array(tol_stats)

    for k in range(3,T_iter):
        index = np.arange(M_array[k-1]) # label the accepted samples in the previous iteration for proposal distribution in PMC

        array_par = array[:,:6]
        weight = array[:,7]
        covmat = 2*np.cov(array_par.T, aweights=weight)
        if accept_rate >= threshold_rate:
            # tolerance value progression
            z += 1
            # re-estimate the parameter values from the previous posterior
            birth1, birth2, death1, death2, q12, q21 = [comp_sumsta.weighted_mean(array[:,i], weights=array[:,7]) for i in range(6)]
            sumsta_sim4tol = np.zeros((nsim4tol,8))
            for i in range(nsim4tol):
                flag = 0
                while flag == 0:
                    t_temp,flag = mbt_bisse_simulator.birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=treesize, start=startphase)
                sumsta_sim4tol[i] = comp_sumsta.generate_mbt_ind_update(t_temp)
            tol_stats = [40*sqrt(variance(sumsta_sim4tol[:,i])) for i in range(8)] # we use large inital tolerance at the start, then shrink them together gradually
            tol_stats = np.array(tol_stats)

            tol_stats = np.exp(-decrease_factor*z)*tol_stats # shrink tolerance value (scale accumulated)
            accept_rate = 0
        
        accepted = np.zeros((M_array[k], 6+1+1+8)) if k<T_nltt else np.zeros((M_array[k], 6+1+1+8+1)) # accepted parameter values, growth rate, weights, and associated statistics
        count_accepted = 0 # count the number of accepted samples
        N = 0 # count the number of simulations
        start = time.time()
        num_accept = M_array[k] if k<T_nltt else 2*M_array[k]
        while count_accepted < num_accept:
            do = True
            while do:
                do = False
                ind = random.choices(index, weights=weight, k=1)[0]
                sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                # make sure it is a supercritical process (w>0)
                while w<=0 or (sim_par-prior_bound_l<=0).any() or (prior_bound_u-sim_par<=0).any():
                    ind = random.choices(index, weights=weight, k=1)[0]
                    sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                    w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                if k < T_nltt:
                    sumsta_vals = comp_sumsta.generate_mbt_data_update([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim], num_tree=1, treesize=treesize, startphase=startphase,r=20)
                else:
                    sumsta_vals = comp_sumsta.generate_mbt_single_tree_update([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim], size=treesize, startphase=startphase, start_nLTT=start_nLTT, max_time_bound=max_tspan_bound)
                if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                    do = True
            N += 1 # only count the survived processes

            if (abs(sumsta_vals[:8]-ex_stats)<=tol_stats).all():
                accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim])
                accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim])]))

                invw=0
                parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
                for item in range(M_array[k-1]):
                    par = array_par[item]
                    invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                weightnew_sim = 1/invw

                accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                accepted[count_accepted] = accepted_ind
                count_accepted += 1

        if k>=T_nltt:
            # we further consider the nLTT statistic
            accept_nLTT_index = np.argpartition(accepted[:,6+1+1+8+1-1], M_array[k])[:M_array[k]]
            accepted = accepted[accept_nLTT_index]
                    
        end = time.time()
        accept_rate = count_accepted/N
        accepted[:,7] = accepted[:,7]/sum(accepted[:,7]) # normalised the weight
        array = accepted

        rate.append(accept_rate)
        # print rate and save results on the path, including the printed results (acceptance rate)
        save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
        save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
        print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s, tolerance has decreased {z} times.', flush=True)



def _irreducible_infer_multitree(obs_trees, experiment, trial_path):
    """
    obs_trees: a list of observed tree
    experiment: an ExpDescriptor object
    """
    trial_path = os.path.join(trial_path, 'results') # the path to the results

    n_trees = experiment.n_trees # the number of trees in the observed dataset
    assert n_trees == len(obs_trees)

    # inference parameters
    T_iter = experiment.n_iter # the number of iterations
    T_nltt = experiment.T_nltt # the number of iteration where we start to consider nLTT statistic
    M_array = experiment.num_accept # a list for the number of accepted samples at each iteration
    startphase = experiment.start_phase # the starting phase
    treesize = experiment.n_leaves # the number of leaves for each tree
    n_trees_iter = experiment.n_trees_iter # the number of trees at each iteration

    prior_bound_l = experiment.prior_bound_l
    prior_bound_u = experiment.prior_bound_u

    threshold_rate = experiment.threshold_rate # threshold acceptance rate
    decrease_factor = experiment.decrease_factor
    tol_nltt = experiment.tol_nltt # the tolerance value for nLTT statistic

    # compute the observed summary statistics
    obs_vals = comp_sumsta.generate_mbt_ind_update(obs_trees[0])
    for i in range(1,n_trees):
        obs_vals = np.vstack((obs_vals, comp_sumsta.generate_mbt_ind_update(obs_trees[i])))

    # nLTT computation (stack multiple trees)
    sim_array, sim_resp = comp_sumsta.nabsdiff(obs_trees[0])  # nLTT for the first tree
    for i in range(1,n_trees):
        init_array, init_resp = comp_sumsta.nabsdiff(obs_trees[i])
        sim_array, sim_resp = comp_sumsta.sumdist_array(init_array, sim_array, init_resp, sim_resp)
    obs_array = sim_array
    obs_resp = sim_resp/n_trees

    ex_stats = np.mean(obs_vals, axis=0)
    max_stats = np.max(obs_vals, axis=0)
    min_stats = np.min(obs_vals, axis=0)
    tol_stats = [2*sqrt(variance(obs_vals[:,i])) for i in range(8)]
    tol_stats = np.array(tol_stats)

    accepted = np.zeros((M_array[0], 6+1+1+8)) # accepted parameter values, growth rate, weights, and associated statistics
    count_accepted = 0 # count the number of accepted samples
    N = 0 # count the number of simulations
    k = 0 # the iteration index
    start = time.time()
    while count_accepted < M_array[0]:
        do = True
        while do:
            do = False
            birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim = [random.uniform(prior_bound_l[i],prior_bound_u[i]) for i in range(6)]
            w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            # make sure it is a supercritical process (w>0)
            while w<=0:
                birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim = [random.uniform(prior_bound_l[i],prior_bound_u[i]) for i in range(6)]
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            sumsta_vals = comp_sumsta.generate_mbt_data_update([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim], num_tree=n_trees_iter[0], treesize = treesize, startphase = startphase, r=20)
            if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                do = True
        N += 1 # only count the survived processes

        if (min(max_stats-sumsta_vals)>=0 and min(sumsta_vals-min_stats)>=0) or (abs(sumsta_vals-ex_stats)<=tol_stats).all():
            accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim])
            accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]), 1/M_array[0]]))
            accepted_ind = np.concatenate((accepted_ind, sumsta_vals))
            accepted[count_accepted] = accepted_ind
            count_accepted += 1

    end = time.time()
    accept_rate = count_accepted/N
    rate = [accept_rate]
    array = accepted

    # print rate and save results on the path, including the printed results (acceptance rate)
    save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
    save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
    print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s.', flush=True)

    z = 0 # record the tolerance level (count the number of times we decrease the tolerance values)
    accept_rate = 0 # initialise rate

    for k in range(1,T_nltt):
        index = np.arange(M_array[k-1]) # label the accepted samples in the previous iteration for proposal distribution in PMC

        array_par = array[:,:6]
        weight = array[:,7]
        covmat = 2*np.cov(array_par.T, aweights=weight)
        if accept_rate >= threshold_rate:
            # tolerance value progression
            tol_stats = np.exp(-decrease_factor)*tol_stats
            accept_rate = 0
            z += 1
        
        accepted = np.zeros((M_array[k], 6+1+1+8)) # accepted parameter values, growth rate, weights, and associated statistics
        count_accepted = 0 # count the number of accepted samples
        N = 0 # count the number of simulations
        start = time.time()
        while count_accepted < M_array[k]:
            do = True
            while do:
                do = False
                ind = random.choices(index, weights=weight, k=1)[0]
                sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                # make sure it is a supercritical process (w>0)
                while w<=0 or (sim_par-prior_bound_l<=0).any() or (prior_bound_u-sim_par<=0).any():
                    ind = random.choices(index, weights=weight, k=1)[0]
                    sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                    w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                sumsta_vals = comp_sumsta.generate_mbt_data_update([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim], num_tree=n_trees_iter[k], treesize = treesize, startphase = startphase, r=20)
                if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                    do = True
            N += 1 # only count the survived processes

            if (abs(sumsta_vals-ex_stats)<=tol_stats).all():
                accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim])
                accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim])]))

                invw=0
                parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
                for item in range(M_array[k-1]):
                    par = array_par[item]
                    invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                weightnew_sim = 1/invw

                accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                accepted[count_accepted] = accepted_ind
                count_accepted += 1

        end = time.time()
        accept_rate = count_accepted/N
        accepted[:,7] = accepted[:,7]/sum(accepted[:,7]) # normalised the weight
        array = accepted

        rate.append(accept_rate)
        # print rate and save results on the path, including the printed results (acceptance rate)
        save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
        save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
        print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s, tolerance has decreased {z} times.', flush=True)

    accept_rate = 0 # initialise rate
    for k in range(T_nltt, T_iter):
        index = np.arange(M_array[k-1]) # label the accepted samples in the previous iteration for proposal distribution in PMC

        array_par = array[:,:6]
        weight = array[:,7]
        covmat = 2*np.cov(array_par.T, aweights=weight)
        if accept_rate >= threshold_rate:
            # tolerance value progression
            tol_stats = np.exp(-decrease_factor)*tol_stats
            tol_nltt = np.exp(-decrease_factor)*tol_nltt
            accept_rate = 0
            z += 1
        
        accepted = np.zeros((M_array[k], 6+1+1+8)) # accepted parameter values, growth rate, weights, and associated statistics
        count_accepted = 0 # count the number of accepted samples
        N = 0 # count the number of simulations
        start = time.time()
        while count_accepted < M_array[k]:
            do = True
            while do:
                do = False
                ind = random.choices(index, weights=weight, k=1)[0]
                sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                # make sure it is a supercritical process (w>0)
                while w<=0 or (sim_par-prior_bound_l<=0).any() or (prior_bound_u-sim_par<=0).any():
                    ind = random.choices(index, weights=weight, k=1)[0]
                    sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim = sim_par
                    w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                sumsta_vals, sim_array, sim_resp = comp_sumsta.generate_mbt_data_update_nLTT([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim], num_tree=n_trees_iter[k], treesize = treesize, startphase = startphase, r=20)
                if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                    do = True
            N += 1 # only count the survived processes

            if (abs(sumsta_vals-ex_stats)<=tol_stats).all() and comp_sumsta.absdist_array(obs_array, sim_array, obs_resp, sim_resp) <= tol_nltt:
                accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim])
                accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim])]))

                invw=0
                parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
                for item in range(M_array[k-1]):
                    par = array_par[item]
                    invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                weightnew_sim = 1/invw

                accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                accepted[count_accepted] = accepted_ind
                count_accepted += 1

        end = time.time()
        accept_rate = count_accepted/N
        accepted[:,7] = accepted[:,7]/sum(accepted[:,7]) # normalised the weight
        array = accepted

        rate.append(accept_rate)
        # print rate and save results on the path, including the printed results (acceptance rate)
        save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
        save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
        print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s, tolerance has decreased {z} times.', flush=True)






    
