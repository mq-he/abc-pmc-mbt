# inference process for reducible processes
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
    _reducible_infer_multitree(obs_trees, experiment, trial_path)




def _reducible_infer_multitree(obs_trees, experiment, trial_path):
    """
    obs_tree: a list of observed trees
    experiment: an ExpDescriptor object
    """
    trial_path = os.path.join(trial_path, 'results') # the path to the results

    n_trees = experiment.n_trees # the number of trees in the observed dataset
    if n_trees < 3:
        raise ValueError("{n_trees} observed trees are insufficient for inference under reducible processes.")
    assert n_trees == len(obs_trees)

    N_PAR = 5
    N_STATS = 7

    # inference parameters
    T_iter = experiment.n_iter # the number of iterations
    T_nltt = experiment.T_nltt # the number of iteration where we start to consider nLTT statistic
    M_array = experiment.num_accept # a list for the number of accepted samples at each iteration
    startphase = experiment.start_phase # the starting phase
    treesize = experiment.n_leaves # the number of leaves for each tree
    n_trees_iter = experiment.n_trees_iter # the number of trees at each iteration

    assert startphase==1, "The process has to start from phase 1, since only transitions from phase 1 to phase 2 are allowed."

    prior_bound_l = experiment.prior_bound_l
    prior_bound_u = experiment.prior_bound_u

    threshold_rate = experiment.threshold_rate # threshold acceptance rate
    decrease_factor = experiment.decrease_factor
    tol_nltt = experiment.tol_nltt # the tolerance value for nLTT statistic
    
    # compute the observed summary statistics
    obs_vals = comp_sumsta.generate_mbt_ind_reducible_update(obs_trees[0])
    for i in range(1,n_trees):
        obs_vals = np.vstack((obs_vals, comp_sumsta.generate_mbt_ind_reducible_update(obs_trees[i])))


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
    tol_stats = [2*sqrt(variance(obs_vals[:,i])) for i in range(N_STATS)]
    tol_stats = np.array(tol_stats)

    # filter the zeros in R1, R2 statistic
    ex_stats[5] = sum(obs_vals[:,5])/np.count_nonzero(obs_vals[:,5])
    ex_stats[6] = sum(obs_vals[:,6])/np.count_nonzero(obs_vals[:,6])
    tol_stats[5] = 2*sqrt(variance(obs_vals[np.nonzero(obs_vals[:,5]),5][0]))
    tol_stats[6] = 2*sqrt(variance(obs_vals[np.nonzero(obs_vals[:,6]),6][0]))

    accepted = np.zeros((M_array[0], N_PAR+1+1+N_STATS)) # accepted parameter values, growth rate, weights, and associated statistics
    count_accepted = 0 # count the number of accepted samples
    N = 0 # count the number of simulations
    k = 0 # the iteration index
    start = time.time()
    # for the first iteration with n=1, we do not consider the R1 and R2 statistic, we only consider them when n>=10
    while count_accepted < M_array[k]:
        do = True
        while do:
            do = False
            birth1sim, birth2sim,death1sim,death2sim,q12sim = [random.uniform(prior_bound_l[i],prior_bound_u[i]) for i in range(5)]
            w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])  
            # make sure it is a supercritical process (w>0)
            while w<=0:
                birth1sim, birth2sim,death1sim,death2sim,q12sim = [random.uniform(prior_bound_l[i],prior_bound_u[i]) for i in range(5)]
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])  
            sumsta_vals = comp_sumsta.generate_mbt_data_reducible_update([birth1sim, birth2sim,death1sim,death2sim,q12sim,0], num_tree=n_trees_iter[k], treesize = treesize, startphase = startphase, r=20)
            if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                do = True
        N += 1 # only count the survived processes
        if n_trees_iter[k]<10:
            if (min(max_stats[:5]-sumsta_vals[:5])>=0 and min(sumsta_vals[:5]-min_stats[:5])>=0) or (abs(sumsta_vals[:5]-ex_stats[:5])<=tol_stats[:5]).all():
                accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim])
                accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim, 0]), 1/M_array[0]]))
                accepted_ind = np.concatenate((accepted_ind, sumsta_vals))
                accepted[count_accepted] = accepted_ind
                count_accepted += 1
        elif sumsta_vals[5]>0: # if we have observed phase-1 leaves
            if (abs(sumsta_vals-ex_stats)<=tol_stats).all():
                accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim])
                accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim, 0]), 1/M_array[0]]))
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

        array_par = array[:,:N_PAR]
        weight = array[:,N_PAR+1]
        covmat = 2*np.cov(array_par.T, aweights=weight)
        if accept_rate >= threshold_rate:
            # tolerance value progression
            tol_stats = np.exp(-decrease_factor)*tol_stats
            accept_rate = 0
            z += 1
        
        accepted = np.zeros((M_array[k], N_PAR+1+1+N_STATS)) # accepted parameter values, growth rate, weights, and associated statistics
        count_accepted = 0 # count the number of accepted samples
        N = 0 # count the number of simulations
        start = time.time()
        while count_accepted < M_array[k]:
            do = True
            while do:
                do = False
                ind = random.choices(index, weights=weight, k=1)[0]
                sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(N_PAR),covmat)
                birth1sim,birth2sim,death1sim,death2sim,q12sim = sim_par
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])  
                # make sure it is a supercritical process (w>0)
                while w<=0 or (sim_par-prior_bound_l<=0).any() or (prior_bound_u-sim_par<=0).any():
                    ind = random.choices(index, weights=weight, k=1)[0]
                    sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(N_PAR),covmat)
                    birth1sim,birth2sim,death1sim,death2sim,q12sim = sim_par
                    w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])  
                sumsta_vals = comp_sumsta.generate_mbt_data_reducible_update([birth1sim,birth2sim,death1sim,death2sim,q12sim,0], num_tree=n_trees_iter[k], treesize = treesize, startphase = startphase, r=20)
                if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                    do = True
            N += 1 # only count the survived processes

            if n_trees_iter[k]<10:
                if (abs(sumsta_vals[:5]-ex_stats[:5])<=tol_stats[:5]).all():
                    accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim])
                    accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,0])]))

                    invw=0
                    parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim])
                    for item in range(M_array[k-1]):
                        par = array_par[item]
                        invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                    weightnew_sim = 1/invw

                    accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                    accepted[count_accepted] = accepted_ind
                    count_accepted += 1
            elif sumsta_vals[5]>0: # if we have observed phase-1 leaves
                if (abs(sumsta_vals-ex_stats)<=tol_stats).all():
                    accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim])
                    accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,0])]))

                    invw=0
                    parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim])
                    for item in range(M_array[k-1]):
                        par = array_par[item]
                        invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                    weightnew_sim = 1/invw

                    accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                    accepted[count_accepted] = accepted_ind
                    count_accepted += 1

        end = time.time()
        accept_rate = count_accepted/N
        accepted[:,N_PAR+1] = accepted[:,N_PAR]/sum(accepted[:,N_PAR+1]) # normalised the weight
        array = accepted

        rate.append(accept_rate)
        # print rate and save results on the path, including the printed results (acceptance rate)
        save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
        save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
        print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s, tolerance has decreased {z} times.', flush=True)

    accept_rate = 0 # initialise rate
    for k in range(T_nltt, T_iter):
        index = np.arange(M_array[k-1]) # label the accepted samples in the previous iteration for proposal distribution in PMC

        array_par = array[:,:N_PAR]
        weight = array[:,N_PAR+1]
        covmat = 2*np.cov(array_par.T, aweights=weight)
        if accept_rate >= threshold_rate:
            # tolerance value progression
            tol_stats = np.exp(-decrease_factor)*tol_stats
            tol_nltt = np.exp(-decrease_factor)*tol_nltt
            accept_rate = 0
            z += 1

        accepted = np.zeros((M_array[k], N_PAR+1+1+N_STATS)) # accepted parameter values, growth rate, weights, and associated statistics
        count_accepted = 0 # count the number of accepted samples
        N = 0 # count the number of simulations
        start = time.time()
        while count_accepted < M_array[k]:
            do = True
            while do:
                do = False
                ind = random.choices(index, weights=weight, k=1)[0]
                sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(N_PAR),covmat)
                birth1sim,birth2sim,death1sim,death2sim,q12sim = sim_par
                w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])  
                # make sure it is a supercritical process (w>0)
                while w<=0 or (sim_par-prior_bound_l<=0).any() or (prior_bound_u-sim_par<=0).any():
                    ind = random.choices(index, weights=weight, k=1)[0]
                    sim_par = array_par[ind]+multivariate_normal.rvs(np.zeros(N_PAR),covmat)
                    birth1sim,birth2sim,death1sim,death2sim,q12sim = sim_par
                    w = mbt_bisse_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,0])  
                sumsta_vals, sim_array, sim_resp = comp_sumsta.generate_mbt_data_reducible_update_nLTT([birth1sim,birth2sim,death1sim,death2sim,q12sim,0], num_tree=n_trees_iter[k], treesize = treesize, startphase = startphase, r=20)
                if sum(sumsta_vals) == 0: # autorejection when the process extinct after r attempts (r=20), that is, if the stats are all 0
                    do = True
            N += 1 # only count the survived processes

            if n_trees_iter[k]<10:
                if (abs(sumsta_vals[:5]-ex_stats[:5])<=tol_stats[:5]).all() and comp_sumsta.absdist_array(obs_array, sim_array, obs_resp, sim_resp) <= tol_nltt:
                    accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim])
                    accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,0])]))

                    invw=0
                    parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim])
                    for item in range(M_array[k-1]):
                        par = array_par[item]
                        invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                    weightnew_sim = 1/invw

                    accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                    accepted[count_accepted] = accepted_ind
                    count_accepted += 1
            elif sumsta_vals[5]>0: # if we have observed phase-1 leaves
                if (abs(sumsta_vals-ex_stats)<=tol_stats).all() and comp_sumsta.absdist_array(obs_array, sim_array, obs_resp, sim_resp) <= tol_nltt:
                    accepted_ind = np.array([birth1sim, birth2sim,death1sim,death2sim,q12sim])
                    accepted_ind = np.concatenate((accepted_ind, [mbt_bisse_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,0])]))

                    invw=0
                    parsim = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim])
                    for item in range(M_array[k-1]):
                        par = array_par[item]
                        invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
                    weightnew_sim = 1/invw

                    accepted_ind = np.concatenate((accepted_ind, [weightnew_sim], sumsta_vals))
                    accepted[count_accepted] = accepted_ind
                    count_accepted += 1

        end = time.time()
        accept_rate = count_accepted/N
        accepted[:,N_PAR+1] = accepted[:,N_PAR+1]/sum(accepted[:,N_PAR+1]) # normalised the weight
        array = accepted

        rate.append(accept_rate)
        # print rate and save results on the path, including the printed results (acceptance rate)
        save(os.path.join(trial_path, f'accepted_iter{k}.csv'), array)
        save(os.path.join(trial_path, 'accept_rate.csv'), rate) # update the acceptance rate
        print(f'At iteration {k}: rate={accept_rate}, duration is {end-start}s, tolerance has decreased {z} times.', flush=True)



