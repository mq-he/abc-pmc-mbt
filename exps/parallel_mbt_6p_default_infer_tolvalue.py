# This script is for inferring parameters in the irreducible process using
# ABC-PMC with nine selected summary statistics *in parallel*

# here, we set r=20, allowing up to 20 attempts for extinction
# we use the psedo-observed values which are generated from a separate file

# *when we calculate the acceptance rate, we include the extinct process as well*

from mpi4py import MPI
from ete3 import Tree
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, variance
from math import sqrt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import sys
import time
import gc
import os

# Get the path of the current script
FILEPATH = os.path.abspath(__file__)
PATH = os.path.dirname(FILEPATH)

import mbt_6p_simulator

# (need to change) change the observed dataset 

# read obs values from csv file
trial_index = int(sys.argv[1]) # trial id
treesize = 5000 #int(sys.argv[2]) # the number of leaves of the tree

M=100 # number of the accepted samples in each iteration (increase this later)
T=10
startphase = 1


# the required functions for the large mbt tree
def break_tree(t, min_tree, max_tree):
    """ Break a large tree into small chunks, whose tree sizes stay between
    min_tree and max_tree
    """
    chunks = []
    if len(t.get_leaves()) < min_tree:
        return chunks
    elif min_tree <= len(t.get_leaves()) <= max_tree:
        chunks.append(t)
        return chunks
    else:
        t_child1 = t.get_children()[0]
        t_child2 = t.get_children()[1]
        chunks1 = break_tree(t_child1,min_tree,max_tree)
        chunks2 = break_tree(t_child2,min_tree,max_tree)
        chunks = chunks1+chunks2
        return chunks

# required functions for PMC
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def weighted_mean(array,weights):
    """
    Return the weighted average.

    array, weights -- lists.
    """
    numerator = sum([array[i]*weights[i] for i in range(len(array))])
    denominator = sum(weights)

    return numerator/denominator


def weighted_var(array,weights):
    """
    Return the unbiased weighted variance.

    array, weights -- lists.
    """
    v1 = sum(weights)
    v2 = sum([weights[i]**2 for i in range(len(weights))])

    wmean = weighted_mean(array,weights)
    numerator = sum([((array[i]-wmean)**2)*weights[i] for i in range(len(array))])
    denominator = v1-(v2/v1)

    return numerator/denominator

# need a function that use inputs as parameter values, size_nodes, and outputs summary statistics
def generate_ss_subtrees(samples, sizeobs, startphase, r=20):
    assert len(samples.shape)==2
    n_sim = samples.shape[0]
    # only record the eight ss
    output_all = np.empty(0)
    for trial in range(n_sim):
        sample = samples[trial]
        b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)
        for i,size in enumerate(sizeobs):
            if i==0:
                tree_sim, flag_sim = mbt_6p_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r)
                if flag_sim == 0:
                    output_samp = np.zeros(8)
                    break
                else:
                    output_samp = mbt_6p_simulator.generate_mbt_ind_update(tree_sim)
            else:
                flag_sim = 0
                while flag_sim == 0:
                    tree_sim, flag_sim = mbt_6p_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r)
                output = mbt_6p_simulator.generate_mbt_ind_update(tree_sim) # stats calculation
                output_samp = np.vstack((output_samp, output))
            
        if flag_sim == 1: # the process survives
            output_samp = np.mean(output_samp,axis=0)

        output_all = np.concatenate((output_all, output_samp))
    
    return output_all

# need a function that use inputs as parameter values, size, only generate one large tree, and outputs its summary statistics
def generate_ss_single_tree(samples, size, startphase, r=20, max_time_bound=None):
    assert len(samples.shape)==2
    n_sim = samples.shape[0]
    # only record the eight ss
    output_all = np.empty(0)
    for trial in range(n_sim):
        sample = samples[trial]
        b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)
        tree_sim, flag_sim = mbt_6p_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r, max_time=max_time_bound)
        if flag_sim == 0:
            output_samp = np.zeros(8)
        elif len(tree_sim.get_leaves())!=size:
            output_samp = np.zeros(8)
        else:
            output_samp = mbt_6p_simulator.generate_mbt_ind_update(tree_sim)

        output_all = np.concatenate((output_all, output_samp))
    
    return output_all

# need a function that use inputs as parameter values, size, observed nLTT curve, only generate one large tree, and outputs its summary statistics as well as its nLTT statistic
def generate_ss_single_tree_nLTT(samples, size, obs_array, obs_resp, startphase, r=20, start_nLTT=20, max_time_bound=None):
    assert len(samples.shape)==2
    n_sim = samples.shape[0]
    # only record the eight ss
    output_all = np.empty(0)
    for trial in range(n_sim):
        sample = samples[trial]
        b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)
        tree_sim, flag_sim = mbt_6p_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r, max_time=max_time_bound)
        if flag_sim == 0:
            output_samp = np.zeros(8+1) # add one for the nLTT stat
        elif len(tree_sim.get_leaves())!=size:
            output_samp = np.zeros(8+1)
        else:
            output_samp_stat = mbt_6p_simulator.generate_mbt_ind_update(tree_sim) # stats calculation
            sim_array, sim_resp = mbt_6p_simulator.nabsdiff(tree_sim, start=start_nLTT)  # nLTT computation
            nLTTstat = mbt_6p_simulator.absdist_array(obs_array, sim_array, obs_resp, sim_resp)
            output_samp = np.append(output_samp_stat, nLTTstat) # append a value to the end of the array and create a new one

        output_all = np.concatenate((output_all, output_samp))
    
    return output_all

def convert2ss(data, partition_item, n_covs):
    """
    Convert the cpus times partition_time*n_covs matrix to
    a n_samples time n_covs matrix
    """

    cpus = np.shape(data)[0]
    output = np.empty((cpus*partition_item, n_covs))
    # the points partitioned with distance = n_covs, and we have partition_item num of repeats
    arr = np.array(range(0, n_covs*partition_item, n_covs)) 
    for p in range(n_covs): # for each covariate
        for i in range(cpus):
            output[i*partition_item:(i+1)*partition_item, p] = data[i,p+arr]
    return output

def compute_dist(data, weight):
    # compute the Euclidean distance after the reweighting
    # data must a 2d array, with shape = (N_SAMP, N_STA)
    # weight must be a 1d array with shape = (N_STA,)
    N_SAMP = data.shape[0]
    assert data.shape[1] == len(weight)

    weight = np.array(weight) # make sure the weight is an array

    dist_data = np.zeros(N_SAMP)
    for i in range(N_SAMP):
        dist_data[i] = sqrt(sum((data[i]/weight)**2))

    return dist_data
        


# step 1 (R0): find parameter values from previous samples and their weights
# step 2 (R): distribute the parameter values in the array (distribute the index) [comm.Bcast, comm.Gather]

n_covs = 8

comm = MPI.COMM_WORLD
cpus = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    # we use pseudo-observation generated from the default parameter values
    # read the tree from newick file
    t_read = Tree(PATH+'/newick_obs/tree_size'+str(treesize)+'_test'+str(trial_index)+'.txt')
    t_read_phase = np.loadtxt(PATH+'/newick_obs/phase_size'+str(treesize)+'_test'+str(trial_index)+'.csv', delimiter=",")
    # assign the phases
    for leaf in t_read.get_leaves():
        index = int(leaf.name)-1
        leaf.phase = t_read_phase[index]
    # after assigning the phases, the mbt tree is now good to go

    t_obs=t_read
    size1,size2=20,50
    tarr = break_tree(t_obs,size1,size2)

    # make sure all statistics have been scaled w.r.t. number of leaves
    balobs=[]
    bal1obs=[]
    bal2obs=[]
    sizeobs=[]
    tspanobs=[]
    distanceobs=[]
    statobs=[]
    stat2obs=[]
    propobs=[]
    for node in tarr:
        size_node = len(node.get_leaves())
        sizeobs.append(size_node)
        sumsta_vals = mbt_6p_simulator.generate_mbt_ind_update(node)
        b,b1,b2, tspan_ind, dist_ind, prop_ind, t1, t2 = sumsta_vals

        balobs.append(b)
        bal1obs.append(b1)
        bal2obs.append(b2)
        distanceobs.append(dist_ind)
        statobs.append(t1)
        stat2obs.append(t2)
        propobs.append(prop_ind)
        tspanobs.append(tspan_ind)


    #initialise the size of the simulation
    size= len(t_obs.get_leaves()) # tree size


    ex_dist = mean(distanceobs)
    ex_imb = mean(balobs)
    ex_imb1 = mean(bal1obs)
    ex_imb2 = mean(bal2obs)
    ex_span = mean(tspanobs)
    ex_prop = mean(propobs)
    ex_stat = mean(statobs)
    ex_stat2 = mean(stat2obs)


    tol_dist = 2*sqrt(variance(distanceobs))
    # tol = 0.05 # for large trees, nLTT are hard to match and it is difficult to set a threshold value for arbitrary process with reasonable acceptance rate
    tol_imb = 2*sqrt(variance(balobs))
    tol_imb1 = 2*sqrt(variance(bal1obs))
    tol_imb2 = 2*sqrt(variance(bal2obs))
    tol_tspan = 2*sqrt(variance(tspanobs))
    tol_prop = 2*sqrt(variance(propobs))
    tol_stat = 2*sqrt(variance(statobs))
    tol_stat2 = 2*sqrt(variance(stat2obs))

    max_dist = max(distanceobs)
    max_imb = max(balobs)
    max_imb1 = max(bal1obs)
    max_imb2 = max(bal2obs)
    max_span = max(tspanobs)
    max_prop = max(propobs)
    max_stat = max(statobs)
    max_stat2 = max(stat2obs)

    min_dist = min(distanceobs)
    min_imb = min(balobs)
    min_imb1 = min(bal1obs)
    min_imb2 = min(bal2obs)
    min_span = min(tspanobs)
    min_prop = min(propobs)
    min_stat = min(statobs)
    min_stat2 = min(stat2obs)

    N=0
    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    domtemp = []

    start = time.time()

    while len(accept1)<200:
        do = True
        while do:
            do = False
            birth1sim = random.uniform(0,5)
            birth2sim = random.uniform(0,5)
            death1sim = random.uniform(0,5)
            death2sim = random.uniform(0,5)
            q12sim = random.uniform(0,5)
            q21sim = random.uniform(0,5)
            w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0:
                birth1sim = random.uniform(0,5)
                birth2sim = random.uniform(0,5)
                death1sim = random.uniform(0,5)
                death2sim = random.uniform(0,5)
                q12sim = random.uniform(0,5)
                q21sim = random.uniform(0,5)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            # only use a single tree (the first one)
            t, flag = mbt_6p_simulator.birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=sizeobs[0], start=startphase, r=20)
            if flag == 0: # autorejection when the process extinct after 20 attempts
                do = True
                
        assert flag == 1
        sumsta_vals = mbt_6p_simulator.generate_mbt_ind_update(t)
        # sim_array, sim_resp = mbt_6p_simulator.nabsdiff(t) # compute nLTT after the 20th iteration
        mbala, mbala1, mbala2, mtspan, mdist, mprop, mstat, mstat2 = sumsta_vals
        N+=1
        
        if min_span <= mtspan <= max_span and min_dist<=mdist<=max_dist and min_imb<=mbala<=max_imb and min_imb1<=mbala1<=max_imb1 and min_imb2<=mbala2<=max_imb2 and min_prop <= mprop <= max_prop and min_stat <= mstat <= max_stat and min_stat2 <= mstat2 <= max_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)

            domtemp.append(mbt_6p_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
            
    accept_rate = len(accept1)/N

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6

    array_all = np.stack((array1,array2,array3,array4,array5,array6)) # array_all records all accepted parameter values from the past iteration
    array_all = array_all.T # in shape (200,6) 

    k=0 # iteration index

    end=time.time()
    duration=[end-start]
    rate = [accept_rate]

    domtemp = np.array(domtemp)
    domtemp = domtemp[:, np.newaxis]

    results_all = np.hstack((array_all, domtemp, np.ones((200,1))/200)) # 6 parameters, growth rate, weights, with shape=(200,8)
    np.savetxt(PATH+'/results/mbt_tolvalue_size'+str(treesize)+'_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")
    print(f'duration={duration}')
    print('\n\n'+'rate='+str(rate), flush=True)

if rank == 0:
    num_sim = int((10*M)//cpus+1)*cpus # for t=2,3, mean sure it is larger than int(num_sim/cpus)
    num_subtree = len(sizeobs) # need to preallocate memory for array-like object
    accept_rate = 0
    tolvalue_scale = 0 # increase if the accept_rate>0.03
    print('num_sim'+str(num_sim))
else: 
    num_sim = None
    num_subtree = None
    startphase = None
num_sim = comm.bcast(num_sim, root=0)
num_subtree = comm.bcast(num_subtree, root=0)
startphase = comm.bcast(startphase, root=0)

if rank == 0:
    sizeobs = np.array(sizeobs, dtype='int64')
else:
    sizeobs = np.empty(num_subtree, dtype='int64')
comm.Bcast(sizeobs, root=0)

for k in range(1,3):
    len_accept = 0 # record the number of accepted samples
    if rank == 0: # compute the parameter values
        print('len_accept'+str(len_accept))
        start = time.time()
        N_total = 0 # count the number of nonextinct process
        n = array_all.shape[0]
        index = list(range(n))
        if k==1:
            weight = [1]*n

        weightnew = []
        
        # for covmat, the first argument, the dataset need to have shape=(N_PAR, M), array_all has shape = (M,N_PAR), so we use its transpose.
        covmat = 2*np.cov(array_all.T, aweights=weight) 

        # resample from prior until we reach the number of accepted samples

        accept_all = np.empty((0,6)) # parameters; dominant value and weights are added later

    # Determine the workload for each process
    partition_item = num_sim // cpus 

    start_index = rank * partition_item 
    end_index = start_index + partition_item

    while len_accept<M:
        if rank==0:
            mbt_par_sim = np.zeros((num_sim, 6), dtype="float") 
            print('mbt_par_sim'+str(mbt_par_sim))
            for i in range(num_sim):
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=array_all[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                    ind = random.choices(index, weights=weight, k=1)[0]
                    [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=array_all[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
                # save the generated parameter values
                mbt_par_sim[i] = [birth1sim, birth2sim,death1sim, death2sim,q12sim,q21sim]
            print('mbt_par_sim'+str(mbt_par_sim))

            # need to distributed the parameter value
        else:
            mbt_par_sim = np.empty((num_sim, 6), dtype="float")
        comm.Bcast(mbt_par_sim, root=0)
        
        sendbuf = generate_ss_subtrees(mbt_par_sim[start_index:end_index], sizeobs, startphase, r=20)
        sendbuf = np.array(sendbuf, dtype="float")

        recvbuf = None
        if rank == 0:
            recvbuf = np.zeros((cpus, partition_item*n_covs), dtype="float")
        comm.Gather(sendbuf, recvbuf, root=0) # at rank 0, recvbuf has all the statistics value for the parameters we generated

    
        if rank == 0:
            output = convert2ss(recvbuf, partition_item, n_covs) # output now has shape (partition_item*cpus, n_covs)
            # remove the simulation that extincts from both statistics and parameter sets
            survived_index = output[:,3]!=0
            N_total += sum(survived_index) # record the total number of survived processes
            print(f"the number of extincted processes is {partition_item*cpus- sum(survived_index)}")
            print(f"the number of accepted samples is {len_accept}")
            output = output[survived_index]
            mbt_par_sim = mbt_par_sim[survived_index]

            assert output.shape[1]==8

            # compute the statistic, reweighting according to the estimated variance of each statistic
            ex_stats = np.array([ex_imb, ex_imb1, ex_imb2, ex_span, ex_dist, ex_prop, ex_stat, ex_stat2])
            tol_stats = np.array([tol_imb, tol_imb1, tol_imb2, tol_tspan, tol_dist, tol_prop, tol_stat, tol_stat2])
            
            # update tolerance values if accept_rate>0.03
            if accept_rate>0.03:
                tolvalue_scale += 1
                accept_rate = 0
            tol_stats = tol_stats*math.exp(-0.2*tolvalue_scale)

            check_tol = abs(output-ex_stats)/tol_stats
            check_max = np.max(check_tol, axis=1)

            assert len(check_max)==sum(survived_index)

            accept_index = check_max<1

            accept_all = np.vstack((accept_all, mbt_par_sim[accept_index]))

            len_accept = accept_all.shape[0]

        len_accept = comm.bcast(len_accept, root=0)

    if rank==0:
        accept_rate = len_accept/N_total
        if len_accept>M:
            accept_all = accept_all[:M]
            len_accept = M
    len_accept = comm.bcast(len_accept, root=0)

    if rank==0:
        # for each sample, compute its dominant eigen value and its weight
        domtemp = np.zeros(M)
        for i in range(M):
            domtemp[i] = mbt_6p_simulator.dom(accept_all[i,0:2],accept_all[i,2:4],accept_all[i,4:6])
            invw=0
            parsim = accept_all[i]
            for item in range(len(weight)):
                par = array_all[item]
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)

        weightnew = np.array(weightnew)/sum(weightnew)
        end = time.time()
        duration.append(end-start)
        rate.append(accept_rate)
        # hstack only works for array with same number of dimension, shape=(M, 1), same for domtemp 

        results_all = np.hstack((accept_all, domtemp[:,np.newaxis], weightnew[:,np.newaxis])) # 6 parameters, growth rate, weights, with shape=(M,8)
        np.savetxt(PATH+'/results/mbt_tolvalue_size'+str(treesize)+'_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")   
        print('\n\n'+'k='+str(k))
        print(f'duration={duration}')
        print('\n\n'+'rate='+str(rate), flush=True)

        array_all = accept_all
        weight = weightnew

if rank == 0:
    num_sim = int((10*M)//cpus+1)*cpus # for mean sure it is larger than int(num_sim/cpus)
    accept_rate = 0
    tolvalue_scale = 0
else: 
    num_sim = None
    size = None # tree size for the large tree
num_sim = comm.bcast(num_sim, root=0)
size = comm.bcast(size, root=0)

if rank == 0:
    # recompute the tolerance
    # after the third iteration, we start to work with the entire tree
    # re-calculate the observed statistics
    final_array, final_resp = mbt_6p_simulator.nabsdiff(t_obs,20)
    sumsta_vals_obs = mbt_6p_simulator.generate_mbt_ind_update(t_obs)
    ex_imb_full,ex_imb1_full,ex_imb2_full, ex_tspan_full, ex_dist_full, ex_prop_full, ex_stat_full,ex_stat2_full = sumsta_vals_obs

    # re-calculate the tolerance values

    birth1=weighted_mean(array_all[:,0], weight)
    birth2=weighted_mean(array_all[:,1], weight)
    death1=weighted_mean(array_all[:,2], weight)
    death2=weighted_mean(array_all[:,3], weight)
    q12=weighted_mean(array_all[:,4], weight)
    q21=weighted_mean(array_all[:,5], weight)

    balobs=[]
    bal1obs=[]
    bal2obs=[]
    tspanobs=[]
    distanceobs=[]
    statobs=[]
    stat2obs=[]
    propobs=[]

    for i in range(20):
        flag = 0
        while flag == 0:
            tsim,flag = mbt_6p_simulator.birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size, start=startphase)
        sumsta_vals = mbt_6p_simulator.generate_mbt_ind_update(tsim)
        b,b1,b2, tspan_ind, dist_ind, prop_ind, t1,t2 = sumsta_vals

        balobs.append(b)
        bal1obs.append(b1)
        bal2obs.append(b2)
        distanceobs.append(dist_ind)
        tspanobs.append(tspan_ind)
        propobs.append(prop_ind)
        statobs.append(t1)
        stat2obs.append(t2)

    # since we use tol only to weight the importance of the statistics, the scale does not matter
    tol_dist = 40*sqrt(variance(distanceobs))
    tol_imb = 40*sqrt(variance(balobs))
    tol_imb1 = 40*sqrt(variance(bal1obs))
    tol_imb2 = 40*sqrt(variance(bal2obs))
    tol_tspan = 40*sqrt(variance(tspanobs))
    tol_prop = 40*sqrt(variance(propobs))
    tol_stat = 40*sqrt(variance(statobs))
    tol_stat2 = 40*sqrt(variance(stat2obs))

# broadcast the upper bound for tree height
if rank == 0:
    max_tspan_bound = 5*ex_tspan_full
else: 
    max_tspan_bound = None
max_tspan_bound = comm.bcast(max_tspan_bound, root=0) # later, we need to ensure tree_size is successfully simulated

for k in range(3,T):
    gc.collect()
    len_accept = 0 # record the number of accepted samples\
    if rank == 0: # compute the parameter values
        start = time.time()
        N_total = 0 # count the number of nonextinct process
        n = array_all.shape[0]
        index = list(range(n))
        if k==1:
            weight = [1]*n

        weightnew = []
        
        # for covmat, the first argument, the dataset need to have shape=(N_PAR, M), array_all has shape = (M,N_PAR), so we use its transpose.
        covmat = 2*np.cov(array_all.T, aweights=weight) 

        # resample from prior until we reach the number of accepted samples

        accept_all = np.empty((0,6)) # parameters; dominant value and weights are added later

    # Determine the workload for each process
    partition_item = num_sim // cpus 

    start_index = rank * partition_item 
    end_index = start_index + partition_item

    while len_accept<M:
        if rank==0:
            mbt_par_sim = np.zeros((num_sim, 6), dtype="float") # tolerance rate is 2%, including extinct process, only the eight stats
            for i in range(num_sim):
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=array_all[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                    ind = random.choices(index, weights=weight, k=1)[0]
                    [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=array_all[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
                # save the generated parameter values
                mbt_par_sim[i] = [birth1sim, birth2sim,death1sim, death2sim,q12sim,q21sim]

            # need to distributed the parameter value
        else:
            mbt_par_sim = np.empty((num_sim, 6), dtype="float")
        comm.Bcast(mbt_par_sim, root=0)
        
        sendbuf = generate_ss_single_tree(mbt_par_sim[start_index:end_index], size, startphase, r=20, max_time_bound=max_tspan_bound)
        sendbuf = np.array(sendbuf, dtype="float")

        recvbuf = None
        if rank == 0:
            recvbuf = np.zeros((cpus, partition_item*n_covs), dtype="float")
        comm.Gather(sendbuf, recvbuf, root=0) # at rank 0, recvbuf has all the statistics value for the parameters we generated

    
        if rank == 0:
            output = convert2ss(recvbuf, partition_item, n_covs) # output now has shape (partition_item*cpus, n_covs)
            # remove the simulation that extincts from both statistics and parameter sets
            survived_index = output[:,3]!=0
            N_total += sum(survived_index) # record the total number of survived processes
            print(f"the number of extincted processes is {partition_item*cpus- sum(survived_index)}")
            print(f"the number of accepted samples is {len_accept}")
            output = output[survived_index]
            mbt_par_sim = mbt_par_sim[survived_index]

            assert output.shape[1]==8

            # update tolerance values if accept_rate>0.03
            if accept_rate>0.03:
                tolvalue_scale += 1
                accept_rate = 0

                 # re-calculate the tolerance values
                birth1=weighted_mean(array_all[:,0], weight)
                birth2=weighted_mean(array_all[:,1], weight)
                death1=weighted_mean(array_all[:,2], weight)
                death2=weighted_mean(array_all[:,3], weight)
                q12=weighted_mean(array_all[:,4], weight)
                q21=weighted_mean(array_all[:,5], weight)

                balobs=[]
                bal1obs=[]
                bal2obs=[]
                tspanobs=[]
                distanceobs=[]
                statobs=[]
                stat2obs=[]
                propobs=[]

                for i in range(20):
                    flag = 0
                    while flag == 0:
                        tsim,flag = mbt_6p_simulator.birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size, start=startphase)
                    sumsta_vals = mbt_6p_simulator.generate_mbt_ind_update(tsim)
                    b,b1,b2, tspan_ind, dist_ind, prop_ind, t1,t2 = sumsta_vals

                    balobs.append(b)
                    bal1obs.append(b1)
                    bal2obs.append(b2)
                    distanceobs.append(dist_ind)
                    tspanobs.append(tspan_ind)
                    propobs.append(prop_ind)
                    statobs.append(t1)
                    stat2obs.append(t2)

                # since we use tol only to weight the importance of the statistics, the scale does not matter
                tol_dist = 40*sqrt(variance(distanceobs))*math.exp(-0.2*tolvalue_scale)
                tol_imb = 40*sqrt(variance(balobs))*math.exp(-0.2*tolvalue_scale)
                tol_imb1 = 40*sqrt(variance(bal1obs))*math.exp(-0.2*tolvalue_scale)
                tol_imb2 = 40*sqrt(variance(bal2obs))*math.exp(-0.2*tolvalue_scale)
                tol_tspan = 40*sqrt(variance(tspanobs))*math.exp(-0.2*tolvalue_scale)
                tol_prop = 40*sqrt(variance(propobs))*math.exp(-0.2*tolvalue_scale)
                tol_stat = 40*sqrt(variance(statobs))*math.exp(-0.2*tolvalue_scale)
                tol_stat2 = 40*sqrt(variance(stat2obs))*math.exp(-0.2*tolvalue_scale)

            # compute the statistic, reweighting according to the estimated variance of each statistic
            ex_stats = np.array([ex_imb_full, ex_imb1_full, ex_imb2_full, ex_tspan_full, ex_dist_full, ex_prop_full, ex_stat_full, ex_stat2_full])
            tol_stats = np.array([tol_imb, tol_imb1, tol_imb2, tol_tspan, tol_dist, tol_prop, tol_stat, tol_stat2])
        
            check_tol = abs(output-ex_stats)/tol_stats
            check_max = np.max(check_tol, axis=1)

            assert len(check_max)==sum(survived_index)

            accept_index = check_max<1

            accept_all = np.vstack((accept_all, mbt_par_sim[accept_index]))

            len_accept = accept_all.shape[0]

        len_accept = comm.bcast(len_accept, root=0)

    if rank==0:
        accept_rate = len_accept/N_total
        if len_accept>M:
            accept_all = accept_all[:M]
            len_accept = M
    len_accept = comm.bcast(len_accept, root=0)

    if rank==0:
        # for each sample, compute its dominant eigen value and its weight
        domtemp = np.zeros(M)
        for i in range(M):
            domtemp[i] = mbt_6p_simulator.dom(accept_all[i,0:2],accept_all[i,2:4],accept_all[i,4:6])
            invw=0
            parsim = accept_all[i]
            for item in range(len(weight)):
                par = array_all[item]
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)

        weightnew = np.array(weightnew)/sum(weightnew)
        end = time.time()
        duration.append(end-start)
        rate.append(accept_rate)
        # hstack only works for array with same number of dimension, shape=(M, 1), same for domtemp 

        results_all = np.hstack((accept_all, domtemp[:,np.newaxis], weightnew[:,np.newaxis])) # 6 parameters, growth rate, weights, with shape=(M,8)
        np.savetxt(PATH+'/results/mbt_tolvalue_size'+str(treesize)+'_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")   
        print('\n\n'+'k='+str(k))
        print(f'duration={duration}')
        print('\n\n'+'rate='+str(rate), flush=True)

        array_all = accept_all
        weight = weightnew

# since we consider nLTT stat after 10th iteration, we need to broadcast the nLTT curve (two arrays) for later nLTT stat computation
if rank == 0:
    len_array = len(final_array)
    len_resp = len(final_resp)
else:
    len_array = None
    len_resp = None
len_array = comm.bcast(len_array, root=0)
len_resp = comm.bcast(len_resp, root=0)

if rank == 0:
    final_array = np.array(final_array, dtype="float")
    final_resp = np.array(final_resp, dtype="float")
else:
    final_array = np.empty(len_array, dtype="float")
    final_resp = np.empty(len_resp, dtype="float")
comm.Bcast(final_array, root=0)
comm.Bcast(final_resp, root=0)

# update the tolerance rate
if rank == 0:
    num_sim = int((10*2*M)//cpus+1)*cpus # for t=2,3, mean sure it is larger than int(num_sim/cpus)
else: 
    num_sim = None
num_sim = comm.bcast(num_sim, root=0)

M_nLTT = 2*M

for k in range(T,3*T):
    gc.collect()
    len_accept = 0 # record the number of accepted samples\
    if rank == 0: # compute the parameter values
        start = time.time()
        N_total = 0 # count the number of nonextinct process
        n = array_all.shape[0]
        index = list(range(n))
        if k==1:
            weight = [1]*n

        weightnew = []
        
        # for covmat, the first argument, the dataset need to have shape=(N_PAR, M), array_all has shape = (M,N_PAR), so we use its transpose.
        covmat = 2*np.cov(array_all.T, aweights=weight) 

        # resample from prior until we reach the number of accepted samples

        accept_all = np.empty((0,6)) # a 2d array, recording all the accepted samples, in the shape of (M_nLTT,6); dominant value and weights are added later
        accept_output = np.empty((0,n_covs+1)) # a 2d array, recording all 9 stats of the accepted samples, in the shape of (M_nLTT,9)

    # Determine the workload for each process
    partition_item = num_sim // cpus 

    start_index = rank * partition_item 
    end_index = start_index + partition_item

    while len_accept<M_nLTT:
        if rank==0:
            mbt_par_sim = np.zeros((num_sim, 6), dtype="float") # tolerance rate is 2%, including extinct process, only the eight stats
            for i in range(num_sim):
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=array_all[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
                while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                    ind = random.choices(index, weights=weight, k=1)[0]
                    [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=array_all[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                    w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
                # save the generated parameter values
                mbt_par_sim[i] = [birth1sim, birth2sim,death1sim, death2sim,q12sim,q21sim]

            # need to distributed the parameter value
        else:
            mbt_par_sim = np.empty((num_sim, 6), dtype="float")
        comm.Bcast(mbt_par_sim, root=0)

        sendbuf = generate_ss_single_tree_nLTT(mbt_par_sim[start_index:end_index], size=size, obs_array=final_array, obs_resp=final_resp, startphase=startphase, r=20, start_nLTT=20, max_time_bound=max_tspan_bound)
        sendbuf = np.array(sendbuf, dtype="float")

        recvbuf = None
        if rank == 0:
            recvbuf = np.zeros((cpus, partition_item*(n_covs+1)), dtype="float") # add 1 for n_covs for nLTTstat
        comm.Gather(sendbuf, recvbuf, root=0) # at rank 0, recvbuf has all the statistics value for the parameters we generated

    
        if rank == 0:
            output = convert2ss(recvbuf, partition_item, n_covs+1) # output now has shape (partition_item*cpus, n_covs)
            # remove the simulation that extincts from both statistics and parameter sets
            survived_index = output[:,3]!=0
            N_total += sum(survived_index) # record the total number of survived processes
            print(f"the number of extincted processes is {partition_item*cpus- sum(survived_index)}")
            print(f"the number of accepted samples is {len_accept}")
            output = output[survived_index]
            mbt_par_sim = mbt_par_sim[survived_index]

            assert output.shape[1]==9
            output_stats = output[:,:8] # only keep the stats for distance

            # update tolerance values if accept_rate>0.03
            if accept_rate>0.03:
                tolvalue_scale += 1
                accept_rate = 0

                 # re-calculate the tolerance values
                birth1=weighted_mean(array_all[:,0], weight)
                birth2=weighted_mean(array_all[:,1], weight)
                death1=weighted_mean(array_all[:,2], weight)
                death2=weighted_mean(array_all[:,3], weight)
                q12=weighted_mean(array_all[:,4], weight)
                q21=weighted_mean(array_all[:,5], weight)

                balobs=[]
                bal1obs=[]
                bal2obs=[]
                tspanobs=[]
                distanceobs=[]
                statobs=[]
                stat2obs=[]
                propobs=[]

                for i in range(20):
                    flag = 0
                    while flag == 0:
                        tsim,flag = mbt_6p_simulator.birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size, start=startphase)
                    sumsta_vals = mbt_6p_simulator.generate_mbt_ind_update(tsim)
                    b,b1,b2, tspan_ind, dist_ind, prop_ind, t1,t2 = sumsta_vals

                    balobs.append(b)
                    bal1obs.append(b1)
                    bal2obs.append(b2)
                    distanceobs.append(dist_ind)
                    tspanobs.append(tspan_ind)
                    propobs.append(prop_ind)
                    statobs.append(t1)
                    stat2obs.append(t2)

                # since we use tol only to weight the importance of the statistics, the scale does not matter
                tol_dist = 40*sqrt(variance(distanceobs))*math.exp(-0.2*tolvalue_scale)
                tol_imb = 40*sqrt(variance(balobs))*math.exp(-0.2*tolvalue_scale)
                tol_imb1 = 40*sqrt(variance(bal1obs))*math.exp(-0.2*tolvalue_scale)
                tol_imb2 = 40*sqrt(variance(bal2obs))*math.exp(-0.2*tolvalue_scale)
                tol_tspan = 40*sqrt(variance(tspanobs))*math.exp(-0.2*tolvalue_scale)
                tol_prop = 40*sqrt(variance(propobs))*math.exp(-0.2*tolvalue_scale)
                tol_stat = 40*sqrt(variance(statobs))*math.exp(-0.2*tolvalue_scale)
                tol_stat2 = 40*sqrt(variance(stat2obs))*math.exp(-0.2*tolvalue_scale)

            # compute the statistic, reweighting according to the estimated variance of each statistic
            ex_stats = np.array([ex_imb_full, ex_imb1_full, ex_imb2_full, ex_tspan_full, ex_dist_full, ex_prop_full, ex_stat_full, ex_stat2_full])
            tol_stats = np.array([tol_imb, tol_imb1, tol_imb2, tol_tspan, tol_dist, tol_prop, tol_stat, tol_stat2])
        
            check_tol = abs(output_stats-ex_stats)/tol_stats
            check_max = np.max(check_tol, axis=1)

            assert len(check_max)==sum(survived_index)

            accept_index = check_max<1

            accept_all = np.vstack((accept_all, mbt_par_sim[accept_index]))
            accept_output = np.vstack((accept_output, output[accept_index]))

            len_accept = accept_all.shape[0]

        len_accept = comm.bcast(len_accept, root=0)

    if rank==0:
        accept_rate = len_accept/N_total
        if len_accept>M_nLTT:
            accept_all = accept_all[:M_nLTT]
            accept_output = accept_output[:M_nLTT]
            len_accept = M_nLTT
    len_accept = comm.bcast(len_accept, root=0)

    if rank==0:
        # among the accepted samples, choose the top M ones with the smallest nLTTstat
        accept_nLTT_index = np.argpartition(accept_output[:,8], M)[:M]
        assert len(accept_nLTT_index)==M, f"len(accept_index)={len(accept_nLTT_index)}, and M={M}"
        accept_all = accept_all[accept_nLTT_index] # update the 2d array, recording all the accepted samples, in the shape of (M,6)

        # for each sample, compute its dominant eigen value and its weight
        domtemp = np.zeros(M)
        for i in range(M):
            domtemp[i] = mbt_6p_simulator.dom(accept_all[i,0:2],accept_all[i,2:4],accept_all[i,4:6])
            invw=0
            parsim = accept_all[i]
            for item in range(len(weight)):
                par = array_all[item]
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)

        weightnew = np.array(weightnew)/sum(weightnew)
        end = time.time()
        duration.append(end-start)
        rate.append(accept_rate)
        # hstack only works for array with same number of dimension, shape=(M, 1), same for domtemp 

        results_all = np.hstack((accept_all, domtemp[:,np.newaxis], weightnew[:,np.newaxis])) # 6 parameters, growth rate, weights, with shape=(M,8)
        np.savetxt(PATH+'/results/mbt_tolvalue_size'+str(treesize)+'_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")   
        print('\n\n'+'k='+str(k))
        print(f'duration={duration}')
        print('\n\n'+'rate='+str(rate), flush=True)

        array_all = accept_all
        weight = weightnew
        
        













