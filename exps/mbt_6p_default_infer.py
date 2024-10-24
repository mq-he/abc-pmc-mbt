# This script is for inferring parameters in the irreducible process using
# ABC-PMC with nine selected summary statistics

# consider nLTT from the 20th iteration

# here, we set r=20, allowing up to 20 attempts for extinction
# we use the psedo-observed values which are generated from a separate file

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
import os

# Get the path of the current script
FILEPATH = os.path.abspath(__file__)
PATH = os.path.dirname(FILEPATH)

# sys.path.append(PATH)

import mbt_6p_simulator

# read obs values from csv file
trial_index = int(sys.argv[1]) # filename
obs_vals = np.loadtxt(PATH+'/obs/mbt_n100_s50_default_t'+str(trial_index)+'.csv', delimiter=",") # shape=(num_tree, N_SUMSTA+2*treesize+8)

# true parameter values used in the simulation study
birth1, birth2, death1, death2, q12, q21 = [3,1,2,0.5,0.5,0.25]

#initialise the size of the simulation
iternum = 100 # number of trees in the observed dataset
num = 100 # number of trees in each simulated dataset
size= 50 # tree size
M=100 # number of the accepted samples in each iteration (increase this later)
T=10
startphase = 1

# np.array([balsim, bal1sim, bal2sim, tspansim, distancesim, propsim, statsim, stat2sim])
balobs = obs_vals[:,0]
bal1obs = obs_vals[:,1]
bal2obs = obs_vals[:,2]
tspanobs = obs_vals[:,3]
distanceobs = obs_vals[:,4]
propobs = obs_vals[:,5]
statobs = obs_vals[:,6]
stat2obs = obs_vals[:,7]

# compute the average values for summary statistics and the nLTT curve
sim_array = obs_vals[0,8:(8+size)]
sim_resp = obs_vals[0,(8+size):]
assert len(sim_resp) == size, "the nLTT curve is not correctly extracted from the observed value"

for i in range(1,iternum):
    init_array = obs_vals[i,8:(8+size)]
    init_resp = obs_vals[i,(8+size):]
    sim_array, sim_resp = mbt_6p_simulator.sumdist_array(init_array, sim_array, init_resp, sim_resp)

final_array1 = sim_array
final_resp1 = sim_resp/iternum

# compute the average values for summary statistics and the nLTT curve

# compute the variance of the observed dataset for tolerance value

# compute the max and min of the variance for boundary


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

exdist = mean(distanceobs)
ex_imb = mean(balobs)
ex_imb1 = mean(bal1obs)
ex_imb2 = mean(bal2obs)
ex_span = mean(tspanobs)
ex_prop = mean(propobs)
ex_stat = mean(statobs)
ex_stat2 = mean(stat2obs)


tol_dist = 2*sqrt(variance(distanceobs))
tol = 0.05
tol_imb = 2*sqrt(variance(balobs))
tol_imb1 = 2*sqrt(variance(bal1obs))
tol_imb2 = 2*sqrt(variance(bal2obs))
tol_span = 2*sqrt(variance(tspanobs))
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
        t, flag = mbt_6p_simulator.birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size, start=startphase, r=20)
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

k=0 # iteration index

rate = [accept_rate]
results_all = np.vstack((accept1, accept2, accept3, accept4, accept5, accept6, domtemp, np.ones(200)/200)) # 6 parameters, growth rate, weights, with shape=(200,8)
np.savetxt(PATH+'/results/mbt_n100_s50_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")
print('\n\n'+'rate='+str(rate), flush=True)

z=0 # record the tolerance level

for k in range(1,T):
    n = len(array1)
    index = list(range(n))
    if k==1:
        weight = [1]*n

    weightnew = []

    scale1 = sqrt(2*weighted_var(array1, weight))
    scale2 = sqrt(2*weighted_var(array2, weight))
    scale3 = sqrt(2*weighted_var(array3, weight))
    scale4 = sqrt(2*weighted_var(array4, weight))
    scale5 = sqrt(2*weighted_var(array5, weight))
    scale6 = sqrt(2*weighted_var(array6, weight))
    
    accept = np.stack((accept1,accept2,accept3,accept4,accept5,accept6))
    covmat = 2*np.cov(accept, aweights=weight)

    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    domtemp = []

    if accept_rate >= 0.03:
        tol_dist = math.exp(-0.2)*tol_dist
        tol_imb = math.exp(-0.2)*tol_imb
        tol_imb1 = math.exp(-0.2)*tol_imb1
        tol_imb2 = math.exp(-0.2)*tol_imb2
        tol_span = math.exp(-0.2)*tol_span
        tol_prop = math.exp(-0.2)*tol_prop
        tol_stat = math.exp(-0.2)*tol_stat
        tol_stat2 = math.exp(-0.2)*tol_stat2
        z += 1
        accept_rate=0
    
    
    N=0
    while len(accept1)<M:
        do = True
        while do:
            do = False
            ind = random.choices(index, weights=weight, k=1)[0]
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            # only consider supercritical process
            sample = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
            sumsta_vals = mbt_6p_simulator.generate_mbt_data_update(sample, num_tree=10, treesize = size, startphase = startphase, r=20)
            if sum(sumsta_vals)==0: # if the process extincts after 20 attempts, redraw the sample
                do=True

        mbala, mbala1, mbala2, mtspan, mdist, mprop, mstat, mstat2 = sumsta_vals
        N+=1
        
        if abs(mtspan-ex_span) <= tol_span and abs(mdist-exdist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            domtemp.append(mbt_6p_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
    accept_rate = len(accept1)/N
    weightnew = list(np.array(weightnew)/sum(weightnew))

    rate.append(accept_rate)
    results_all = np.vstack((accept1, accept2, accept3, accept4, accept5, accept6, domtemp, weightnew)) # 6 parameters, growth rate, weights, with shape=(200,8)
    np.savetxt(PATH+'/results/mbt_n100_s50_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")    
    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'rate='+str(rate), flush=True)

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6

    weight = weightnew


for k in range(T,2*T):

    n = len(array1)
    index = list(range(n))
    weightnew = []

    scale1 = sqrt(2*weighted_var(array1, weight))
    scale2 = sqrt(2*weighted_var(array2, weight))
    scale3 = sqrt(2*weighted_var(array3, weight))
    scale4 = sqrt(2*weighted_var(array4, weight))
    scale5 = sqrt(2*weighted_var(array5, weight))
    scale6 = sqrt(2*weighted_var(array6, weight))
    
    accept = np.stack((accept1,accept2,accept3,accept4,accept5,accept6))
    covmat = 2*np.cov(accept, aweights=weight)

    if accept_rate >= 0.03:
        tol_dist = math.exp(-0.2)*tol_dist
        tol_imb = math.exp(-0.2)*tol_imb
        tol_imb1 = math.exp(-0.2)*tol_imb1
        tol_imb2 = math.exp(-0.2)*tol_imb2
        tol_span = math.exp(-0.2)*tol_span
        tol_prop = math.exp(-0.2)*tol_prop
        tol_stat = math.exp(-0.2)*tol_stat
        tol_stat2 = math.exp(-0.2)*tol_stat2
        z += 1
        accept_rate=0
    

    #tol_span = 1.5

    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    domtemp = []
    
    
    N=0
    while len(accept1)<M:
        do = True
        while do:
            do = False
            flag = 0
            ind = random.choices(index, weights=weight, k=1)[0]
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            # only consider supercritical process
            sample = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
            sumsta_vals = mbt_6p_simulator.generate_mbt_data_update(sample, num_tree=num, treesize = size, startphase = startphase, r=20)
            if sum(sumsta_vals)==0: # if the process extincts after 20 attempts, redraw the sample
                do=True

        mbala, mbala1, mbala2, mtspan, mdist, mprop, mstat, mstat2 = sumsta_vals
        N+=1

        if abs(mtspan-ex_span) <= tol_span and abs(mdist-exdist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            domtemp.append(mbt_6p_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))

            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
    accept_rate = len(accept1)/N
    weightnew = list(np.array(weightnew)/sum(weightnew))

    rate.append(accept_rate)
    results_all = np.vstack((accept1, accept2, accept3, accept4, accept5, accept6, domtemp, weightnew)) # 6 parameters, growth rate, weights, with shape=(200,8)
    np.savetxt(PATH+'/results/mbt_n100_s50_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")    
    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'rate='+str(rate), flush=True)


    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6

    weight = weightnew

for k in range(2*T,3*T):

    n = len(array1)
    index = list(range(n))
    weightnew = []
    
    accept = np.stack((accept1,accept2,accept3,accept4,accept5,accept6))
    covmat = 2*np.cov(accept, aweights=weight)

    if accept_rate >= 0.03 and k>2*T: # introduce nLTT after the 20th iteration
        tol = math.exp(-0.2)*tol
        tol_dist = math.exp(-0.2)*tol_dist
        tol_imb = math.exp(-0.2)*tol_imb
        tol_imb1 = math.exp(-0.2)*tol_imb1
        tol_imb2 = math.exp(-0.2)*tol_imb2
        tol_span = math.exp(-0.2)*tol_span
        tol_prop = math.exp(-0.2)*tol_prop
        tol_stat = math.exp(-0.2)*tol_stat
        tol_stat2 = math.exp(-0.2)*tol_stat2
        z += 1
        accept_rate=0
    

    #tol_span = 1.5

    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    domtemp = []
    
    
    N=0
    while len(accept1)<M:
        do = True
        while do:
            do = False
            flag = 0
            ind = random.choices(index, weights=weight, k=1)[0]
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = mbt_6p_simulator.dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            # only consider supercritical process
            sample = np.array([birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim])
            sumsta_vals, sim_array, sim_resp = mbt_6p_simulator.generate_mbt_data_update_nLTT(sample, num_tree=num, treesize = size, startphase = startphase, r=20)
            if sum(sumsta_vals)==0: # if the process extincts after 20 attempts, redraw the sample
                do=True

        mbala, mbala1, mbala2, mtspan, mdist, mprop, mstat, mstat2 = sumsta_vals
        N+=1

        if mbt_6p_simulator.absdist_array(final_array1, sim_array, final_resp1, sim_resp) <= tol and abs(mtspan-ex_span) <= tol_span and abs(mdist-exdist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            domtemp.append(mbt_6p_simulator.dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))

            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
    accept_rate = len(accept1)/N
    weightnew = list(np.array(weightnew)/sum(weightnew))

    rate.append(accept_rate)
    results_all = np.vstack((accept1, accept2, accept3, accept4, accept5, accept6, domtemp, weightnew)) # 6 parameters, growth rate, weights, with shape=(200,8)
    np.savetxt(PATH+'/results/mbt_n100_s50_default_t'+str(trial_index)+'_iter'+str(k)+'.csv', results_all, delimiter=",")    
    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'rate='+str(rate), flush=True)

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6

    weight = weightnew