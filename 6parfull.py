# This script is for inferring parameters in the irreducible process using
# ABC-PMC with nine selected summary statistics


from ete3 import Tree
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, variance
from math import sqrt
from scipy.stats import norm
from scipy.stats import multivariate_normal

#true parameter values used in the simulation study
birth1 = 3
birth2 = 1
death1 = 2
death2 = 0.5
q12 = 0.5
q21 = 0.25

#initialise the size of the simulation
iternum = 100
num = 100
size=50
M=100
T=10

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


def dist2root(tree, node):
    """Compute the distance between the internal node
    and its root (root.dist=0)/the first speciation event
    Arguments:
        - ``tree`` : a birth-death tree (binary, no extinct branch)
        - ``node`` : a internal node of this tree
    """
    #assert tree.get_tree_root().dist == 0
    if node.is_leaf() or node.is_root():
        raise ValueError("It is not an internal node")
    dist = node.dist
    for item in node.get_ancestors():
        if not item.is_root():
            dist += item.dist
    return dist


def dist2tip(tree, node):
    """Compute the distance between the internal node
    and its leaves
    Arguments:
        - ``tree`` : a birth-death tree (binary, no extinct branch)
        - ``node`` : a internal node of this tree
    """
    if node.is_leaf() or node.is_root():
        raise ValueError("It is not an internal node")
    item = node.children[0]
    dist = item.dist
    while not item.is_leaf():
        item = item.get_children()[0]
        dist += item.dist
    return dist

def ratiodist(tree, phase=1, shift=0.02):
    """Compute a new statistics that is the sum of
    -log(proportion)/(dist2tip(node)+shift) for all internal node"""
    stat = 0
    for node in tree.traverse():
        if (not node.is_leaf()) and (not node.is_root()):
            total_leaf_num = 0
            leaf_num1 = 0
            leaf_num2 = 0
            for leaf in node.get_leaves():
                if leaf.phase == 1:
                    leaf_num1 += 1
                    total_leaf_num += 1
                elif leaf.phase == 2:
                    leaf_num2 += 1
                    total_leaf_num += 1
            if total_leaf_num == leaf_num1 or total_leaf_num == leaf_num2:
                add = 0
            elif phase == 1:
                add = -math.log(leaf_num1/total_leaf_num)/(dist2tip(tree, node)+shift)
            else:
                add = -math.log(leaf_num2/total_leaf_num)/(dist2tip(tree, node)+shift)
            stat += add
    return stat

def ratio(t):
    ''' The proportion of the phase 1 tips'''
    n1 = numphase1(t)
    total = len(t.get_leaves())
    return n1/total

def NodeDelete(root):
    for node in root.traverse():
        flag = 0
        for leaf in node.get_leaves():
            if not leaf.extinct:
                flag = 1
        if flag == 0:
            node.detach()

def ignoreRoot(array):
    n = len(array)
    output = [0]
    for i in range(1,n):
        difference = array[i]-array[0]
        output.append(difference)
    return output


def dom(birth, death, transition):
    """Compute the dominant eigenvalue (i.e., growth rate)
    Arguments:
        - ``birth`` : a list/array of birth rates in each phase
        - ``death`` : a list/array of death rates in each phase
        - ``transition`` : a list/array of transition rates
    """
    b0, b1, d0, d1, q01, q10 = birth[0], birth[1], death[0], death[1], transition[0], transition[1]
    a = -q01-b0-d0
    b = -q10-b1-d1
    omega = np.array([[2*b0+a, q01], [q10, 2*b1+b]])
    eig_va, eig_vec = np.linalg.eig(omega)
    if eig_va[0] > eig_va[1]:
        return eig_va[0]
    else:
        return eig_va[1]



def delete_single_child_internal(t):
    """Utility function that removes internal nodes
    with a single child from tree"""

    for node in t.traverse("postorder"):
        
        if(not node.is_leaf() and len(node.get_children())<2):
            child = node.get_children()[0]
            child.dist = child.dist + node.dist
            node.delete()

    if len(t.get_children()) == 1:
        t.dist = t.children[0].dist
        t.children = t.children[0].children






def absdist_array(a,b,ya,yb):
    """Compute the absolute distance between two 
    step functions which is shown in the form of array"""

    abs_dist = 0
    
    #initialise the start value and the indices for the step functions
    i, j = 1,1
    #the value for the closest discontinuity point
    start = 0
    while i < len(a) and j < len(b):
        if a[i] >= b[j]:
            abs_dist = abs_dist + abs(ya[i]-yb[j])*(b[j]-start)
            start = b[j]

            j = j + 1

        else:
            abs_dist = abs_dist + abs(ya[i]-yb[j])*(a[i]-start)
            start = a[i]

            i = i + 1

    return abs_dist

def sumdist_array(a,b,ya,yb):
    """Compute the sum of the distance of two 
    step functions which is shown in the form of array
    Both time span and lineage are in [0,1]"""

    array = np.concatenate([a,b[1:len(b)-1]])
    array.sort()
    init_value = ya[0] + yb[0]
    resp = [init_value]
    
    #initialise the start value and the indices for the step functions
    i, j = 1,1
    while i < len(a) and j < len(b):
        if a[i] >= b[j]:
            sum_value = ya[i] + yb[j]
            resp.append(sum_value)

            j = j + 1
        else:
            sum_value = ya[i] + yb[j]
            resp.append(sum_value)

            i = i + 1

    resp = np.array(resp)
    return array, resp


def delete_single_child_internal(t):
    """Utility function that removes internal nodes
    with a single child from tree"""

    for node in t.traverse("postorder"):
        
        if(not node.is_leaf() and len(node.get_children()) < 2):
            child = node.get_children()[0]
            child.dist = child.dist + node.dist
            node.delete()

    if len(t.get_children()) == 1:
        t.children = t.children[0].children


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


def birth_death_tree2(birth, death, transition, nsize=None, start=1, max_time=None, remlosses=True, r=0):
    """Generates a birth-death tree.
    Arguments:
        - ``birth`` : birth rate, list type with length=2
        - ``death`` : death rate, list type with length=2
        - ``transition`` : transition rate
        - ``nsize`` : desired number of leaves
        - ``max_time`` : maximum time of evolution
        - ``remlosses`` : whether lost leaves (extinct taxa) should be pruned from tree
        - ``r`` : repeat until success
    """
    # initialize tree with root node
    tree = Tree()
    tree.add_features(extinct=False)
    tree.add_features(phase=start)
    tree.dist = 0.0
    done = False

    # get current list of leaves
    leaf_nodes = tree.get_leaves()
    curr_num_leaves1 = 0
    curr_num_leaves2 = 0
    nodes_phase1 = []
    nodes_phase2 = []
    for node in leaf_nodes:
        if node.phase == 1:
            curr_num_leaves1 += 1
            nodes_phase1.append(node)
        elif node.phase == 2:
            curr_num_leaves2 += 1
            nodes_phase2.append(node)

    total_time = 0

    # total event rate to compute waiting time
    event_rate1 = float(birth[0] + death[0])
    event_rate2 = float(birth[1] + death[1])

    while True:
        # waiting time based on event_rate
        total_rate = curr_num_leaves1*(event_rate1+transition[0])+curr_num_leaves2*(event_rate2+transition[1])
        event1 = curr_num_leaves1*event_rate1
        birth_rate1 = curr_num_leaves1*birth[0]
        death_rate1 = curr_num_leaves1*death[0]
        event2 = curr_num_leaves2*event_rate2
        birth_rate2 = curr_num_leaves2*birth[1]
        death_rate2 = curr_num_leaves2*death[1]
        transition_rate12 = curr_num_leaves1*transition[0]
        transition_rate21 = curr_num_leaves2*transition[1]
        wtime = random.expovariate(total_rate)
        total_time += wtime
        for leaf in nodes_phase1:
            # extinct leaves cannot update their branches length
            if not leaf.extinct:
                leaf.dist += wtime
        for leaf in nodes_phase2:
            # extinct leaves cannot update their branches length
            if not leaf.extinct:
                leaf.dist += wtime
        if nsize is not None and (curr_num_leaves1 + curr_num_leaves2) >= nsize:
            done = True
            flag = 1
        if max_time is not None and total_time >= max_time:
            done = True
            flag=1

        if done:
            break

        # if event occurs within time constraints
        if max_time is None or total_time <= max_time:
            eprob = random.random()
            # for node in phase 1
            # select node at random, then find chance it died or give birth
            # (speciation)

            if eprob < event1/total_rate:
                node = random.choice(nodes_phase1)
                
                nodes_phase1.remove(node)
                curr_num_leaves1 -= 1

                # birth event (speciation) creating two children nodes in phase 1
                if eprob < birth_rate1/total_rate:
                    child1 = Tree()
                    child1.dist = 0
                    child1.add_features(extinct=False)
                    child1.add_features(phase=1)
                    child2 = Tree()
                    child2.dist = 0
                    child2.add_features(extinct=False)
                    child2.add_features(phase=1)
                    node.add_child(child1)
                    node.add_child(child2)
                    nodes_phase1.append(child1)
                    nodes_phase1.append(child2)
                    # update add two new leave
                    # (remember that parent was removed)
                    curr_num_leaves1 += 2

                else:
                    # death of the chosen node
                    if (curr_num_leaves1 + curr_num_leaves2) > 0:
                        node.extinct = True
                    else:
                        if not (r>0):
                            flag=0
                            tree = Tree()
                            length = 0
                            return tree,flag
                        # Restart the simulation because the tree has gone
                        # extinct
                        tree = Tree()
                        leaf_nodes = tree.get_leaves()
                        tree.add_features(phase=start)

                        tree.add_features(extinct=False)
                        tree.dist = 0.0
                        curr_num_leaves1 = 0
                        curr_num_leaves2 = 0
                        nodes_phase1 = []
                        nodes_phase2 = []
                        for node in leaf_nodes:
                            if node.phase == 1:
                                curr_num_leaves1 += 1
                                nodes_phase1.append(node)
                            elif node.phase == 2:
                                curr_num_leaves2 += 1
                                nodes_phase2.append(node)
                        
                        total_time = 0

            elif event1/total_rate <= eprob < (event1+transition_rate12)/total_rate:
                node = random.choice(nodes_phase1)
                child = Tree()
                child.dist = 0
                child.add_features(extinct=False)
                child.add_features(phase=2)
                nodes_phase1.remove(node)
                nodes_phase2.append(child)
                node.add_child(child)
                curr_num_leaves1 -= 1
                curr_num_leaves2 += 1
            elif (event1+transition_rate12)/total_rate <= eprob < (event1+transition_rate12+transition_rate21)/total_rate:
                node = random.choice(nodes_phase2)
                child = Tree()
                child.dist = 0
                child.add_features(extinct=False)
                child.add_features(phase=1)
                nodes_phase1.append(child)
                nodes_phase2.remove(node)
                node.add_child(child)
                curr_num_leaves1 += 1
                curr_num_leaves2 -= 1
            elif (event1+transition_rate12+transition_rate21)/total_rate <= eprob < (event1+transition_rate12+transition_rate21+event2)/total_rate:
                node = random.choice(nodes_phase2)
                
                nodes_phase2.remove(node)
                curr_num_leaves2 -= 1

                # birth event (speciation) creating two children nodes in phase 2
                if (event1+transition_rate12+transition_rate21)/total_rate <= eprob < (event1+transition_rate12+transition_rate21+birth_rate2)/total_rate:
                    child1 = Tree()
                    child1.dist = 0
                    child1.add_features(extinct=False)
                    child1.add_features(phase=2)
                    child2 = Tree()
                    child2.dist = 0
                    child2.add_features(extinct=False)
                    child2.add_features(phase=2)
                    node.add_child(child1)
                    node.add_child(child2)
                    nodes_phase2.append(child1)
                    nodes_phase2.append(child2)
                    # update add two new leave
                    # (remember that parent was removed)
                    curr_num_leaves2 += 2

                else:
                    # death of the chosen node
                    if (curr_num_leaves1 + curr_num_leaves2) > 0:
                        node.extinct = True
                    else:
                        if not (r>0):
                            flag = 0
                            tree = Tree()
                            length = 0
                            return tree, flag
                        # Restart the simulation because the tree has gone
                        # extinct
                        r -= 1
                        tree = Tree()
                        leaf_nodes = tree.get_leaves()
                        tree.add_features(phase=start)

                        tree.add_features(extinct=False)
                        tree.dist = 0.0
                        curr_num_leaves1 = 0
                        curr_num_leaves2 = 0
                        nodes_phase1 = []
                        nodes_phase2 = []
                        for node in leaf_nodes:
                            if node.phase == 1:
                                curr_num_leaves1 += 1
                                nodes_phase1.append(node)
                            elif node.phase == 2:
                                curr_num_leaves2 += 1
                                nodes_phase2.append(node)
                        
                        total_time = 0
            else:
                raise ValueError("eprob is larger than 1")
            # this should always hold true
            assert (curr_num_leaves1 + curr_num_leaves2) == (len(nodes_phase1)+len(nodes_phase2))

    if remlosses:
        # prune lost leaves from tree
        NodeDelete(tree)
        # remove all non binary nodes
        delete_single_child_internal(tree)

    leaf_nodes = tree.get_leaves()
    leaf_compteur = 1
    for ind, node in enumerate(leaf_nodes):
        # label only extant leaves
        if not node.extinct:
            # node.dist += wtime
            node.name = "T%d" % leaf_compteur
            leaf_compteur += 1
    length = len(tree.get_leaves())
    return tree, flag


def gamma_stat(tree):
    n = len(tree.get_leaves())
    if n == 1 or n == 2:
        return 0
    else:
        total_stat = 0
        stat = 0
        seq=[]
        for node in tree.traverse():
            dis = node.dist
            if node.is_root():
                seq.append(dis)
            else:
                for parent in node.get_ancestors():
                    dis += parent.dist
                seq.append(dis)
        seq.sort()
        data=seq[0: n]
        for i in range(n):
            if i == 0:
                T=0
            else:
                stepsize = data[i]-data[i-1]
                T += stepsize*(i+1)
        for j in range(2,n):
            stat=0
            for k in range(1,j):
                stat += (k+1)* (data[k]-data[k-1])
            total_stat += stat
        result = (total_stat/(n-2)-T/2)/(T*math.sqrt(1/(12*(n-2))))
        return result

def step_array(tsim):
    seq=[]
    for node in tsim.traverse():
        dis = node.dist
        if node.is_root():
            seq.append(dis)
        else:
            for parent in node.get_ancestors():
                dis += parent.dist
            seq.append(dis)
    seq.sort()
    return seq

def nabsdiff(t):
    size = len(t.get_leaves())
    initial_array = step_array(t)
    init_data = ignoreRoot(initial_array)
    init_data=init_data[0:size]
    init_x = np.array(init_data)
    init_nx = init_x/init_data[-1]
    lineage = [2]
    for num in range(2, size+1):
        lineage.append(num)
    y = np.array(lineage)
    init_ny = y/size
    return init_nx, init_ny


def PD(tree):
    distance = []
    for node in tree.traverse():
        if not node.is_root():
            distance.append(node.dist)
    return mean(distance)

def mode(data, step = 0.05, width = 0.05):
    start = min(data)
    end = max(data)
    max_count = 0
    max_i = -1
    for i in np.arange(start+width, end-width, step):
        count = 0
        for point in data:
            if i - width < point < i + width:
                count = count + 1
        if count > max_count:
            max_i = i
            max_count = count
    return max_i



def imbalance(tree):
    array = []
    for node in tree.traverse():
        if not node.is_leaf():
            diff = abs(len(node.children[0].get_leaves())-len(node.children[1].get_leaves()))
            array.append(diff)
    return sum(array)/((100-1)*(100-2)/2)

def numphase1(parent):
    num = 0
    for node in parent.get_leaves():
        if node.phase == 1:
            num = num + 1
    return num

def numphase2(parent):
    num = 0
    for node in parent.get_leaves():
        if node.phase == 2:
            num = num + 1
    return num

def imb1(tree):
    array = []
    for node in tree.traverse():
        if not node.is_leaf():
            diff = abs(numphase1(node.children[0])-numphase1(node.children[1]))
            array.append(diff)
    return sum(array)/((100-1)*(100-2)/2)

def imb2(tree):
    array = []
    for node in tree.traverse():
        if not node.is_leaf():
            diff = abs(numphase2(node.children[0])-numphase2(node.children[1]))
            array.append(diff)
    return sum(array)/((100-1)*(100-2)/2)

def mdistp1(t):
    dist=[]
    for node in t.traverse():
        if numphase1(node)!=0:
            dist.append(node.dist)
    if len(dist)==0:
        return 0
    else:
        return mean(dist)


t, flag = birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size)
while flag==0:
    t, flag = birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size)
init_array, init_resp = nabsdiff(t)
balobs=[imbalance(t)]
bal1obs=[imb1(t)]
bal2obs=[imb2(t)]
tspanobs=[ignoreRoot(step_array(t))[-1]]
distanceobs=[PD(t)]

#new summary statistics
propobs = [ratio(t)] # the proportion of phase 1 tips
statobs = [ratiodist(t)] # introduce new summary statistics
stat2obs = [ratiodist(t,phase=2)] # introduce symmetric summary statistics


#about imb1=0
tspann0obs=[]
distp1obs=[]
if numphase1(t)!=0:
    tspann0obs.append(ignoreRoot(step_array(t))[-1])
    distp1obs.append(mdistp1(t))

for i in range(1,iternum):
    flag = 0
    while flag == 0:
        tsim,flag = birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size)
    sim_array, sim_resp = nabsdiff(tsim)
    init_array, init_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
    balobs.append(imbalance(tsim))
    bal1obs.append(imb1(tsim))
    bal2obs.append(imb2(tsim))
    distanceobs.append(PD(tsim))
    tspanobs.append(ignoreRoot(step_array(tsim))[-1])
    propobs.append(ratio(tsim))
    statobs.append(ratiodist(tsim))
    stat2obs.append(ratiodist(tsim,phase=2))
    if numphase1(tsim)!=0:
        tspann0obs.append(ignoreRoot(step_array(tsim))[-1])
        distp1obs.append(mdistp1(tsim))

final_array1 = init_array
final_resp1 = init_resp/iternum
    

exdist = mean(distanceobs)
ex_imb = mean(balobs)
ex_imb1 = mean(bal1obs)
ex_imb2 = mean(bal2obs)
ex_span = mean(tspanobs)
ex_prop = mean(propobs)
ex_stat = mean(statobs)
ex_stat2 = mean(stat2obs)

mtn0=mean(tspann0obs)
mdp1 = mean(distp1obs)

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
        flag = 0
        birth1sim = random.uniform(0,5)
        birth2sim = random.uniform(0,5)
        death1sim = random.uniform(0,5)
        death2sim = random.uniform(0,5)
        q12sim = random.uniform(0,5)
        q21sim = random.uniform(0,5)
        w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
        while w<=0.1:
            birth1sim = random.uniform(0,5)
            birth2sim = random.uniform(0,5)
            death1sim = random.uniform(0,5)
            death2sim = random.uniform(0,5)
            q12sim = random.uniform(0,5)
            q21sim = random.uniform(0,5)
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
        r=0
        length = 0
        while flag == 0 or length != 50:
            t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size, max_time=30)
            r+=1
            length = len(t.get_leaves())
            if r == 5:
                do = True
                break
            
    assert flag == 1
    sim_array, sim_resp = nabsdiff(t)
    tspan=[ignoreRoot(step_array(t))[-1]]
    dist=[PD(t)]
    bala = [imbalance(t)]
    bala1 = [imb1(t)]
    bala2 = [imb2(t)]
    prop = [ratio(t)] # the proportion of phase 1 tips
    stat = [ratiodist(t)]
    stat2 = [ratiodist(t,phase=2)]
    N+=1
    for j in range(1,1):
        flag = 0
        while flag == 0:
            tsim, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size)
        assert flag == 1
        init_array, init_resp = nabsdiff(tsim)
        sim_array, sim_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
        tspan.append(ignoreRoot(step_array(tsim))[-1])
        dist.append(PD(tsim))
        bala.append(imbalance(tsim))
        bala1.append(imb1(tsim))
        bala2.append(imb2(tsim))
        prop.append(ratio(tsim))
        stat.append(ratiodist(tsim))
        stat2.append(ratiodist(tsim,phase=2))
    sim_resp = sim_resp/num
    
    mdist = mean(dist)
    mbala = mean(bala)
    mbala1 = mean(bala1)
    mbala2 = mean(bala2)
    mtspan = mean(tspan)
    mprop = mean(prop)
    mstat = mean(stat)
    mstat2 = mean(stat2)
    
    if min_span <= mtspan <= max_span and min_dist<=mdist<=max_dist and min_imb<=mbala<=max_imb and min_imb1<=mbala1<=max_imb1 and min_imb2<=mbala2<=max_imb2 and min_prop <= mprop <= max_prop and min_stat <= mstat <= max_stat and min_stat2 <= mstat2 <= max_stat2:
        accept1.append(birth1sim)
        accept2.append(birth2sim)
        accept3.append(death1sim)
        accept4.append(death2sim)
        accept5.append(q12sim)
        accept6.append(q21sim)

        domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
        
accept_rate = len(accept1)/N

meanseq1=[mean(accept1)]
meanseq2=[mean(accept2)]
meanseq3=[mean(accept3)]
meanseq4=[mean(accept4)]
meanseq5=[mean(accept5)]
meanseq6=[mean(accept6)]
dommean=[mean(domtemp)]

wmeanseq1=[mean(accept1)]
wmeanseq2=[mean(accept2)]
wmeanseq3=[mean(accept3)]
wmeanseq4=[mean(accept4)]
wmeanseq5=[mean(accept5)]
wmeanseq6=[mean(accept6)]
rate=[accept_rate]

dommean_u=[np.quantile(domtemp,0.75)]
weighted_q1u=[np.quantile(accept1,0.75)]
weighted_q2u=[np.quantile(accept2,0.75)]
weighted_q3u=[np.quantile(accept3,0.75)]
weighted_q4u=[np.quantile(accept4,0.75)]
weighted_q5u=[np.quantile(accept5,0.75)]
weighted_q6u=[np.quantile(accept6,0.75)]

dommean_l=[np.quantile(domtemp,0.25)]
weighted_q1l=[np.quantile(accept1,0.25)]
weighted_q2l=[np.quantile(accept2,0.25)]
weighted_q3l=[np.quantile(accept3,0.25)]
weighted_q4l=[np.quantile(accept4,0.25)]
weighted_q5l=[np.quantile(accept5,0.25)]
weighted_q6l=[np.quantile(accept6,0.25)]

array1=accept1
array2=accept2
array3=accept3
array4=accept4
array5=accept5
array6=accept6

print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u))
print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l))
print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6))
print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6))
print('\n\n'+'rate='+str(rate), flush=True)

z=0

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
            flag = 0
            ind = random.choices(index, weights=weight, k=1)[0]
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.01 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            r=0
            length = 0
            while flag == 0 or length != 50:
                t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size, max_time=30)
                r+=1
                length = len(t.get_leaves())
                if r == 5:
                    do = True
                    break
                
        assert flag == 1
        sim_array, sim_resp = nabsdiff(t)
        tspan=[ignoreRoot(step_array(t))[-1]]
        dist=[PD(t)]
        bala = [imbalance(t)]
        bala1 = [imb1(t)]
        bala2 = [imb2(t)]
        prop = [ratio(t)] # the proportion of phase 1 tips
        stat = [ratiodist(t)]
        stat2 = [ratiodist(t,phase=2)]
        N+=1
        for j in range(1,10):
            flag = 0
            while flag == 0:
                tsim, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size)
            assert flag == 1
            init_array, init_resp = nabsdiff(tsim)
            sim_array, sim_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
            tspan.append(ignoreRoot(step_array(tsim))[-1])
            dist.append(PD(tsim))
            bala.append(imbalance(tsim))
            bala1.append(imb1(tsim))
            bala2.append(imb2(tsim))
            prop.append(ratio(tsim))
            stat.append(ratiodist(tsim))
            stat2.append(ratiodist(tsim,phase=2))
        sim_resp = sim_resp/num
        
        mdist = mean(dist)
        mbala = mean(bala)
        mbala1 = mean(bala1)
        mbala2 = mean(bala2)
        mtspan = mean(tspan)
        mprop = mean(prop)
        mstat = mean(stat)
        mstat2 = mean(stat2)
        
        if abs(mtspan-ex_span) <= tol_span and abs(mdist-exdist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
    accept_rate = len(accept1)/N
    weightnew = list(np.array(weightnew)/sum(weightnew))

    meanseq1.append(mean(accept1))
    meanseq2.append(mean(accept2))
    meanseq3.append(mean(accept3))
    meanseq4.append(mean(accept4))
    meanseq5.append(mean(accept5))
    meanseq6.append(mean(accept6))
    dommean.append(weighted_mean(domtemp, weightnew))

    wmeanseq1.append(weighted_mean(accept1, weightnew))
    wmeanseq2.append(weighted_mean(accept2, weightnew))
    wmeanseq3.append(weighted_mean(accept3, weightnew))
    wmeanseq4.append(weighted_mean(accept4, weightnew))
    wmeanseq5.append(weighted_mean(accept5, weightnew))
    wmeanseq6.append(weighted_mean(accept6, weightnew))
    rate.append(accept_rate)

    dommean_u.append(weighted_quantile(domtemp, 0.75, sample_weight=weightnew))
    weighted_q1u.append(weighted_quantile(accept1, 0.75, sample_weight=weightnew))
    weighted_q2u.append(weighted_quantile(accept2, 0.75, sample_weight=weightnew))
    weighted_q3u.append(weighted_quantile(accept3, 0.75, sample_weight=weightnew))
    weighted_q4u.append(weighted_quantile(accept4, 0.75, sample_weight=weightnew))
    weighted_q5u.append(weighted_quantile(accept5, 0.75, sample_weight=weightnew))
    weighted_q6u.append(weighted_quantile(accept6, 0.75, sample_weight=weightnew))

    dommean_l.append(weighted_quantile(domtemp, 0.25, sample_weight=weightnew))
    weighted_q1l.append(weighted_quantile(accept1, 0.25, sample_weight=weightnew))
    weighted_q2l.append(weighted_quantile(accept2, 0.25, sample_weight=weightnew))
    weighted_q3l.append(weighted_quantile(accept3, 0.25, sample_weight=weightnew))
    weighted_q4l.append(weighted_quantile(accept4, 0.25, sample_weight=weightnew))
    weighted_q5l.append(weighted_quantile(accept5, 0.25, sample_weight=weightnew))
    weighted_q6l.append(weighted_quantile(accept6, 0.25, sample_weight=weightnew))

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6

    weight = weightnew

    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
    print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u))
    print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l))
    print('\n\n'+'weighted_meanseq1='+str(wmeanseq1)+'\n\n'+'weighted_meanseq2='+str(wmeanseq2)+'\n\n'+'weighted_meanseq3='+str(wmeanseq3)+'\n\n'+'weighted_meanseq4='+str(wmeanseq4)+'\n\n'+'weighted_meanseq5='+str(wmeanseq5), 'weighted_meanseq6='+str(wmeanseq6))
    print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6))
    print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6))
    print('\n\n'+'weight='+str(list(weight))+'\n\n'+'rate='+str(rate), flush=True)



for k in range(T,3*T):

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
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0.1 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.01 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=5:
                ind = random.choices(index, weights=weight, k=1)[0]
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            r=0
            length = 0
            while flag == 0 or length != size:
                t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size, max_time=30)
                r+=1
                length = len(t.get_leaves())
                if r == 5:
                    do = True
                    break
                
        assert flag == 1
        sim_array, sim_resp = nabsdiff(t)
        tspan=[ignoreRoot(step_array(t))[-1]]
        dist=[PD(t)]
        bala = [imbalance(t)]
        bala1 = [imb1(t)]
        bala2 = [imb2(t)]
        proptemp = [ratio(t)]
        stattemp = [ratiodist(t)]
        stat2temp = [ratiodist(t,phase=2)]
        #about imb1!=0
        tspann0=[]
        distp1=[]
        if numphase1(t)!=0:
            tspann0.append(ignoreRoot(step_array(t))[-1])
            distp1.append(mdistp1(t))
        N+=1
        for j in range(1,num):
            flag = 0
            while flag == 0:
                tsim, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], nsize=size)
            assert flag == 1
            init_array, init_resp = nabsdiff(tsim)
            sim_array, sim_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
            tspan.append(ignoreRoot(step_array(tsim))[-1])
            dist.append(PD(tsim))
            bala.append(imbalance(tsim))
            bala1.append(imb1(tsim))
            bala2.append(imb2(tsim))
            proptemp.append(ratio(tsim))
            stattemp.append(ratiodist(tsim))
            stat2temp.append(ratiodist(tsim,phase=2))
            if numphase1(tsim)!=0:
                tspann0.append(ignoreRoot(step_array(tsim))[-1])
                distp1.append(mdistp1(tsim))
        sim_resp = sim_resp/num
        
        mdist = mean(dist)
        mbala = mean(bala)
        mbala1 = mean(bala1)
        mbala2 = mean(bala2)
        mtspan = mean(tspan)
        mprop = mean(proptemp)
        mstat = mean(stattemp)
        mstat2 = mean(stat2temp)
        if absdist_array(final_array1, sim_array, final_resp1, sim_resp) <= tol and abs(mtspan-ex_span) <= tol_span and abs(mdist-exdist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))

            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
    accept_rate = len(accept1)/N
    weightnew = list(np.array(weightnew)/sum(weightnew))

    meanseq1.append(mean(accept1))
    meanseq2.append(mean(accept2))
    meanseq3.append(mean(accept3))
    meanseq4.append(mean(accept4))
    meanseq5.append(mean(accept5))
    meanseq6.append(mean(accept6))
    dommean.append(weighted_mean(domtemp, weightnew))

    wmeanseq1.append(weighted_mean(accept1, weightnew))
    wmeanseq2.append(weighted_mean(accept2, weightnew))
    wmeanseq3.append(weighted_mean(accept3, weightnew))
    wmeanseq4.append(weighted_mean(accept4, weightnew))
    wmeanseq5.append(weighted_mean(accept5, weightnew))
    wmeanseq6.append(weighted_mean(accept6, weightnew))
    rate.append(accept_rate)

    dommean_u.append(weighted_quantile(domtemp, 0.75, sample_weight=weightnew))
    weighted_q1u.append(weighted_quantile(accept1, 0.75, sample_weight=weightnew))
    weighted_q2u.append(weighted_quantile(accept2, 0.75, sample_weight=weightnew))
    weighted_q3u.append(weighted_quantile(accept3, 0.75, sample_weight=weightnew))
    weighted_q4u.append(weighted_quantile(accept4, 0.75, sample_weight=weightnew))
    weighted_q5u.append(weighted_quantile(accept5, 0.75, sample_weight=weightnew))
    weighted_q6u.append(weighted_quantile(accept6, 0.75, sample_weight=weightnew))

    dommean_l.append(weighted_quantile(domtemp, 0.25, sample_weight=weightnew))
    weighted_q1l.append(weighted_quantile(accept1, 0.25, sample_weight=weightnew))
    weighted_q2l.append(weighted_quantile(accept2, 0.25, sample_weight=weightnew))
    weighted_q3l.append(weighted_quantile(accept3, 0.25, sample_weight=weightnew))
    weighted_q4l.append(weighted_quantile(accept4, 0.25, sample_weight=weightnew))
    weighted_q5l.append(weighted_quantile(accept5, 0.25, sample_weight=weightnew))
    weighted_q6l.append(weighted_quantile(accept6, 0.25, sample_weight=weightnew))

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6

    weight = weightnew

    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
    print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u))
    print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l))
    print('\n\n'+'weighted_meanseq1='+str(wmeanseq1)+'\n\n'+'weighted_meanseq2='+str(wmeanseq2)+'\n\n'+'weighted_meanseq3='+str(wmeanseq3)+'\n\n'+'weighted_meanseq4='+str(wmeanseq4)+'\n\n'+'weighted_meanseq5='+str(wmeanseq5), 'weighted_meanseq6='+str(wmeanseq6))
    print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6))
    print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6))
    print('\n\n'+'weight='+str(list(weight))+'\n\n'+'rate='+str(rate), flush=True)

        

