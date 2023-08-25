"""
Use inferred tree from paper 'Early origin of viviparity and multiple
reversions to oviparity in squamate reptiles' to infer diversification rates
in the Markovian binary tree using ABC-PMC
"""



import os
from ete3 import Tree
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, variance
from math import sqrt
from scipy.stats import norm
from scipy.stats import multivariate_normal

# read the supertree
parity_tree = Tree("parity_phy_nwk_dated.txt", format=0)

# read csv file
import csv

with open('trait_parity2.csv', 'r', encoding='utf-8') as f:
    data = {line[0]: line[1] for line in csv.reader(f)}

with open('parity_name_replace.csv', 'r', encoding='utf-8') as f:
    newname = {line[1]: line[0] for line in csv.reader(f)}

def search(item):
    """Find the parity of the species
    Argument:
    item: name of the species
    Outputs:
    0: oviparous
    1: viviparous
    -1: no information or neither 0/1
    """
    if newname.get(item):
        itemname = newname.get(item)
    else:
        itemname = item

    trait = data.get(item)
    if trait:
        if trait == '0':
            return 0
        elif trait == '1':
            return 1
    
    return -1

def NodeDeleteInfo(root):
    for node in root.traverse():
        flag = 0
        for leaf in node.get_leaves():
            if leaf.trait!=-1:
                flag = 1
        if flag == 0:
            node.detach()


def Addphase(t):
    """Attach phase info to leaves
    Argument:
    t: tree structure
    Phase:
    1: oviparous
    2: viviparous
    """
    for leaf in t.get_leaves():
        if leaf.trait==0:
            leaf.phase=1
        elif leaf.trait==1:
            leaf.phase=2

def delete_single_child_internal_dist(t):
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
            
for leaf in parity_tree.get_leaves():
    leaf.trait = search(leaf.name)

NodeDeleteInfo(parity_tree)

delete_single_child_internal_dist(parity_tree)

Addphase(parity_tree)

tobs=parity_tree

M=100
T=10
size1,size2=20,50

def tree_height(t, ignore_root=True):
    """Compute the tree height of the given tree
    """
    height=0
    node=t
    while not node.is_leaf():
        height+=node.dist
        node=node.get_children()[0]
    height+=node.dist
    if ignore_root:
        height-=t.dist
    return height

def split_height(t,min_height):
    """Split a time into pieces with the given tree height, the hieght is the
    distance from the spliting point to its leaves 
    """
    chunks = []
    if tree_height(t)<=min_height<=tree_height(t,False):
        chunks.append(t)
        return chunks
    elif tree_height(t,False)<min_height:
        return chunks
    else:
        t_child1 = t.get_children()[0]
        t_child2 = t.get_children()[1]
        chunks1 = split_height(t_child1,min_height)
        chunks2 = split_height(t_child2,min_height)
        chunks = chunks1+chunks2
        return chunks

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

#mod
def ratiodist(tree, shift=0.02):
    """Compute transition statistics that is the sum of
    -log(proportion)/(dist2tip(node)+shift) for all internal node"""
    stat,stat2 = 0,0
    for node in tree.traverse():
        if (not node.is_leaf()) and (not node.is_root()):
            total_leaf_num = node.size
            leaf_num1 = node.size1
            leaf_num2 = node.size2
            if total_leaf_num == leaf_num1 or total_leaf_num == leaf_num2:
                add1,add2 = 0,0
            else:
                add1 = -math.log(leaf_num1/total_leaf_num)/(dist2tip(tree, node)+shift)
                add2 = -math.log(leaf_num2/total_leaf_num)/(dist2tip(tree, node)+shift)
            stat += add1
            stat2 += add2
    return stat,stat2


def trans(t):
    size=len(t.get_leaves())
    tran1,tran2=ratiodist(t)
    return tran1/(size-1),tran2/(size-1)

def ratio(t):
    ''' The proportion of the phase 1 tips'''
    n1 = numphase1(t)
    total = len(t.get_leaves())
    return n1/total

def ignoreRoot(array, start=2):
    """Ignore part of the nLTT curve to let the curve start from the "start"
    number of lineages, by default, the array contains the information for the
    curve to start from two lineages, since the information before the first
    coalescence is not included in the inferred tree.
    """
    n = len(array)
    output = [0]
    for i in range(start-1,n):
        difference = array[i]-array[start-2]
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


def NodeDelete(root):
    for leaf in root.get_leaves():
        if not leaf.extinct:
            for anc in leaf.get_ancestors():
                if anc.extinct:
                    anc.extinct = False
    remove_extinct(root)


def remove_extinct(root):
    """Utility function that removes extinct subtrees
    """
    for node in root.traverse():
        if node.extinct:
            node.detach()

def delete_single_child_internal(t):
    """Utility function that removes internal nodes
    with a single child from tree"""

    for node in t.traverse("postorder"):
        
        if(not node.is_leaf() and len(node.get_children())<2):
            child = node.get_children()[0]
            #child.dist = child.dist + node.dist
            node.delete()

    if len(t.get_children()) == 1:
        t.height = t.children[0].height
        #t.dist = t.children[0].dist
        t.children = t.children[0].children

def Add_dist(t):
    """Add dist using attribute height
    """
    for node in t.traverse():
        if node.is_root():
            node.dist==node.height
        else:
            node.dist=node.height-node.get_ancestors()[0].height


            

def birth_death_tree2(birth, death, transition, nsize=None, start=1, max_time=None, remlosses=True, r=5):
    """Generates a birth-death tree. (use minimal computational time)
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
        if nsize is not None and (curr_num_leaves1 + curr_num_leaves2) >= nsize:
            for leaf in nodes_phase1:
                leaf.height = total_time
            for leaf in nodes_phase2:
                leaf.height = total_time
                    
            done = True
            flag = 1
        if max_time is not None and total_time >= max_time:
            for leaf in nodes_phase1:
                # extinct leaves cannot update their branches length
                if not leaf.extinct:
                    leaf.dist -= total_time-max_time
                    leaf.height = max_time
            for leaf in nodes_phase2:
                # extinct leaves cannot update their branches length
                if not leaf.extinct:
                    leaf.dist -= total_time-max_time
                    leaf.height = max_time
                    
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
                ran = random.uniform(0,1)
                ind = int(ran*len(nodes_phase1))
                node = nodes_phase1[ind]
                del nodes_phase1[ind]
                
                curr_num_leaves1 -= 1
                # add height (no matter birth or death, the branch length stops growing)
                node.height=total_time

                # birth event (speciation) creating two children nodes in phase 1
                if eprob < birth_rate1/total_rate:
                    node.extinct = True # disappear of the current node
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
                            return tree, flag
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

            # transit from phase 1 to phase 2
            elif event1/total_rate <= eprob < (event1+transition_rate12)/total_rate:
                ran = random.uniform(0,1)
                ind = int(ran*len(nodes_phase1))
                node = nodes_phase1[ind]
                del nodes_phase1[ind]
                
                child = Tree()
                child.dist = 0
                child.add_features(extinct=False)
                child.add_features(phase=2)
                nodes_phase2.append(child)
                node.add_child(child)
                node.extinct = True # disappear of the current node
                curr_num_leaves1 -= 1
                curr_num_leaves2 += 1
                # add height
                node.height=total_time
                
            # transit from phase 2 to phase 1
            elif (event1+transition_rate12)/total_rate <= eprob < (event1+transition_rate12+transition_rate21)/total_rate:
                ran = random.uniform(0,1)
                ind = int(ran*len(nodes_phase2))
                node = nodes_phase2[ind]
                del nodes_phase2[ind]
                
                child = Tree()
                child.dist = 0
                child.add_features(extinct=False)
                child.add_features(phase=1)
                nodes_phase1.append(child)
                node.add_child(child)
                node.extinct = True # disappear of the current node
                curr_num_leaves1 += 1
                curr_num_leaves2 -= 1
                # add height
                node.height=total_time
            # birth or death event in phase 2
            elif (event1+transition_rate12+transition_rate21)/total_rate <= eprob < (event1+transition_rate12+transition_rate21+event2)/total_rate:
                ran = random.uniform(0,1)
                ind = int(ran*len(nodes_phase2))
                node = nodes_phase2[ind]
                del nodes_phase2[ind]
                
                curr_num_leaves2 -= 1

                # add height
                node.height=total_time

                # birth event (speciation) creating two children nodes in phase 2
                if (event1+transition_rate12+transition_rate21)/total_rate <= eprob < (event1+transition_rate12+transition_rate21+birth_rate2)/total_rate:

                    node.extinct = True # disappear of the current node
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
        # add attribute dist
        Add_dist(tree)

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
    """Compute the time stamp of the speciation event and ordered by time
    """
    root2=False
    for parent in tsim.get_leaves()[0].get_ancestors():
        if (len(parent.get_leaves())==len(tsim.get_leaves()) and (not parent.is_root())):
            root2=True
    seq=[]
    for node in tsim.traverse():
        dis = node.dist
        if node.is_root():
            seq.append(dis)
        else:
            for parent in node.get_ancestors():
                dis += parent.dist
            if root2:
                dis-=tsim.dist
            seq.append(dis)
    seq.sort()
    return seq

def nabsdiff(t, start=2, fix_size=True):
    """Compute two arrays
    Outputs:
    init_nx: the array of the time stamp for the speciation events in order
    init_ny: the array of the number of lineages
    
    """
    if fix_size:
        size = len(t.get_leaves())
        initial_array = step_array(t)
        init_data = ignoreRoot(initial_array,start=start)
        init_data=init_data[0:(size-start+2)]
        init_x = np.array(init_data)
        init_nx = init_x/init_data[-1]
        lineage = [start]
        for num in range(start, size+1):
            lineage.append(num)
        y = np.array(lineage)
        init_ny = y/size
        return init_nx, init_ny
    else:
        size = len(t.get_leaves())
        init_data = step_array(t)
        init_data=[0]+init_data[0:size]
        init_x = np.array(init_data)
        init_nx = init_x/init_data[-1]
        lineage = [1]
        for num in range(1, size+1):
            lineage.append(num)
        y = np.array(lineage)
        init_ny = y/size
        return init_nx, init_ny


def PD(tree, no_root=True):
    """ Average branch length excluding the branch length of the root
    when no_root is True
    """
    distance = []
    if no_root:
        for node in tree.traverse():
            if not node.is_root():
                distance.append(node.dist)
    else:
        for node in tree.traverse():
            distance.append(node.dist)
    if len(distance)==0:
        return 0
    else:
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


def balance(tree):
    """Return three balance index all at once
    """
    # assign the number of leaves for subtrees
    for node in tree.traverse():
        if node.is_leaf():
            node.add_features(size=1)
            if node.phase==1:
                node.add_features(size1=1)
                node.add_features(size2=0)
            else:
                node.add_features(size1=0)
                node.add_features(size2=1)
        else:
            node.add_features(size=0)
            node.add_features(size1=0)
            node.add_features(size2=0)

    for ancestor in tree.get_leaves()[0].get_ancestors():
        ancestor.add_features(size=0)
        ancestor.add_features(size1=0)
        ancestor.add_features(size2=0)
            
    for leaf in tree.get_leaves():
        for anc in leaf.get_ancestors():
            anc.size += 1
            anc.size1 += leaf.size1
            anc.size2 += leaf.size2

    n = len(tree.get_leaves())
    bal,bal1,bal2=0,0,0
    for node in tree.traverse():
        if not node.is_leaf():
            child1=node.children[0]
            child2=node.children[1]
            bal+=abs(child1.size-child2.size)
            bal1+=abs(child1.size1-child2.size1)
            bal2+=abs(child1.size2-child2.size2)
    return bal/((n-1)*(n-2)/2), bal1/((n-1)*(n-2)/2), bal2/((n-1)*(n-2)/2)
            
            

def imbalance(tree):
    n = len(tree.get_leaves())
    if n<=3:
        return 0
    array = []
    for node in tree.traverse():
        if not node.is_leaf():
            diff = abs(len(node.children[0].get_leaves())-len(node.children[1].get_leaves()))
            array.append(diff)
    return sum(array)/((n-1)*(n-2)/2)

def imb1(tree):
    n = len(tree.get_leaves())
    if n<=3:
        return 0
    array = []
    for node in tree.traverse():
        if not node.is_leaf():
            diff = abs(numphase1(node.children[0])-numphase1(node.children[1]))
            array.append(diff)
    return sum(array)/((n-1)*(n-2)/2)

def imb2(tree):
    n = len(tree.get_leaves())
    if n<=3:
        return 0
    array = []
    for node in tree.traverse():
        if not node.is_leaf():
            diff = abs(numphase2(node.children[0])-numphase2(node.children[1]))
            array.append(diff)
    return sum(array)/((n-1)*(n-2)/2)

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


def mdistp1(t):
    seq=[]
    for node in t.traverse():
        if not node.is_root() and numphase1(node)!=0:
            seq.append(node.dist)
    return mean(seq)








t=tobs
init_array, init_resp = nabsdiff(t,20)
tarr = break_tree(t,size1,size2)
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
    b,b1,b2=balance(node)
    balobs.append(b)
    bal1obs.append(b1)
    bal2obs.append(b2)
    distanceobs.append(PD(node))
    t1,t2=trans(node)
    statobs.append(t1)
    stat2obs.append(t2)
    propobs.append(numphase1(node)/size_node)
    tspanobs.append(tree_height(node))
    

final_array = init_array
final_resp = init_resp
    

ex_dist = mean(distanceobs)
ex_imb = mean(balobs)
ex_imb1 = mean(bal1obs)
ex_imb2 = mean(bal2obs)
ex_tspan = mean(tspanobs)
ex_stat = mean(statobs)
ex_stat2 = mean(stat2obs)
ex_prop = mean(propobs)

ex_wdist = weighted_mean(distanceobs,list(np.array(sizeobs)*2-2))
ex_wprop = weighted_mean(propobs,sizeobs)


tol_dist = 2*sqrt(variance(distanceobs))
tol_imb = 2*sqrt(variance(balobs))
tol_imb1 = 2*sqrt(variance(bal1obs))
tol_imb2 = 2*sqrt(variance(bal2obs))
tol_tspan = 2*sqrt(variance(tspanobs))
tol_stat = 2*sqrt(variance(statobs))
tol_stat2 = 2*sqrt(variance(stat2obs))
tol_prop = 2*sqrt(variance(propobs))


max_dist = max(distanceobs)
max_imb = max(balobs)
max_imb1 = max(bal1obs)
max_imb2 = max(bal2obs)
max_tspan = max(tspanobs)
max_stat = max(statobs)
max_stat2 = max(stat2obs)
max_prop = max(propobs)



min_dist = min(distanceobs)
min_imb = min(balobs)
min_imb1 = min(bal1obs)
min_imb2 = min(bal2obs)
min_tspan = min(tspanobs)
min_stat = min(statobs)
min_stat2 = min(stat2obs)
min_prop = min(propobs)


N=0
accept1 = []
accept2 = []
accept3 = []
accept4 = []
accept5 = []
accept6 = []
acceptp = []
domtemp = []
import time
start=time.time()
while len(accept1)<200:
    do = True
    while do:
        do = False
        flag = 0
        birth1sim = random.uniform(0,0.2)
        birth2sim = random.uniform(0,0.2)
        death1sim = random.uniform(0,0.2)
        death2sim = random.uniform(0,0.2)
        q12sim = random.uniform(0,0.2)
        q21sim = random.uniform(0,0.2)
        psim = random.uniform(0,1)
        w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
        while w<=0.001:
            birth1sim = random.uniform(0,0.2)
            birth2sim = random.uniform(0,0.2)
            death1sim = random.uniform(0,0.2)
            death2sim = random.uniform(0,0.2)
            q12sim = random.uniform(0,0.2)
            q21sim = random.uniform(0,0.2)
            psim = random.uniform(0,1)
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
        r=0
        length = 0
        while flag == 0:
            startphase=2
            t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], start=startphase, nsize=sizeobs[0])
            r+=1
            length = len(t.get_leaves())
            if r == 5:
                do = True
                break
            
    assert flag == 1
    b,b1,b2=balance(t)
    bal=[b]
    bal1=[b1]
    bal2=[b2]
    tspan=[tree_height(t)]
    dist=[PD(t)]
    t1,t2=trans(t)
    stat=[t1]
    stat2=[t2]
    prop=[numphase1(t)/size_node]
    N+=1
    for j in range(1,1):
        flag = 0
        while flag == 0:
            startphase=2
            tsim, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], start=startphase, nsize=sizeobs[j])
        assert flag == 1
        b,b1,b2=balance(tsim)
        bal.append(b)
        bal1.append(b1)
        bal2.append(b2)
        dist.append(PD(tsim))
        t1,t2=trans(tsim)
        stat.append(t1)
        stat2.append(t2)
        prop.append(numphase1(tsim)/sizeobs[j])
        tspan.append(tree_height(tsim))
    
    mbal=mean(bal)
    mbal1=mean(bal1)
    mbal2=mean(bal2)
    mdist=mean(dist)
    mstat=mean(stat)
    mstat2=mean(stat2)
    mprop=mean(prop)
    mtspan=mean(tspan)

    if min_tspan <= mtspan <= max_tspan and min_dist<=mdist<=max_dist and min_imb<=mbal<=max_imb and min_imb1<=mbal1<=max_imb1 and min_imb2<=mbal2<=max_imb2 and min_stat<=mstat<=max_stat and min_stat2<=mstat2<=max_stat2 and min_prop<=mprop<=max_prop:
        accept1.append(birth1sim)
        accept2.append(birth2sim)
        accept3.append(death1sim)
        accept4.append(death2sim)
        accept5.append(q12sim)
        accept6.append(q21sim)
        acceptp.append(psim)

        domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
        
accept_rate = len(accept1)/N

meanseq1=[mean(accept1)]
meanseq2=[mean(accept2)]
meanseq3=[mean(accept3)]
meanseq4=[mean(accept4)]
meanseq5=[mean(accept5)]
meanseq6=[mean(accept6)]
meanseqp=[mean(acceptp)]
dommean=[mean(domtemp)]

wmeanseq1=[mean(accept1)]
wmeanseq2=[mean(accept2)]
wmeanseq3=[mean(accept3)]
wmeanseq4=[mean(accept4)]
wmeanseq5=[mean(accept5)]
wmeanseq6=[mean(accept6)]
wmeanseqp=[mean(acceptp)]
rate=[accept_rate]

dommean_u=[np.quantile(domtemp,0.75)]
weighted_q1u=[np.quantile(accept1,0.75)]
weighted_q2u=[np.quantile(accept2,0.75)]
weighted_q3u=[np.quantile(accept3,0.75)]
weighted_q4u=[np.quantile(accept4,0.75)]
weighted_q5u=[np.quantile(accept5,0.75)]
weighted_q6u=[np.quantile(accept6,0.75)]
weighted_qpu=[np.quantile(acceptp,0.75)]

dommean_l=[np.quantile(domtemp,0.25)]
weighted_q1l=[np.quantile(accept1,0.25)]
weighted_q2l=[np.quantile(accept2,0.25)]
weighted_q3l=[np.quantile(accept3,0.25)]
weighted_q4l=[np.quantile(accept4,0.25)]
weighted_q5l=[np.quantile(accept5,0.25)]
weighted_q6l=[np.quantile(accept6,0.25)]
weighted_qpl=[np.quantile(acceptp,0.25)]

array1=accept1
array2=accept2
array3=accept3
array4=accept4
array5=accept5
array6=accept6
arrayp=acceptp

print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u)+'\n\n'+'weighted_qpu='+str(weighted_qpu))
print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l)+'\n\n'+'weighted_qpl='+str(weighted_qpl))
print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6)+'\n\n'+'meanseqp='+str(meanseqp))
print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6)+'\n\n'+'acceptp='+str(arrayp))
print('\n\n'+'rate='+str(rate), flush=True)

end = time.time()
duration=[end-start]

z=0
accept_rate=0

for k in range(1,3):
    start=time.time()
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
    scalep = sqrt(2*weighted_var(arrayp, weight))
    
    accept = np.stack((accept1,accept2,accept3,accept4,accept5,accept6))
    covmat = 2*np.cov(accept, aweights=weight)

    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    acceptp = []
    domtemp = []

    if accept_rate >= 0.03:
        tol_dist = math.exp(-0.2)*tol_dist
        tol_imb = math.exp(-0.2)*tol_imb
        tol_imb1 = math.exp(-0.2)*tol_imb1
        tol_imb2 = math.exp(-0.2)*tol_imb2
        tol_tspan = math.exp(-0.2)*tol_tspan
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
            psim = arrayp[ind]+np.random.normal(loc=0.0, scale=scalep, size=1)[0]
            if psim<0:
                psim=0
            elif psim>1:
                psim=1
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0.001 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=0.2:
                ind = random.choices(index, weights=weight, k=1)[0]
                psim = arrayp[ind]+np.random.normal(loc=0.0, scale=scalep, size=1)[0]
                if psim<0:
                    psim=0
                elif psim>1:
                    psim=1
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            r=0
            while flag == 0:
                startphase=2
                t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], start=startphase, nsize=sizeobs[0])
                r+=1
                length = len(t.get_leaves())
                if r == 5:
                    do = True
                    break
                    
        assert flag == 1
        b,b1,b2=balance(t)
        bal=[b]
        bal1=[b1]
        bal2=[b2]
        tspan=[tree_height(t)]
        dist=[PD(t)]
        t1,t2=trans(t)
        stat=[t1]
        stat2=[t2]
        prop=[numphase1(t)/sizeobs[0]]
        N+=1
        for j in range(1,len(sizeobs)):
            flag = 0
            while flag == 0:
                startphase=2
                tsim, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], start=startphase, nsize=sizeobs[j])
            assert flag == 1
            b,b1,b2=balance(tsim)
            bal.append(b)
            bal1.append(b1)
            bal2.append(b2)
            dist.append(PD(tsim))
            t1,t2=trans(tsim)
            stat.append(t1)
            stat2.append(t2)
            prop.append(numphase1(tsim)/sizeobs[j])
            tspan.append(tree_height(tsim))
        
        mbal=mean(bal)
        mbal1=mean(bal1)
        mbal2=mean(bal2)
        mdist=mean(dist)
        mstat=mean(stat)
        mstat2=mean(stat2)
        mprop=mean(prop)
        mtspan=mean(tspan)

        
        wmdist=weighted_mean(dist,list(np.array(sizeobs)*2-2))
        wmprop=weighted_mean(prop,sizeobs)
        
        if abs(mtspan-ex_tspan) <= tol_tspan and abs(mdist-ex_dist) <= tol_dist and abs(mbal-ex_imb)<=tol_imb and abs(mbal1-ex_imb1)<=tol_imb1 and abs(mbal2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            acceptp.append(psim)

            domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                parp = (psim-arrayp[item])/scalep
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)*norm.pdf(parp)
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
    meanseqp.append(mean(acceptp))
    dommean.append(weighted_mean(domtemp, weightnew))

    wmeanseq1.append(weighted_mean(accept1, weightnew))
    wmeanseq2.append(weighted_mean(accept2, weightnew))
    wmeanseq3.append(weighted_mean(accept3, weightnew))
    wmeanseq4.append(weighted_mean(accept4, weightnew))
    wmeanseq5.append(weighted_mean(accept5, weightnew))
    wmeanseq6.append(weighted_mean(accept6, weightnew))
    wmeanseqp.append(weighted_mean(acceptp, weightnew))
    rate.append(accept_rate)

    dommean_u.append(weighted_quantile(domtemp, 0.75, sample_weight=weightnew))
    weighted_q1u.append(weighted_quantile(accept1, 0.75, sample_weight=weightnew))
    weighted_q2u.append(weighted_quantile(accept2, 0.75, sample_weight=weightnew))
    weighted_q3u.append(weighted_quantile(accept3, 0.75, sample_weight=weightnew))
    weighted_q4u.append(weighted_quantile(accept4, 0.75, sample_weight=weightnew))
    weighted_q5u.append(weighted_quantile(accept5, 0.75, sample_weight=weightnew))
    weighted_q6u.append(weighted_quantile(accept6, 0.75, sample_weight=weightnew))
    weighted_qpu.append(weighted_quantile(acceptp, 0.75, sample_weight=weightnew))

    dommean_l.append(weighted_quantile(domtemp, 0.25, sample_weight=weightnew))
    weighted_q1l.append(weighted_quantile(accept1, 0.25, sample_weight=weightnew))
    weighted_q2l.append(weighted_quantile(accept2, 0.25, sample_weight=weightnew))
    weighted_q3l.append(weighted_quantile(accept3, 0.25, sample_weight=weightnew))
    weighted_q4l.append(weighted_quantile(accept4, 0.25, sample_weight=weightnew))
    weighted_q5l.append(weighted_quantile(accept5, 0.25, sample_weight=weightnew))
    weighted_q6l.append(weighted_quantile(accept6, 0.25, sample_weight=weightnew))
    weighted_qpl.append(weighted_quantile(acceptp, 0.25, sample_weight=weightnew))

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6
    arrayp=acceptp

    weight = weightnew

    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
    print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u)+'\n\n'+'weighted_qpu='+str(weighted_qpu))
    print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l)+'\n\n'+'weighted_qpl='+str(weighted_qpl))
    print('\n\n'+'weighted_meanseq1='+str(wmeanseq1)+'\n\n'+'weighted_meanseq2='+str(wmeanseq2)+'\n\n'+'weighted_meanseq3='+str(wmeanseq3)+'\n\n'+'weighted_meanseq4='+str(wmeanseq4)+'\n\n'+'weighted_meanseq5='+str(wmeanseq5)+'\n\n'+'weighted_meanseq6='+str(wmeanseq6)+'\n\n'+'weighted_meanseqp='+str(wmeanseqp))
    print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6)+'\n\n'+'meanseqp='+str(meanseqp))
    print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6)+'\n\n'+'acceptp='+str(arrayp))
    print('\n\n'+'weight='+str(list(weight))+'\n\n'+'rate='+str(rate), flush=True)

    end=time.time()
    duration.append(end-start)

size=len(tobs.get_leaves())

t=tobs
ex_dist = PD(t)
ex_imb,ex_imb1,ex_imb2 = balance(t)
ex_tspan = tree_height(t)
ex_stat,ex_stat2 = trans(t)
ex_prop = numphase1(t)/size

final_array, final_resp = nabsdiff(t,start=20)

birth1=weighted_mean(accept1, weight)
birth2=weighted_mean(accept2, weight)
death1=weighted_mean(accept3, weight)
death2=weighted_mean(accept4, weight)
q12=weighted_mean(accept5, weight)
q21=weighted_mean(accept6, weight)
ptest=weighted_mean(acceptp, weight)

balobs=[]
bal1obs=[]
bal2obs=[]
tspanobs=[]
distanceobs=[]
statobs=[]
stat2obs=[]
propobs=[]

for i in range(10):
    flag = 0
    while flag == 0:
        startphase=2
        tsim,flag = birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size, start=startphase)
    b,b1,b2=balance(tsim)
    balobs.append(b)
    bal1obs.append(b1)
    bal2obs.append(b2)
    distanceobs.append(PD(tsim))
    tspanobs.append(tree_height(tsim))
    propobs.append(numphase1(tsim)/size)
    t1,t2=trans(tsim)
    statobs.append(t1)
    stat2obs.append(t2)

tol_dist = 40*sqrt(variance(distanceobs))
tol_imb = 40*sqrt(variance(balobs))
tol_imb1 = 40*sqrt(variance(bal1obs))
tol_imb2 = 40*sqrt(variance(bal2obs))
tol_tspan = 40*sqrt(variance(tspanobs))
tol_prop = 40*sqrt(variance(propobs))
tol_stat = 40*sqrt(variance(statobs))
tol_stat2 = 40*sqrt(variance(stat2obs))

accept_rate=0
z=0
time_elapse=[]

for k in range(3,T):
    start=time.time()
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
    scalep = sqrt(2*weighted_var(arrayp, weight))
    
    accept = np.stack((accept1,accept2,accept3,accept4,accept5,accept6))
    covmat = 2*np.cov(accept, aweights=weight)

    if accept_rate >= 0.03:
        z += 1
        birth1=weighted_mean(accept1, weight)
        birth2=weighted_mean(accept2, weight)
        death1=weighted_mean(accept3, weight)
        death2=weighted_mean(accept4, weight)
        q12=weighted_mean(accept5, weight)
        q21=weighted_mean(accept6, weight)
        ptest=weighted_mean(acceptp, weight)

        balobs=[]
        bal1obs=[]
        bal2obs=[]
        tspanobs=[]
        distanceobs=[]
        statobs=[]
        stat2obs=[]
        propobs=[]
        for i in range(10):
            flag = 0
            while flag == 0:
                startphase=2
                tsim,flag = birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size, start=startphase)
            b,b1,b2=balance(tsim)
            balobs.append(b)
            bal1obs.append(b1)
            bal2obs.append(b2)
            distanceobs.append(PD(tsim))
            tspanobs.append(tree_height(tsim))
            propobs.append(numphase1(tsim)/size)
            t1,t2=trans(tsim)
            statobs.append(t1)
            stat2obs.append(t2)
            
        tol_dist = 40*math.exp(-0.2*z)*sqrt(variance(distanceobs))
        tol_imb = 40*math.exp(-0.2*z)*sqrt(variance(balobs))
        tol_imb1 = 40*math.exp(-0.2*z)*sqrt(variance(bal1obs))
        tol_imb2 = 40*math.exp(-0.2*z)*sqrt(variance(bal2obs))
        tol_tspan = 40*math.exp(-0.2*z)*sqrt(variance(tspanobs))
        tol_prop = 40*math.exp(-0.2*z)*sqrt(variance(propobs))
        tol_stat = 40*math.exp(-0.2*z)*sqrt(variance(statobs))
        tol_stat2 = 40*math.exp(-0.2*z)*sqrt(variance(stat2obs))
        accept_rate=0

    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    acceptp = []
    domtemp = []
    time_elapse=[]
    
    
    N=0
    start=time.time()
    while len(accept1)<M:
        do = True
        while do:
            do = False
            flag = 0
            ind = random.choices(index, weights=weight, k=1)[0]
            psim = arrayp[ind]+np.random.normal(loc=0.0, scale=scalep, size=1)[0]
            if psim<0:
                psim=0
            elif psim>1:
                psim=1
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0.001 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=0.2:
                ind = random.choices(index, weights=weight, k=1)[0]
                psim = arrayp[ind]+np.random.normal(loc=0.0, scale=scalep, size=1)[0]
                if psim<0:
                    psim=0
                elif psim>1:
                    psim=1
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            r=0
            while flag == 0:
                startphase=2
                t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], start=startphase, nsize=size)
                r+=1
                if r == 5:
                    do = True
                    break
                
        assert flag == 1
        tspan=[tree_height(t)]
        dist=[PD(t)]
        b,b1,b2=balance(t)
        bala = [b]
        bala1 = [b1]
        bala2 = [b2]
        prop = [numphase1(t)/size] # the proportion of phase 1 tips
        t1,t2=trans(t)
        stat = [t1]
        stat2 = [t2]
        N+=1

        if N%10==0:
            print("N="+str(N),flush=True)
        
        mdist = mean(dist)
        mbala = mean(bala)
        mbala1 = mean(bala1)
        mbala2 = mean(bala2)
        mtspan = mean(tspan)
        mprop = mean(prop)
        mstat = mean(stat)
        mstat2 = mean(stat2)
        
        if abs(mtspan-ex_tspan) <= tol_tspan and abs(mdist-ex_dist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            acceptp.append(psim)
            domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                parp = (psim-arrayp[item])/scalep
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)*norm.pdf(parp)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
            # time
            end=time.time()
            time_elapse.append(end-start)
            if len(accept1)%10==0:
                print('\n\n'+'accept1='+str(accept1)+'\n\n'+'accept2='+str(accept2)+'\n\n'+'accept3='+str(accept3)+'\n\n'+'accept4='+str(accept4)+'\n\n'+'accept5='+str(accept5)+'\n\n'+'accept6='+str(accept6))
                print('\n\n'+'time_elapse='+str(time_elapse),flush=True)
            
    accept_rate = len(accept1)/N
    weightnew = list(np.array(weightnew)/sum(weightnew))

    meanseq1.append(mean(accept1))
    meanseq2.append(mean(accept2))
    meanseq3.append(mean(accept3))
    meanseq4.append(mean(accept4))
    meanseq5.append(mean(accept5))
    meanseq6.append(mean(accept6))
    meanseqp.append(mean(acceptp))
    dommean.append(weighted_mean(domtemp, weightnew))

    wmeanseq1.append(weighted_mean(accept1, weightnew))
    wmeanseq2.append(weighted_mean(accept2, weightnew))
    wmeanseq3.append(weighted_mean(accept3, weightnew))
    wmeanseq4.append(weighted_mean(accept4, weightnew))
    wmeanseq5.append(weighted_mean(accept5, weightnew))
    wmeanseq6.append(weighted_mean(accept6, weightnew))
    wmeanseqp.append(weighted_mean(acceptp, weightnew))
    rate.append(accept_rate)

    dommean_u.append(weighted_quantile(domtemp, 0.75, sample_weight=weightnew))
    weighted_q1u.append(weighted_quantile(accept1, 0.75, sample_weight=weightnew))
    weighted_q2u.append(weighted_quantile(accept2, 0.75, sample_weight=weightnew))
    weighted_q3u.append(weighted_quantile(accept3, 0.75, sample_weight=weightnew))
    weighted_q4u.append(weighted_quantile(accept4, 0.75, sample_weight=weightnew))
    weighted_q5u.append(weighted_quantile(accept5, 0.75, sample_weight=weightnew))
    weighted_q6u.append(weighted_quantile(accept6, 0.75, sample_weight=weightnew))
    weighted_qpu.append(weighted_quantile(acceptp, 0.75, sample_weight=weightnew))

    dommean_l.append(weighted_quantile(domtemp, 0.25, sample_weight=weightnew))
    weighted_q1l.append(weighted_quantile(accept1, 0.25, sample_weight=weightnew))
    weighted_q2l.append(weighted_quantile(accept2, 0.25, sample_weight=weightnew))
    weighted_q3l.append(weighted_quantile(accept3, 0.25, sample_weight=weightnew))
    weighted_q4l.append(weighted_quantile(accept4, 0.25, sample_weight=weightnew))
    weighted_q5l.append(weighted_quantile(accept5, 0.25, sample_weight=weightnew))
    weighted_q6l.append(weighted_quantile(accept6, 0.25, sample_weight=weightnew))
    weighted_qpl.append(weighted_quantile(acceptp, 0.25, sample_weight=weightnew))

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6
    arrayp=acceptp

    weight = weightnew

    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
    print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u)+'\n\n'+'weighted_qpu='+str(weighted_qpu))
    print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l)+'\n\n'+'weighted_qpl='+str(weighted_qpl))
    print('\n\n'+'weighted_meanseq1='+str(wmeanseq1)+'\n\n'+'weighted_meanseq2='+str(wmeanseq2)+'\n\n'+'weighted_meanseq3='+str(wmeanseq3)+'\n\n'+'weighted_meanseq4='+str(wmeanseq4)+'\n\n'+'weighted_meanseq5='+str(wmeanseq5)+'\n\n'+'weighted_meanseq6='+str(wmeanseq6)+'\n\n'+'weighted_meanseqp='+str(wmeanseqp))
    print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6)+'\n\n'+'meanseqp='+str(meanseqp))
    print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6)+'\n\n'+'acceptp='+str(arrayp))
    print('\n\n'+'weight='+str(list(weight))+'\n\n'+'rate='+str(rate), flush=True)

    end=time.time()
    duration.append(end-start)


for k in range(T,3*T):
    start=time.time()
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
    scalep = sqrt(2*weighted_var(arrayp, weight))
    
    accept = np.stack((accept1,accept2,accept3,accept4,accept5,accept6))
    covmat = 2*np.cov(accept, aweights=weight)

    if accept_rate >= 0.03:
        z += 1
        birth1=weighted_mean(accept1, weight)
        birth2=weighted_mean(accept2, weight)
        death1=weighted_mean(accept3, weight)
        death2=weighted_mean(accept4, weight)
        q12=weighted_mean(accept5, weight)
        q21=weighted_mean(accept6, weight)
        ptest=weighted_mean(acceptp, weight)

        balobs=[]
        bal1obs=[]
        bal2obs=[]
        tspanobs=[]
        distanceobs=[]
        statobs=[]
        stat2obs=[]
        propobs=[]
        for i in range(10):
            flag = 0
            while flag == 0:
                startphase=2
                tsim,flag = birth_death_tree2([birth1,birth2], [death1,death2], [q12,q21], nsize=size, start=startphase)
            b,b1,b2=balance(tsim)
            balobs.append(b)
            bal1obs.append(b1)
            bal2obs.append(b2)
            distanceobs.append(PD(tsim))
            tspanobs.append(tree_height(tsim))
            propobs.append(numphase1(tsim)/size)
            t1,t2=trans(tsim)
            statobs.append(t1)
            stat2obs.append(t2)
            
        tol_dist = 40*math.exp(-0.2*z)*sqrt(variance(distanceobs))
        tol_imb = 40*math.exp(-0.2*z)*sqrt(variance(balobs))
        tol_imb1 = 40*math.exp(-0.2*z)*sqrt(variance(bal1obs))
        tol_imb2 = 40*math.exp(-0.2*z)*sqrt(variance(bal2obs))
        tol_tspan = 40*math.exp(-0.2*z)*sqrt(variance(tspanobs))
        tol_prop = 40*math.exp(-0.2*z)*sqrt(variance(propobs))
        tol_stat = 40*math.exp(-0.2*z)*sqrt(variance(statobs))
        tol_stat2 = 40*math.exp(-0.2*z)*sqrt(variance(stat2obs))
        accept_rate=0

    accept1 = []
    accept2 = []
    accept3 = []
    accept4 = []
    accept5 = []
    accept6 = []
    acceptp = []
    domtemp = []
    time_elapse=[]
    nLTTstat=[]
    
    
    N=0
    start=time.time()
    while len(accept1)<2*M:
        do = True
        while do:
            do = False
            flag = 0
            ind = random.choices(index, weights=weight, k=1)[0]
            psim = arrayp[ind]+np.random.normal(loc=0.0, scale=scalep, size=1)[0]
            if psim<0:
                psim=0
            elif psim>1:
                psim=1
            [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
            w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim])  
            while w<=0.001 or min(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)<=0.0 or max(birth1sim, birth2sim,death1sim,death2sim,q12sim,q21sim)>=0.2:
                ind = random.choices(index, weights=weight, k=1)[0]
                psim = arrayp[ind]+np.random.normal(loc=0.0, scale=scalep, size=1)[0]
                if psim<0:
                    psim=0
                elif psim>1:
                    psim=1
                [birth1sim,birth2sim,death1sim,death2sim,q12sim,q21sim]=accept.T[ind]+multivariate_normal.rvs(np.zeros(6),covmat)
                w = dom([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim]) 
            r=0
            while flag == 0:
                startphase=2
                t, flag = birth_death_tree2([birth1sim, birth2sim], [death1sim, death2sim], [q12sim,q21sim], start=startphase, nsize=size)
                r+=1
                if r == 5:
                    do = True
                    break
                
        assert flag == 1
        tspan=[tree_height(t)]
        dist=[PD(t)]
        b,b1,b2=balance(t)
        bala = [b]
        bala1 = [b1]
        bala2 = [b2]
        prop = [numphase1(t)/size] # the proportion of phase 1 tips
        t1,t2=trans(t)
        stat = [t1]
        stat2 = [t2]
        N+=1

        if N%10==0:
            print("N="+str(N),flush=True)
        
        mdist = mean(dist)
        mbala = mean(bala)
        mbala1 = mean(bala1)
        mbala2 = mean(bala2)
        mtspan = mean(tspan)
        mprop = mean(prop)
        mstat = mean(stat)
        mstat2 = mean(stat2)

        sim_array, sim_resp = nabsdiff(t,20)
        
        if abs(mtspan-ex_tspan) <= tol_tspan and abs(mdist-ex_dist) <= tol_dist and abs(mbala-ex_imb)<=tol_imb and abs(mbala1-ex_imb1)<=tol_imb1 and abs(mbala2-ex_imb2)<=tol_imb2 and abs(mprop-ex_prop)<=tol_prop and abs(mstat-ex_stat)<=tol_stat and abs(mstat2-ex_stat2)<=tol_stat2:
            accept1.append(birth1sim)
            accept2.append(birth2sim)
            accept3.append(death1sim)
            accept4.append(death2sim)
            accept5.append(q12sim)
            accept6.append(q21sim)
            acceptp.append(psim)
            domtemp.append(dom([birth1sim,birth2sim],[death1sim,death2sim],[q12sim,q21sim]))
            invw=0
            parsim = np.array([birth1sim,birth2sim,death1sim,death2sim, q12sim,q21sim])
            for item in range(len(weight)):
                par = np.array([array1[item],array2[item],array3[item],array4[item],array5[item],array6[item]])
                parp = (psim-arrayp[item])/scalep
                invw += weight[item]*multivariate_normal.pdf(par, mean=parsim, cov=covmat)*norm.pdf(parp)
            weightnewsim = 1/invw
            weightnew.append(weightnewsim)
            # time
            end=time.time()
            time_elapse.append(end-start)
            nLTTstat.append(absdist_array(final_array, sim_array, final_resp, sim_resp))
            if len(accept1)%10==0:
                print('\n\n'+'accept1='+str(accept1)+'\n\n'+'accept2='+str(accept2)+'\n\n'+'accept3='+str(accept3)+'\n\n'+'accept4='+str(accept4)+'\n\n'+'accept5='+str(accept5)+'\n\n'+'accept6='+str(accept6))
                print('\n\n'+'time_elapse='+str(time_elapse),flush=True)
            
    accept_rate = len(accept1)/N

    accept_nLTT = sorted(nLTTstat)[0:M]
    accept_index = []
    for nltt in accept_nLTT:
        accept_index.append(nLTTstat.index(nltt))
    accept1=[accept1[item] for item in accept_index]
    accept2=[accept2[item] for item in accept_index]
    accept3=[accept3[item] for item in accept_index]
    accept4=[accept4[item] for item in accept_index]
    accept5=[accept5[item] for item in accept_index]
    accept6=[accept6[item] for item in accept_index]
    acceptp=[acceptp[item] for item in accept_index]
    weightnew=[weightnew[item] for item in accept_index]
    domtemp=[domtemp[item] for item in accept_index]
    
    
    weightnew = list(np.array(weightnew)/sum(weightnew))

    meanseq1.append(mean(accept1))
    meanseq2.append(mean(accept2))
    meanseq3.append(mean(accept3))
    meanseq4.append(mean(accept4))
    meanseq5.append(mean(accept5))
    meanseq6.append(mean(accept6))
    meanseqp.append(mean(acceptp))
    dommean.append(weighted_mean(domtemp, weightnew))

    wmeanseq1.append(weighted_mean(accept1, weightnew))
    wmeanseq2.append(weighted_mean(accept2, weightnew))
    wmeanseq3.append(weighted_mean(accept3, weightnew))
    wmeanseq4.append(weighted_mean(accept4, weightnew))
    wmeanseq5.append(weighted_mean(accept5, weightnew))
    wmeanseq6.append(weighted_mean(accept6, weightnew))
    wmeanseqp.append(weighted_mean(acceptp, weightnew))
    rate.append(accept_rate)

    dommean_u.append(weighted_quantile(domtemp, 0.75, sample_weight=weightnew))
    weighted_q1u.append(weighted_quantile(accept1, 0.75, sample_weight=weightnew))
    weighted_q2u.append(weighted_quantile(accept2, 0.75, sample_weight=weightnew))
    weighted_q3u.append(weighted_quantile(accept3, 0.75, sample_weight=weightnew))
    weighted_q4u.append(weighted_quantile(accept4, 0.75, sample_weight=weightnew))
    weighted_q5u.append(weighted_quantile(accept5, 0.75, sample_weight=weightnew))
    weighted_q6u.append(weighted_quantile(accept6, 0.75, sample_weight=weightnew))
    weighted_qpu.append(weighted_quantile(acceptp, 0.75, sample_weight=weightnew))

    dommean_l.append(weighted_quantile(domtemp, 0.25, sample_weight=weightnew))
    weighted_q1l.append(weighted_quantile(accept1, 0.25, sample_weight=weightnew))
    weighted_q2l.append(weighted_quantile(accept2, 0.25, sample_weight=weightnew))
    weighted_q3l.append(weighted_quantile(accept3, 0.25, sample_weight=weightnew))
    weighted_q4l.append(weighted_quantile(accept4, 0.25, sample_weight=weightnew))
    weighted_q5l.append(weighted_quantile(accept5, 0.25, sample_weight=weightnew))
    weighted_q6l.append(weighted_quantile(accept6, 0.25, sample_weight=weightnew))
    weighted_qpl.append(weighted_quantile(acceptp, 0.25, sample_weight=weightnew))

    array1=accept1
    array2=accept2
    array3=accept3
    array4=accept4
    array5=accept5
    array6=accept6
    arrayp=acceptp

    weight = weightnew

    print('\n\n'+'z='+str(z)+',k='+str(k))
    print('\n\n'+'weighted_domu='+str(dommean_u)+'\n\n'+'weighted_dom='+str(dommean)+'\n\n'+'weighted_doml='+str(dommean_l))
    print('\n\n'+'weighted_q1u='+str(weighted_q1u)+'\n\n'+'weighted_q2u='+str(weighted_q2u)+'\n\n'+'weighted_q3u='+str(weighted_q3u)+'\n\n'+'weighted_q4u='+str(weighted_q4u)+'\n\n'+'weighted_q5u='+str(weighted_q5u)+'\n\n'+'weighted_q6u='+str(weighted_q6u)+'\n\n'+'weighted_qpu='+str(weighted_qpu))
    print('\n\n'+'weighted_q1l='+str(weighted_q1l)+'\n\n'+'weighted_q2l='+str(weighted_q2l)+'\n\n'+'weighted_q3l='+str(weighted_q3l)+'\n\n'+'weighted_q4l='+str(weighted_q4l)+'\n\n'+'weighted_q5l='+str(weighted_q5l)+'\n\n'+'weighted_q6l='+str(weighted_q6l)+'\n\n'+'weighted_qpl='+str(weighted_qpl))
    print('\n\n'+'weighted_meanseq1='+str(wmeanseq1)+'\n\n'+'weighted_meanseq2='+str(wmeanseq2)+'\n\n'+'weighted_meanseq3='+str(wmeanseq3)+'\n\n'+'weighted_meanseq4='+str(wmeanseq4)+'\n\n'+'weighted_meanseq5='+str(wmeanseq5)+'\n\n'+'weighted_meanseq6='+str(wmeanseq6)+'\n\n'+'weighted_meanseqp='+str(wmeanseqp))
    print('\n\n'+'meanseq1='+str(meanseq1)+'\n\n'+'meanseq2='+str(meanseq2)+'\n\n'+'meanseq3='+str(meanseq3)+'\n\n'+'meanseq4='+str(meanseq4)+'\n\n'+'meanseq5='+str(meanseq5)+'\n\n'+'meanseq6='+str(meanseq6)+'\n\n'+'meanseqp='+str(meanseqp))
    print('\n\n'+'accept1='+str(array1)+'\n\n'+'accept2='+str(array2)+'\n\n'+'accept3='+str(array3)+'\n\n'+'accept4='+str(array4)+'\n\n'+'accept5='+str(array5)+'\n\n'+'accept6='+str(array6)+'\n\n'+'acceptp='+str(arrayp))
    print('\n\n'+'weight='+str(list(weight))+'\n\n'+'rate='+str(rate), flush=True)

    end=time.time()
    duration.append(end-start)



