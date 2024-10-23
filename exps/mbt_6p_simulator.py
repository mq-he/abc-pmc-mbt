# the corrected calculation for tree height, the standardised transition statistics (including root), the standardised balanced index

import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, variance
from math import sqrt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from ete3 import Tree
import random

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

# updated
def dist2tip(tree, node):
    """Compute the distance between the internal node
    and its leaves
    Arguments:
        - ``tree`` : a birth-death tree (binary, no extinct branch)
        - ``node`` : a internal node of this tree
    """
    if node.is_leaf():
        raise ValueError("It is a leaf node")
    item = node.children[0]
    dist = item.dist
    while not item.is_leaf():
        item = item.get_children()[0]
        dist += item.dist
    return dist

# updated
def ratiodist(tree, shift=0.02):
    """Compute transition statistics that is the sum of
    -log(proportion)/(dist2tip(node)+shift) for all internal node"""
    stat,stat2 = 0,0
    for node in tree.traverse():
        if (not node.is_leaf()):
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

# updated
def trans(t):
    size=len(t.get_leaves())
    tran1,tran2=ratiodist(t)
    return tran1/size,tran2/size

def ratio(t):
    ''' The proportion of the phase 1 tips'''
    n1 = numphase1(t)
    total = len(t.get_leaves())
    return n1/total

def ignoreRoot(array, start=2):
    n = len(array)
    output = [0]
    for i in range(start-1,n):
        difference = array[i]-array[start-2]
        output.append(difference)
    return output

# modified growth rate
def dom(birth, death, transition):
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

# mod2
def NodeDelete(root):
    for leaf in root.get_leaves():
        if not leaf.extinct:
            for anc in leaf.get_ancestors():
                if anc.extinct:
                    anc.extinct = False
    remove_extinct(root)

#def remove_extinct(root):
#    if root.extinct:
#        root.detach()
#    else:
#        if not root.is_leaf():
#            for node in root.get_children():
#                remove_extinct(node)

# mod remove
def remove_extinct(root):
    for node in root.traverse():
        if node.extinct:
            node.detach()


def delete_single_child_internal_false(t): 
    """Utility function that removes internal nodes
    with a single child from tree"""

    for node in t.traverse("postorder"):
        
        if((not node.is_leaf() and not node.is_root()) and len(node.get_children())<2):
            #child = node.get_children()[0]
            #child.dist = child.dist + node.dist
            node.delete()

    if len(t.get_children()) == 1:
        t.height = t.children[0].height
        #t.dist = t.children[0].dist
        t.children = t.children[0].children

def delete_single_child_internal(t): 
    """Utility function that removes internal nodes
    with a single child from tree"""

    for node in t.traverse("postorder"):
        
        if((not node.is_leaf() and not node.is_root()) and len(node.get_children())<2):
            #child = node.get_children()[0]
            #child.dist = child.dist + node.dist
            node.delete()

    if len(t.get_children()) == 1: # corrected
        t.height = t.children[0].height
        #t.dist = t.children[0].dist
        t.children[0].delete()

def Add_dist(t):
    """Add dist using attribute height
    """
    for node in t.traverse():
        if node.is_root():
            node.dist=0 # corrected
        else:
            node.dist=node.height-node.get_ancestors()[0].height


            

def birth_death_tree2(birth, death, transition, nsize=None, start=1, max_time=None, remlosses=True, r=20):
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

# only for reducible process
def mdistp1(t):
    seq=[]
    for node in t.traverse():
        if not node.is_root() and numphase1(node)!=0:
            seq.append(node.dist)
    return mean(seq)


def generate_mbt_ind_update(tree_ind):
    '''
    Each simulated dataset has one tre. 
    Input:
        samples: a tree object
    Output:
        a 1d array, which records the associated summary statistics for one sample, with shape=(N_SUMSTA,)
    '''
    balsim, bal1sim, bal2sim=balance(tree_ind)
    tspansim=tree_height(tree_ind)
    distancesim=PD(tree_ind)

    propsim = ratio(tree_ind)
    statsim,stat2sim=trans(tree_ind)

    return np.array([balsim, bal1sim, bal2sim, tspansim, distancesim, propsim, statsim, stat2sim])

 
def generate_mbt_data_update(sample, num_tree=100, treesize = 50, startphase = 1, r=20):
    '''
    Each simulated dataset has `num_tree` tree each with `treesize` leaves, starting in phase `startphase`. 
    The process will run `r` times until it finds one surviving tree, otherwise it will terminate, giving all zeros.
    Input:
        sample: a 1d array, which records the parameter values for one sample, with shape=(N_PAR,)
    Output:
        an arrays, which records the associated summary statistics for one sample, with shape=(N_SUMSTA,)
    
    '''
    
    # unpack the parameter values
    b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)

    # simulation process, keep generate sample until we have one surviving tree
    output = np.zeros(8) 
    tree_sim, flag_sim = birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
    if flag_sim==0:
        return output
    else:
        output = generate_mbt_ind_update(tree_sim) # stats calculation
        for j in range(1, num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
            output += generate_mbt_ind_update(tree_sim) # stats calculation
        output = output/num_tree
        return output

def generate_mbt_data_update_nLTT(sample, num_tree=100, treesize = 50, startphase = 1, r=20, start_nLTT=2):
    '''
    Each simulated dataset has `num_tree` tree each with `treesize` leaves, starting in phase `startphase`. 
    The process will run `r` times until it finds one surviving tree, otherwise it will terminate, giving all zeros.
    Input:
        sample: a 1d array, which records the parameter values for one sample, with shape=(N_PAR,)
    Output:
        three arrays, one of which records the associated summary statistics for one sample, with shape=(N_SUMSTA,),
        the other two record the nLTT curve, zero if the tree go extinct
    
    '''
    
    # unpack the parameter values
    b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)

    # simulation process, keep generate sample until we have one surviving tree
    output = np.zeros(8) 
    sim_array, sim_resp = np.zeros(2) 
    tree_sim, flag_sim = birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
    if flag_sim==0:
        return output, sim_array, sim_resp
    else:
        output = generate_mbt_ind_update(tree_sim) # stats calculation
        sim_array, sim_resp = nabsdiff(tree_sim, start=start_nLTT)  # nLTT computation
        for j in range(1, num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
            output += generate_mbt_ind_update(tree_sim) # stats calculation
            init_array, init_resp = nabsdiff(tree_sim, start=start_nLTT) # nLTT computation
            sim_array, sim_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
        sim_resp = sim_resp/num_tree
        output = output/num_tree
        return output, sim_array, sim_resp

def generate_mbt_n100_s50_array_survival_update_full(samples):
    '''
    Each simulated dataset has 100 tree each with 50 leaves. 
    The process will run until it finds one surviving tree.
    Input:
        samples: an array, where each row record the parameter values for one sample, with shape=(N_SAMP, N_PAR)
    Output:
        an arrays, which records the summary statistics for each tree in the dataset, with shape=(N_SAMP, num_tree, N_SUMSTA+2*treesize), where num_tree=100, treesize=50
    
    '''

    if np.ndim(samples)==1:
        samples = samples[np.newaxis, :]
    elif np.ndim(samples)!=2:
        raise ValueError('Incorrect number of dimensions')

    treesize = 50
    num_tree = 100
    startphase = 1
    for i in range(np.shape(samples)[0]):
        sample = samples[i]
        # unpack the parameter values
        b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)

        # simulation process, keep generate samples until we have one surviving tree
        output = np.zeros(8) 
        for j in range(num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=0)
            output = generate_mbt_ind_update(tree_sim) # stats calculation
            sim_array, sim_resp = nabsdiff(tree_sim)  # nLTT computation
            output = np.concatenate((output, sim_array, sim_resp))
            if j==0:
                output_samp = output
            else:
                output_samp = np.vstack((output_samp, output))
        
        output_samp = output_samp[np.newaxis,:,:] # expand the axis for later stacking

        if i==0:
            output_all = output_samp
        else:
            output_all = np.concatenate((output_all, output_samp), axis=0)

        
    return output_all

def generate_mbt_n100_s50_array_survival_update_full_withtrees(samples):
    '''
    Each simulated dataset has 100 tree each with 50 leaves. 
    The process will run until it finds one surviving tree.
    Input:
        samples: an array, where each row record the parameter values for one sample, with shape=(N_SAMP, N_PAR)
    Output:
        an arrays, which records the summary statistics for each tree in the dataset, with shape=(N_SAMP, num_tree, N_SUMSTA+2*treesize), where num_tree=100, treesize=50
    
    '''

    if np.ndim(samples)==1:
        samples = samples[np.newaxis, :]
    elif np.ndim(samples)!=2:
        raise ValueError('Incorrect number of dimensions')

    tree_sim_list = [[] for _ in range(np.shape(samples)[0])] # added, for recording the simulated trees
    treesize = 50
    num_tree = 100
    startphase = 1
    for i in range(np.shape(samples)[0]):
        sample = samples[i]
        # unpack the parameter values
        b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)

        # simulation process, keep generate samples until we have one surviving tree
        output = np.zeros(8) 
        for j in range(num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=0)
            output = generate_mbt_ind_update(tree_sim) # stats calculation
            sim_array, sim_resp = nabsdiff(tree_sim)  # nLTT computation
            output = np.concatenate((output, sim_array, sim_resp))
            tree_sim_list[i].append(tree_sim) # added, for recording the simulated trees
            if j==0:
                output_samp = output
            else:
                output_samp = np.vstack((output_samp, output))
        
        output_samp = output_samp[np.newaxis,:,:] # expand the axis for later stacking

        if i==0:
            output_all = output_samp
        else:
            output_all = np.concatenate((output_all, output_samp), axis=0)

        
    return output_all, tree_sim_list
