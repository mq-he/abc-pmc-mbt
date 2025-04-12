import math
import numpy as np
from statistics import mean
import simulator.mbt_bisse_simulator as mbt_bisse_simulator

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

def nabsdiff(t, start=2):
    """Compute two arrays
    Inputs:
    t: tree
    start: the number of lineages when we start counting (default to 2, we assume tree starts with two lineages for nLTT computation)
    Outputs:
    init_nx: the array of the time stamp for the speciation events in order
    init_ny: the array of the number of lineages 
    """
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

###################### for distributions with weighted samples #########################################

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


####################### only for reducible process ######################################################
def mdistp1(t):
    seq=[]
    for node in t.traverse():
        if not node.is_root() and numphase1(node)!=0:
            seq.append(node.dist)
    return mean(seq)

def generate_mbt_ind_reducible_update(tree_ind):
    '''
    Compute the associated statistics for a single tree. 
    Input:
        samples: a tree object
    Output:
        a 1d array, which records the associated summary statistics for one sample, with shape=(N_SUMSTA,)
        For reducible process, two statistics are designed for trees that contain phase-1 leaves, 
        if the simulated tree does not have phase-1 leaves, return 0 for these two statistics.
    '''
    balsim, bal1sim, bal2sim=balance(tree_ind)
    tspansim=tree_height(tree_ind)
    distancesim=PD(tree_ind)
    tspann0, distp1 = 0, 0
    if numphase1(tree_ind)>0:
        tspann0 = tree_height(tree_ind) # the tree height for trees with phase-1 leaves
        distp1 = mdistp1(tree_ind)

    return np.array([balsim, bal1sim, bal2sim, tspansim, distancesim, tspann0, distp1])

def generate_mbt_data_reducible_update(sample, num_tree=100, treesize = 50, startphase = 1, r=20):
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
    output = np.zeros(7) # seven statistics for reducible process (exclude nLTT) 
    tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
    if flag_sim==0:
        return output
    else:
        output = generate_mbt_ind_reducible_update(tree_sim) # stats calculation
        count_tree_p1 = numphase1(tree_sim)>0 # count the number of trees with phase-1 leaves
        for j in range(1, num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
            count_tree_p1 += numphase1(tree_sim)>0
            output += generate_mbt_ind_reducible_update(tree_sim) # stats calculation
        output[:5] = output[:5]/num_tree
        if output[5]>0:
            assert count_tree_p1>0, f'count_tree_p1={count_tree_p1}'
            output[5:] = output[5:]/count_tree_p1
        return output

def generate_mbt_data_reducible_update_nLTT(sample, num_tree=100, treesize = 50, startphase = 1, r=20, start_nLTT=2):
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
    output = np.zeros(7) 
    sim_array, sim_resp = np.zeros(2) 
    tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
    if flag_sim==0:
        return output, sim_array, sim_resp
    else:
        output = generate_mbt_ind_reducible_update(tree_sim) # stats calculation
        sim_array, sim_resp = nabsdiff(tree_sim, start=start_nLTT)  # nLTT computation
        count_tree_p1 = numphase1(tree_sim)>0 # count the number of trees with phase-1 leaves
        for j in range(1, num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
            count_tree_p1 += numphase1(tree_sim)>0
            output += generate_mbt_ind_reducible_update(tree_sim) # stats calculation
            init_array, init_resp = nabsdiff(tree_sim, start=start_nLTT) # nLTT computation
            sim_array, sim_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
        output[:5] = output[:5]/num_tree
        if output[5]>0:
            assert count_tree_p1>0, f'count_tree_p1={count_tree_p1}'
            output[5:] = output[5:]/count_tree_p1
        return output, sim_array, sim_resp

############################################################################################################

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
    tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
    if flag_sim==0:
        return output
    else:
        output = generate_mbt_ind_update(tree_sim) # stats calculation
        for j in range(1, num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
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
    tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
    if flag_sim==0:
        return output, sim_array, sim_resp
    else:
        output = generate_mbt_ind_update(tree_sim) # stats calculation
        sim_array, sim_resp = nabsdiff(tree_sim, start=start_nLTT)  # nLTT computation
        for j in range(1, num_tree):
            flag_sim = 0
            while flag_sim==0:
                tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=treesize, start=startphase, r=r)
            output += generate_mbt_ind_update(tree_sim) # stats calculation
            init_array, init_resp = nabsdiff(tree_sim, start=start_nLTT) # nLTT computation
            sim_array, sim_resp = sumdist_array(init_array, sim_array, init_resp, sim_resp)
        sim_resp = sim_resp/num_tree
        output = output/num_tree
        return output, sim_array, sim_resp

################################# functions for a single observed tree ######################################
# for a large single tree, we need to break it into smaller subtrees for the first few iteratiions
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


def generate_mbt_subtrees_update(sample, subtrees_size, startphase, r=20):
    b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)
    # simulation process, keep generate sample until we have one surviving tree
    for i,size in enumerate(subtrees_size):
        if i==0:
            tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r)
            if flag_sim == 0:
                output_samp = np.zeros(8)
                break
            else:
                output_samp = generate_mbt_ind_update(tree_sim)
        else:
            flag_sim = 0
            while flag_sim == 0:
                tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r)
            output = generate_mbt_ind_update(tree_sim) # stats calculation
            output_samp = np.vstack((output_samp, output))
            
    if flag_sim == 1 and len(subtrees_size)>1: # the process survives
        output_samp = np.mean(output_samp,axis=0)
    
    return output_samp

def generate_mbt_single_tree_update_nLTT(sample, size, obs_array, obs_resp, startphase, r=20, start_nLTT=20, max_time_bound=None):
    """
    Generate the summary statistics including the nLTT statistic
    """
    b1sim, b2sim, d1sim, d2sim, q12sim, q21sim = list(sample)
    tree_sim, flag_sim = mbt_bisse_simulator.birth_death_tree2([b1sim, b2sim], [d1sim, d2sim], [q12sim, q21sim], nsize=size, start=startphase, r=r, max_time=max_time_bound)
    if flag_sim == 0:
        output_samp = np.zeros(8+1) # add one for the nLTT stat
    elif len(tree_sim.get_leaves())!=size:
        output_samp = np.zeros(8+1)
    else:
        output_samp_stat = generate_mbt_ind_update(tree_sim) # stats calculation
        sim_array, sim_resp = nabsdiff(tree_sim, start=start_nLTT)  # nLTT computation
        nLTTstat = absdist_array(obs_array, sim_array, obs_resp, sim_resp)
        output_samp = np.append(output_samp_stat, nLTTstat) # append a value to the end of the array and create a new one
    
    return output_samp