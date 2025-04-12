# generate processes for BiSSE models (a special case of MBT)
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from ete3 import Tree
import random

# growth rate for MBTs (the dominant eigenvalues)
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

def NodeDelete(root):
    """
    Remove the extinct lineages
    """
    for leaf in root.get_leaves():
        if not leaf.extinct:
            for anc in leaf.get_ancestors():
                if anc.extinct:
                    anc.extinct = False
    remove_extinct(root)

def remove_extinct(root):
    for node in root.traverse():
        if node.extinct:
            node.detach()

def delete_single_child_internal(t): 
    """
    Removes internal nodes with a single child from tree
    """

    for node in t.traverse("postorder"):
        
        if((not node.is_leaf() and not node.is_root()) and len(node.get_children())<2):
            #child = node.get_children()[0]
            #child.dist = child.dist + node.dist
            node.delete()

    if len(t.get_children()) == 1: # corrected, the case where the root only has one child
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
