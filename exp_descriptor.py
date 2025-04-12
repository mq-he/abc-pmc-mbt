# parse the string in the txt file and return a list of arguments for algorithm
import re
import os

class ParseError(Exception):
    """
    Exception to be thrown when there is a parsing error.
    """

    def __init__(self, info):
        self.info = info

    def __str__(self):
        return self.info

class ExpDescriptor:
    """
    Set up the model and the inference process
    """

    def __init__(self, filename):

        self.parse(filename)
    
    def parse(self, filename):
        with open(filename, 'r') as f:
            content = f.read()
        raw = re.sub(r'\s+', '', content) # remove white space
        m = re.match(r'experiment\{model:(reducible|irreducible),n_trees:(.*),n_leaves:(.*),true_ps:(.*),prior_bound_l:(.*),prior_bound_u:(.*),start_phase:(.*),abc:(.*)\}\Z', raw)
        if m is None:
            raise ParseError(raw)
        m_abc = re.match(r'\{num_accept:(.*),n_iter:(.*),n_trees_iter:(.*),print_iter:(.*),T_nltt:(.*),tol_nltt:(.*),threshold_rate:(.*),decrease_factor:(.*)\}\Z', m.group(8))
        if m_abc is None:
            raise ParseError(m.group(7))

        self.model = m.group(1)
        self.n_trees = int(m.group(2))
        self.n_leaves = int(m.group(3))
        self.true_ps = eval(m.group(4))
        self.prior_bound_l = eval(m.group(5))
        self.prior_bound_u = eval(m.group(6))
        self.start_phase = int(m.group(7))
        self.num_accept = eval(m_abc.group(1)) # a list for accepted
        self.n_iter = int(m_abc.group(2))
        self.n_trees_iter = eval(m_abc.group(3))
        self.print_iter = True if m_abc.group(4).lower()=='true' else False
        self.T_nltt = int(m_abc.group(5))
        self.tol_nltt = float(m_abc.group(6))
        self.threshold_rate = float(m_abc.group(7))
        self.decrease_factor = float(m_abc.group(8))

        # sanity checks
        assert len(self.num_accept) == self.n_iter
        if self.n_trees > 1:
            assert len(self.n_trees_iter) == self.n_iter

    def get_dir(self):
        # path to save the results
        return os.path.join(f'{self.model}', 'simulation_study', f'mbt_n{self.n_trees}_s{self.n_leaves}_T{self.n_iter}')

    def info(self):
        info_string = f"Dataset has {self.n_trees} trees with {self.n_leaves} leaves, starting in phase {self.start_phase}.\nTrue values:{self.true_ps}\nPrior lower bound:{self.prior_bound_l}\nPrior upper bound:{self.prior_bound_u}"
        return info_string

# def parse(filename):
#     #filename = 'demo_exp_multitrees.txt'
#     with open(filename, 'r') as f:
#         content = f.read()
#     raw = re.sub(r'\s+', '', content) # remove white space
#     m = re.match(r'experiment\{model:(reducible|irreducible),n_trees:(.*),n_leaves:(.*),true_ps:(.*),prior_bound_l:(.*),prior_bound_u:(.*),abc:(.*)\}\Z', raw)
#     m_abc = re.match(r'\{num_accept:(.*),n_iter:(.*),n_trees_iter:(.*),print_iter:(.*),T_nltt:(.*),tol_nltt:(.*),threshold_rate:(.*),decrease_factor:(.*)\}\Z', m.group(7))

#     # Create output container
#     output = SimpleNamespace()

#     # make sure all variables has the correct type 
#     output.n_trees = int(m.group(2))
#     output.n_leaves = int(m.group(3))
#     output.true_ps = eval(m.group(4))
#     output.prior_bound_l = eval(m.group(5))
#     output.prior_bound_u = eval(m.group(6))
#     output.num_accept = eval(m_abc.group(1)) # a list for accepted
#     output.n_iter = int(m_abc.group(2))
#     output.n_trees_iter = eval(m_abc.group(3))
#     output.print_iter = True if m_abc.group(4).lower()=='true' else False
#     output.T_nltt = int(m_abc.group(5))
#     output.tol_nltt = float(m_abc.group(6))
#     output.threshold_rate = float(m_abc.group(7))
#     output.decrease_factor = float(m_abc.group(8))

#     # sanity checks
#     assert len(output.num_accept) == output.n_iter
#     assert len(output.n_trees_iter) == output.n_iter

#     return output

