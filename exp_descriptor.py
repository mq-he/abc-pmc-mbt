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

    def __init__(self, filename, test):

        self.parse(filename, test)
    
    def parse(self, filename, test):
        with open(filename, 'r') as f:
            content = f.read()
        raw = re.sub(r'\s+', '', content) # remove white space
        self.test = test
        if test == "simulation":
            m = re.match(r'experiment\{model:(reducible|irreducible),n_trees:(.*),n_leaves:(.*),true_ps:(.*),prior_bound_l:(.*),prior_bound_u:(.*),start_phase:(.*),abc:(.*)\}\Z', raw)
            self.model = m.group(1)
            self.n_trees = int(m.group(2))
            self.n_leaves = int(m.group(3))
            self.true_ps = eval(m.group(4))
            self.prior_bound_l = eval(m.group(5))
            self.prior_bound_u = eval(m.group(6))
            self.start_phase = int(m.group(7))
            m_abc_raw = m.group(8)

        elif test == "tree":
            m = re.match(r'experiment\{model:(reducible|irreducible),n_trees:(.*),n_leaves:(.*),obs_simulated:(True|False),prior_bound_l:(.*),prior_bound_u:(.*),start_phase:(.*),abc:(.*)\}\Z', raw)
            self.model = m.group(1)
            self.n_trees = int(m.group(2))
            self.n_leaves = int(m.group(3))
            obs_simulated = m.group(4).lower()
            self.obs_simulated = obs_simulated in ['true','t'] # True if the simulated observed values are used (e.g., from the experiments we did earlier)
            self.prior_bound_l = eval(m.group(5))
            self.prior_bound_u = eval(m.group(6))
            self.start_phase = int(m.group(7))
            m_abc_raw = m.group(8)
        else:
            raise ValueError("Incorrect input for parse function, the test can either be 'simulation' or 'tree'.")
        if m is None:
            raise ParseError(raw)
        
        m_abc = re.match(r'\{num_accept:(.*),n_iter:(.*),n_trees_iter:(.*),print_iter:(.*),T_nltt:(.*),tol_nltt:(.*),threshold_rate:(.*),decrease_factor:(.*)\}\Z', m_abc_raw)
        if m_abc is None:
            raise ParseError(m_abc_raw)

        
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
        if self.test == "simulation":
            return os.path.join(f'{self.model}', 'simulation_study', f'mbt_n{self.n_trees}_s{self.n_leaves}_sf{self.start_phase}_T{self.n_iter}')
        else:
            return os.path.join(f'{self.model}', 'infer_from_tree', f'sf{self.start_phase}_T{self.n_iter}')

    def info(self):
        if self.test == "simulation":
            info_string = f"Dataset has {self.n_trees} trees with {self.n_leaves} leaves, starting in phase {self.start_phase}.\nTrue values:{self.true_ps}\nPrior lower bound:{self.prior_bound_l}\nPrior upper bound:{self.prior_bound_u}"
        else: 
            info_string = f"Dataset has {self.n_trees} trees with {self.n_leaves} leaves, starting in phase {self.start_phase}.\nPrior lower bound:{self.prior_bound_l}\nPrior upper bound:{self.prior_bound_u}"
        return info_string



