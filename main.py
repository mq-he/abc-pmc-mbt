import argparse
from exp_descriptor import ExpDescriptor

def run_experiment(args):
    """
    Runs experiments.
    """
    from exp_runner import ExpRunner

    experiment = ExpDescriptor(args.files[0], "simulation")

    ExpRunner(experiment).run(trial=0)

    print('ALL DONE')


def run_trials(args):
    """
    Runs experiments for multiple trials with the same ground truth.
    """

    from exp_runner import ExpRunner

    if args.start < 1:
        raise ValueError('trial # must be a positive integer')

    if args.end < args.start:
        raise ValueError('end trial can''t be less than start trial')

    experiment = ExpDescriptor(args.files[0], "simulation")

    for trial in range(args.start, args.end + 1):

        ExpRunner(experiment).run(trial=trial)

    print('ALL DONE')

def run_tree(args):
    """
    Runs experiments with the observed tree(s) and the associated phases.
    """
    from exp_runner import ExpRunner

    experiment = ExpDescriptor(args.files[0], "tree")

    ExpRunner(experiment).run(trial=0, sim_obs=False, tree_file=args.tree_file, phase_file=args.phase_file)

    print('ALL DONE')

def run_tree_trials(args):
    """
    Runs experiments with the observed tree(s) and the associated phases for multiple trials.
    """
    from exp_runner import ExpRunner

    experiment = ExpDescriptor(args.files[0], "tree")

    if args.start < 1:
        raise ValueError('trial # must be a positive integer')

    if args.end < args.start:
        raise ValueError('end trial can''t be less than start trial')

    for trial in range(args.start, args.end + 1):

        ExpRunner(experiment).run(trial=trial, sim_obs=False, tree_file=args.tree_file, phase_file=args.phase_file)

    print('ALL DONE')

def parse_args():
    """
    Returns an object describing the command line.
    """
    parser = argparse.ArgumentParser(description='ABC experiments.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_run = subparsers.add_parser('run', help='run experiments')
    parser_run.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_run.set_defaults(func=run_experiment)

    parser_trials = subparsers.add_parser('trials', help='run multiple experiment trials')
    parser_trials.add_argument('start', type=int, help='# of first trial')
    parser_trials.add_argument('end', type=int, help='# of last trial')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.set_defaults(func=run_trials)

    parser_trials = subparsers.add_parser('infer', help='infer the parameter values using the observed tree(s)')
    parser_trials.add_argument('tree_file', type=str, help='file for the observed tree structure (newick format)')
    parser_trials.add_argument('phase_file', type=str, help='file for the tip phases (leave name match with the ones in tree_file)')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.set_defaults(func=run_tree)

    parser_trials = subparsers.add_parser('infer_trials', help='infer the parameter values using the observed tree(s) -- repeated trials')
    parser_trials.add_argument('tree_file', type=str, help='file for the observed tree structure (newick format)')
    parser_trials.add_argument('phase_file', type=str, help='file for the tip phases (leave name match with the ones in tree_file)')
    parser_trials.add_argument('start', type=int, help='# of first trial')
    parser_trials.add_argument('end', type=int, help='# of last trial')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.set_defaults(func=run_tree_trials)

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()