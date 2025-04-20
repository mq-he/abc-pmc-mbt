import argparse
from exp_descriptor import ExpDescriptor

def run_experiment(args):
    """
    Runs experiments.
    """
    from exp_runner import ExpRunner

    experiment = ExpDescriptor(args.files[0])

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

    experiment = ExpDescriptor(args.files[0])

    for trial in range(args.start, args.end + 1):

        ExpRunner(experiment).run(trial=trial)

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

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()