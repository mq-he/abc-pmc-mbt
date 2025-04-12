# import the packages
import argparse
from exp_descriptor import ExpDescriptor
# read the exp_descriptor (class)

# # use the object as input as insert in the inference function (depend on whether the process is reducible or irreducible)

# # all outputs will be generated or printed automatically in the inference algorithm

# # set a random seed 

# def parse_args():
#     """
#     Returns an object describing the command line.
#     """
#     parser = argparse.ArgumentParser(description='ABC experiments.')
#     subparsers = parser.add_subparsers(dest='command', required=True)

#     parser_run = subparsers.add_parser('run', help='run experiments')
#     parser_run.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
#     parser_run.set_defaults(func=run_experiment)

# # read arguments from exps/demo.txt, parse using exp_descriptor

def run_experiment(args):
    """
    Runs experiments.
    """
    from exp_runner import ExpRunner

    experiment = ExpDescriptor(args.files[0])

    ExpRunner(experiment).run(trial=0)

    print('ALL DONE')


# # run arguments using exp_runner

# def main():

#     args = parse_args()
#     args.func(args)


# if __name__ == '__main__':
#     main()


# def run_experiment(args):
#     print("Running experiment with files:", args.files)

def parse_args():
    """
    Returns an object describing the command line.
    """
    parser = argparse.ArgumentParser(description='ABC experiments.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_run = subparsers.add_parser('run', help='run experiments')
    parser_run.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_run.set_defaults(func=run_experiment)

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()