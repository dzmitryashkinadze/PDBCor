import argparse
import json
import os

from . import CorrelationExtraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation extraction from multistate protein bundles')
    parser.add_argument('bundle', type=str,
                        help='protein bundle file path')
    parser.add_argument('--nstates', type=int,
                        default=2,
                        help='number of states')
    parser.add_argument('--graphics', type=bool,
                        default=True,
                        help='generate graphical output')
    parser.add_argument('--mode', type=str,
                        default='backbone',
                        help='correlation mode')
    parser.add_argument('--therm_fluct', type=float,
                        default=0.5,
                        help='Thermal fluctuation of distances in the protein bundle')
    parser.add_argument('--therm_iter', type=int,
                        default=5,
                        help='Number of thermal simulations')
    parser.add_argument('--loop_start', type=int,
                        default=-1,
                        help='Start of the loop')
    parser.add_argument('--loop_end', type=int,
                        default=-1,
                        help='End of the loop')
    args = parser.parse_args()
    # create correlations folder
    corPath = os.path.join(os.path.dirname(args.bundle), 'correlations')
    try:
        os.mkdir(corPath)
    except:
        pass
    # write parameters of the correlation extraction
    args_dict = vars(args)
    args_path = os.path.join(corPath, 'args.json')
    with open(args_path, 'w') as outfile:
        json.dump(args_dict, outfile)
    # correlation mode
    if args.mode == 'backbone':
        modes = ['backbone']
    elif args.mode == 'sidechain':
        modes = ['sidechain']
    elif args.mode == 'combined':
        modes = ['combined']
    elif args.mode == 'full':
        modes = ['backbone', 'sidechain', 'combined']
    else:
        parser.error('Mode has to be either backbone, sidechain, combined or full')
    for mode in modes:
        print('###############################################################################')
        print('############################   {} CORRELATIONS   ########################'.format(mode.upper()))
        print('###############################################################################')
        print()
        a = CorrelationExtraction(args.bundle,
                                  mode=mode,
                                  nstates=args.nstates,
                                  therm_fluct=args.therm_fluct,
                                  therm_iter=args.therm_iter,
                                  loop_start=args.loop_start,
                                  loop_end=args.loop_end)
        a.calc_cor(graphics=args.graphics)