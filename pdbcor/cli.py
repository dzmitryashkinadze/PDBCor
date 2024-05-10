import argparse
import json
import os

from .correlation_extraction import CorrelationExtraction


def cli():
    """Provide a commandline interface to CorrelationExtraction for use as a standalone program."""
    parser = argparse.ArgumentParser(
        description="Correlation extraction from multistate protein bundles"
    )
    parser.add_argument("bundle", type=str, help="protein bundle file path")
    parser.add_argument(
        "--format",
        type=str,
        default="",
        help="Input file format (or leave blank to determine from file extension)",
        choices=["PDB", "mmCIF"],
    )
    parser.add_argument("--nstates", type=int, default=2, help="number of states")
    parser.add_argument(
        "--graphics", type=bool, default=True, help="generate graphical output"
    )
    parser.add_argument("--mode", type=str, default="backbone", help="correlation mode")
    parser.add_argument(
        "--therm_fluct",
        type=float,
        default=0.5,
        help="Thermal fluctuation of distances in the protein bundle",
    )
    parser.add_argument(
        "--therm_iter", type=int, default=5, help="Number of thermal simulations"
    )
    parser.add_argument("--loop_start", type=int, default=-1, help="Start of the loop")
    parser.add_argument("--loop_end", type=int, default=-1, help="End of the loop")
    args = parser.parse_args()

    # create correlations folder
    cor_path = os.path.join(os.path.dirname(args.bundle), "correlations")
    os.makedirs(cor_path, exist_ok=True)

    # write parameters of the correlation extraction
    args_dict = vars(args)
    args_path = os.path.join(cor_path, "args.json")
    with open(args_path, "w") as outfile:
        json.dump(args_dict, outfile)

    # correlation mode
    if args.mode == "backbone":
        modes = ["backbone"]
    elif args.mode == "sidechain":
        modes = ["sidechain"]
    elif args.mode == "combined":
        modes = ["combined"]
    elif args.mode == "full":
        modes = ["backbone", "sidechain", "combined"]
    else:
        modes = []
        parser.error("Mode has to be either backbone, sidechain, combined or full")

    for mode in modes:
        print(
            "###############################################################################\n"
            f"############################   {mode.upper()} CORRELATIONS   ########################\n"
            "###############################################################################"
        )
        print()
        a = CorrelationExtraction(
            args.bundle,
            input_file_format=(args.format if len(args.format) > 0 else None),
            mode=mode,
            nstates=args.nstates,
            therm_fluct=args.therm_fluct,
            therm_iter=args.therm_iter,
            loop_start=args.loop_start,
            loop_end=args.loop_end,
        )
        a.calculate_correlation(graphics=args.graphics)
