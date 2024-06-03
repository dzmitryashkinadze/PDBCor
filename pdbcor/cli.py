import argparse
import json
import os

from .console import console
from .correlation_extraction import CorrelationExtraction
from .io import CorrelationExtractionIOParams


class CLI:
    """Provide a commandline interface to CorrelationExtraction for use as a standalone program."""

    def __init__(self, *args):
        """
        Parse current CLI arguments and save to disk, then create list of `CorrelationExtraction` objects.

        If an optional list of arguments is passed, these will be used instead of the default (`sys.argv`).
        """
        parser = self.new_arg_parser()
        if len(args) > 0:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        console.set_quiet(quiet=self.args.quiet)

        self.residue_subsets = (
            ["backbone", "sidechain", "combined"]
            if self.args.residue_subset == "full"
            else [self.args.residue_subset]
        )

        io_params = CorrelationExtractionIOParams(
            **{
                "create_plots": self.args.create_plots,
                "create_archive": self.args.create_archive,
                "create_vis_scripts": self.args.create_vis_scripts,
                "create_cluster_plots": self.args.create_cluster_plots,
            }
        )

        if io_params.create_cluster_plots and not io_params.create_plots:
            console.warn(
                "Incompatible CLI options --no-plots and --draw-clusters passed - ignoring --draw-clusters."
            )
            io_params.create_cluster_plots = False

        self.extractors = [
            CorrelationExtraction(
                self.args.bundle,
                input_file_format=(
                    self.args.format if len(self.args.format) > 0 else None
                ),
                output_directory=self.args.output,
                io_params=io_params,
                residue_subset=residue_subset,
                features=self.args.features,
                nstates=self.args.num_states,
                therm_fluct=self.args.therm_fluct,
                therm_iter=self.args.therm_iter,
                loop_start=self.args.loop[0],
                loop_end=self.args.loop[1],
                fast=self.args.fast,
            )
            for residue_subset in self.residue_subsets
        ]

        args_dict = vars(self.args)
        args_path = os.path.join(self.extractors[0].io.output_parent_path, "args.json")
        with open(args_path, "w") as outfile:
            json.dump(args_dict, outfile)

    def calculate_correlation(self):
        """Run the correlation extraction for each enabled subset of residue."""
        for residue_subset, extractor in zip(self.residue_subsets, self.extractors):
            extractor.calculate_correlation()

    @classmethod
    def run(cls):
        """Run the CLI as a standalone program."""
        cli = cls()
        cli.calculate_correlation()

    @staticmethod
    def new_arg_parser():
        """Create a new `argparse.ArgumentParser` instance for the CLI."""
        parser = argparse.ArgumentParser(
            description="Correlation extraction from multistate protein bundles"
        )

        parser.add_argument("bundle", type=str, help="protein bundle file path")

        io_args = parser.add_argument_group("input/output settings")
        io_args.add_argument(
            "-f",
            "--format",
            dest="format",
            type=str,
            default="",
            help="input file format (default: determine from file extension)",
            choices=["PDB", "mmCIF"],
        )
        io_args.add_argument(
            "-o",
            "--output",
            dest="output",
            type=str,
            default="",
            help='filename for output directory (default: "correlations_<name of structure file>")',
        )
        io_args.add_argument(
            "--no-plots",
            dest="create_plots",
            action="store_false",
            help="do not plot any graphical output",
        )
        io_args.add_argument(
            "--create-archive",
            dest="create_archive",
            action="store_true",
            help="create .zip archive of output directory",
        )
        io_args.add_argument(
            "--no-vis",
            dest="create_vis_scripts",
            action="store_false",
            help="do not create scripts for visualisation in PyMOL/Chimera",
        )
        io_args.add_argument(
            "--draw-clusters",
            dest="create_cluster_plots",
            action="store_true",
            help="create plots of clustering results",
        )
        io_args.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="quiet mode (only output errors to console)",
        )

        corr_args = parser.add_argument_group("correlation extraction settings")
        corr_args.add_argument(
            "-n",
            "--num-states",
            dest="num_states",
            type=int,
            default=2,
            help="number of states (default: 2)",
        )
        corr_args.add_argument(
            "--residue-subset",
            dest="residue_subset",
            type=str,
            default="backbone",
            help="subset of residue atoms used for clustering, or 'full' to iterate over each (default: backbone)",
            choices=["backbone", "sidechain", "combined", "full"],
        )
        corr_args.add_argument(
            "--features",
            dest="features",
            type=str,
            default="both",
            help="features used for clustering (default: both)",
            choices=["distance", "angle", "both"],
        )
        corr_args.add_argument(
            "--fast",
            dest="fast",
            action="store_true",
            help="run in fast mode (see documentation for details)",
        )
        corr_args.add_argument(
            "-i",
            "--therm-iter",
            dest="therm_iter",
            type=int,
            default=5,
            help="number of thermal iterations to average for distance-based correlations (default: 5)",
        )
        corr_args.add_argument(
            "--therm-fluct",
            dest="therm_fluct",
            type=float,
            default=0.5,
            help="thermal fluctuation of distances in the protein bundle "
            "-> scaling factor for added random noise "
            "(default: 0.5)",
        )
        corr_args.add_argument(
            "--loop",
            nargs=2,
            type=int,
            default=[-1, -1],
            help="residue numbers of start & end of loop to exclude from analysis (default: none)",
        )

        return parser
