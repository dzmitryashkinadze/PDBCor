from __future__ import annotations

import json
import os
import re
import shutil
from abc import ABC
from copy import copy
from typing import Any, TYPE_CHECKING, Optional, Union, SupportsFloat, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter

from pydantic import BaseModel
from sklearn.decomposition import PCA

from .console import console

if TYPE_CHECKING:
    from .correlation_extraction import CorrelationExtraction
    from .clustering import (
        AngleClusterCalculator,
        ClusterCalculator,
        DistanceClusterCalculator,
    )


class CorrelationExtractionIOParams(BaseModel):
    create_plots: bool = True
    create_archive: bool = False
    create_vis_scripts: bool = True
    create_cluster_plots: bool = False


class CorrelationExtractionIO:
    def __init__(
        self,
        correlation_extraction: "CorrelationExtraction",
        output_directory: str | os.PathLike | None = None,
        params: CorrelationExtractionIOParams | None = None,
    ):
        self.correlation_extraction = correlation_extraction
        if output_directory is None or output_directory == "":
            output_directory = f"correlations_{os.path.splitext(self.correlation_extraction.pdb_file_name)[0]}"
        self.output_parent_path: str | os.PathLike = os.path.join(
            os.path.dirname(self.correlation_extraction.pdb_file_path), output_directory
        )
        os.makedirs(self.output_parent_path, exist_ok=True)
        self.params = CorrelationExtractionIOParams() if params is None else params

        # write readme file
        shutil.copyfile(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.txt"),
            os.path.join(self.output_parent_path, "README.txt"),
        )

    def chain_io(self, chain_id):
        return CorrelationExtractionChainIO(
            general_io=self,
            chain_id=chain_id,
        )

    def finalize_analysis(self):
        if self.params.create_archive:
            self._archive_output()

    def _archive_output(self):
        console.print("Archiving results...")
        shutil.make_archive(
            str(self.output_parent_path),
            "zip",
            str(self.output_parent_path) + "/",
        )


class CorrelationExtractionChainIO:
    def __init__(
        self,
        general_io: CorrelationExtractionIO,
        chain_id: str | None,
    ):
        self.correlation_extraction = general_io.correlation_extraction
        self.chain_id = chain_id
        self.save_path = (
            self.output_path(general_io.output_parent_path, chain_id)
            if self.chain_id is not None
            else general_io.output_parent_path
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.params = general_io.params

    @classmethod
    def output_path(cls, parent_path, chain_id):
        return os.path.join(
            parent_path,
            f"chain_{chain_id}",
        )

    def generate_single_feature_output(
        self,
        corr_list: np.ndarray,
        clusters: np.ndarray,
        corr_matrix: np.ndarray,
        best_clust: np.ndarray,
        feature_name: str | tuple[str, str],
    ) -> None:
        """Generate output for a single set of features (distances or angles)."""
        # Ensure tuple (short_name, long_name)
        if isinstance(feature_name, str):
            feature_name = (feature_name, feature_name)

        console.h3(f"{feature_name[1].capitalize()}-based correlation results")

        console.print("Saving correlation values to disk...")
        self._create_data_file(corr_list, feature_name)

        if self.params.create_plots:
            console.print("Plotting results...")
            self._create_plots(corr_list, corr_matrix, feature_name)

        if self.params.create_vis_scripts:
            console.print("Creating visualisation scripts...")
            self._create_vis_scripts(best_clust, feature_name)

    def _create_data_file(self, corr_list, feature_name):
        df = pd.DataFrame(
            {
                "residue_id_1": corr_list[:, 0].astype("int"),
                "residue_id_2": corr_list[:, 1].astype("int"),
                "correlation": corr_list[:, 2],
            }
        )
        df.to_feather(
            os.path.join(
                self.save_path,
                f"{feature_name[0]}_corr_{self.correlation_extraction.residue_subset}.feather",
            ),
        )

    def _create_plots(self, corr_list, corr_matrix, feature_name):
        self._plot_heatmaps(
            corr_matrix,
            os.path.join(
                self.save_path,
                f"heatmap_{feature_name[0]}_{self.correlation_extraction.residue_subset}.png",
            ),
        )
        self._plot_hist(
            corr_list,
            os.path.join(
                self.save_path,
                f"hist_{feature_name[0]}_{self.correlation_extraction.residue_subset}.png",
            ),
        )
        self._corr_per_resid_bar_plot(
            corr_matrix,
            os.path.join(
                self.save_path,
                f"seq_{feature_name[0]}_{self.correlation_extraction.residue_subset}.png",
            ),
        )

    def _create_vis_scripts(self, best_clust, feature_name):
        self._write_chimera_script(
            best_clust,
            os.path.join(
                self.save_path,
                f"bundle_vis_chimera_{feature_name[0]}_{self.correlation_extraction.residue_subset}.py",
            ),
        )
        self._write_pymol_script(
            best_clust,
            os.path.join(
                self.save_path,
                f"bundle_vis_pymol_{feature_name[0]}_{self.correlation_extraction.residue_subset}.pml",
            ),
        )

    def _write_chimera_script(
        self, best_clust: np.ndarray, file_path: str | os.PathLike
    ) -> None:
        """Construct a Chimera script to view the calculated clusters in separate colours."""
        state_color = ["#00FFFF", "#00008b", "#FF00FF", "#FFFF00", "#000000"]
        with open(file_path, "w") as f:
            f.write("from chimera import runCommand as rc\n")
            f.write(
                'rc("open ../../{}")\n'.format(
                    self.correlation_extraction.pdb_file_name
                )
            )
            f.write('rc("background solid white")\n')
            f.write('rc("~ribbon")\n')
            f.write('rc("show :.a@ca")\n')
            # bundle coloring
            for i in range(len(self.correlation_extraction.structure)):
                f.write(
                    'rc("color {} #0.{}")\n'.format(
                        state_color[int(best_clust[i])], int(i + 1)
                    )
                )

    def _write_pymol_script(
        self, best_clust: np.ndarray, file_path: str | os.PathLike
    ) -> None:
        """Construct a PyMOL script to view the calculated clusters in separate colours."""
        state_color = ["0x00FFFF", "0x00008b", "0xFF00FF", "0xFFFF00", "0x000000"]
        with open(file_path, "w") as f:
            f.write(f"load ../../{self.correlation_extraction.pdb_file_name}\n")
            f.write("bg_color white\n")
            f.write("hide\n")
            f.write("show ribbon\n")
            f.write("set all_states, true\n")
            f.writelines(
                [
                    f"set ribbon_color, {state_color[int(best_clust[i])]}, all, {i+1}\n"
                    for i in range(len(self.correlation_extraction.structure))
                ]
            )

    def _plot_heatmaps(self, hm: np.ndarray, path: str | os.PathLike) -> None:
        """Plot the correlation matrix as a heatmap."""
        # change color map to display nans as gray
        cmap = copy(get_cmap("viridis"))
        cmap.set_bad("gray")
        # plot distance heatmap
        fig, ax = plt.subplots()
        pos = ax.imshow(
            hm,
            origin="lower",
            extent=[
                self.correlation_extraction.aaS,
                self.correlation_extraction.aaF,
                self.correlation_extraction.aaS,
                self.correlation_extraction.aaF,
            ],
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        fig.colorbar(pos, ax=ax)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        plt.xlabel("Residue ID")
        plt.ylabel("Residue ID")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _plot_hist(ami: np.ndarray, path: str | os.PathLike) -> None:
        """Plot a histogram of correlation parameters (from `ndarray` of pairwise AMI values)."""
        # plot distance correlation histogram
        start = np.floor(min(ami[:, 2]) * 50) / 50
        nbreaks = int((1 - start) / 0.02) + 1
        fig, ax = plt.subplots()
        ax.hist(ami[:, 2], bins=np.linspace(start, 1, nbreaks), density=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Density")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def _corr_per_resid_bar_subplot(
        self, cor_seq: np.ndarray, ax: plt.Axes, ind: int
    ) -> None:
        """
        Create a barplot of average correlation parameters per residue,
        to be used as a single row in `self._corr_per_resid_bar_plot()`.
        """
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        range_loc = range(
            self.correlation_extraction.aaS + 50 * ind,
            min(
                self.correlation_extraction.aaF + 1,
                self.correlation_extraction.aaS + 50 * (ind + 1),
            ),
        )
        cor_seq_loc = cor_seq[(50 * ind) : min(len(cor_seq), 50 * (ind + 1))]
        if min(cor_seq) < 0:
            cor_range = max(cor_seq) - min(cor_seq)
            ax.set_ylim(min(cor_seq) - cor_range / 8, max(cor_seq) + cor_range / 8)
        else:
            ax.set_ylim(0, max(cor_seq))
            cor_range = max(cor_seq)
        ax.bar(range_loc, cor_seq_loc, width=0.8)
        for ind2, cor in enumerate(cor_seq_loc):
            if cor >= 0:
                ax.text(
                    range_loc[ind2] - 0.25,
                    cor + cor_range / 50,
                    "{:.3f}".format(cor),
                    fontsize=10,
                    rotation=90,
                )
            else:
                ax.text(
                    range_loc[ind2] - 0.25,
                    cor - cor_range / 11,
                    "{:.3f}".format(cor),
                    fontsize=10,
                    rotation=90,
                )
        if len(cor_seq) < 50:
            ax.set_xlim(
                self.correlation_extraction.aaS + 50 * ind - 1,
                self.correlation_extraction.aaF + 1,
            )
        else:
            ax.set_xlim(
                self.correlation_extraction.aaS + 50 * ind - 1,
                self.correlation_extraction.aaS + 50 * (ind + 1),
            )
        ax.set_xticks(range_loc)
        ax.set_xticklabels([f"{i}" for i in range_loc], fontsize=10, rotation=90)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel("Correlation")
        if ind == np.ceil(len(cor_seq) / 50) - 1:
            ax.set_xlabel("Residue ID")

    def _corr_per_resid_bar_plot(self, hm: np.ndarray, path: str | os.PathLike) -> None:
        """Save a bar plot of the average correlation parameter for each residue."""
        cor_seq = np.mean(np.nan_to_num(hm), axis=0)
        fig, axs = plt.subplots(
            nrows=int(np.ceil(len(cor_seq) / 50)),
            ncols=1,
            figsize=(16, 4 * int(np.ceil(len(cor_seq) / 50))),
        )
        if len(cor_seq) < 50:
            self._corr_per_resid_bar_subplot(cor_seq, axs, 0)
        else:
            for ind, ax in enumerate(axs):
                self._corr_per_resid_bar_subplot(cor_seq, ax, ind)
        plt.subplots_adjust(
            left=0.125, bottom=-0.8, right=0.9, top=0.9, wspace=0.2, hspace=0.2
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_combined_output(
        self,
        dist_corr_list: np.ndarray,
        ang_corr_list: np.ndarray,
        best_dist_clusters: np.ndarray,
    ) -> None:
        """Generate combined output incorporating both distance and angle features, if both used."""

        console.h3("Combined output")

        # correlation parameters
        avg_dist_cor = (
            np.mean(dist_corr_list[:, 2]) if dist_corr_list is not None else None
        )
        avg_ang_cor = (
            np.mean(ang_corr_list[:, 2]) if ang_corr_list is not None else None
        )

        self._write_combined_txt(avg_ang_cor, avg_dist_cor, best_dist_clusters)
        self._write_combined_json(avg_ang_cor, avg_dist_cor, best_dist_clusters)

    def _write_combined_json(
        self,
        avg_ang_cor: float | None,
        avg_dist_cor: float | None,
        best_clusters: np.ndarray | None,
    ) -> None:
        """
        Write clustering results to a JSON file.

        Writes the following values, if available:

        * Average distance-based correlation over all residues
        * Average angle-based correlation over all residues
        * Cluster assignment of each conformer, as given in `best_clusters`
        """
        json_dump_data: dict[str, dict[str, Any]] = {"distribution": {}}
        if avg_dist_cor is not None:
            json_dump_data["distribution"]["dist_cor"] = np.around(avg_dist_cor, 4)
        if avg_ang_cor is not None:
            json_dump_data["distribution"]["ang_cor"] = np.around(avg_ang_cor, 4)
        if best_clusters is not None:
            json_dump_data["cluster_assignment"] = {
                str(i + 1): int(best_clusters[i])
                for i in range(len(self.correlation_extraction.structure))
            }
        with open(
            os.path.join(
                self.save_path,
                f"results_{self.correlation_extraction.residue_subset}.json",
            ),
            "w",
        ) as outfile:
            json.dump(json_dump_data, outfile)

    def _write_combined_txt(
        self,
        avg_ang_cor: float | None,
        avg_dist_cor: float | None,
        best_clusters: np.ndarray | None,
    ) -> None:
        """
        Write clustering results to a plaintext file.

        Writes the following values, if available:

        * Average distance-based correlation over all residues
        * Average angle-based correlation over all residues
        * Population of each state (calculated as the fraction occupied over all conformers in `best_clusters`)
        """
        with open(
            os.path.join(
                self.save_path,
                "correlations_" + self.correlation_extraction.residue_subset + ".txt",
            ),
            "w",
        ) as f:
            if avg_dist_cor is not None:
                f.write(f"Distance correlations: {avg_dist_cor}\n")
            if avg_ang_cor is not None:
                f.write(f"Angle correlations: {avg_ang_cor}\n")
            if best_clusters is not None:
                for i in range(self.correlation_extraction.nstates):
                    pop = len([j for j in best_clusters if j == i]) / len(best_clusters)
                    f.write("State {} population: {} \n".format(i + 1, pop))


class ClusterIO(ABC):
    feature_name = "base"

    def __init__(
        self,
        calculator: "ClusterCalculator",
        params: Optional[CorrelationExtractionIOParams],
        output_parent_path: Optional[Union[str, os.PathLike]] = None,
    ):
        self.calculator = calculator
        self.params = CorrelationExtractionIOParams() if params is None else params
        self.output_parent_path = output_parent_path

    @classmethod
    def output_path(cls, parent_path, chain_id):
        return os.path.join(
            CorrelationExtractionChainIO.output_path(parent_path, chain_id),
            f"clustering_{cls.feature_name}",
        )

    @classmethod
    def draw_features(
        cls,
        features: np.ndarray,
        clusters: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
        xlim: Optional[tuple[SupportsFloat, SupportsFloat]] = None,
        ylim: Optional[tuple[SupportsFloat, SupportsFloat]] = None,
        axis_labels: Optional[tuple[str, str]] = None,
    ):
        """
        Draw `features` on `ax` or a new `Axes` object,
        optionally colour-coded by cluster using the order specified in `clusters`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.scatter(
            x=features[:, 0],
            y=features[:, 1],
            c=clusters,  # Also works if clusters is None
            marker=".",
        )
        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

    def _create_clustering_results_plot(
        self,
        features_all_aa: np.ndarray,
        clusters: np.ndarray,
        res_ids: Sequence[str],
        combine: Optional[int] = None,
        xlim: Optional[tuple[SupportsFloat, SupportsFloat]] = None,
        ylim: Optional[tuple[SupportsFloat, SupportsFloat]] = None,
        axis_labels: Optional[tuple[str, str]] = None,
    ) -> tuple[dict[plt.Figure, list[str]], list[plt.Figure]]:
        """
        Base function to plot each residue's calculated features with colour-coding for the assigned clusters.

        Intended to be inherited for use in child classes' plotting function.

        :param features_all_aa:
            `ndarray` of calculated features, with shape (num_residues, num_conformers, num_features).
            If more than two features are contained, only the first two will be plotted.
        :param clusters: `ndarray` of cluster assignments, with shape (num_residues, num_conformers).
        :param res_ids: Iterable of residue IDs to use in subplot titles
        :param combine: Number of subplots to combine per figure (default: 9, or num_residues if fewer)
        :param xlim: x-axis limits (default: selected automatically by Matplotlib)
        :param ylim: y-axis limits (default: selected automatically by Matplotlib)
        :param axis_labels: Tuple of (x, y)-axis labels (default: left empty)
        """
        if features_all_aa.shape[0] != clusters.shape[0]:
            raise ValueError(
                "Inconsistent number of residues in features_all_aa and clusters."
            )
        if features_all_aa.shape[1] != clusters.shape[1]:
            raise ValueError(
                "Inconsistent number of conformers in features_all_aa and clusters."
            )
        num_resi = clusters.shape[0]

        if combine is None:
            combine = min(9, num_resi)

        # Set up figures & axes
        figs = []
        axs = []
        nrows = int(np.sqrt(combine))
        ncols = combine // nrows
        for i in range(num_resi):
            idx = i % combine
            if idx == 0:
                figs.append(
                    plt.figure(
                        layout="compressed",
                        figsize=(float(ncols) * 4.8, float(nrows) * 3.6),
                    )
                )
            axs.append(figs[-1].add_subplot(nrows, ncols, idx + 1))

        # Draw clusters on each figure
        fig_res_ids: dict[plt.Figure, list[str]] = {fig: [] for fig in figs}
        for i in console.trange(num_resi, desc="Drawing clusters"):
            ax = axs[i]
            res_id = res_ids[i]
            features_single_aa = features_all_aa[i, :, :2]
            clusters_single_aa = clusters[i, :]
            assert isinstance(ax.figure, plt.Figure)  # for mypy
            fig_res_ids[ax.figure].append(res_id)
            self.draw_features(
                features=features_single_aa,
                clusters=clusters_single_aa,
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                axis_labels=axis_labels,
            )
            ax.set_title(f"Residue {res_id}")

        return fig_res_ids, figs


class DistanceClusterIO(ClusterIO):
    feature_name = "distances"

    def draw_clustering_results(
        self,
        coord_matrix: np.ndarray,
        clusters: np.ndarray,
        chain_id: str,
    ) -> Optional[list[plt.Figure]]:
        """
        Plot PCA-encoded distance features for each residue, colour-coded by cluster.

        Uses `ClusterIO._create_clustering_results_plot()` to plot the generated features,
        saving the resulting plots to disk if required.
        """
        self.calculator: "DistanceClusterCalculator"  # for MyPy

        # Skip plot generation & return early if switched off
        if not self.params.create_cluster_plots:
            return None

        features_2d = np.stack(
            [
                PCA(n_components=2).fit_transform(  # Reduce to 2D
                    np.linalg.norm(  # Normalize over Cartesian coordinates
                        np.delete(  # Delete zero column corresponding to residue "ind"
                            (
                                # Take slice of self.coord_matrix belonging to residue "ind"
                                # and transform from shape (num_conformers, 3) to shape (num_conformers, 1, 3)
                                # to allow subtraction via broadcasting
                                coord_matrix
                                - np.expand_dims(coord_matrix[:, i, :], axis=1)
                            ),
                            i,
                            axis=1,
                        ),
                        axis=-1,
                    )
                )
                for i in range(coord_matrix.shape[1])
            ]
        )

        fig_res_ids, figs = self._create_clustering_results_plot(
            features_all_aa=features_2d,
            clusters=clusters,
            res_ids=[str(r) for r in self.calculator.resid],
            xlim=None,
            ylim=None,
            axis_labels=(
                "Principal component 1",
                "Principal component 2",
            ),
        )

        if self.output_parent_path is not None:
            output_path = self.output_path(self.output_parent_path, chain_id)
            os.makedirs(output_path, exist_ok=True)

            for fig in figs:
                res_ids = fig_res_ids[fig]
                if len(res_ids) > 1:
                    filename = f"dist_pca_resid_{res_ids[0]}-{res_ids[-1]}.svg"
                else:
                    filename = f"dist_pca_resid_{res_ids[0]}.svg"
                fig.savefig(os.path.join(output_path, filename))

        return figs


class AngleClusterIO(ClusterIO):
    feature_name = "angles"

    @staticmethod
    def angle_to_tex(angle, enclose_in_formula=True):
        """Represent an angle name in TeX algebra notation."""
        angle_re = r"(?P<angle_name>[a-z]+)(?P<angle_index>[0-9]+)?"
        angle_match = re.match(angle_re, angle.lower())
        f = "$" if enclose_in_formula else ""
        if angle_match.group("angle_index") is None:
            return f"{f}\\{angle_match.group('angle_name')}{f}"
        else:
            return f"{f}\\{angle_match.group('angle_name')}_{{{angle_match.group('angle_index')}}}{f}"

    def draw_clustering_results(
        self,
        angles_all_aa: np.ndarray,
        clusters: np.ndarray,
        chain_id: str,
    ) -> Optional[list[plt.Figure]]:
        """
        Plot the angle features for each residue, colour-coded by cluster.

        Primarily uses `ClusterIO._create_clustering_results_plot()`,
        adding class-specific features such as angle names and saving the resulting plots to disk if required.
        """
        self.calculator: "AngleClusterCalculator"  # for MyPy

        # Skip plot generation & return early if switched off
        if not self.params.create_cluster_plots:
            return None

        angle_names = (
            self.calculator.angleDict[0],
            self.calculator.angleDict[1],
        )

        fig_res_ids, figs = self._create_clustering_results_plot(
            features_all_aa=angles_all_aa,
            clusters=clusters,
            res_ids=[str(r) for r in self.calculator.resid],
            xlim=(-180, 180),
            ylim=(-180, 180),
            axis_labels=(
                self.angle_to_tex(angle_names[0]),
                self.angle_to_tex(angle_names[1]),
            ),
        )

        if self.output_parent_path is not None:
            output_path = self.output_path(self.output_parent_path, chain_id)
            os.makedirs(output_path, exist_ok=True)

            for fig in figs:
                res_ids = fig_res_ids[fig]
                if len(res_ids) > 1:
                    filename = f"angles_{angle_names[0]}-{angle_names[1]}_resid_{res_ids[0]}-{res_ids[-1]}.svg"
                else:
                    filename = f"angles_{angle_names[0]}-{angle_names[1]}_resid_{res_ids[0]}.svg"
                fig.savefig(os.path.join(output_path, filename))

        return figs
