import os
from typing import List, Optional, Union

import matplotlib
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score  # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore

from .io import CorrelationExtractionIOParams, CorrelationExtractionIO
from .clustering import DistanceClusterCalculator, AngleClusterCalculator
from .console import console

matplotlib.use("agg")


class CorrelationExtraction:
    """Main wrapper class for correlation extraction."""

    def __init__(
        self,
        pdb_file_path: Union[str, os.PathLike],
        input_file_format: Optional[str] = None,
        output_directory: Optional[Union[str, os.PathLike]] = None,
        io_params: Optional[CorrelationExtractionIOParams] = None,
        residue_subset: str = "backbone",
        features: str = "both",
        nstates: int = 2,
        therm_fluct: float = 0.5,
        therm_iter: int = 5,
        loop_start: int = -1,
        loop_end: int = -1,
        fast: bool = False,
    ):
        """
        Initialize the `CorrelationExtraction` object and clustering estimators.

        Fast mode
        ---------

        If `fast` is `True`, "fast mode" will be activated for correlation calculation.

        This changes the following:

        * Adjusted Rand score used for calculating correlation instead of adjusted mutual information
        * GMM initialized using k-means++ instead of full k-means
        * GMM fitting replicated 10x instead of 25x per clustering
        """
        console.h1("PDBCor")

        # HYPERVARIABLES
        self.residue_subset: str = residue_subset
        self.features: str = features
        self.pdb_file_path: Union[str, os.PathLike] = pdb_file_path
        self.pdb_file_name: Union[str, os.PathLike] = os.path.basename(
            self.pdb_file_path
        )
        self.io = CorrelationExtractionIO(
            correlation_extraction=self,
            output_directory=output_directory,
            params=io_params,
        )
        self.therm_iter: int = therm_iter
        self.resid: list[int] = []
        self.aaS: int = 0
        self.aaF: int = 0
        self.nstates: int = nstates  # number of states
        self.fast = fast

        # CREATE CORRELATION ESTIMATORS WITH STRUCTURE ANG CLUSTERING MODEL
        console.h2("Setup")
        console.h3("Structure file import")
        structure_parser: Union[MMCIFParser, PDBParser]  # for mypy
        if input_file_format is None:
            if os.path.splitext(self.pdb_file_name)[1] in {".pdb", ".ent"}:
                console.print("PDB format identified.")
                structure_parser = PDBParser()
            else:
                console.print("Assuming PDBx/mmCIF format.")
                structure_parser = MMCIFParser()
        elif input_file_format.lower() == "mmcif":
            console.print("PDBx/mmCIF format set via CLI.")
            structure_parser = MMCIFParser()
        elif input_file_format.lower() == "pdb":
            console.print("PDB format set via CLI.")
            structure_parser = PDBParser()
        else:
            raise ValueError(f"Invalid structure file format: {input_file_format}")
        console.print("Parsing structure file...")
        self.structure = structure_parser.get_structure(
            os.path.splitext(self.pdb_file_name)[0], pdb_file_path
        )
        self.chain_ids = [chain.id for chain in self.structure[0].get_chains()]
        console.print(
            "Structure parsed successfully "
            f"(identified {len(self.chain_ids)} chain{'' if len(self.chain_ids) == 1 else 's'})."
        )

        console.h3("Clustering model initialization")
        console.print("Initializing GMM...")
        clust_model = GaussianMixture(
            n_components=self.nstates,
            n_init=10 if self.fast else 25,
            covariance_type="diag",
            init_params="k-means++" if self.fast else "kmeans",
        )

        console.print(f"Initializing {self.features} cluster calculators...")
        self.dist_clust_calc = (
            DistanceClusterCalculator(
                self.structure,
                residue_subset,
                nstates,
                clust_model,
                therm_fluct,
                loop_start,
                loop_end,
                output_parent_path=self.io.output_parent_path,
                io_params=io_params,
            )
            if self.uses_distances
            else None
        )
        self.ang_clust_calc = (
            AngleClusterCalculator(
                self.structure,
                residue_subset,
                nstates,
                clust_model,
                output_parent_path=self.io.output_parent_path,
                io_params=io_params,
            )
            if self.uses_angles
            else None
        )

        console.print("Setup complete.", style="green")

        self.correlation_metric = (
            adjusted_rand_score if self.fast else adjusted_mutual_info_score
        )
        console.print(f"Fast mode {'enabled' if self.fast else 'disabled'}.")

    @property
    def uses_angles(self) -> bool:
        return self.features in {"both", "angle"}

    @property
    def uses_distances(self) -> bool:
        return self.features in {"both", "distance"}

    def _calc_ami(self, clusters: np.ndarray, resid: List[int]) -> np.ndarray:
        """
        Calculate adjusted mutual information between 2 clustering sets in the form of a correlation list.

        Uses `self.aaS`/`self.aaF` (set by `self.calculate_correlation()`)
        to specify the shape of the correlation matrix.

        :returns: `ndarray` containing AMI values in pairwise list form with rows `[ resi_i, resi_j, ami_i_j ]`
        """
        return np.vstack(
            [
                [
                    resid[i],  # residue ID i
                    resid[j],  # residue ID j
                    self.correlation_metric(
                        clusters[i, :], clusters[j, :]
                    ),  # correlation
                ]
                for i in console.tqdm(
                    range(clusters.shape[0]), desc="Extracting mutual information"
                )
                for j in range(i + 1, clusters.shape[0])
            ]
        )

    def _ami_list_to_matrix(
        self, ami_list: np.ndarray, banres: List[int]
    ) -> np.ndarray:
        """
        Convert list of correlations into matrix form.

        Uses `self.aaS`/`self.aaF` (set by `self.calculate_correlation()`)
        to specify the shape of the correlation matrix.
        Index 0 corresponds to `aaS` and the final row/column to `aaF`,
        with skipped residue numbers within this range being represented by `np.nan`.

        :param ami_list: `ndarray` containing AMI values in pairwise list form with rows `[ resi_i, resi_j, ami_i_j ]`
        :param banres: list of excluded residues (set to `NaN`)
        :returns: `ndarray` containing correlations as a 2D matrix
        """

        ami_matrix = np.zeros(
            (int(self.aaF - self.aaS + 1), int(self.aaF - self.aaS + 1))
        )

        for i in range(self.aaS, int(self.aaF + 1)):
            if i not in self.resid or i in banres:
                for j in range(self.aaS, int(self.aaF + 1)):
                    ami_matrix[int(i - self.aaS), int(j - self.aaS)] = np.nan
                    ami_matrix[int(j - self.aaS), int(i - self.aaS)] = np.nan
            else:
                ami_matrix[int(i - self.aaS), int(i - self.aaS)] = 1

        for i in range(ami_list.shape[0]):
            ami_matrix[
                int(ami_list[i, 0] - self.aaS), int(ami_list[i, 1] - self.aaS)
            ] = ami_list[i, 2]
            ami_matrix[
                int(ami_list[i, 1] - self.aaS), int(ami_list[i, 0] - self.aaS)
            ] = ami_list[i, 2]

        return ami_matrix

    def calculate_correlation(self) -> None:
        """
        Main function for calculating correlation.

        Iterates over each chain in `self.chain_ids` and runs `self._calc_cor_chain()`.
        Saves output for each chain to a separate directory,
        and copies the file `README.txt` to the parent output directory.

        Also calculates minimum/maximum residue nr. and saves to `self.aaS`/`self.aaF`
        """
        for chain in self.chain_ids:
            self.resid = []
            for res in self.structure[0][chain].get_residues():
                if is_aa(res, standard=True):
                    self.resid.append(res.id[1])
            if len(self.resid) == 0:
                console.warn(
                    f"No valid (AA) residues identified for chain {chain} of file {self.pdb_file_path}. "
                    "Skipping this chain."
                )
                continue
            self.aaS = min(self.resid)
            self.aaF = max(self.resid)
            console.h2(f"Chain {chain}")
            self._calc_cor_chain(chain, self.resid)

        console.h3("Final processing")
        self.io.finalize_analysis()
        console.print("PDBCor complete!", style="green")

    def _calc_cor_chain(self, chain: str, resid: List[int]) -> None:
        """
        Execute correlation extraction for a single chain.

        #. Perform angle clustering using `self.angCor`.
        #. Extract correlations using `self._calc_ami()`.
        #. Perform distance clustering using `self.distCor`.
        #. Extract correlations using `self._calc_ami()`.
        #. Average over `self.therm_iter` replicates.
        #. Clean up data etc.
        #. Output data & generate figures.
        """
        output_handler = self.io.chain_io(
            chain_id=chain,
        )

        if self.uses_angles:
            ang_ami, ang_clusters, ang_hm = self._calc_cor_chain_ang(chain, resid)
            best_ang_clusters = self.best_clusters(ang_hm, ang_clusters)
            output_handler.generate_single_feature_output(
                corr_list=ang_ami,
                clusters=ang_clusters,
                corr_matrix=ang_hm,
                best_clust=best_ang_clusters,
                feature_name=("ang", "angle"),
            )
        else:
            ang_ami, ang_clusters, ang_hm = None, None, None

        if self.uses_distances:
            dist_ami, dist_clusters, dist_hm = self._calc_cor_chain_dist(chain, resid)
            best_dist_clusters = self.best_clusters(dist_hm, dist_clusters)
            output_handler.generate_single_feature_output(
                corr_list=dist_ami,
                clusters=dist_clusters,
                corr_matrix=dist_hm,
                best_clust=best_dist_clusters,
                feature_name=("dist", "distance"),
            )
        else:
            dist_ami, dist_clusters, dist_hm = None, None, None
            best_dist_clusters = None

        output_handler.generate_combined_output(dist_ami, ang_ami, best_dist_clusters)
        console.print(f"Chain {chain} complete!", style="green")

    def _calc_cor_chain_dist(self, chain, resid):
        assert isinstance(self.dist_clust_calc, DistanceClusterCalculator)
        assert self.therm_iter > 0

        # Initialize arrays as empty to ensure they exist
        dist_ami = np.empty((0, 0))
        dist_hm = np.empty((0, 0))
        dist_clusters = np.empty(0)

        # Run a series of thermally corrected distance correlation extractions
        for i in range(self.therm_iter):
            console.h3(f"Distance clustering (run {i + 1} of {self.therm_iter})")
            dist_clusters, dist_banres = self.dist_clust_calc.cluster(chain, resid)
            dist_ami_loc = self._calc_ami(dist_clusters, resid)
            dist_hm_loc = self._ami_list_to_matrix(dist_ami_loc, dist_banres)
            if i == 0:
                dist_ami = dist_ami_loc
                dist_hm = dist_hm_loc
            else:
                dist_ami = np.dstack((dist_ami, dist_ami_loc))
                dist_hm = np.dstack((dist_hm, dist_hm_loc))
        if dist_ami.size == 0 or dist_hm.size == 0 or dist_clusters.size == 0:
            raise ValueError("Calculation not completed correctly.")

        # Average over iterations
        if self.therm_iter > 1:
            dist_ami = np.mean(dist_ami, axis=2)
            dist_hm = np.mean(dist_hm, axis=2)

        # Round to 3 sign. digits
        dist_ami[:, 2] = np.around(dist_ami[:, 2], 4)

        return dist_ami, dist_clusters, dist_hm

    def _calc_cor_chain_ang(self, chain, resid):
        assert isinstance(self.ang_clust_calc, AngleClusterCalculator)

        # extract angle correlation matrices
        console.h3("Angle clustering")
        ang_clusters, ang_banres = self.ang_clust_calc.cluster(chain, resid)
        ang_ami = self._calc_ami(ang_clusters, resid)
        ang_hm = self._ami_list_to_matrix(ang_ami, ang_banres)

        # Round to 3 sign. digits
        ang_ami[:, 2] = np.around(ang_ami[:, 2], 4)

        return ang_ami, ang_clusters, ang_hm

    def best_clusters(self, corr_matrix, clusters):
        """
        Calculate the best clusters, i.e. those belonging to the residue with the highest total correlation.

        Requires "conversion" of index from the correlation matrix to the cluster assignment list:
        The matrix contains `(self.aaF - self.aaS + 1)` rows/columns, with "skipped" residues represented as `np.nan`,
        whereas there are only `len(self.resid)` defined cluster assignments.
        Therefore, `(<matrix_row_id> + self.aaS)` gives any row's actual residue number,
        and the index of this value in `self.resid` is the same index
        that should be used to retrieve that residue's cluster assignments.
        """
        corr_sum = np.nansum(corr_matrix, axis=1)
        best_row_id = np.argmax(corr_sum)
        best_cluster_idx = self.resid.index(best_row_id + self.aaS)  # index conversion
        return clusters[best_cluster_idx, :]
