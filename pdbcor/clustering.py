import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture  # type: ignore
from Bio.PDB import is_aa
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue

from .io import DistanceClusterIO, AngleClusterIO, CorrelationExtractionIOParams
from .console import console


class ClusterCalculator(ABC):
    @abstractmethod
    def _clust_aa(self, aa_id: int, prob=False) -> np.ndarray:
        """
        Perform clustering within a single residue.

        If `prob`, return cluster probabilities, else return assignments.
        """
        return np.empty((0, 0) if prob else (0,))

    @abstractmethod
    def cluster(
        self, chain: str, resid: List[int], prob=False
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get clustering matrix for the specified chain by calling `self._clust_aa()` on each amino acid.

        The returned array of cluster indices has the shape (number of residues, number of conformers).
        Alongside this, a list of excluded residues is returned.

        If `prob`, the weights of each cluster are returned instead of assignments.
        The returned array has the shape
        (number of conformers, number of residues) if `prob` is `False`
        and (number of conformers, number of residues, number of clusters) otherwise.
        """
        return np.empty((0, 0, 0) if prob else (0, 0)), []


class DistanceClusterCalculator(ClusterCalculator):
    """Distance-based clustering."""

    def __init__(
        self,
        structure: Structure,
        residue_subset: str,
        nstates: int,
        clust_model: GaussianMixture,
        therm_fluct: float,
        loop_start: int,
        loop_end: int,
        output_parent_path: Optional[Union[str, os.PathLike]] = None,
        io_params: Optional[CorrelationExtractionIOParams] = None,
    ):
        """
        Set hypervariables and import structure.

        :param structure: `Structure` object to perform clustering on
        :param residue_subset: Subset of residue atoms used for clustering (can be `"backbone"`, `"sidechain"` or `"combined"`)
        :param nstates: Number of states to use for clustering
        :param clust_model: `GaussianMixture` model to use for clustering
        :param therm_fluct: Parameter for amplitude of thermal noise added to residues
        :param loop_start: Residue nr. for start of loop region to exclude from analysis
        :param loop_end: Residue nr. for end of loop region to exclude from analysis
        """
        # HYPERVARIABLES
        self.nstates: int = nstates  # number of states
        self.clust_model: GaussianMixture = clust_model
        self.therm_fluct: float = therm_fluct
        self.loop_start: int = loop_start
        self.loop_end: int = loop_end

        # IMPORT THE STRUCTURE
        self.structure: Structure = structure
        self.residue_subset: str = residue_subset
        self.backboneAtoms: list[str] = ["N", "H", "CA", "HA", "HA2", "HA3", "C", "O"]
        self.banres: list[int] = []
        self.resid: list[int] = []
        self.coord_matrix: np.ndarray = np.empty((0, 0))
        self.io: DistanceClusterIO = DistanceClusterIO(
            calculator=self,
            output_parent_path=output_parent_path,
            params=io_params,
        )

    def _residue_center(self, res: Residue) -> np.ndarray:
        """
        Calculate the center of mass of a given residue.

        Only backbone/sidechain coordinates are used if the corresponding subset is set.
        Thermal noise is added to the resulting coordinates as specified by `self.therm_fluct`.
        """
        coord_flat = []
        atom_number = 0
        for atom in res.get_atoms():
            if self.residue_subset == "backbone":
                if atom.id in self.backboneAtoms:
                    atom_number += 1
                    coord_flat.extend(list(atom.get_coord()))
            elif self.residue_subset == "sidechain":
                if atom.id not in self.backboneAtoms:
                    atom_number += 1
                    coord_flat.extend(list(atom.get_coord()))
            else:
                atom_number += 1
                coord_flat.extend(list(atom.get_coord()))
        coord = np.array(coord_flat).reshape(-1, 3)

        # average coordinates of all atoms
        coord = np.mean(coord, axis=0)

        # add thermal noise to the residue center
        coord += np.random.normal(0, self.therm_fluct / np.sqrt(atom_number), 3)
        return coord

    def _residue_coords(self, chain_id: str) -> np.ndarray:
        """
        Get coordinates of all residues in the specified chain.

        Residues between `self.loop_start` and `self.loop_end` are skipped,
        as well as glycines if `self.residue_subset == "sidechain"`.
        The coordinates of these residues are set to `[0, 0, 0]`
        and they are added to `self.banres`.

        The resulting `np.ndarray` has dimensions `(num_conformers, num_residues, 3)`
        (positions for each conformer, for each residue's center, in Cartesian coordinates).
        """
        coord_list: list[list[np.ndarray]] = []
        for model in self.structure.get_models():
            chain = model[chain_id]
            model_coord: list[np.ndarray] = []
            for res in chain.get_residues():
                if is_aa(res, standard=True):
                    if not (
                        self.residue_subset == "sidechain"
                        and res.get_resname() == "GLY"
                    ) and not (self.loop_end >= res.id[1] >= self.loop_start):
                        model_coord.append(self._residue_center(res))
                    else:
                        self.banres.append(res.id[1])
                        model_coord.append(np.array([0, 0, 0]))
            coord_list.append(model_coord)
        return np.array(coord_list).reshape(len(self.structure), len(self.resid), 3)

    def _clust_aa(self, ind: int, prob=False) -> np.ndarray:
        """
        Cluster conformers according to the distances between the specified residue and all other residues.

        Euclidean distances are calculated from `self.coord_matrix`, normalized and passed to `self.clust_model`.
        """
        features = self._calc_norm_dist_matrix(ind)

        # clustering
        if prob:
            self.clust_model.fit(features)
            return self.clust_model.predict_proba(features)
        else:
            return self.clust_model.fit_predict(features)

    def _calc_norm_dist_matrix(self, ind):
        """
        Calculate normalized distances between residue `ind` and all others.

        Creates a `np.ndarray` of shape `(num_conformers, num_residues - 1)`.
        """
        features = np.linalg.norm(  # Normalize over Cartesian coordinates
            np.delete(  # Delete zero column corresponding to residue "ind"
                (
                    # Take slice of self.coord_matrix belonging to residue "ind"
                    # and transform from shape (num_conformers, 3) to shape (num_conformers, 1, 3)
                    # to allow subtraction via broadcasting
                    self.coord_matrix
                    - np.expand_dims(self.coord_matrix[:, ind, :], axis=1)
                ),
                ind,
                axis=1,
            ),
            axis=-1,
        )
        return features / np.linalg.norm(features)

    def cluster(
        self, chain: str, resid: List[int], prob=False
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get clustering matrix for the specified chain by calling `self._clust_aa()` on each amino acid.

        Residues in `self.banres` are assigned 0 for each conformer.

        The returned array has the shape
        (number of residues, number of conformers) if `prob` is `False`
        and (number of residues, number of conformers, number of clusters) otherwise.
        """
        self.resid = resid
        self.coord_matrix = self._residue_coords(chain)
        clusters = np.stack(
            [
                self._clust_aa(i, prob=prob)
                if self.resid[i] not in self.banres
                else np.zeros(len(self.structure))
                for i in console.tqdm(
                    range(len(self.resid)), desc="Calculating clusters"
                )
            ],
            axis=0,
        )
        figs = self.io.draw_clustering_results(self.coord_matrix, clusters, chain)
        if figs is not None:
            for f in figs:
                plt.close(f)
        return clusters, self.banres


# angle correlations estimator
class AngleClusterCalculator(ClusterCalculator):
    """Angle-based clustering."""

    def __init__(
        self,
        structure: Structure,
        residue_subset: str,
        nstates: int,
        clust_model: GaussianMixture,
        featurize: bool = True,
        output_parent_path: Optional[Union[str, os.PathLike]] = None,
        io_params: Optional[CorrelationExtractionIOParams] = None,
    ):
        """
        Set hypervariables and import structure.

        Structure coordinates are converted from Cartesian to internal
        using `Structure.atom_to_internal_coordinates()`.

        :param structure: `Structure` object to perform clustering on
        :param residue_subset: Subset of residue atoms used for clustering (can be `"backbone"`, `"sidechain"` or `"combined"`)
        :param nstates: Number of states to use for clustering
        :param clust_model: `GaussianMixture` model to use for clustering
        :param featurize: "Featurize" angles by transforming into sin/cos instead of using directly for clustering
            (default: `True`)
        """
        # HYPERVARIABLES
        self.nstates: int = nstates  # number of states
        self.clust_model: GaussianMixture = clust_model
        self.structure: Structure = structure
        self.nConf: int = len(self.structure)  # number of PDB models
        self.structure.atom_to_internal_coordinates()
        allowed_angles: dict[str, list[str]] = {
            "backbone": ["phi", "psi"],
            "sidechain": ["chi1", "chi2", "chi3", "chi4", "chi5"],
            "combined": ["phi", "psi", "chi1", "chi2", "chi3", "chi4", "chi5"],
        }
        banned_res_dict: dict[str, list[str]] = {
            "backbone": [],
            "sidechain": ["GLY", "ALA"],
            "combined": [],
        }
        self.bannedRes: list[str] = banned_res_dict[residue_subset]
        self.angleDict: list[str] = allowed_angles[residue_subset]
        self.banres: list[int] = []
        self.resid: list[int] = []
        self.angle_data: np.ndarray = np.empty((0, 0))
        self.featurize: bool = featurize
        self.io: AngleClusterIO = AngleClusterIO(
            calculator=self,
            output_parent_path=output_parent_path,
            params=io_params,
        )

    def _all_residues_angles(self, chain_id: str) -> np.ndarray:
        """
        Get angle data for all residues in the specified chain.

        Angles to include are listed in `self.angleDict`.
        Amino acids specified in `self.bannedRes` are skipped entirely.
        """
        angles = []
        for model in self.structure.get_models():
            chain = model[chain_id]
            for res in chain.get_residues():
                if is_aa(res, standard=True):
                    if res.internal_coord and res.get_resname() not in self.bannedRes:
                        entry = [res.get_full_id()[1], res.id[1]]
                        for angle in self.angleDict:
                            entry.append(res.internal_coord.get_angle(angle))
                        entry = [0 if v is None else v for v in entry]
                        angles.extend(entry)
        return np.array(angles).reshape(-1, 2 + len(self.angleDict))

    def _single_residue_angles(self, aa_id: int) -> np.ndarray:
        """
        Gather angles from one amino acid.

        Angle values are extracted from `self.angle_data`
        (requires this field to have been filled by `self._all_residues_angles()`).
        """
        angles = self.angle_data[self.angle_data[:, 1] == aa_id, 2:]
        return angles

    @staticmethod
    def featurize_angles(angles_deg: np.ndarray) -> np.ndarray:
        """ "Featurize" angles (given in degrees) by transforming into sin/cos."""
        angles_rad = angles_deg * np.pi / 180
        return np.hstack([np.sin(angles_rad), np.cos(angles_rad)])

    @staticmethod
    def correct_cyclic_angle(ang: np.ndarray) -> np.ndarray:
        """
        Shift angles to avoid clusters spreading over the (-180,180) cyclic coordinate closure.

        Takes & returns an array of angle values (in degrees) from the interval [-180, 180].

        Procedure:

        #. Find the two angles with the largest difference:
            #. Sort all angles
            #. Append the smallest angle + 360 to the end of the array
            #. Calculate the maximum difference between a pair of consecutive values
        #. Assign the lower of these two to 360 -> "wrapping" occurs over the largest gap
        #. "Rotate" all other angles correspondingly
        #. Subtract 360 from any angles > 360 after 3.
        #. Subtract the mean value of the resulting array, leaving values centered around 0
        """
        ang_sort = np.sort(ang)
        ang_cycled = np.append(ang_sort, ang_sort[0] + 360)
        ang_max_id = np.argmax(np.diff(ang_cycled))
        ang_max = ang_sort[ang_max_id]
        ang_shift = ang + 360 - ang_max
        ang_shift = np.array([v - 360 if v > 360 else v for v in ang_shift])
        return ang_shift - np.mean(ang_shift)

    def _clust_aa(self, aa_id: int, prob=False) -> np.ndarray:
        """
        Cluster conformers according to the distribution of angles within a single residue.

        Angles are calculated via `self._single_residue_angles()` and normalized via `self._correct_cyclic_angle()`.
        Amino acids containing no angle data are added to `self.banres` and a zero array is returned.
        """
        features = self._generate_aa_features(aa_id)
        return self._calc_result(features, prob)

    def _generate_aa_features(self, aa_id: int) -> np.ndarray:
        """Generate complete array of features for residue `aa_id`"""
        features = self._single_residue_angles(aa_id)
        if features.shape[0] == 0:
            self.banres.append(aa_id)
            return np.zeros(self.nConf)
        # Featurize angles or correct the cyclic angle coordinates
        if self.featurize:
            features = self.featurize_angles(features)
        else:
            for i in range(len(self.angleDict)):
                features[:, i] = self.correct_cyclic_angle(features[:, i])
        return features

    def _generate_aa_angles(self, aa_id: int) -> np.ndarray:
        """
        Generate array of angles for residue `aa_id`.
        Serves as a parallel to `_generate_aa_features` but without featurization (e.g. for plotting).
        """
        features = self._single_residue_angles(aa_id)
        if features.shape[0] == 0:
            if aa_id not in self.banres:
                self.banres.append(aa_id)
            return np.zeros(self.nConf)
        # Return angles directly
        return features

    def _calc_result(self, features: np.ndarray, prob: bool = False) -> np.ndarray:
        if prob:
            self.clust_model.fit(features)
            return self.clust_model.predict_proba(features)
        else:
            return self.clust_model.fit_predict(features)

    def cluster(
        self, chain: str, resid: List[int], prob=False
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get clustering matrix for the specified chain by calling `self._clust_aa()` on each amino acid.

        Residues in `self.banres` are listed in the returned tuple,
        however they are not treated differently here,
        as `self._clust_aa()` automatically identifies them and returns a zero array.

        The returned array has the shape
        (number of residues, number of conformers) if `prob` is `False`
        and (number of residues, number of conformers, number of clusters) otherwise.
        """
        self.angle_data = self._all_residues_angles(chain)
        self.resid = resid
        features_all_aa = np.stack([self._generate_aa_features(r) for r in self.resid])
        clusters = np.stack(
            [
                self._calc_result(features, prob)
                for features in console.tqdm(
                    features_all_aa, desc="Calculating clusters"
                )
            ]
        )
        angles_to_draw = np.stack([self._generate_aa_angles(r) for r in self.resid])
        figs = self.io.draw_clustering_results(angles_to_draw, clusters, chain)
        if figs is not None:
            for f in figs:
                plt.close(f)
        return clusters, self.banres
