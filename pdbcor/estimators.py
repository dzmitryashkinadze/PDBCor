import numpy as np
from Bio.PDB import is_aa
from tqdm import tqdm


class DistanceCor:
    """Distance-based correlation estimator"""
    # constructor
    def __init__(self,
                 structure,
                 mode,
                 nstates,
                 clust_model,
                 pose_estimator,
                 therm_fluct,
                 loop_start,
                 loop_end):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clust_model = clust_model
        self.pose_estimator = pose_estimator
        self.therm_fluct = therm_fluct
        self.loop_start = loop_start
        self.loop_end = loop_end
        # IMPORT THE STRUCTURE
        self.structure = structure
        self.mode = mode
        self.backboneAtoms = ['N', 'H', 'CA', 'HA', 'HA2', 'HA3', 'C', 'O']
        self.banres = []

    # get a center of mass of a given residue
    def get_coord(self, res):
        coord = []
        atom_number = 0
        for atom in res.get_atoms():
            if self.mode == 'backbone':
                if atom.id in self.backboneAtoms:
                    atom_number += 1
                    coord += list(atom.get_coord())
            elif self.mode == 'sidechain':
                if atom.id not in self.backboneAtoms:
                    atom_number += 1
                    coord += list(atom.get_coord())
            else:
                atom_number += 1
                coord += list(atom.get_coord())
        coord = np.array(coord).reshape(-1, 3)
        # average coordinates of all atoms
        coord = np.mean(coord, axis=0)
        # add thermal noise to the residue center
        coord += np.random.normal(0, self.therm_fluct / np.sqrt(atom_number), 3)
        return coord

    # Get coordinates of all residues
    def get_coord_matrix(self, chainID):
        coord_list = []
        for model in self.structure.get_models():
            chain = model[chainID]
            model_coord = []
            for res in chain.get_residues():
                if is_aa(res, standard=True):
                    if not (self.mode == 'sidechain' and
                            res.get_resname() == 'GLY') and not \
                            (self.loop_end >= res.id[1] >= self.loop_start):
                        model_coord.append(self.get_coord(res))
                    else:
                        self.banres += [res.id[1]]
                        model_coord.append([0, 0, 0])
            coord_list.append(model_coord)
        return np.array(coord_list).reshape(len(self.structure), len(self.resid), 3)

    # cluster conformers according to the distances between the ind residue with other residues
    def clust_aa(self, ind):
        features = []
        for model in range(len(self.structure)):
            for i in range(len(self.resid)):
                if i != ind:
                    # calculate an euclidean distance between residue centers
                    dist = np.sqrt(np.sum((self.CM[model][ind] - self.CM[model][i]) ** 2))
                    # save the distance
                    features += [dist]
        # scale features down to the unit norm and rearange into the feature matrix
        features = features / np.linalg.norm(np.array(features))
        features = np.array(features).reshape(len(self.structure), -1)
        # clustering
        return list(self.clust_model.fit_predict(features))

    # get clustering matrix
    def clust_cor(self, chain, resid):
        self.resid = resid
        self.CM = self.get_coord_matrix(chain)
        self.resid = [i for i in self.resid if i not in self.banres]
        clusters = []
        print('DISTANCE CLUSTERING PROCESS:')
        for i in tqdm(range(len(self.resid))):
            if self.resid[i] in self.banres:
                clusters += [self.resid[i]] + list(np.zeros(len(self.structure)))
            else:
                clusters += [self.resid[i]] + self.clust_aa(i)
        return np.array(clusters).reshape(-1, len(self.structure) + 1), self.banres


# angle correlations estimator
class AngleCor:
    """Angle-based correlation estimator"""
    # constructor
    def __init__(self, structure, mode, nstates, clust_model):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clustModel = clust_model
        self.structure = structure
        self.structure.atom_to_internal_coordinates()
        allowedAngles = {
            'backbone': ['phi', 'psi'],
            'sidechain': ['chi1', 'chi2', 'chi3', 'chi4', 'chi5'],
            'combined': ['phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']
        }
        bannedResDict = {
            'backbone': [],
            'sidechain': ['GLY', 'ALA'],
            'combined': []
        }
        self.bannedRes = bannedResDict[mode]
        self.angleDict = allowedAngles[mode]
        self.banres = []

    # collect angle data
    def get_angle_data(self, chainID):
        angles = []
        for model in self.structure.get_models():
            chain = model[chainID]
            for res in chain.get_residues():
                if is_aa(res, standard=True):
                    if res.internal_coord and res.get_resname() not in self.bannedRes:
                        entry = [res.get_full_id()[1],
                                 res.id[1]]
                        for angle in self.angleDict:
                            entry += [res.internal_coord.get_angle(angle)]
                        entry = [0 if v is None else v for v in entry]
                        angles += entry
        return np.array(angles).reshape(-1, 2 + len(self.angleDict))

    # gather angles from one amino acid
    def group_aa(self, aa_id):
        return self.angle_data[self.angle_data[:, 1] == aa_id, 2:]

    # shift angles to avoid clusters spreading over the (-180,180) cyclic coordinate closure
    @staticmethod
    def correct_cyclic_angle(ang):
        ang_sort = np.sort(ang)
        ang_cycled = np.append(ang_sort, ang_sort[0] + 360)
        ang_max_id = np.argmax(np.diff(ang_cycled))
        ang_max = ang_sort[ang_max_id]
        ang_shift = ang + 360 - ang_max
        ang_shift = [v - 360 if v > 360 else v for v in ang_shift]
        return ang_shift - np.mean(ang_shift)

    # execute clustering of single residue
    def clust_aa(self, aa_id):
        aa_data = self.group_aa(aa_id)
        if aa_data.shape == (0, 5):
            self.banres += [aa_id]
            return np.zeros(self.nConf)
        # correct the cyclic angle coordinates
        for i in range(len(self.angleDict)):
            aa_data[:, i] = self.correct_cyclic_angle(aa_data[:, i])
        # CLUSTERING
        return self.clustModel.fit_predict(aa_data)

    # get clustering matrix
    def clust_cor(self, chain, resid):
        self.angle_data = self.get_angle_data(chain)
        self.resid = resid
        # extract number of pdb models
        self.nConf = len(self.structure)
        # collect all clusterings
        clusters = []
        print('ANGLE CLUSTERING PROCESS:')
        for i in tqdm(range(len(self.resid))):
            clusters += [self.resid[i]]
            clusters += list(self.clust_aa(self.resid[i]))
        return np.array(clusters).reshape(-1, self.nConf + 1), self.banres