# Clustering algorithm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

# Import other libraries
from Bio.PDB.PDBParser import PDBParser    # pdb extraction
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
import os
import pandas as pd
import argparse
from copy import copy
from tqdm import tqdm


# Visualization
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.cm import get_cmap
import networkx as nx
plt.ioff()                       # turn off interactive plotting


# correlation extraction wrapper class
class CorrelationExtraction:
    # constructor
    def __init__(self, path, mode, nstates, cornet_cutoff, graphics, therm_fluct):
        # HYPERVARIABLES
        directory = 'correlations'
        self.mode = mode
        self.cornet_cutoff = cornet_cutoff
        self.savePath = os.path.join(os.path.dirname(path), directory)
        self.PDBfilename = os.path.basename(path)
        try:
            os.mkdir(self.savePath)
        except:
            pass
        self.nstates = nstates  # number of states
        self.graphics = graphics
        # CREATE CORRELATION ESTIMATORS WITH STRUCTURE ANG CLUSTERING MODEL
        self.structure = PDBParser().get_structure('test', path)
        self.resid = []
        for res in self.structure[0].get_residues():
            self.resid.append(res._id[1])
        self.aaS = min(self.resid)
        self.aaF = max(self.resid)
        clust_model = GaussianMixture(n_components=self.nstates, n_init=25, covariance_type='diag')
        pose_estimator = GaussianMixture(n_init=25, covariance_type='full')
        self.distCor = DistanceCor(self.structure, mode, self.resid, nstates, clust_model, pose_estimator, therm_fluct)
        self.angCor = AngleCor(self.structure, mode, self.resid, nstates, clust_model)

    # calculate information gain between 2 clustering sets
    def calc_ami(self, clusters1, banres1, clusters2, banres2, symmetrical=False):
        # calculate mutual information
        ami_matrix = np.zeros((clusters1.shape[0], clusters1.shape[0]))
        if symmetrical:
            for i in tqdm(range(clusters1.shape[0])):
                if clusters1[i, 0] not in banres1:
                    ami_matrix[i, i] = 1
                    for j in range(i + 1, clusters1.shape[0]):
                        if (clusters1[i, 0] not in banres1) and (clusters1[j, 0] not in banres1):
                            ami_loc = adjusted_mutual_info_score(clusters1[i, 1:], clusters1[j, 1:])
                            ami_matrix[i, j] = ami_loc
                            ami_matrix[j, i] = ami_loc
                        else:
                            ami_matrix[i, j] = None
                            ami_matrix[j, i] = None
                else:
                    ami_matrix[i, i] = None
                    for j in range(i + 1, clusters1.shape[0]):
                        ami_matrix[i, j] = None
                        ami_matrix[j, i] = None
            ami_list = []
            for i in range(clusters1.shape[0]):
                for j in range(i + 1, clusters1.shape[0]):
                    if (clusters1[i, 0] not in banres1) and (clusters1[j, 0] not in banres1):
                        ami_list += list(clusters1[i, :1]) + list(clusters1[j, :1]) + [ami_matrix[i, j]]
        else:
            for i in tqdm(range(clusters1.shape[0])):
                if clusters1[i, 0] not in banres1:
                    for j in range(clusters2.shape[0]):
                        if clusters2[j, 0] not in banres2:
                            ami_loc = adjusted_mutual_info_score(clusters1[i, 1:], clusters2[j, 1:])
                            ami_matrix[i, j] = ami_loc
                        else:
                            ami_matrix[i, j] = None
                else:
                    for j in range(1, clusters2.shape[0]):
                        ami_matrix[i, j] = None
            ami_list = []
            for i in range(clusters1.shape[0]):
                for j in range(clusters2.shape[0]):
                    if (clusters1[i, 0] not in banres1) and (clusters2[j, 0] not in banres2):
                        ami_list += list(clusters1[i, :1]) + list(clusters2[j, :1]) + [ami_matrix[i, j]]
        return np.array(ami_list).reshape(-1, 3), ami_matrix

    # execute correlation extraction
    def calc_cor(self):
        # extract correlation matrices
        ang_clusters, ang_banres = self.angCor.clust_cor()
        dist_clusters, dist_banres = self.distCor.clust_cor()
        print('########################       MUTUAL INFORMATION       #######################')
        print('ANGULAR MUTUAL INFORMATION EXTRACTION:')
        ang_ami, ang_hm = self.calc_ami(ang_clusters, ang_banres, ang_clusters, ang_banres, symmetrical=True)
        print('DISTANCE MUTUAL INFORMATION EXTRACTION:')
        dist_ami, dist_hm = self.calc_ami(dist_clusters, dist_banres, dist_clusters, dist_banres, symmetrical=True)
        # Calculate best coloring vector
        ami_sum = np.nansum(dist_hm, axis=1)
        best_res = [i for i in range(len(ami_sum)) if ami_sum[i] == np.nanmax(ami_sum)]
        best_res = best_res[0]
        best_clust = dist_clusters[best_res, 1:]
        print()
        print('############################       FINALIZING       ###########################')
        print('PROCESSING CORRELATION MATRICES')
        pd.DataFrame(ang_ami).to_csv(self.savePath + '/ang_ami_' + self.mode + '.csv',
                                    index=False,
                                    header=['ID1',
                                            'ID2',
                                            'AMI'])
        pd.DataFrame(dist_ami).to_csv(self.savePath + '/dist_ami_' + self.mode + '.csv',
                                     index=False,
                                     header=['ID1',
                                             'ID2',
                                             'AMI'])
        # write correlation parameters
        self.write_correlations(dist_ami, ang_ami)
        # plot everything if graphics is enabled
        if self.graphics:
            print('PLOTTING')
            self.plot_heatmaps(dist_hm, ang_hm)
            self.plot_hist(dist_ami, ang_ami)
            self.plot_cor_per_aa(dist_hm, ang_hm)
            self.color_pdb(best_clust)
        print('DONE')
        print()
        print()
        print()

    # write a file with correlation parameters
    def write_correlations(self, dist_ami, ang_ami):
        # correlation parameters
        dist_cor = (np.mean(dist_ami[:, 2]))
        ang_cor = (np.mean(ang_ami[:, 2]))
        f = open(os.path.join(self.savePath, 'correlations_' + self.mode + '.txt'), "w")
        f.write('Distance correlations: {}\nAngle correlations: {} '.format(dist_cor, ang_cor))
        f.close()

    # construct a chimera executive to view a colored bundle
    def color_pdb(self, best_clust):
        state_color = ['#00FFFF', '#FF00FF', '#FFFF00', '#000000']
        chimera_path = os.path.join(self.savePath, 'bundle_vis_' + self.mode + '.py')
        with open(chimera_path, 'w') as f:
            f.write('from chimera import runCommand as rc\n')
            f.write('rc("open ../{}")\n'.format(self.PDBfilename))
            f.write('rc("background solid white")\n')
            f.write('rc("~ribbon")\n')
            f.write('rc("show :.a@ca")\n')
            # bundle coloring
            for i in range(len(self.structure)):
                f.write('rc("color {} #0.{}")\n'.format(state_color[int(best_clust[i])], int(i + 1)))

    # plot the correlation matrix heatmap
    def plot_heatmaps(self, dist_hm, ang_hm):
        # change color map to display nans as gray
        cmap = copy(get_cmap("viridis"))
        cmap.set_bad('gray')
        # plot distance heatmap
        fig, ax = plt.subplots()
        ax.imshow(dist_hm, origin='lower', extent=[self.aaS, self.aaF, self.aaS, self.aaF], cmap=cmap)
        plt.xlabel('Distance clustering id')
        plt.ylabel('Distance clustering id')
        plt.savefig(os.path.join(self.savePath, 'heatmap_dist_' + self.mode + '.png'))
        plt.close()
        # plot angular heatmap
        fig, ax = plt.subplots()
        ax.imshow(ang_hm, origin='lower', extent=[self.aaS, self.aaF, self.aaS, self.aaF], cmap=cmap)
        plt.xlabel('Angular clustering id')
        plt.ylabel('Angular clustering id')
        plt.savefig(os.path.join(self.savePath, 'heatmap_ang_' + self.mode + '.png'))
        plt.close()

    # plot histogram of correlation parameter
    def plot_hist(self, dist_ami, ang_ami):
        # plot distance correlation histogram
        plt.hist(dist_ami[:, 2], bins=50)
        plt.xlabel('Information gain')
        plt.ylabel('Density')
        plt.savefig(os.path.join(self.savePath, 'hist_dist_' + self.mode + '.png'))
        plt.close()
        # plot angle correlation histogram
        plt.hist(ang_ami[:, 2], bins=50)
        plt.xlabel('Information gain')
        plt.ylabel('Density')
        plt.savefig(os.path.join(self.savePath, 'hist_ang_' + self.mode + '.png'))
        plt.close()

    # plot the correlations per amino acid barplot
    def plot_cor_per_aa(self, dist_hm, ang_hm):
        # plot sequential distance correlations
        cor_seq = np.mean(np.nan_to_num(dist_hm), axis=0)
        plt.figure()
        plt.bar(range(self.aaS, self.aaF + 1), cor_seq, width=0.8)
        plt.xlabel('Residue')
        plt.ylabel('Correlation')
        plt.savefig(os.path.join(self.savePath, 'seq_dist_' + self.mode + '.png'))
        plt.close()
        # plot sequential angle correlations
        cor_seq = np.mean(np.nan_to_num(ang_hm), axis=0)
        plt.figure()
        plt.bar(range(self.aaS, self.aaF + 1), cor_seq, width=0.8)
        plt.xlabel('Residue')
        plt.ylabel('Correlation')
        plt.savefig(os.path.join(self.savePath, 'seq_ang_' + self.mode + '.png'))
        plt.close()


# distance correlations estimator
class DistanceCor:
    # constructor
    def __init__(self, structure, mode, resid, nstates, clust_model, pose_estimator, therm_fluct):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clust_model = clust_model
        self.pose_estimator = pose_estimator
        self.therm_fluct = therm_fluct
        # IMPORT THE STRUCTURE
        self.structure = structure
        self.resid = resid
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
    def get_coord_matrix(self):
        coord_list = []
        print('############################   DISTANCE CLUSTERING   ##########################')
        for model in self.structure.get_models():
            model_coord = []
            for res in model.get_residues():
                if not (self.mode == 'sidechain' and res.get_resname() == 'GLY'):
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
    def clust_cor(self):
        self.CM = self.get_coord_matrix()
        clusters = []
        print('DISTANCE CLUSTERING PROCESS:')
        for i in tqdm(range(len(self.resid))):
            if self.resid[i] in self.banres:
                clusters += [self.resid[i]] + list(np.zeros(len(self.structure)))
            else:
                clusters += [self.resid[i]] + self.clust_aa(i)
        print()
        return np.array(clusters).reshape(-1, len(self.structure) + 1), self.banres


# angle correlations estimator
class AngleCor:
    # constructor
    def __init__(self, structure, mode, resid, nstates, clust_model):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clustModel = clust_model
        structure.atom_to_internal_coordinates()
        self.resid = resid
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
        bannedRes = bannedResDict[mode]
        self.angleDict = allowedAngles[mode]
        self.banres = []
        # EXTRACT ANGLES TO DATAFRAME
        angles = []
        for model in structure.get_models():
            for res in model.get_residues():
                if res.internal_coord and res.get_resname() not in bannedRes:
                    entry = [res.get_full_id()[1],
                             res.id[1]]
                    for angle in self.angleDict:
                        entry += [res.internal_coord.get_angle(angle)]
                    entry = [0 if v is None else v for v in entry]
                    angles += entry
        self.angle_data = np.array(angles).reshape(-1, 2 + len(self.angleDict))
        # extract number of pdb models
        self.nConf = int(max(self.angle_data[:, 0])) + 1

    # gather angles from one amino acid
    def group_aa(self, aa_id):
        return self.angle_data[self.angle_data[:, 1] == aa_id, 2:]

    # shift angles to avoid clusters spreading over the (-180,180) cyclic coordinate closure
    @staticmethod
    def correct_cyclic_angle(ang):
        ang_sort = np.sort(ang)
        ang_cycled = np.append(ang_sort, ang_sort[0]+360)
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
    def clust_cor(self):
        # collect all clusterings
        clusters = []
        print('#############################   ANGLE CLUSTERING   ############################')
        print('ANGLE CLUSTERING PROCESS:')
        for i in tqdm(range(len(self.resid))):
            clusters += [self.resid[i]]
            clusters += list(self.clust_aa(self.resid[i]))
        print()
        return np.array(clusters).reshape(-1, self.nConf + 1), self.banres


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
    parser.add_argument('--cornet_cutoff', type=int,
                        default=2,
                        help='Minimum sequential difference between residues in the correlation network')
    parser.add_argument('--therm_fluct', type=float,
                        default=0.5,
                        help='Thermal fluctuation of distances in the protein bundle')
    args = parser.parse_args()
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
                                  cornet_cutoff=args.cornet_cutoff,
                                  graphics=args.graphics,
                                  therm_fluct=args.therm_fluct)
        a.calc_cor()
