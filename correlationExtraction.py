# Clustering algorithm
from sklearn.mixture import GaussianMixture

# Import other libraries
from Bio.PDB.PDBParser import PDBParser  # pdb extraction
from Bio.PDB.Polypeptide import is_aa
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
import os
import pandas as pd
import argparse
from copy import copy
from tqdm import tqdm

# Visualization
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt  # Plotting library
from matplotlib.cm import get_cmap


# correlation extraction wrapper class
class CorrelationExtraction:
    # constructor
    def __init__(self,
                 path,
                 mode='backbone',
                 nstates=2,
                 therm_fluct=0.5,
                 therm_iter=5,
                 loop_start=-1,
                 loop_end=-1):
        # HYPERVARIABLES
        directory = 'correlations'
        self.mode = mode
        self.savePath = os.path.join(os.path.dirname(path), directory)
        self.PDBfilename = os.path.basename(path)
        self.therm_iter = therm_iter
        self.resid = []
        self.aaS = 0
        self.aaF = 0
        try:
            os.mkdir(self.savePath)
        except:
            pass
        self.nstates = nstates  # number of states
        # CREATE CORRELATION ESTIMATORS WITH STRUCTURE ANG CLUSTERING MODEL
        self.structure = PDBParser().get_structure('test', path)
        self.chains = []
        for chain in self.structure[0].get_chains():
            self.chains.append(chain.id)
        clust_model = GaussianMixture(n_components=self.nstates, n_init=25, covariance_type='diag')
        pose_estimator = GaussianMixture(n_init=25, covariance_type='full')
        self.distCor = DistanceCor(self.structure,
                                   mode,
                                   nstates,
                                   clust_model,
                                   pose_estimator,
                                   therm_fluct,
                                   loop_start,
                                   loop_end)
        self.angCor = AngleCor(self.structure, mode, nstates, clust_model)

    # calculate information gain between 2 clustering sets
    def calc_ami(self, clusters, banres):
        # calculate mutual information
        ami_list = []
        for i in tqdm(range(clusters.shape[0])):
            for j in range(i+1, clusters.shape[0]):
                cor = adjusted_mutual_info_score(clusters[i, 1:], clusters[j, 1:])
                ami_list += list(clusters[i, :1]) + list(clusters[j, :1]) + [cor]
        ami_list = np.array(ami_list).reshape(-1, 3)
        # construct correlation matrix
        ami_matrix = np.zeros((int(self.aaF - self.aaS + 1), int(self.aaF - self.aaS + 1)))
        for i in range(self.aaS, int(self.aaF + 1)):
            if i not in self.resid or i in banres:
                for j in range(self.aaS, int(self.aaF + 1)):
                    ami_matrix[int(i - self.aaS), int(j - self.aaS)] = None
                    ami_matrix[int(j - self.aaS), int(i - self.aaS)] = None
            else:
                ami_matrix[int(i - self.aaS), int(i - self.aaS)] = 1
        for i in range(ami_list.shape[0]):
            ami_matrix[int(ami_list[i, 0] - self.aaS), int(ami_list[i, 1] - self.aaS)] = ami_list[i, 2]
            ami_matrix[int(ami_list[i, 1] - self.aaS), int(ami_list[i, 0] - self.aaS)] = ami_list[i, 2]
        return ami_list, ami_matrix

    def calc_cor(self, graphics=True):
        for chain in self.chains:
            chainPath = os.path.join(self.savePath, 'chain' + chain)
            try:
                os.mkdir(chainPath)
            except:
                pass
            self.resid = []
            for res in self.structure[0][chain].get_residues():
                if is_aa(res, standard=True):
                    self.resid.append(res._id[1])
            self.aaS = min(self.resid)
            self.aaF = max(self.resid)
            print()
            print()
            print('################################################################################')
            print('#################################   CHAIN {}   #################################'.format(chain))
            print('################################################################################')
            self.calc_cor_chain(chain, chainPath, self.resid, graphics)

    # execute correlation extraction
    def calc_cor_chain(self, chain, chainPath, resid, graphics):
        # extract angle correlation matrices
        print()
        print()
        print('#############################   ANGLE CLUSTERING   ############################')
        ang_clusters, ang_banres = self.angCor.clust_cor(chain, resid)
        print('ANGULAR MUTUAL INFORMATION EXTRACTION:')
        ang_ami, ang_hm = self.calc_ami(ang_clusters, ang_banres)
        # Run a series of thermally corrected distance correlation extractions
        print()
        print()
        print('############################   DISTANCE CLUSTERING   ##########################')
        for i in range(self.therm_iter):
            dist_clusters, dist_banres = self.distCor.clust_cor(chain, resid)
            print('DISTANCE MUTUAL INFORMATION EXTRACTION, RUN {}:'.format(i + 1))
            dist_ami_loc, dist_hm_loc = self.calc_ami(dist_clusters, dist_banres)
            if i == 0:
                dist_ami = dist_ami_loc
                dist_hm = dist_hm_loc
            else:
                dist_ami = np.dstack((dist_ami, dist_ami_loc))
                dist_hm = np.dstack((dist_hm, dist_hm_loc))
        # Average them
        if self.therm_iter > 1:
            dist_ami = np.mean(dist_ami, axis=2)
            dist_hm = np.mean(dist_hm, axis=2)
        # Calculate best coloring vector
        ami_sum = np.nansum(dist_hm, axis=1)
        best_res = [i for i in range(len(ami_sum)) if ami_sum[i] == np.nanmax(ami_sum)]
        best_res = best_res[0]
        best_clust = dist_clusters[best_res, 1:]
        print()
        print()
        print('############################       FINALIZING       ###########################')
        print('PROCESSING CORRELATION MATRICES')
        pd.DataFrame(ang_ami).to_csv(chainPath + '/ang_ami_' + self.mode + '.csv',
                                     index=False,
                                     header=['ID1',
                                             'ID2',
                                             'AMI'])
        pd.DataFrame(dist_ami).to_csv(chainPath + '/dist_ami_' + self.mode + '.csv',
                                      index=False,
                                      header=['ID1',
                                              'ID2',
                                              'AMI'])
        # write correlation parameters
        self.write_correlations(dist_ami, ang_ami, best_clust, chainPath)
        # create a chimera executable
        self.color_pdb(best_clust, chainPath)
        # plot everything if graphics is enabled
        if graphics:
            print('PLOTTING')
            # plot through the try option so that it does not crash server calculations
            # that can not have graphical output
            try:
                self.plot_heatmaps(dist_hm, ang_hm, chainPath)
                self.plot_hist(dist_ami, ang_ami, chainPath)
                self.plot_cor_per_aa(dist_hm, ang_hm, chainPath)
            except:
                None
        print('DONE')
        print()

    # write a file with correlation parameters
    def write_correlations(self, dist_ami, ang_ami, best_clust, path):
        # correlation parameters
        dist_cor = (np.mean(dist_ami[:, 2]))
        ang_cor = (np.mean(ang_ami[:, 2]))
        f = open(os.path.join(path, 'correlations_' + self.mode + '.txt'), "w")
        f.write('Distance correlations: {}\nAngle correlations: {} \n'.format(dist_cor, ang_cor))
        for i in range(self.nstates):
            pop = len([j for j in best_clust if j == i]) / len(best_clust)
            f.write('State {} population: {} \n'.format(i+1, pop))
        f.close()

    # construct a chimera executive to view a colored bundle
    def color_pdb(self, best_clust, path):
        state_color = ['#00FFFF', '#00008b', '#FF00FF', '#FFFF00', '#000000']
        chimera_path = os.path.join(path, 'bundle_vis_' + self.mode + '.py')
        with open(chimera_path, 'w') as f:
            f.write('from chimera import runCommand as rc\n')
            f.write('rc("open ../../{}")\n'.format(self.PDBfilename))
            f.write('rc("background solid white")\n')
            f.write('rc("~ribbon")\n')
            f.write('rc("show :.a@ca")\n')
            # bundle coloring
            for i in range(len(self.structure)):
                f.write('rc("color {} #0.{}")\n'.format(state_color[int(best_clust[i])], int(i + 1)))

    # plot the correlation matrix heatmap
    def plot_heatmaps(self, dist_hm, ang_hm, path):
        # change color map to display nans as gray
        cmap = copy(get_cmap("viridis"))
        cmap.set_bad('gray')
        # plot distance heatmap
        fig, ax = plt.subplots()
        ax.imshow(dist_hm, origin='lower', extent=[self.aaS, self.aaF, self.aaS, self.aaF], cmap=cmap)
        plt.xlabel('Distance clustering id')
        plt.ylabel('Distance clustering id')
        plt.savefig(os.path.join(path, 'heatmap_dist_' + self.mode + '.png'), bbox_inches='tight')
        plt.close()
        # plot angular heatmap
        fig, ax = plt.subplots()
        ax.imshow(ang_hm, origin='lower', extent=[self.aaS, self.aaF, self.aaS, self.aaF], cmap=cmap)
        plt.xlabel('Angular clustering id')
        plt.ylabel('Angular clustering id')
        plt.savefig(os.path.join(path, 'heatmap_ang_' + self.mode + '.png'), bbox_inches='tight')
        plt.close()

    # plot histogram of correlation parameter
    def plot_hist(self, dist_ami, ang_ami, path):
        # plot distance correlation histogram
        plt.hist(dist_ami[:, 2], bins=50)
        plt.xlabel('Information gain')
        plt.ylabel('Density')
        plt.savefig(os.path.join(path, 'hist_dist_' + self.mode + '.png'), bbox_inches='tight')
        plt.close()
        # plot angle correlation histogram
        plt.hist(ang_ami[:, 2], bins=50)
        plt.xlabel('Information gain')
        plt.ylabel('Density')
        plt.savefig(os.path.join(path, 'hist_ang_' + self.mode + '.png'), bbox_inches='tight')
        plt.close()

    # plot the correlations per amino acid barplot
    def plot_cor_per_aa(self, dist_hm, ang_hm, path):
        # plot sequential distance correlations
        cor_seq = np.mean(np.nan_to_num(dist_hm), axis=0)
        plt.figure()
        plt.bar(range(self.aaS, self.aaF + 1), cor_seq, width=0.8)
        plt.xlabel('Residue')
        plt.ylabel('Correlation')
        plt.savefig(os.path.join(path, 'seq_dist_' + self.mode + '.png'), bbox_inches='tight')
        plt.close()
        # plot sequential angle correlations
        cor_seq = np.mean(np.nan_to_num(ang_hm), axis=0)
        plt.figure()
        plt.bar(range(self.aaS, self.aaF + 1), cor_seq, width=0.8)
        plt.xlabel('Residue')
        plt.ylabel('Correlation')
        plt.savefig(os.path.join(path, 'seq_ang_' + self.mode + '.png'), bbox_inches='tight')
        plt.close()


# distance correlations estimator
class DistanceCor:
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
