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

# Visualization
import matplotlib.pyplot as plt  # Plotting library
import networkx as nx
plt.ioff()                       # turn off interactive plotting


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


# allostery extraction wrapper class
class AllosteryExtraction:
    # constructor
    def __init__(self, path, nstates=2):
        # HYPERVARIABLES
        directory = 'allostery'
        self.savePath = os.path.join(os.path.dirname(path), directory)
        self.PDBfilename = os.path.basename(path)
        try:
            os.mkdir(self.savePath)
        except:
            pass
        self.nstates = nstates  # number of states
        # CREATE ALLOSTERY ESTIMATORS WITH STRUCTURE ANG CLUSTERING MODEL
        self.structure = PDBParser().get_structure('test', path)
        self.resid = []
        for res in self.structure[0].get_residues():
            self.resid.append(res._id[1])
        self.aaS = min(self.resid)
        self.aaF = max(self.resid)
        clust_model = GaussianMixture(n_components=self.nstates, n_init=25, covariance_type='diag')
        pose_estimator = GaussianMixture(n_init=25, covariance_type='full')
        self.distAll = DistanceAllostery(self.structure, self.resid, nstates, clust_model, pose_estimator)
        self.angAll = AngleAllostery(self.structure, self.resid, nstates, clust_model)

    # write a file with allosteric parameter
    def write_allostery(self, dist_id, ang_id):
        # allosteric parameter
        dist_all = (np.mean(dist_id[:, 2]))
        ang_all = (np.mean(ang_id[:, 2]))
        f = open(os.path.join(self.savePath, 'allostery.txt'), "w")
        f.write('Distance allostery: {}\nAngle allostery: {} '.format(dist_all, ang_all))
        f.close()
        return None

    # execute correlation extraction
    def calc_all(self, graphics=True):
        # extract allostery matrices
        ang_ig, ang_hm = self.angAll.calc_ig()
        dist_ig, dist_hm, best_clust = self.distAll.calc_ig()
        pd.DataFrame(ang_ig).to_csv(self.savePath + '/ang_ig.csv', index=False)
        pd.DataFrame(dist_ig).to_csv(self.savePath + '/dist_ig.csv', index=False)
        # write allostery parameters
        self.write_allostery(dist_ig, ang_ig)
        # plot everything if graphics is enabled
        if graphics:
            self.plot_heatmaps(dist_hm, ang_hm)
            self.plot_hist(dist_ig, ang_ig)
            self.plot_all_per_aa(dist_hm, ang_hm)
            self.color_pdb(best_clust)
            try:
                os.mkdir(os.path.join(self.savePath, 'cornet_dist'))
                os.mkdir(os.path.join(self.savePath, 'cornet_ang'))
            except:
                pass
            for thr in np.linspace(0.1, 1, 19):
                self.plot_graph(dist_ig, ang_ig, thr)
        print('DONE')

    # construct a chimera executive to view a colored bundle
    def color_pdb(self, best_clust):
        state_color = ['#00FFFF', '#FF00FF', '#FFFF00', '#000000']
        chimera_path = os.path.join(self.savePath, 'bundle_vis.py')
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
        # plot distance heatmap
        fig, ax = plt.subplots()
        ax.imshow(dist_hm, origin='lower', extent=[self.aaS, self.aaF, self.aaS, self.aaF])
        plt.savefig(os.path.join(self.savePath, 'heatmap_dist.png'))
        plt.close()
        # plot distance heatmap
        fig, ax = plt.subplots()
        ax.imshow(ang_hm, origin='lower', extent=[self.aaS, self.aaF, self.aaS, self.aaF])
        plt.savefig(os.path.join(self.savePath, 'heatmap_ang.png'))
        plt.close()

    # plot histogram of correlation parameter
    def plot_hist(self, dist_ig, ang_ig):
        # plot distance allostery histogram
        plt.hist(dist_ig[:, 2], bins=50)
        plt.xlabel('Information gain')
        plt.ylabel('Density')
        plt.savefig(os.path.join(self.savePath, 'hist_dist.png'))
        plt.close()
        # plot angle allostery histogram
        plt.hist(ang_ig[:, 2], bins=50)
        plt.xlabel('Information gain')
        plt.ylabel('Density')
        plt.savefig(os.path.join(self.savePath, 'hist_ang.png'))
        plt.close()

    # plot the allostery per amino  acid barplot
    def plot_all_per_aa(self, dist_hm, ang_hm):
        # plot sequential distance allostery
        all_seq = np.mean(dist_hm, axis=0)
        plt.figure()
        plt.bar(range(self.aaS, self.aaF + 1), all_seq, width=0.8)
        plt.xlabel('Residue')
        plt.ylabel('Allostery')
        plt.savefig(os.path.join(self.savePath, 'seq_dist.png'))
        plt.close()
        # plot sequential angle allostery
        all_seq = np.mean(ang_hm, axis=0)
        plt.figure()
        plt.bar(range(self.aaS, self.aaF + 1), all_seq, width=0.8)
        plt.xlabel('Residue')
        plt.ylabel('Allostery')
        plt.savefig(os.path.join(self.savePath, 'seq_ang.png'))
        plt.close()

    # plot the allosteric graph
    def plot_graph(self, dist_ig, ang_ig, threshold=0.2):
        # plot allosteric graph for distance allostery
        ig_loc_id = [i for i in range(dist_ig.shape[0]) if dist_ig[i, 2] > threshold]
        ig_loc = dist_ig[ig_loc_id, :]
        if ig_loc.shape[0] > 1:
            res_loc = np.unique(ig_loc[:, :2])
            # create the graph
            G = nx.Graph()
            # add nodes
            for i in range(len(res_loc)):
                G.add_node(int(res_loc[i]))
            # add edges
            for i in range(ig_loc.shape[0]):
                G.add_edge(int(ig_loc[i, 0]), int(ig_loc[i, 1]))
            plt.figure()
            nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
            plt.savefig(os.path.join(self.savePath, 'cornet_dist/graph_' + str(np.round(threshold, 2)) + '.png'))
            plt.close()

        # plot allosteric graph for angle allostery
        ig_loc_id = [i for i in range(ang_ig.shape[0]) if ang_ig[i, 2] > threshold]
        ig_loc = ang_ig[ig_loc_id, :]
        if ig_loc.shape[0] > 1:
            res_loc = np.unique(ig_loc[:, :2])
            # create the graph
            G = nx.Graph()
            # add nodes
            for i in range(len(res_loc)):
                G.add_node(int(res_loc[i]))
            # add edges
            for i in range(ig_loc.shape[0]):
                G.add_edge(int(ig_loc[i, 0]), int(ig_loc[i, 1]))
            plt.figure()
            nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
            plt.savefig(os.path.join(self.savePath, 'cornet_ang/graph_' + str(np.round(threshold, 2)) + '.png'))
            plt.close()


# distance allostery estimator
class DistanceAllostery:
    # constructor
    def __init__(self, structure, resid, nstates, clust_model, pose_estimator):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clust_model = clust_model
        self.pose_estimator = pose_estimator
        # IMPORT THE STRUCTURE
        self.structure = structure
        self.resid = resid

    # Get atom coordinate of a single amino acid
    def get_coord(self, res):
        coord = []
        for atom in res.get_atoms():
            coord += list(atom.get_coord())
        self.pose_estimator.fit(np.array(coord).reshape(-1, 3))
        result_m = self.pose_estimator.means_.tolist()[0]
        result_c = self.pose_estimator.covariances_.reshape(1, -1).tolist()[0]
        return result_m + result_c

    # Get coordinates of all residues
    def get_coord_matrix(self):
        coord_list = []
        print('############################   DISTANCE ALLOSTERY   ###########################')
        print('COORDINATE MATRIX BUILDING:')
        print_progress_bar(0, len(self.structure) - 1, prefix='Progress:', suffix='Complete', length=50)
        for model in self.structure.get_models():
            model_coord = []
            print_progress_bar(model.id, len(self.structure) - 1, prefix='Progress:', suffix='Complete', length=50)
            for res in model.get_residues():
                res_coord = []
                model_coord.append(self.get_coord(res))
            coord_list.append(model_coord)
        return np.array(coord_list).reshape(len(self.structure), len(self.resid), 12)

    def clust_aa(self, ind):
        features = []
        for model in range(len(self.structure)):
            for i in range(len(self.resid)):
                m_dist = self.CM[model][ind][:3] - self.CM[model][i][:3]
                c_dist = self.CM[model][ind][3:] + self.CM[model][i][3:]
                dist = list(np.hstack((m_dist, c_dist)))
                features += dist
        features = np.array(features).reshape(len(self.structure), -1)
        return list(self.clust_model.fit_predict(normalize(features)))

    # get clustering matrix
    def clust_all(self):
        self.CM = self.get_coord_matrix()
        clusters = []
        print('DISTANCE CLUSTERING PROCESS:')
        print_progress_bar(0, len(self.resid) - 1, prefix='Progress:', suffix='Complete', length=50)
        for i in range(len(self.resid)):
            print_progress_bar(i, len(self.resid) - 1, prefix='Progress:', suffix='Complete', length=50)
            clusters += [self.resid[i]] + self.clust_aa(i)
        clusters = np.array(clusters).reshape(-1, len(self.structure) + 1)
        return clusters

    # calculate information gain for all residue pairs from the clustering matrix
    def calc_ig(self):
        clusters = self.clust_all()
        # calculate mutual information
        ig_matrix = np.zeros((clusters.shape[0], clusters.shape[0]))
        print('MUTUAL INFORMATION EXTRACTION PROCESS:')
        print_progress_bar(0, clusters.shape[0]-1, prefix='Progress:', suffix='Complete', length=50)
        for i in range(clusters.shape[0]):
            print_progress_bar(i, clusters.shape[0] - 1, prefix='Progress:', suffix='Complete', length=50)
            ig_matrix[i, i] = 1
            for j in range(i+1, clusters.shape[0]):
                ig_loc = adjusted_mutual_info_score(clusters[i, 1:], clusters[j, 1:])
                ig_matrix[i, j] = ig_loc
                ig_matrix[j, i] = ig_loc
        ig_list = []
        for i in range(clusters.shape[0]):
            for j in range(i+1, clusters.shape[0]):
                ig_list += list(clusters[i, :1]) + list(clusters[j, :1]) + [ig_matrix[i, j]]
        print('###############################################################################')
        print()
        print()
        # Calculate best coloring vector
        ig_sum = np.sum(ig_matrix, axis=1)
        best_res = [i for i in range(len(ig_sum)) if ig_sum[i] == max(ig_sum)][0]
        best_clust = clusters[best_res, 1:]
        return np.array(ig_list).reshape(-1, 3), ig_matrix, best_clust


# angle allostery estimator
class AngleAllostery:
    # constructor
    def __init__(self, structure, resid, nstates, clust_model):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clustModel = clust_model
        structure.atom_to_internal_coordinates()
        self.resid = resid
        # EXTRACT ANGLES TO DATAFRAME
        angles = []
        for model in structure.get_models():
            for res in model.get_residues():
                if res.internal_coord:
                    entry = [res.get_full_id()[1],
                             res.id[1],
                             res.internal_coord.get_angle("psi"),
                             res.internal_coord.get_angle("phi"),
                             res.internal_coord.get_angle("omega"),
                             res.internal_coord.get_angle("chi2"),
                             res.internal_coord.get_angle("chi2"),
                             res.internal_coord.get_angle("chi3"),
                             res.internal_coord.get_angle("chi4"),
                             res.internal_coord.get_angle("chi5")]
                    entry = [0 if v is None else v for v in entry]
                    angles += entry
        self.angle_data = np.array(angles).reshape(-1, 10)
        # extract number of pdb models
        self.nConf = int(max(self.angle_data[:, 0])) + 1

    # Gather angles from one amino acid
    def group_aa(self, aa_id):
        ind = [i for i in range(self.angle_data.shape[0]) if self.angle_data[i, 1] == aa_id]
        return self.angle_data[ind, 2:]

    # shift angles to avoid clusters spreading over the (-180,180) cyclic coordinate closure
    @staticmethod
    def correct_cyclic_angle(ang):
        ang_sort = np.sort(ang)
        ang_cycled = np.append(ang_sort, ang_sort[0]+360)
        ang_max_id = np.argmax(np.diff(ang_cycled))
        ang_max = ang_sort[ang_max_id]
        ang_shift = ang + 360 - ang_max
        ang_shift = [v - 360 if v > 360 else v for v in ang_shift]
        ang_center = ang_shift - np.mean(ang_shift)
        return ang_center

    # Execute clustering of single residue
    def clust_aa(self, aa_id, hard_clust=True):
        aa_data = self.group_aa(aa_id)
        # correct the cyclic angle coordinates
        for i in range(8):
            aa_data[:, i] = self.correct_cyclic_angle(aa_data[:, i])
        # CLUSTERING
        self.clustModel.fit(aa_data)
        if hard_clust:
            clust = self.clustModel.predict(aa_data)
        else:
            clust = self.clustModel.predict_proba(aa_data)[:, 0]
        return clust

    # get clustering matrix
    def clust_all(self):
        # collect all clusterings
        clusters = []
        print('#############################   ANGLE ALLOSTERY   #############################')
        print('ANGLE CLUSTERING PROCESS:')
        print_progress_bar(0, len(self.resid) - 1, prefix='Progress:', suffix='Complete', length=50)
        for i in range(len(self.resid)):
            print_progress_bar(i, len(self.resid) - 1, prefix='Progress:', suffix='Complete', length=50)
            clusters += [self.resid[i]]
            clusters += list(self.clust_aa(self.resid[i]))
        return np.array(clusters).reshape(-1, self.nConf + 1)

    # calculate information gain for all residue pairs
    def calc_ig(self):
        # calculate clusters
        clusters = self.clust_all()
        # calculate mutual information
        ig_matrix = np.zeros((clusters.shape[0], clusters.shape[0]))
        print('MUTUAL INFORMATION EXTRACTION PROCESS:')
        print_progress_bar(0, clusters.shape[0] - 1, prefix='Progress:', suffix='Complete', length=50)
        for i in range(clusters.shape[0]):
            print_progress_bar(i, clusters.shape[0] - 1, prefix='Progress:', suffix='Complete', length=50)
            ig_matrix[i, i] = 1
            for j in range(i + 1, clusters.shape[0]):
                ig_loc = adjusted_mutual_info_score(clusters[i, 1:], clusters[j, 1:])
                ig_matrix[i, j] = ig_loc
                ig_matrix[j, i] = ig_loc
        ig_list = []
        for i in range(clusters.shape[0]):
            for j in range(i + 1, clusters.shape[0]):
                ig_list += list(clusters[i, :1]) + list(clusters[j, :1]) + [ig_matrix[i, j]]
        print('###############################################################################')
        print()
        print()
        return np.array(ig_list).reshape(-1, 3), ig_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation extraction from multistate protein bundles')
    parser.add_argument('bundle', type=str,
                        help='protein bundle file path')
    parser.add_argument('--nstates', type=int,
                        help='number of states')
    args = parser.parse_args()
    if args.nstates:
        nstates = args.nstates
    else:
        nstates = 2
    a = AllosteryExtraction(args.bundle, nstates=nstates)
    a.calc_all()