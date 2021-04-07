# pdb_correlation_extraction
Extraction of the correlations from the multistate pdb protein coordinates

Required software:
* Python3

Set up a new python environment
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Analyse correlations of your multistate protein bundle PDB with
```
python allosteryExtraction.py bundle_path --nstates=2 --mode=backbone --allnet_cutoff=2 --graphics=True

# Demo example:
python allosteryExtraction.py demo.pdb
```

| Parameters:   |               |
| ------------- | ------------- |
|               | **bundle:** str, path to the pdb file |
|               | **nstates:** int, number of protein states, deafult=2 |
|               | **mode:** str, mode of allostery, deafult=backbone<br>can be one of the following:<br>backbone - calculate allostery in protein backbone<br>sidechain - calculates allostery in protein sidechain<br>combined - calculates overall allostery<br>full - subsequently calculated backbone, sidechain and combined allostery |
|               | **allnet_cutoff:** int, minimum sequential difference between residues in the allosteric network |
|               | **graphics:** bool, graphical output, default=True |

Outputs are combined in the folder /allostery, that is created in the parent directory of the source pdb file.
Outputs include distance and angular correlation results including:
* text file with allosteric parameters of your bundle (allostery_[mode].txt)
* lists of correlations between each pair of residues ([ang,dist]_ig_[mode].csv)
* correlation heatmaps (heatmap_[ang,dist,cross]_[mode].png)
* histograms of correlation parameters (hist_[ang,dist]_[mode].png)
* sequential correlation parameter charts per residue (seq_[ang,dist]_[mode].png)
* allosteric network generated with range of different thrasholds (cornet_[mode]/graph_thr.png)
* chimera executable to visualize a multistate bundle (bundle_vis_[mode].py)
