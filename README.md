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
python allosteryExtraction.py bundle_path nstates=2

# Demo example:
python allosteryExtraction.py demo.pdb
```

| Parameters:   |               |
| ------------- | ------------- |
|               | **bundle:** path to the pdb file |
|               | **nstates:** number of protein states, deafult=2 |

Outputs are combined in the folder /allostery, that is created in the parent directory of the source pdb file.
Outputs include distance and angular correlation results including:
* text file with allosteric parameters of your bundle (allostery.txt)
* lists of correlations between each pair of residues ([ang,dist]_ig.csv)
* correlation heatmaps (heatmap_[ang,dist].png)
* histograms of correlation parameters (hist_[ang,dist].png)
* sequential correlation parameter charts per residue (seq_[ang,dist].png)
* allosteric networks generated with range of different thrasholds (cornet_[ang,dist]/graph_thr.png)
* chimera executable to visualize a multistate bundle (bundle_vis.py)
