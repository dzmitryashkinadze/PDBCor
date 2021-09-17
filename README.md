# PDBCor
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
# Demo example:
python correlationExtraction.py demo.pdb
```

| Parameters:   |               |
| ------------- | ------------- |
|               | **bundle:** str, path to the pdb file |
|               | **--nstates:** int, number of protein states, deafult=2 |
|               | **--mode:** str, mode of correlations, deafult=backbone<br>can be one of the following:<br>backbone - calculate correlations in protein backbone<br>sidechain - calculates correlations in protein sidechain<br>combined - calculates overall correlations<br>full - subsequently calculated backbone, sidechain and combined correlations |
|               | **--therm_fluct:** float, amplitude of the thermal motion, default=0.5|
|               | **--therm_iter:** int, number of thermal simulations, default=1|
|               | **--loop_start:** int, start of the loop (residue index), default=-1|
|               | **--loop_end:** int, end of the loop (residue index), default=-1|

Outputs are combined in the folder /correlations, that is created in the parent directory of the source pdb file.
Outputs include distance and angular correlation results including:
* text file with correlation parameters of your bundle and state populations (correlations_{mode}.txt)
* lists of correlations between each pair of residues ({ang,dist,cross}_ig_{mode}.csv)
* correlation heatmaps (heatmap_{ang,dist,cross}_{mode}.png)
* histograms of correlation parameters (hist_{ang,dist}_{mode}.png)
* sequential correlation parameter charts per residue (seq_{ang,dist}_{mode}.png)
* chimera executable to visualize a multistate bundle (bundle_vis_{mode}.py)
