# PDBCor
Extraction of correlations from multistate PDB protein coordinates

## Installation
Required software:
* Python 3

1. Download the latest PDBcor wheel (`pdbcor-x.y.z-py3-none-any.whl`)

2. Set up a new python environment & install PDBcor:

```shell
python3 -m venv venv
source venv/bin/activate
pip install pdbcor-x.y.z-py3-none-any.whl
```

## Shell script

Analyse correlations of your multistate protein bundle directly via the shell entry-point to PDBcor:

```shell
# Demo example
pdbcor demo.pdb
```

| Parameters: |               |
|-------------| ------------- |
|             | **bundle:** str, path to the pdb file |
|             | **--nstates:** int, number of protein states, deafult=2 |
|             | **--mode:** str, mode of correlations, deafult=backbone<br>can be one of the following:<br>backbone - calculate correlations in protein backbone<br>sidechain - calculates correlations in protein sidechain<br>combined - calculates overall correlations<br>full - subsequently calculated backbone, sidechain and combined correlations |
|             | **--therm_fluct:** float, amplitude of the thermal motion, default=0.5|
|             | **--therm_iter:** int, number of thermal simulations, default=1|
|             | **--loop_start:** int, start of the loop (residue index), default=-1|
|             | **--loop_end:** int, end of the loop (residue index), default=-1|

## Python package 

Implement your own code in Python:

```python
from pdbcor import CorrelationExtraction

# Prepare correlation extractor (demo example)
a = CorrelationExtraction(
    "demo.pdb",
    mode="backbone",
    nstates=2,
)

# Run calculation
a.calculate_correlation()
```

## Produced output

Whether running from the shell or via Python, 
outputs are combined in the folder `/correlations`, 
which is created in the parent directory of the source PDB file.

Outputs include distance and angular correlation results including:

* text file with correlation parameters of your bundle and state populations (`correlations_{mode}.txt`)
* lists of correlations between each pair of residues (`{ang,dist,cross}_ig_{mode}.csv`)
* correlation heatmaps (`heatmap_{ang,dist,cross}_{mode}.png`)
* histograms of correlation parameters (`hist_{ang,dist}_{mode}.png`)
* sequential correlation parameter charts per residue (`seq_{ang,dist}_{mode}.png`)
* chimera executable to visualize a multistate bundle (`bundle_vis_{mode}.py`)
