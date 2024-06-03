# PDBCor
Extraction of correlations from multistate PDB protein coordinates

## Installation

PDBCor requires Python 3.9 or higher 
(lower versions may also function, but require manual installation via the source code).

### Virtual environment setup

We recommend running PDBCor in an isolated virtual environment.
A variety of tools include this functionality, e.g. 
[`venv`](https://docs.python.org/3/library/venv.html) or
[miniforge](https://github.com/conda-forge/miniforge).

For example, using the built-in `venv` module, 
simply run the following to create and activate a new virtual environment:

```shell
python3 -m venv pdbcor-venv
source pdbcor-venv/bin/activate
```

### Package installation

1. Download the latest PDBcor wheel
    (`pdbcor-x.y.z-py3-none-any.whl`, replacing `x.y.z` with the latest version number).
    This file contains the PDBCor module and additional metadata for installation.
1. Install the downloaded file via `pip install pdbcor-x.y.z-py3-none-any.whl`

## Shell script

Analyse correlations of your multistate protein bundle directly via the shell entry-point to PDBCor:

```shell
# Demo example
pdbcor demo.pdb
```

The default PDBCor CLI includes the following options:

```commandline
  -h, --help            show this help message and exit

input/output settings:
  -f {PDB,mmCIF}, --format {PDB,mmCIF}
                        input file format (default: determine from file extension)
  -o OUTPUT, --output OUTPUT
                        filename for output directory (default: "correlations_<name of structure file>")
  --no-plots            do not plot any graphical output
  --create-archive      create .zip archive of output directory
  --no-vis              do not create scripts for visualisation in PyMOL/Chimera
  --draw-clusters       create plots of clustering results
  -q, --quiet           quiet mode (only output errors to console)

correlation extraction settings:
  -n NUM_STATES, --num-states NUM_STATES
                        number of states (default: 2)
  --residue-subset {backbone,sidechain,combined,full}
                        subset of residue atoms used for clustering, or 'full' to iterate over each (default: backbone)
  --features {distance,angle,both}
                        features used for clustering (default: both)
  --fast                run in fast mode (see documentation for details)
  -i THERM_ITER, --therm-iter THERM_ITER
                        number of thermal iterations to average for distance-based correlations (default: 5)
  --therm-fluct THERM_FLUCT
                        thermal fluctuation of distances in the protein bundle -> scaling factor for added random noise (default: 0.5)
  --loop LOOP LOOP      residue numbers of start & end of loop to exclude from analysis (default: none)
```

This functionality can be extended from Python by deriving the `pdbcor.cli.CLI` class 
and overriding `new_arg_parser()` to yield a parser with custom options.

## Python package 

Implement your own code in Python:

```python
from pdbcor import CorrelationExtraction

# Prepare correlation extractor (demo example)
a = CorrelationExtraction(
    "demo.pdb",
    residue_subset="backbone",
    nstates=2,
)

# Run calculation
a.calculate_correlation()
```

## Produced output

Whether running from the shell or via Python, 
outputs are combined in a folder created in the parent directory of the source PDB file.
This folder's name can be set to a custom value, but defaults to
`correlations_<name of structure file>`.

Outputs include distance and angular correlation results including:

* text file with averaged correlation parameters and individual state populations 
(`correlations_{backbone,sidechain,full}.txt`)
* lists of correlations between each pair of residues in machine-readable Feather format
(`{ang,dist}_corr_{backbone,sidechain,full}.feather`)
* correlation heatmaps (`heatmap_{ang,dist}_{backbone,sidechain,full}.png`)
* histograms of correlation parameters (`hist_{ang,dist}_{backbone,sidechain,full}.png`)
* sequential correlation parameter charts per residue (`seq_{ang,dist}_{backbone,sidechain,full}.png`)
* Chimera & PyMOL executables to visualize the multistate bundle with assigned clusters 
(`bundle_vis_{chimera,pymol}_{ang,dist}_{backbone,sidechain,full}.{py,pml}`)
* plots of features per residue colour-coded by cluster assignment 
(`angle_clusters/angles_{angle1}-{angle2}_resid_{resid1}-{resid2}.svg`)
* ZIP archives of all output (name identical to output folder with extension `.zip`)
