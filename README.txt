README

What is PDBcor? 

	PDBcor is an automated and unbiased method for the detection and analysis of 
	correlated motions from experimental multi-state protein structures using torsion angle and 
	distance statistics that does not require any structure superposition. Clustering of protein 
	conformers allows us to extract correlations in the form of mutual information based on information 
	theory. Correlations extracted with PDBcor can be utilized in subsequent assays including NMR 
	multi-state structure optimization and validation. Further information is available in the reference 
	publication under https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3904349 

How does PDBcor work?

	An input structure bundle (supplied PDB) is subjected
	to the significance thresholding that filters out spurious insignificant correlations. 
	We assume that for the proteins existing in 2 states, distances center around 2 state-specific values. 
	During significance thresholding random displacement of atoms broadens the edges
	of states so that states separated by less than the amplitude of the noise loose separation.
	Then, interresidual distances are used to cluster conformers for each residue with GMM (Gaussian Mixture Model).
	In PDBcor, distances from the selected residue to all other residues are considered for the clustering. 
	By repeating this procedure for each residue a set of N clustering vectors is obtained.
	Finally, a pairwise comparison of the resulting clustering vectors based on their mutual information yields an
	interpretable correlation matrix.
	
What can PDBcor results indicate?

	Correlation matrix indicates statistically significant correlations that stand out from the background.
	There are two types of correlation matrices: distance correlation matrix and angular correlation matrix.
	The difference between them is that distance correlation matrix takes distances as input whereas angular
	correlation matrix takes dihedral angles as input. Both distance and angular correlation analyses are
	able to detect correlated motion. Nevertheless, distance correlation extraction is more sensitive to the
	protein motion.
	
Required software

	Python3
	
Installation instruction

	Set up a new python environment:

	-> python3 -m venv venv
	-> source venv/bin/activate
	-> pip install -r requirements.txt
	
Usage

	-> python correlationExtraction.py demo.pdb
	
Help and parameter desctiption

	-> python correlationExtraction.py --help

CONTACT 

	Prof. Dr. Roland Riek, roland.riek@ethz.ch
