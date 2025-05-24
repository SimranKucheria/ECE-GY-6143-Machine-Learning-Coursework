ReadMe for Problem 2.

Code files

problem2.py - Main file. Run this to produce all graphs. Iterates through different values of C to find most optimum value for linear kernel, c and degree for polynomial kernel, c and sigma for bf kernel. Performs data cross validation by splitting data into train and test datasets.


Graphs

Linear Kernel Graphs
CValuesAlphaLinearKernel.png - Plotting Alpha with varying C Values.
CValuesErrorLinearKernel.png - Plotting Error with varying C Values.
CValuesNSVLinearKernel.png - Plotting Number of support vectors with varying C Values.


Polynomial Kernel Graphs
CValuesAlphaPolyKernelDegree[degree].png - Plotting Alpha with varying C Values while keeping degree of polynomial fixed to [degree].
CValuesErrorPolyKernelDegree[degree].png - Plotting Error with varying C Values while keeping degree of polynomial fixed to [degree].
CValuesNSVPolyKernelDegree[degree].png - Plotting Number of support vectors with varying C Values while keeping degree of polynomial fixed to [degree].

DegreeValuesAlphaPolyKernel[c].png - Plotting Alpha with varying degree Values while keeping C of polynomial fixed to [c].
DegreeValuesErrorPolyKernel[c].png - Plotting Error with varying degree Values while keeping C of polynomial fixed to [c].
DegreeValuesNSVPolyKernel[c].png - Plotting Number of support vectors with varying degree Values while keeping C of polynomial fixed to [c].

RBF Kernel Graphs

CValuesAlphaRBFKernelSigma[sigma].png - Plotting Alpha with varying C Values while keeping sigma of rbf fixed to [sigma].
CValuesErrorPolyKernelDegree[degree].png - Plotting Error with varying C Values while keeping sigma of rbf fixed to [sigma].
CValuesNSVPolyKernelDegree[degree].png - Plotting Number of support vectors with varying C Values while keeping sigma of rbf fixed to [sigma].

SigmaValuesAlphaRBFKernel[c].png - Plotting Alpha with varying sigma Values while keeping C of dbf kernel fixed to [c].
SigmaValuesErrorRBFKernel[c].png - Plotting Error with varying sigma Values while keeping C of dbf kernel fixed to [c].
SigmaValuesNSVRBFKernel[c].png - Plotting Number of support vectors with varying sigma Values while keeping C of ref kernel fixed to [c].

Dataset
dataset.mat
