ReadMe for Problem 1.

Code files

problem1.py - Main file. Run this to produce all images. Calculates mean of the dataset along with covariance matrix on the centred data to further calculate the Eigen values and eigenvectors. Sorts the eigenvalues to pick the top eigenvectors and uses that to create a coefficient matrix that will be used for PCA reconstruction. CoefficientMatrix . Centred Data + Mean is used for reconstruction. We pick 10 random images to illustrate the difference between the original and reconstructed images. Whilst the top 3 eigenvectors are able to encode a lot of the information, this performance can be improved by using more Eigen vectors. The performance is further evaluated by calculating the L2 norm between the original and reconstructed images.


Graphs

Mean.png - The image representation of the calculated mean.
Eigenvector %d.png - The image representation of the top 3 Eigenvectors.
Comparison.png - Compare 10 random images of the dataset along with their reconstruction.

Dataset
teapots.mat
