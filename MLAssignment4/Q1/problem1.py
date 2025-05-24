from sklearn import svm
import numpy as np
import scipy.io as sio
from scipy import ndimage
from matplotlib import pyplot as plt


data = sio.loadmat('/Users/simran/Documents/ML/Assignments/MLAssignment4/Q1/teapots.mat')
X = data['teapotImages']
mean = np.mean(X,axis=0)

plt.imshow(ndimage.rotate(mean[:].reshape(38,50),-90),cmap='gray')
plt.savefig('/Users/simran/Documents/ML/Assignments/MLAssignment4/Q1/mean.png')

centered_data = X - mean

covariance_matrix = np.cov(centered_data.T)

eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

eigenvectors = np.real(eigenvectors)
top_eigenvectors = eigenvectors[:, :3]

for i in range(3):
    EV = top_eigenvectors[:, i]
    plt.imshow(ndimage.rotate(EV.reshape(50, 38),-90), cmap='gray')
    plt.savefig('/Users/simran/Documents/ML/Assignments/MLAssignment4/Q1/' + f'Eigenvector {i+1}')


# Calculate the coeff matrix 
coefficient_matrix = np.dot(centered_data, top_eigenvectors)

reconstructed_data = np.dot(coefficient_matrix, top_eigenvectors.T) + mean

fig, axes = plt.subplots(2, 10, figsize=(15, 5))
for i in range(10):
    idx = np.random.randint(0,100)
    axes[0, i].imshow(ndimage.rotate(X[idx, :].reshape(50,38),-90), cmap='gray')
    axes[0, i].set_title(f'Original {i+1}')
    axes[0, i].axis('off')

    axes[1, i].imshow(ndimage.rotate(reconstructed_data[idx, :].reshape(50,38),-90), cmap='gray')
    axes[1, i].set_title(f'Reconstructed {i+1}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('/Users/simran/Documents/ML/Assignments/MLAssignment4/Q1/Comparison.png')

print(np.linalg.norm(X-reconstructed_data,ord=2))