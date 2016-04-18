# Encoding model demo 
# feature_model = Identity
# response_model = KernelRidgeRegression

#Set your ebrain base directory ***
import os
os.chdir('/home/ed/Documents/Code/PYTHON/ebrain')

import numpy as np
from matplotlib import pyplot as plt
from encoding_model.EncodingModel import EncodingModel
from feature_models.identity import Identity
from response_models.kernel_ridge_regression import KernelRidgeRegression

# Generate stimulus response pairs
n_samples, n_features, n_voxels = 90, 100, 5
rng = np.random.RandomState(0)
real_weights = rng.randn(n_features,n_voxels)
stim_train = rng.randn(n_samples, n_features) 
resp_train = np.dot(stim_train,real_weights)
stim_test = rng.randn(n_samples, n_features) 
resp_test = np.dot(stim_test,real_weights)


# Encoding model
fm=Identity() #Feature model
rm=KernelRidgeRegression() #Response model
em=EncodingModel(fm,rm)

# Fit encoding model
em.fit(stim_train,resp_train)

# Predict encoding model
resp_test_hat = em.predict(stim_test)

# Analyze encoding performance
# Row-wise Correlation Coefficient calculation for two 2D arrays:
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

# Get prediction / ground truth voxel correlations
R = np.diagonal(corr2_coeff(resp_test.T,resp_test_hat[0].T))
print 'encoding performance: ',np.mean(R),' (mean R)'

# Plot encoding performance Pyplot 
fig = plt.figure()
plt.plot(np.arange(len(R))+1,sorted(R, reverse=True))
fig.suptitle('encoding performance')
plt.xlabel('voxel')
yLab=plt.ylabel('R')
yLab.set_rotation(0)
plt.ylim(-1, 1)
plt.xscale('log')

