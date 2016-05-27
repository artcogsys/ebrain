# Encoding model demo using the VIM-1 dataset

#Set your ebrain base directory ***
import os
os.chdir('/vol/ccnlab-scratch1/egrant/ebrain')

import numpy as np
from encoding_models.encoding_model import EncodingModel
from feature_models.gabor_wavelet_pyramid import GaborWaveletPyramid
from feature_models.convolutional_neural_network import CNN
from feature_models.identity import Identity
from response_models.kernel_ridge_regression import KernelRidgeRegression
from matplotlib import pyplot as plt
import tables
import scipy.io


# Import data from the VIM-1 dataset, ROI=V1 (region of interest)
# Dataset available from https://crcns.org/data-sets
# Dataset info from https://crcns.org/files/data/vim-1/crcns-vim-1-readme.pdf
EstimatedResponses = tables.open_file('/vol/ccnlab-scratch1/egrant/ebrain/Data/EstimatedResponses.mat')
Stimuli = scipy.io.loadmat('/vol/ccnlab-scratch1/egrant/ebrain/Data/Stimuli.mat',struct_as_record=True)
data_train = EstimatedResponses.get_node('/dataTrnS1')[:].astype('float64')
data_val = EstimatedResponses.get_node('/dataValS1')[:].astype('float64')
ROI = EstimatedResponses.get_node('/roiS1')[:].flatten()
V1idx = np.nonzero(ROI==1)[0] #ROI 

#remove nan voxels
V1resp_train = data_train[:,V1idx]
V1resp_val = data_val[:,V1idx]
mask = (np.nan_to_num(V1resp_val) != 0 ).all(axis=0) | (np.nan_to_num(V1resp_train) != 0 ).all(axis=0)
V1resp_train=V1resp_train[:,mask]
V1resp_train[np.isnan(V1resp_train)]=0
V1resp_val=V1resp_val[:,mask]
V1resp_val[np.isnan(V1resp_val)]=0

stim_train = Stimuli["stimTrn"].astype('float64')+0.55
stim_train = np.reshape(stim_train,[stim_train.shape[0],stim_train.shape[1]*stim_train.shape[2]],order="F")
stim_val = Stimuli["stimVal"].astype('float64')+0.55
stim_val = np.reshape(stim_val,[stim_val.shape[0],stim_val.shape[1]*stim_val.shape[2]],order="F")

# Select n random voxels for demo set
n_vox=V1resp_train.shape[1]   #set to 'V1resp_train.shape[1]' for all voxels
np.random.seed(0)
target_vox=np.random.randint(n_vox, size=n_vox)
V1resp_train=V1resp_train[:,target_vox]
V1resp_val=V1resp_val[:,target_vox]

# Define encoding model
weights='/vol/ccnlab-scratch1/egrant/ebrain/Data/vgg16_weights.h5' #path to CNN weights
#fm=CNN(weights) #CNN feature model
#fm=Identity() #Identity feature model
fm=GaborWaveletPyramid() #GWP feature model
rm=KernelRidgeRegression() #Response model
em=EncodingModel(fm,rm)

# Fit encoding model 
em.fit(stim_train,V1resp_train)

# Predict encoding model 
V1resp_val_hat=em.predict(stim_val)

# Remove insignificant voxels
significant_vox=em.rm.H_0==False
V1resp_val=V1resp_val[:,np.squeeze(significant_vox)]
V1resp_train=V1resp_train[:,np.squeeze(significant_vox)]

# Analyze encoding performance

#Get num of significant voxels
print '\nsigfniciant voxels:',n_vox-np.sum(em.rm.H_0),'/', n_vox, 'at alpha =',em.rm.alpha

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
R = np.diagonal(corr2_coeff(V1resp_val.T,V1resp_val_hat[0].T))

if V1resp_val_hat[0].shape[1]==0:
    print 'no responses can be predicted at this alpha setting'
else:
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
    
# Get identification accuracy
dPoints=V1resp_val.shape[0]
ranking=np.zeros(dPoints)
for i in range(0,dPoints):
    C=corr2_coeff (np.expand_dims(V1resp_val_hat[0][i,:],axis=1).T ,  V1resp_val)
    ranking[i]=np.sum(C>C[:,i])+1
print 'identification performance: ',np.mean(ranking),'/ 120 (mean ranking)'
