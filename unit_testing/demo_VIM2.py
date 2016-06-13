## Encoding model demo using the VIM-2 dataset

## Set your ebrain base directory ***
import os
os.chdir('/vol/ccnlab-scratch1/egrant/ebrain')

import numpy as np
import h5py
import skimage.transform
from encoding_models.encoding_model import EncodingModel
from feature_models.gabor_wavelet_pyramid import GaborWaveletPyramid
from feature_models.convolutional_neural_network import CNN
from feature_models.identity import Identity
from response_models.kernel_ridge_regression import KernelRidgeRegression
from encoding_models.ring_buffer import RingBuffer
from matplotlib import pyplot as plt
import tables
import scipy.io


## Import data from the VIM-2 dataset, ROI=V1 (region of interest)
## Dataset available from https://crcns.org/data-sets
## Dataset info from https://crcns.org/files/data/vim-2/crcns-vim-2-readme.pdf
print('Importing data (May take some time)')
f = tables.openFile('/vol/ccnlab-scratch1/egrant/ebrain/Data/VIM2/VoxelResponses_subject1.mat')
f.listNodes # Show all variables available
data_train = f.getNode('/rt')[:]
data_val = f.getNode('/rv')[:]
roi = f.getNode('/roi/v1lh')[:].flatten()
v1lh_idx = np.nonzero(roi==1)[0]
V1LHresp_train = data_train[v1lh_idx].T
V1LHresp_val = data_val[v1lh_idx].T
Stimuli= h5py.File('/vol/ccnlab-scratch1/egrant/ebrain/Data/VIM2/Stimuli.mat')
stim_train = Stimuli['st'][0::15,:,:,:].astype('float')/255.0 #15Hz
stim_val = Stimuli['sv'][0::15,:,:,:].astype('float')/255.0 #15Hz
#convert video to grayscale and resize
stim_train=0.21*stim_train[:,0,:,:]+0.72*stim_train[:,1,:,:]+0.07*stim_train[:,2,:,:]
stim_val=0.21*stim_val[:,0,:,:]+0.72*stim_val[:,1,:,:]+0.07*stim_val[:,2,:,:]
stim_train=np.transpose(stim_train,(1,2,0))
stim_val=np.transpose(stim_val,(1,2,0))
stim_train = skimage.transform.resize(stim_train, (64, 64))
stim_val = skimage.transform.resize(stim_val, (64, 64))
stim_train=np.transpose(stim_train,(2,0,1))
stim_val=np.transpose(stim_val,(2,0,1))
#Reshape
stim_train=np.reshape(stim_train,(stim_train.shape[0],stim_train.shape[1]*stim_train.shape[2]))
stim_val=np.reshape(stim_val,(stim_val.shape[0],stim_val.shape[1]*stim_val.shape[2]))
del Stimuli #Cleanup
del f
del data_train
del data_val


## Clean response data
print('Cleaning  data')
mask = (np.nan_to_num(V1LHresp_val) != 0 ).all(axis=0) | (np.nan_to_num(V1LHresp_train) != 0 ).all(axis=0)
V1LHresp_train=V1LHresp_train[:,mask]
V1LHresp_train[np.isnan(V1LHresp_train)]=0
V1LHresp_val=V1LHresp_val[:,mask]
V1LHresp_val[np.isnan(V1LHresp_val)]=0


## Select n random voxels for demo set
n_vox=V1LHresp_train.shape[1]   #set to 'V1LHresp_train.shape[1]' for all voxels
np.random.seed(0)
target_vox=np.random.randint(n_vox, size=n_vox)
V1LHresp_train=V1LHresp_train[:,target_vox]
V1LHresp_val=V1LHresp_val[:,target_vox]


## Define encoding model
print('Creating encoding model')
#weights='/vol/ccnlab-scratch1/egrant/ebrain/Data/vgg16_weights.h5' #path to CNN weights
#fm=CNN(weights) #CNN feature model
#fm=Identity() #Identity feature model
fm=GaborWaveletPyramid() #GWP feature model
rm=KernelRidgeRegression() #Response model
buff_size=5 #Size of ring buffer
em=EncodingModel(fm,rm,buff_size)


## Fit encoding model 
print('Training encoding model')
em.fit(stim_train,V1LHresp_train)


## Predict encoding model 
print('Predicting validation responses')
V1LHresp_val_hat=em.predict(stim_val)


## Align responses with buffere features
V1LHresp_val=V1LHresp_val[em.val_feature_idx[:,0]+buff_size-1,:] 


## Remove insignificant voxels
print('Analyzing encoding performance')
significant_vox=em.rm.H_0==False
V1LHresp_val=V1LHresp_val[:,np.squeeze(significant_vox)]


## Analyze encoding performance
#Get num of significant voxels
print '\nsigfniciant voxels:',n_vox-np.sum(em.rm.H_0),'/', n_vox, 'at alpha =',em.rm.alpha


## Row-wise Correlation Coefficient calculation for two 2D arrays:
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
R = np.diagonal(corr2_coeff(V1LHresp_val.T,V1LHresp_val_hat[0].T))


if V1LHresp_val_hat[0].shape[1]==0:
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
dPoints=V1LHresp_val.shape[0]
ranking=np.zeros(dPoints)
for i in range(0,dPoints):
    C=corr2_coeff (np.expand_dims(V1LHresp_val_hat[0][i,:],axis=1).T ,  V1LHresp_val)
    ranking[i]=np.sum(C>C[:,i])+1
print 'identification performance: ',np.mean(ranking),'/',V1LHresp_val.shape[0], '(mean ranking)'

# Online mode create video
em.mode='online'
import cv2
import cv2.cv as cv
import numpy as np
Stimuli= h5py.File('/vol/ccnlab-scratch1/egrant/ebrain/Data/VIM2/Stimuli.mat')
stim_fullres = Stimuli['sv']
counter=1
counter2=1
resp_size=V1LHresp_val_hat[0].shape[1]
resp_size_hat=ceil(sqrt(V1LHresp_val_hat[0].shape[1]))
resp=np.zeros((128,128,3))
filler=int(resp_size_hat**2-resp_size)
writer = cv2.VideoWriter('test1.avi',cv.CV_FOURCC('P','I','M','1'),25,(256,128))
for i in range(0,stim_fullres.shape[0]-1):
    s = stim_fullres[i,:,:,:].astype('uint8')
    s=np.transpose(s,(2,1,0))
    s=s[:,:,[2,1,0]]    
    if counter%15==0 or counter==1:  
        prediction=em.predict(np.reshape(stim_val[counter2-1,:],(1,stim_val.shape[1])))
        counter2=counter2+1
        if counter>(buff_size-1)*15:
            prediction=prediction[0]
            prediction=np.reshape(prediction,(1,resp_size))
            prediction=np.concatenate((prediction,np.zeros((1,filler))),axis=1)
            prediction=np.reshape(prediction,(resp_size_hat,resp_size_hat,1))
            prediction=(prediction-np.min(V1LHresp_val_hat[0]))/(np.max(V1LHresp_val_hat[0])-np.min(V1LHresp_val_hat[0]))
            prediction=skimage.transform.resize(prediction, (128, 128))
            prediction=prediction*255
            prediction=prediction.astype('uint8')            
            prediction=np.reshape(prediction,(128,128,1))
            resp=np.repeat(prediction,3,axis=2).astype('uint8')
    fram=np.concatenate((s,resp),axis=1).astype('uint8')
    counter=counter+1
    writer.write(fram)


 




