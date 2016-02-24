# Encoding model demo using the VIM-1 dataset

#Set your ebrain base directory ***
import os
os.chdir('/home/ed/Documents/Code/ebrain')

import numpy as np
from encoding_model.EncodingModel import EncodingModel
from matplotlib import pyplot as plt
import tables
import scipy.io


# Import data from the VIM-1 dataset, ROI=V1 (region of interest)
# Dataset available from https://crcns.org/data-sets
# Dataset info from https://crcns.org/files/data/vim-1/crcns-vim-1-readme.pdf
EstimatedResponses = tables.open_file('/home/ed/Documents/Code/ebrain/Data/EstimatedResponses.mat')
Stimuli = scipy.io.loadmat('/home/ed/Documents/Code/ebrain/Data/Stimuli.mat',struct_as_record=True)
data_train = EstimatedResponses.get_node('/dataTrnS1')[:].astype('float64')
data_val = EstimatedResponses.get_node('/dataValS1')[:].astype('float64')
ROI = EstimatedResponses.get_node('/roiS1')[:].flatten()
V1idx = np.nonzero(ROI==1)[0] #ROI 

#remove nan voxels
V1resp_train = data_train[:,V1idx]
V1resp_val = data_val[:,V1idx]
mask = (np.nan_to_num(V1resp_val) != 0 ).all(axis=0) | (np.nan_to_num(V1resp_train) != 0 ).all(axis=0)
V1resp_train=V1resp_train[:,mask]
V1resp_val=V1resp_val[:,mask]
#V1resp_train[np.isnan(V1resp_train)] = 0 #nan to zero
#V1resp_val[np.isnan(V1resp_val)] = 0


stim_train = Stimuli["stimTrn"]
stim_train = np.reshape(stim_train,[stim_train.shape[0],stim_train.shape[1]*stim_train.shape[2]],order="F")
stim_val = Stimuli["stimVal"]
stim_val = np.reshape(stim_val,[stim_val.shape[0],stim_val.shape[1]*stim_val.shape[2]],order="F")

# Select n random voxels for demo set to 
#n_vox=V1resp_train.shape[1]   #set to 'V1resp_train.shape[1]' for all
#np.random.seed(3000)
#target_vox=np.random.randint(n_vox, size=n_vox)
#V1resp_train=V1resp_train[:,target_vox]
#V1resp_val=V1resp_val[:,target_vox]


# Encoding model
em=EncodingModel()

# Fit and predict features
em.fit_feature_model(stim_train.astype('float64'))
feature_train=em.predict_feature_model(stim_train.astype('float64'))
feature_val=em.predict_feature_model(stim_val.astype('float64'))
feature_train=stim_train.astype('float64')
feature_val=stim_val.astype('float64')

# Fit and predict response
em.fit_response_model(feature_train.astype('float64'),V1resp_train.astype('float64'))
V1resp_val_hat = em.predict_response_model(feature_val.astype('float64'))
V1resp_train_hat = em.predict_response_model(feature_train.astype('float64'))


# Remove insignificant voxels
significant_vox=em.rm.model.H_0==False
V1resp_val=V1resp_val[:,np.squeeze(significant_vox)]
V1resp_train=V1resp_train[:,np.squeeze(significant_vox)]

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
R = np.diagonal(corr2_coeff(V1resp_val.T,V1resp_val_hat[0].T))

if V1resp_val_hat[0].shape[1]==0:
    print 'No responses can be predicted at this alpha setting'
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
    

dPoints=V1resp_val.shape[0]
ranking=np.zeros(dPoints)
for i in range(0,dPoints):
    C=corr2_coeff (np.expand_dims(V1resp_val_hat[0][i,:],axis=1).T ,  V1resp_val)
    ranking[i]=np.sum(C>C[:,i])+1
print 'Identification performance: ',np.mean(ranking),'/ 120 (mean ranking)'
## Plot encoding performance Bokeh (Nicer but may require $ pip install bokeh)
#from bokeh.plotting import figure, output_file, show
#output_file("encoding_performance.html", title="encoding performance")
#p = figure(title="econding performance", x_axis_label='voxel', 
#            y_range=[-1, 1], y_axis_label='R', x_axis_type="log", x_range=[1, len(R)])
#p.line(np.arange(len(R))+1,sorted(R, reverse=True), line_width=2)
#show(p)