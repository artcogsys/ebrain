# Encoding model demo 
# feature_model = Identity
# response_model = KernelRidgeRegression

import numpy as np
from matplotlib import pyplot as plt
import tables
import scipy.io

# Import models
from feature_models.Identity import Identity
from response_models.KernelRidgeRegression import KernelRidgeRegression
from response_models.RidgeRegressionCV import RidgeRegressionCV

# Import data from the VIM-1 dataset, ROI=V1
# Dataset available from https://crcns.org/data-sets
# Dataset info from https://crcns.org/files/data/vim-1/crcns-vim-1-readme.pdf
EstimatedResponses = tables.open_file('/home/ed/Documents/Code/ebrain/Data/EstimatedResponses.mat')
Stimuli = scipy.io.loadmat('/home/ed/Documents/Code/ebrain/Data/Stimuli.mat',struct_as_record=True)
data_train = EstimatedResponses.get_node('/dataTrnS1')[:]
data_val = EstimatedResponses.get_node('/dataValS1')[:]
ROI = EstimatedResponses.get_node('/roiS1')[:].flatten()
V1idx = np.nonzero(ROI==1)[0]
V1resp_train = data_train[:,V1idx]
V1resp_val = data_val[:,V1idx]
stim_train = Stimuli["stimTrn"]
stim_train = np.reshape(stim_train,[stim_train.shape[0],stim_train.shape[1]*stim_train.shape[2]])
stim_val = Stimuli["stimVal"]
stim_val = np.reshape(stim_val,[stim_val.shape[0],stim_val.shape[1]*stim_val.shape[2]])

# Select n random voxels for demo
n_vox=10
np.random.seed(0)
target_vox=np.random.randint(len(V1idx), size=n_vox)
V1resp_train=V1resp_train[:,target_vox]
V1resp_val=V1resp_val[:,target_vox]

# Define feature model
fm = Identity()

# Train feature model
fm.fit(stim_train)

# Simulate feature model
feature_train = fm.predict(stim_train)
feature_val = fm.predict(stim_val)

# Define response model
rm = RidgeRegressionCV()
    
# Train response model
rm.fit(feature_train, V1resp_train)

# Simulate response model
V1resp_val_hat = rm.predict(feature_val);

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
R = np.diagonal(corr2_coeff(V1resp_val.T,V1resp_val_hat.T))
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

## Plot encoding performance Bokeh (Nicer but may require $ pip install bokeh)
#from bokeh.plotting import figure, output_file, show
#output_file("encoding_performance.html", title="encoding performance")
#p = figure(title="econding performance", x_axis_label='voxel', 
#            y_range=[-1, 1], y_axis_label='R', x_axis_type="log", x_range=[1, len(R)])
#p.line(np.arange(len(R))+1,sorted(R, reverse=True), line_width=2)
#show(p)