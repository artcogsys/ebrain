#Gabor Wavelet Pyramid feature model

# NOTE: The feature models will likely be prepared outside the toolbox. The toolbox
# will provide a way to access them, e.g. by loading preexisting features or by 
# passing the input stimuli through a pretrained model. 

from feature_models.FeatureModel import FeatureModel
from external.GWP.GABOR_WAVELET_PYRAMID import GABOR_WAVELET_PYRAMID
import numpy as np
import scipy

class GWP(FeatureModel):
    
    def __init__(self):      
        self.b      = np.array([1])
        self.FOV    = np.linspace(-63.5,63.5,128)
        self.gamma  = 1
        self.lam    = 2**np.linspace(2,7,6)
        self.sigma  = np.array([])
        self.theta  = np.linspace(0,7 * np.pi/8, 8)
        
    def fit(self): #Fit model
        self.G = GABOR_WAVELET_PYRAMID(self.b, self.FOV, self.gamma, self.lam, self.sigma, self.theta)
    
    def predict(self,X): #Return predictions  
        g=[0]*2
        print(self.G[0].shape[0])
        g[0] = np.reshape(self.G[0], (self.G[0].shape[0], self.G[0].shape[1]))
        g[1] = np.reshape(self.G[1], (g[0].shape))
        X = scipy.misc.imresize(X, [len(self.FOV), len(self.FOV)])
        x = np.reshape(X, (X.shape[0] * X.shape[1],1))
        
        #Simple cell responses
        s=np.dot(g[0],x) 
        
        #Complex cell responses
        c=np.sqrt( np.power(np.dot(g[0], x) ,2) + np.power(np.dot(g[1], x), 2))
        
        return s,c



