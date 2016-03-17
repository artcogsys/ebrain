#Gabor Wavelet Pyramid feature model

# NOTE: The feature models will likely be prepared outside the toolbox. The toolbox
# will provide a way to access them, e.g. by loading preexisting features or by 
# passing the input stimuli through a pretrained model. 

from feature_models.feature_model import FeatureModel
import numpy as np
import scipy

class GaborWaveletPyramid(FeatureModel):
    
    def __init__(self):      
        self.b      = np.array([1])
        self.FOV    = np.linspace(-63.5,63.5,128)
        self.gamma  = 1
        self.lam    = 2**np.linspace(2,7,6)
        self.sigma  = np.array([])
        self.theta  = np.linspace(0,7*np.pi/8,8)
        
    def GABOR_WAVELET(self,b, FOV, gamma, lam, phi, sigma, theta, x_0, y_0):           
        [X, Y] = np.meshgrid(FOV,FOV);
        if ~np.logical_xor(b.size==0, sigma.size==0):
            raise ValueError('~np.logical_xor(b.size==0, sigma.size==0)')    
        elif b.size==0:
            b = (sigma / lam * np.pi + np.sqrt(np.log(2) / 2)) / (sigma / lam * np.pi - np.sqrt(np.log(2) / 2))
            output_arg = b       
        elif sigma.size==0:            
            sigma = lam * 1 / np.pi * np.sqrt(np.log(2) / 2) * (2 ** b + 1) / (2 ** b - 1)
            output_arg = sigma;     
        X_prime =  (X - x_0) * np.cos(theta) + (Y + y_0) * np.sin(theta)
        Y_prime = -(X - x_0) * np.sin(theta) + (Y + y_0) * np.cos(theta)
        G = np.multiply(np.exp(-(np.power(X_prime,2) + gamma ** 2 * np.power(Y_prime,2)) /
            (2 * sigma **2)), np.cos(2*np.pi*X_prime/lam+phi))
        return G, output_arg
    
    def GABOR_WAVELET_PYRAMID(self,b, FOV, gamma, lam, sigma, theta):
        numberOfElements_1 = len(FOV)
        numberOfElements_2 = len(theta)
        numberOfElements_3 = np.divide(numberOfElements_1,lam).astype('int')
        G                  = [0]*2
        G[0]               = np.zeros((np.sum(numberOfElements_2 * np.power(numberOfElements_3,2)).astype('int'), 
                                        numberOfElements_1 * numberOfElements_1))        
        G[1]               = np.zeros(G[0].shape)
        i1                 = 0            
        for i2 in range (0 , len(lam)):
            x_0 = np.linspace(FOV[0], FOV[-1], numberOfElements_3[i2])
            print(i2)
            for i3 in range (0, numberOfElements_2):   
                for i4 in range ( 0 , numberOfElements_3[i2]):
                    for i5 in range( 0 , numberOfElements_3[i2]):
                        G[0][i1,:] = np.ndarray.flatten(self.GABOR_WAVELET(b,FOV,gamma,lam[i2],0,sigma,theta[i3],x_0[i4],x_0[i5])[0])
                        G[1][i1,:] = np.ndarray.flatten(self.GABOR_WAVELET(b,FOV,gamma,lam[i2],np.pi/2,sigma,theta[i3],x_0[i4],x_0[i5])[0])           
                        i1         = i1 + 1
    
        return G
        
    def fit(self): #Fit model
        self.G = self.GABOR_WAVELET_PYRAMID(self.b, self.FOV, self.gamma, self.lam, self.sigma, self.theta)
    
    def predict(self,X): #Return predictions  
        g=[0]*2
        g[0]=np.reshape(self.G[0],(self.G[0].shape[0],self.G[0].shape[1]))
        g[1]=np.reshape(self.G[1],(g[0].shape))
        #reshape images for resize, rows=images
        X=np.reshape(X,(np.sqrt(X.shape[1]),np.sqrt(X.shape[1]),X.shape[0]))
        #resize images      
        X=skimage.transform.resize(X,(len(self.FOV),len(self.FOV)))
        x=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
        
        #Simple cell responses
        s=np.dot(g[0],x) 
        
        #Complex cell responses
        c=np.sqrt( np.power(np.dot(g[0],x),2) + np.power(np.dot(g[1],x),2))
        
        return s,c



