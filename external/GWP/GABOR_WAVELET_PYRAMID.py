#Gabor Wavelet Pyramid Function
import numpy as np
from external.GWP.GABOR_WAVELET import GABOR_WAVELET

def GABOR_WAVELET_PYRAMID(b, FOV, gamma, lam, sigma, theta):
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
                    G[0][i1,:] = np.ndarray.flatten(GABOR_WAVELET(b,FOV,gamma,lam[i2],0,sigma,theta[i3],x_0[i4],x_0[i5])[0])
                    G[1][i1,:] = np.ndarray.flatten(GABOR_WAVELET(b,FOV,gamma,lam[i2],np.pi/2,sigma,theta[i3],x_0[i4],x_0[i5])[0])           
                    i1         = i1 + 1

    return G
