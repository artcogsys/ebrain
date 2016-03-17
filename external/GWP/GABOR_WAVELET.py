#Gabor Wavelet Function
import numpy as np

def GABOR_WAVELET(b, FOV, gamma, lam, phi, sigma, theta, x_0, y_0):
        
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