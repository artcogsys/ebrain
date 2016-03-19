import os
os.chdir('/home/ed/Documents/Code/PYTHON/ebrain')

from feature_models.feature_model import FeatureModel
from feature_models.gabor_wavelet_pyramid import GaborWaveletPyramid

import scipy
fm=GaborWaveletPyramid()
fm.fit()

import numpy as np
from scipy import misc
arr = misc.imread('/home/ed/camman.tif').astype('float32')
arr=arr/255
arr=np.reshape(arr,(128*128,1)).T

p=fm.predict(arr)