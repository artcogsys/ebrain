import os
os.chdir('/home/ed/Documents/Code/PYTHON/ebrain')

#from feature_models.feature_model import FeatureModel
from feature_models.gabor_wavelet_pyramid import GWP

import scipy
x=scipy.misc.imread('/home/ed/Documents/Code/PYTHON/ebrain/feature_models/cameraman.png')
fm=GWP()
fm.fit()


