from feature_models.FeatureModel import FeatureModel
from feature_models.GWP import GWP

import scipy
x = scipy.misc.imread('cameraman.png')
fm = GWP()
fm.fit()


