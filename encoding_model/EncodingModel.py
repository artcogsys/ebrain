# Encoding model class

# Import required feature and response models ***
from feature_models.identity import Identity
from feature_models.gabor_wavelet_pyramid import GaborWaveletPyramid
from feature_models.convolutional_neural_network import CNN
from response_models.kernel_ridge_regression import KernelRidgeRegression

class EncodingModel(object):
    
    # Select feature and response models ***
    def __init__(self,fm,rm):
        self.fm = fm       
        self.rm = rm
    
    # Train encoding model
    def fit(self,stimulus,response):
        self.fm.fit() #Fit feature model
        #Predict features (get first set if feature maps > 1)
        self.train_feature=self.fm.predict(stimulus)
        if isinstance(self.train_feature, list):
            self.train_feature=self.train_feature[1]
        self.rm.fit(self.train_feature,response) #Fit response model

    # Predict encoding model
    def predict(self,stimulus):  
        #Predict features (get first set if feature maps > 1)
        self.test_feature=self.fm.predict(stimulus)
        if isinstance(self.test_feature, list):
            self.test_feature=self.test_feature[1]
        return self.rm.predict(self.test_feature); #Predict test response
        

        
    