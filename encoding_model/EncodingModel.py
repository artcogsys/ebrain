# Encoding model class

# Import required feature and response models ***
from feature_models.identity import Identity
from response_models.kernel_ridge_regression import KernelRidgeRegression

class EncodingModel(object):
    
    # Select feature and response models ***
    def __init__(self):
        self.fm = Identity()
        self.rm = KernelRidgeRegression()
    
    # Train encoding model
    def fit(self,stimulus,response):
        self.fm.fit(stimulus) #Fit feature model
        self.train_feature=self.fm.predict(stimulus) #Predict train features
        self.rm.fit(self.train_feature,response) #Fit response model

    # Predict encoding model
    def predict(self,stimulus):  
        self.test_feature=self.fm.predict(stimulus) #Predict test features
        return self.rm.predict(self.test_feature); #Predict test response
        

        
    