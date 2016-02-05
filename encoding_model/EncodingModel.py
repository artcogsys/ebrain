# Encoding model class

# Import required feature and response models ***
from feature_models.Identity import Identity
from response_models.KernelRidgeRegression import KernelRidgeRegression

class EncodingModel(object):
    
    # Select feature and response models ***
    def __init__(self):
        self.fm = Identity()
        self.rm = KernelRidgeRegression()
    
    # Train feature model
    def fit_feature_model(self,stimulus):
        self.fm.fit(stimulus)

    # Return feature model predictions
    def predict_feature_model(self,stimulus):
        return self.fm.predict(stimulus)

    # Train response model
    def fit_response_model(self,feature,response):
        self.rm.fit(feature,response)
        
    # Return response model predictions
    def predict_response_model(self,feature):
        return self.rm.predict(feature);
    