#Identity feature model

# NOTE: The feature models will likely be prepared outside the toolbox. The toolbox
# will provide a way to access them, e.g. by loading preexisting features or by 
# passing the input stimuli through a pretrained model. 

from feature_models.feature_model import FeatureModel

class Identity(FeatureModel):
        
    def fit(self): #Fit model
        pass
    
    def predict(self,X): #Return predictions as in list
        return X

 

 