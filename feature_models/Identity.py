#Identity feature model

# NOTE: The feature models will likely be prepared outside the toolbox. The toolbox
# will provide a way to access them, e.g. by loading preexisting features or by 
# passing the input stimuli through a pretrained model. 

class Identity(object):
        
    def fit(self,X): #Fit model
        pass
    
    def predict(self,X): #Return predictions
        return X

 

 