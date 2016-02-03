# CV ridge regression - response model


from response_models.ResponseModel import ResponseModel
from sklearn.linear_model import RidgeCV

class RidgeRegressionCV(ResponseModel):

    def __init__(self):  #Define model
        self.model = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, 
                             normalize=False, scoring=None, cv=3, 
                             gcv_mode=None, store_cv_values=False)
                      
    def fit(self,X,Y): #Fit model
        self.model.fit(X, Y) 

    def predict(self,X): # Return predictions 
        return self.model.predict(X)
