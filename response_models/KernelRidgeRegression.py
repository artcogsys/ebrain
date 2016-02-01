# Kernal ridge regression - response model
# Fore more info see:
# http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html

from response_models.ResponseModel import ResponseModel
from sklearn.kernel_ridge import KernelRidge

class KernelRidgeRegression(ResponseModel):

    def __init__(self):  #Define model
        self.alpha_=1.0
        self.coef0_=1
        self.degree_=3
        self.gamma_=None
        self.kernel_='linear'
        self.kernel_params_=None
        self.model = KernelRidge(alpha=self.alpha_,coef0=self.coef0_, 
                                 degree=self.degree_,gamma=self.gamma_,
                                 kernel=self.kernel_,kernel_params=self.kernel_params_)
                      
    def fit(self,X,Y): #Fit model
        self.model.fit(X, Y) 

    def predict(self,X): # Return predictions 
        return self.model.predict(X)
