# Kernal ridge regression - response model
# Fore more info see:
# http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html

from sklearn.kernel_ridge import KernelRidge
class KernelRidgeRegression(object):

    def __init__(self):  #Define model
        alpha_=1.0
        coef0_=1
        degree_=3
        gamma_=None
        kernel_='linear'
        kernel_params_=None
        self.model = KernelRidge(alpha=alpha_, coef0=coef0_, degree=degree_, 
               gamma=gamma_,kernel=kernel_, kernel_params=kernel_params_)
                      
    def fit(self,X,Y): #Fit model
        self.model.fit(X, Y) 

    def predict(self,X): # Return predictions 
        return self.model.predict(X)
