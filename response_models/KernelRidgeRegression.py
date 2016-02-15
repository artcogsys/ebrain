from response_models.ResponseModel import ResponseModel
from external.KernelRidge import KERNEL_RIDGE_REGRESSION

class KernelRidgeRegression(ResponseModel):
    
    def __init__(self):
        alpha=1
        n=3
        k=10
        self.model=KERNEL_RIDGE_REGRESSION(alpha,k,n)
          
    def fit(self,X,Y):
        self.model.fit(X,Y)
            
    def predict(self,X):
        return self.model.predict(X)
            