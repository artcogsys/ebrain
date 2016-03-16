import numpy as np
from scipy import stats
from scipy.stats import t
from utils import ind2sub
from utils import get_lambda
from utils import get_R_and_lambda
from utils import get_BETA_hat


class KERNEL_RIDGE_REGRESSION_obj(object):

    def __init__(self,alpha,k,n):
        self.alpha=alpha
        self.k=k
        self.n=n
    
    def fit(self,X,Y):
        alpha=self.alpha
        k=self.k
        n=self.n
        if isinstance(X,list) is False: #convert single feature to list
            tmpX=[0]
            tmpX[0]=X
            X=tmpX
            del tmpX
            
        L      = len(X)
        MU     = [0]*L   
        SIGMA  = [0]*L
        K      = [0]*L
        LAMBDA = np.full([n, L],np.nan)
        d      = Y.shape
        R      = np.full([d[1],n, L],np.nan)  

        for i in range(0,L):
            print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (a / c): %d / %d" % (i+1,L)
     

            MU[i] = np.mean(X[i], axis=0)
            SIGMA[i] = np.std(X[i], axis=0)
            X[i] = stats.mstats.zscore(X[i], axis=0)
            X[i][np.isnan(X[i])] = 0  #Correct for division by zero
            K[i] = np.dot(X[i],X[i].T)
            R[:, :, i], LAMBDA[:, i] = get_R_and_lambda(K[i], Y, k, n)
         
    
            r_hat      = np.full([d[1], 1],np.nan)      
            lambda_hat = np.full([d[1], 1],np.nan)    
            X_hat      = np.full([d[1], 1],np.nan)    


        for i in range(0,d[1]):
            print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (b / c): %d / %d" % (i+1,d[1])
            r_hat[i] = np.amax( np.reshape(R,(d[1],n*L))[i,:])
            I = np.argmax( np.reshape(R,(d[1],n*L))[i,:] )
            I,J = ind2sub((n, L), np.asarray(I))
    
            lambda_hat[i] = LAMBDA[I, J];
            X_hat[i]      = J;
            BETA_hat = [0]*d[1]  
            
            H_0 = 1.0 - t.cdf(r_hat * np.sqrt((d[0] - 2.0) / (1.0 - r_hat** 2.0)), d[0] - 2.0) >= alpha;

            X_hat[H_0]      = np.nan;
            lambda_hat[H_0] = np.nan;

        for i in range(0,L):
            print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (c / c): %d / %d" % (i+1,L)

    
            BETA_hat[i]= get_BETA_hat(K[i], X[i], Y[:, np.squeeze(X_hat == i)], lambda_hat[X_hat == i])
        
        self.BETA_hat =BETA_hat
        self.H_0 =H_0
        self.MU = MU
        self.SIGMA =SIGMA
        self.X_hat =X_hat
        self.lambda_hat =lambda_hat
        self.r_hat=r_hat

    def predict(self,X):
        BETA_hat = self.BETA_hat
        MU = self.MU
        SIGMA = self.SIGMA 

        
        if isinstance(X,list) is False: #convert single feature to list
            tmpX=[0]
            tmpX[0]=X
            X=tmpX
            del tmpX
            
        L = len(X)
        Y_hat = [0]*L
        for i in range(0,L):
            print "PREDICT_LINEAR_KERNEL_RIDGE_REGRESSION: %d / %d" % (i+1,L)
            X[i]=(X[i]-MU[i])/SIGMA[i]
            X[i][np.isnan(X[i])] = 0
            Y_hat[i] = np.dot(X[i],BETA_hat[i]) 
        return Y_hat