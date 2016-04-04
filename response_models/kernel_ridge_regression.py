from response_models.response_model import ResponseModel
#from external.KernelRidge import KERNEL_RIDGE_REGRESSION
import numpy as np
import time
import warnings
import math
from scipy import stats
from scipy.stats import t


class KernelRidgeRegression(ResponseModel):
    
    def __init__(self):
        self.alpha=2.5e-4
        self.n=10
        self.k=3
        
    def ind2sub(self,array_shape, ind):
        ind[np.asarray(ind < 0)] = -1
        ind[np.asarray(ind >= array_shape[0]*array_shape[1])] = -1
        rows = (ind.astype('int') / array_shape[1])
        cols = ind % array_shape[1]
        return (rows, cols)
    
    def get_lambda(self,K,n): 
        s= np.linalg.svd(K,compute_uv=False)
        s = s[s>0]
        lam = np.full([n],np.nan)
        L = len(s)
        df = np.linspace(L, 1, n)
        M = np.mean(1.0/s)
        f = lambda df,lam: df-np.sum(  np.divide (s,(s+lam) )   )
        f_prime = lambda lam: np.sum(   np.divide(    s,np.square(s+lam)    )     )
    
        for i in range(0,n):
            
            if i==0:
                lam[i]=0
            else:
                lam[i]=lam[i-1]
                
            lam[i] = max(lam[i], (L / df[i] - 1) / M)        
            temp = f(df[i], lam[i])
            
            tic = time.time()        
            while abs(temp) > 1e-10:
                lam[i] = max(0, lam[i] - temp / f_prime(lam[i]))
                temp = f(df[i], lam[i])
                if time.time()-tic > 1:
                    warnings.warn("GET_LAMBDA did not converge.")
                    break
                
    
        return lam+0.0001 #Prevents singularness. ask Umut 
                
    
    def get_R_and_lambda(self,K,Y,k,n):
        d = Y.shape
        folds = list(xrange(k))
        Indices = np.kron(np.ones( math.ceil(float(d[0])/k)),folds)
        Indices = np.sort(Indices[:d[0]])
        lam = self.get_lambda(K,n)
        Y_hat = np.full([d[0],d[1],n],np.nan)
    
        for i in range(0,k):
            Train = (Indices<>i)
            S = np.sum(Train)
            N = np.full([S,d[1],n],np.nan)
            I = np.eye(S)
            Test = (Indices==i)
            foo = K[np.ix_(Train,Train)] 
            bar = Y[Train,:]
            for ii in range(0,n):
                N[:,:,ii]=np.linalg.solve( foo+np.multiply(lam[ii],I) , bar )
    
            Y_hat[Test, :, :] = np.reshape( np.dot(K[np.ix_(Test,Train)], np.reshape(N, (S, d[1] * n))) , (sum(Test), d[1], n ))
    
        R = np.full([d[1],n],np.nan)
        
        for i in range(0,n):
                C_1 = Y-np.mean(Y,axis=0)
                C_2 = Y_hat[:,:,i] - np.mean(Y_hat[:,:,i],axis=0)
                R[:,i] = np.divide(np.sum(np.multiply(C_1,C_2),axis=0) , np.multiply(np.sqrt(np.sum(C_1**2,axis=0)), np.sqrt(np.sum(C_2**2,axis=0)))) 
    
        return (R,lam)
    
    def get_BETA_hat(self,K, X, Y, lambda_hat):
        C        = np.unique(lambda_hat)
        d        = Y.shape 
        BETA_hat = np.full([d[0],d[1]],np.nan)  
        I        = np.eye(d[0])
    
        for i in range(0,len(C)):    
            BETA_hat[:,np.squeeze(lambda_hat==C[i])] = np.linalg.solve ( K + C[i]* I  ,  Y[:, np.squeeze(lambda_hat==C[i])])
    
        BETA_hat = np.dot(X.T,BETA_hat)   
    
    
        return BETA_hat
          
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
            R[:, :, i], LAMBDA[:, i] = self.get_R_and_lambda(K[i], Y, k, n)
         
    
            r_hat      = np.full([d[1], 1],np.nan)      
            lambda_hat = np.full([d[1], 1],np.nan)    
            X_hat      = np.full([d[1], 1],np.nan)    


        for i in range(0,d[1]):
            print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (b / c): %d / %d" % (i+1,d[1])
            r_hat[i] = np.amax( np.reshape(R,(d[1],n*L))[i,:])
            I = np.argmax( np.reshape(R,(d[1],n*L))[i,:] )
            I,J = self.ind2sub((n, L), np.asarray(I))
    
            lambda_hat[i] = LAMBDA[I, J];
            X_hat[i]      = J;
            BETA_hat = [0]*d[1]  
            
            H_0 = 1.0 - t.cdf(r_hat * np.sqrt((d[0] - 2.0) / (1.0 - r_hat** 2.0)), d[0] - 2.0) >= alpha;

            X_hat[H_0]      = np.nan;
            lambda_hat[H_0] = np.nan;

        for i in range(0,L):
            print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (c / c): %d / %d" % (i+1,L)
            BETA_hat[i]= self.get_BETA_hat(K[i], X[i], Y[:, np.squeeze(X_hat == i)], lambda_hat[X_hat == i])
        
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
            X[i][np.isinf(X[i])] = 0
            Y_hat[i] = np.dot(X[i],BETA_hat[i]) 
        return Y_hat
            