import numpy as np
import time
import warnings
import math


def ind2sub(array_shape, ind):
    ind[np.asarray(ind < 0)] = -1
    ind[np.asarray(ind >= array_shape[0]*array_shape[1])] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def get_lambda(K,n): 
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
            

    return lam+0.00001 #Prevents singularness. ask Umut 
            

def get_R_and_lambda(K,Y,k,n):
    d = Y.shape
    folds = list(xrange(k))
    Indices = np.kron(np.ones( math.ceil(float(d[0])/k)),folds)
    Indices = np.sort(Indices[:d[0]])
    lam = get_lambda(K,n)
    Y_hat = np.full([d[0],d[1],n],np.nan)

    for i in range(0,k):
        Train = (Indices<>i)
        S = np.sum(Train)
        N = np.full([S,d[1],n],np.nan)
        I = np.eye(S)
        Test = (Indices==i)
        foo = K[Train,Train]
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

def get_BETA_hat(K, X, Y, lambda_hat):

    C        = np.unique(lambda_hat)
    d        = Y.shape 
    BETA_hat = np.full([d[0],d[1]],np.nan)  
    I        = np.eye(d[0])

    for i in range(0,len(C)):    
        BETA_hat[:,np.squeeze(lambda_hat==C[i])] = np.linalg.solve ( K + np.dot(C[i], I)  ,  Y[:, np.squeeze(lambda_hat==C[i])])

    BETA_hat = np.dot(X.T,BETA_hat)   


    return BETA_hat