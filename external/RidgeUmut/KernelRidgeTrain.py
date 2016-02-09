import numpy as np
from scipy import stats
from scipy.stats import t

alpha=1
k=3
n=10
X=[np.random.randn(2000,300).astype(float)]
rlwt=np.random.randn(300,30)
Y=np.dot(X[0],rlwt).astype(float) #   np.random.randn(2000,30)

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
    K[i] = np.dot(X[i],X[i].T)
    R[:, :, i], LAMBDA[:, i] = get_R_and_lambda(K[i], Y, k, n);
    
r_hat      = np.full([d[1], 1],np.nan)      
lambda_hat = np.full([d[1], 1],np.nan)    
X_hat      = np.full([d[1], 1],np.nan)    


for i in range(0,d[1]):
    print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (b / c): %d / %d" % (i+1,d[1])
    r_hat[i] = np.amax( np.reshape(R,(d[1],n*L))[i,:] )
    I = np.argmax( np.reshape(R,(d[1],n*L))[i,:] )
    I,J = ind2sub((n, L), np.asarray(I))
    
    lambda_hat[i] = LAMBDA[I, J];
    X_hat[i]      = J;
    

BETA_hat = [0]*L  

H_0=1-t.cdf(np.divide((r_hat),np.sqrt( np.divide ((d[0]-2), (1-(r_hat)**2)))) , d[0] - 2)>= alpha

X_hat[H_0]      = np.nan;
lambda_hat[H_0] = np.nan;

for i in range(0,L):
    print "'TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (c / c): %d / %d" % (i+1,L)

    
    BETA_hat[i]= get_BETA_hat(K[i], X[i], Y[:, np.squeeze(X_hat == i)], lambda_hat[X_hat == i])
        


