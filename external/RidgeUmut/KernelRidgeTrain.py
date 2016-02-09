import numpy as np
from scipy import stats

alpha=1
k=3
n=10
X=[np.random.rand(20,300)]
Y=np.random.rand(20,30)

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
    
    
    
BETA_hat = [0]*d[1]  

tcdf(      (r_hat)   .*   sqrt(      (d(1) - 2)   ./   (1 - (r_hat) .^ 2)  )        , d(1) - 2)                  >= alpha;








H_0      = 1 - tcdf(double(r_hat) .* sqrt((d(1) - 2) ./ (1 - double(r_hat) .^ 2)), d(1) - 2) >= alpha;

X_hat(H_0)      = NaN;
lambda_hat(H_0) = NaN;

for index = 1 : L
    
    fprintf('TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (c / c): %d / %d\n', index, L);
    
    BETA_hat(X_hat == index) = subsref(get_BETA_hat(K{index}, X{index}, Y(:, X_hat == index), lambda_hat(X_hat == index)), substruct('()', {':'}));
    
end

end