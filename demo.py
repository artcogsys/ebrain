# Encoding model demo 
# feature_model = Identity
# response_model = KernelRidgeRegression

import numpy as np
from matplotlib import pyplot as plt

# Import models
from feature_models.Identity import Identity
from response_models.KernelRidgeRegression import KernelRidgeRegression

#Generate stimulus response pairs
n_samples, n_features, n_voxels = 100, 15, 3
rng = np.random.RandomState(0)
stimulus = rng.randn(n_samples, n_features) 
response = rng.randn(n_samples,n_voxels) 

# Define feature model
fm = Identity()

# Train feature model
fm.fit(stimulus)

# Simulate feature model
feature = fm.predict(stimulus)

# Define response model
rm = KernelRidgeRegression()
    
# Train response model
rm.fit(feature, response)

# Simulate response model
response_hat = rm.predict(feature);

# Analyze encoding performance
R = np.diagonal(np.corrcoef(response,response_hat))
print 'encoding performance: ',np.mean(R),' (mean R)'

## Plot encoding performance Pyplot 
fig = plt.figure()
plt.plot(sorted(R, reverse=True), color='blue', lw=2)
fig.suptitle('encoding performance')
plt.xlabel('voxel')
yLab=plt.ylabel('R')
yLab.set_rotation(0)
plt.autoscale(enable=True, axis='both', tight=True)
plt.xscale('log')

## Plot encoding performance Bokeh (Nicer but may require $ pip install bokeh)
#from bokeh.plotting import figure, output_file, show
#output_file("encoding_performance.html", title="encoding performance")
#p = figure(title="econding performance", x_axis_label='voxel', y_axis_label='R',x_axis_type="log",x_range=[1, len(R)+1])
#p.line(np.arange(len(R))+1,sorted(R, reverse=True), line_width=2)
#show(p)