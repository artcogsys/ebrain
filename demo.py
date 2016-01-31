# Encoding model demo 
# feature_model = Identity
# response_model = KernelRidgeRegression

# Import models
from feature_models.Identity import Identity
from response_models.KernelRidgeRegression import KernelRidgeRegression
import numpy as np
    
#Generate examples
n_samples, n_features = 10, 50
rng = np.random.RandomState(0)
response = rng.randn(n_samples) 
stimulus = rng.randn(n_samples, n_features) 

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

