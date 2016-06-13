# Encoding model class

# Import required feature and response models ***
import numpy as np
from feature_models.identity import Identity
from feature_models.gabor_wavelet_pyramid import GaborWaveletPyramid
from feature_models.convolutional_neural_network import CNN
from response_models.kernel_ridge_regression import KernelRidgeRegression
from encoding_models.ring_buffer import RingBuffer

class EncodingModel(object):
    
    # Select feature and response models ***
    def __init__(self,fm,rm,buff_size = 1,mode = 'full_batch'):
        self.fm = fm       
        self.rm = rm
        self.buff_size=buff_size
        self.first=True
        self.mode=mode
        
    # Hemodynamic response model
    def ringBuff(self,feature):
        # Full batch mode
        if self.mode == 'full_batch':
            self.nexamples=feature.shape[0] # Number of training examples
            self.nfeatures=feature.shape[1]  # Number of features
            self.buffer_model=RingBuffer(self.nexamples,self.nfeatures,'float')# Create training ring buffer instance        
            for i in range(0,self.nexamples): # Append examples to buffer
                self.buffer_model.append(feature[i,:])
                
            # Get buffered features and indexes for responses 
            feature_buff,feature_idx=self.buffer_model.getRandom(self.nexamples-self.buff_size+1,self.buff_size)
            feature_buff=np.reshape(feature_buff,(feature_buff.shape[0],feature_buff.shape[1]*feature_buff.shape[2]))
                
            return feature_buff,feature_idx
            
        # Online mode
        if self.mode == 'online':
            if self.first == True: # Init online buffer
                self.buffer_model = RingBuffer(self.buff_size,self.nfeatures,'float')
                self.first = False
            self.buffer_model.append(feature)
            if self.buffer_model.full==False: # If buffer is not full
                return 0
            if self.buffer_model.full==True: 
                feature_buff=self.buffer_model.get()
                return np.reshape(feature_buff,(1,feature_buff.shape[0]*feature_buff.shape[1]))
                
   # Train encoding model
    def fit(self,stimulus,response):
        self.fm.fit() # Fit feature model       
        self.train_feature=self.fm.predict(stimulus) # Predict features
        
        if  self.buff_size>1: # Ring Buffer features
            self.train_feature,self.train_feature_idx = self.ringBuff(self.train_feature)       
            response=response[self.train_feature_idx[:,0]+self.buff_size-1,:] # Align response with features
        
        self.rm.fit(self.train_feature,response) # Fit response model

    # Predict encoding model
    def predict(self,stimulus):  
        # Predict features (get first set if feature maps > 1)
        self.val_feature=self.fm.predict(stimulus)

        if  self.buff_size>1: # Ring Buffer features
            if self.mode == 'full_batch':
                self.val_feature,self.val_feature_idx=self.ringBuff(self.val_feature)        
            if self.mode == 'online':
                self.val_feature=self.ringBuff(self.val_feature)       
            if self.buffer_model.full==False: # Return 0 if buffer not full
                return 0            
            if self.buffer_model.full==True:  # Predict using buffered features
                return self.rm.predict(self.val_feature); # Predict response
                
        if  self.buff_size==1: # Static predictions (no buffer)
            return self.rm.predict(self.val_feature);
        
    