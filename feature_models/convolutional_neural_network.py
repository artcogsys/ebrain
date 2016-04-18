
from feature_models.feature_model import FeatureModel
import theano as th
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import numpy as np
import skimage.transform

class CNN(FeatureModel):
    
    def __init__(self,model_path,channels=1,layer=5):  
        # path to vgg16_weights.h5
        self.model_path      = model_path
        # grayscale(channels=1)
        self.channels        = channels  
        #target layer activations
        self.layer           = layer
 
    def get_activations(self,model, layer, X_batch):
        get_activations = th.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
        activations = get_activations(X_batch) # same result as above
        return activations
    
    def VGG_16(self,weights_path=None):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
    
        if weights_path:
            model.load_weights(weights_path)
    
        return model
        
    def fit(self):
        pass
    
    def predict(self,X):
        #only grayscale
        s=X.shape
        X=np.reshape(X.T,(np.sqrt(X.shape[1]),np.sqrt(X.shape[1]),X.shape[0]))
        X = skimage.transform.resize(X, (224, 224)).astype(np.float32)*255
        X=np.reshape(X,(224,224,1,s[0]))
        X = np.repeat(X, 3,2)
        X[:,:,0,:] -= 103.939
        X[:,:,1,:] -= 116.779
        X[:,:,2,:] -= 123.68
        X = X.transpose((3,2,0,1))
    
        # Get CNN features
        model = self.VGG_16(self.model_path)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = self.get_activations(model,self.layer,np.expand_dims(X[1,:,:,:],axis=0))
        s2=out[0].shape[0]*out[0].shape[1]*out[0].shape[2]
        features=np.zeros((s[0],s2))
        bsize=100
        batches=s[0]//bsize
        last_bsize=s[0]%bsize
        for i in range(0,batches):
            features[i*bsize:i*bsize+bsize,:] = np.reshape(self.get_activations(model,self.layer,X[i*bsize:i*bsize+bsize,:,:,:]),(bsize,s2))
            print("MAKING CNN FEATURES: %d / %d" % ((i+1)*bsize,s[0]))  
        if last_bsize>0:    
            features[-last_bsize:,:] == np.reshape(self.get_activations(model,self.layer,np.reshape(X[-last_bsize:,:,:,:],(last_bsize,3,224,224))),(last_bsize,s2))
        return features

