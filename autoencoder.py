'''
Based on http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
Autoencoder that can be used in a sklearn-pipeline
'''

import math
import inspect
import keras
import tensorflow

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model

from sklearn.base import BaseEstimator, TransformerMixin

class Autoencoder(BaseEstimator, TransformerMixin):
    # Class of an autoencoder inheriting from BaseEstimator and TransformerMixin
    def __init__(self,
                 n_features=None,
                 n_epochs=None,
                 batch_size=None,
                 enc_dimension=None):
        '''
        Required parameters are
        :param n_features: number of features on the input layer
        :param n_epochs: number of epochs to train the model
        :param batch_size: batch size
        :param enc_dimension: encoding dimension
        '''

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)

        self.input_data = Input(shape=(self.n_features,))
        
        self.encoded = Dense(units=self.enc_dimension, activation="relu")(self.input_data)
        
        self.decoded = Dense(units=self.n_features, activation="sigmoid")(self.encoded)
        
        self.autoencoder = Model(self.input_data, self.decoded)
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                 loss="mean_squared_error")
           
    def fit(self,
            X,
            y=None):
        self.autoencoder.fit(X, X,
                             validation_split=0.3,
                             epochs=self.n_epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             verbose=1)

        self.encoder = Model(self.input_data, self.encoded)
        
        return self
    
    def transform(self,
                  X):
        return self.encoder.predict(X)